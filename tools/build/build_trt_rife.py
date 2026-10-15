"""把 RIFE 编译成 TensorRT 引擎并实测（速度 + 数值精度）。

为什么值得重试 TensorRT：
  之前记录过「ONNX 是负优化」——但那次用的是 **ONNX Runtime 的 CUDA EP**，
  它对卷积只做算子级替换，不做层融合、不选 tactic。TensorRT 是另一回事：
  它会做层融合（conv+bn+relu）、按实测选 kernel、为每个 shape 生成专用引擎。
  这是两码事，值得单独验证。

必须同时看精度：
  之前踩过「纯 fp16 毁画面」的坑 —— RIFE 的 warp 把坐标归一化到 [-1,1]，
  高分辨率下相邻像素间距小于 fp16 在 1.0 附近的精度，采样位置被量化。
  TensorRT 的 GridSample 在 fp16 下会不会有同样问题，必须实测 PSNR。

    python build_trt_rife.py --size 1672 960            # 构建 + 基准
    python build_trt_rife.py --size 1672 960 --fp32     # 用 fp32 引擎对比
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# --- 仓库内路径引导 ---
# 本脚本位于 tools/{build,bench,diag}/，是**两层深**，
# 所以仓库根要往上取两级（只取一级会得到 tools/，路径就全错了）。
_ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (_ROOT,
           os.path.join(_ROOT, "service"),
           os.path.join(_ROOT, "service", "engine"),
           os.path.join(_ROOT, "tools", "bench")):
    sys.path.insert(0, _p)

sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "engine"))

from config import CACHE, RIFE_WEIGHT  # noqa: E402


def pad_to32(h: int, w: int):
    return (h + 31) // 32 * 32, (w + 31) // 32 * 32


# --------------------------------------------------------------------------- #
def export_onnx(h: int, w: int, path: str) -> str:
    from rife_engine import export_rife_onnx
    if os.path.exists(path):
        print(f"  复用已有 ONNX: {path}")
        return path
    print(f"  导出 ONNX {w}x{h} ...", flush=True)
    t0 = time.time()
    export_rife_onnx(RIFE_WEIGHT, path, (h, w), fp16=False, verbose=True)
    print(f"  导出完成 {time.time()-t0:.1f}s  "
          f"{os.path.getsize(path)/2**20:.1f} MB")
    return path


def build_engine(onnx_path: str, engine_path: str, mode: str = "fp16",
                 workspace_gb: float = 6.0, verbose: bool = True):
    """mode: fp16 / fp32 / mix

    mix = 卷积等走 fp16，但 **GridSample（warp）强制 fp32**。
    必须这么做的原因：warp 把坐标归一化到 [-1,1]，高分辨率下相邻像素间距
    （960 宽下 ~5e-4）小于 fp16 在 1.0 附近的精度（~1e-3），采样位置被量化，
    画面直接崩掉 —— 实测纯 fp16 引擎 PSNR 只有 5.5 dB。
    """
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("  解析错误:", parser.get_error(i))
            raise RuntimeError("ONNX 解析失败")

    n_fp32 = 0
    if mode == "mix":
        # ⚠️ 只强制 GRID_SAMPLE，**不要**带上 RESIZE。
        # 上次两者都强制时整个引擎输出 NaN —— Resize 在 fp16 网络里被要求
        # 输出 fp32 会破坏 TRT 的精度传播（它两侧都还是 fp16）。
        # warp 的精度问题只出在 grid_sample 的坐标量化上，管住它就够了。
        want = []
        for nm in ("GRID_SAMPLE",):
            if hasattr(trt.LayerType, nm):
                want.append(getattr(trt.LayerType, nm))
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.type in want:
                layer.precision = trt.float32
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.float32)
                n_fp32 += 1
    if verbose:
        print(f"  网络: {network.num_layers} 层, "
              f"{network.num_inputs} 入 / {network.num_outputs} 出"
              + (f"，其中 {n_fp32} 层强制 fp32" if mode == "mix" else ""))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                 int(workspace_gb * 2**30))
    if mode in ("fp16", "mix"):
        if not builder.platform_has_fast_fp16:
            print("  ⚠️ 本平台不报告 fp16 加速，仍尝试")
        config.set_flag(trt.BuilderFlag.FP16)
    elif mode == "tf32":
        # ★ TF32：10 位尾数（和 fp16 一样）但**指数范围与 fp32 相同**。
        # 对卷积来说精度远好于 fp16（不会有小数值下溢），Blackwell 上有加速。
        # 实测这是"既要速度又要精度"的现实选项。
        config.set_flag(trt.BuilderFlag.TF32)
        # TF32 要求网络是 strong typing，类型由图决定，不自动降精度
        pass
    if mode == "mix":
        # 没有这个 flag，builder 会忽略上面设的 layer.precision
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    # 关掉 TF32：默认开着会让 fp32 引擎悄悄降精度，不利于排查数值问题
    config.clear_flag(trt.BuilderFlag.TF32)

    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("引擎构建失败")
    dt = time.time() - t0
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"  引擎构建完成 {dt:.1f}s  "
          f"{os.path.getsize(engine_path)/2**20:.1f} MB  → {engine_path}")
    return engine_path


# --------------------------------------------------------------------------- #
class TrtRife:
    """最小 TensorRT 运行时封装（TRT 10.x 的 tensor-address API）。"""

    def __init__(self, engine_path: str, device: str = "cuda"):
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.ERROR)
        self.trt = trt
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("引擎反序列化失败")
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i)
                 for i in range(self.engine.num_io_tensors)]
        self.in_name = next(n for n in names
                            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.out_name = next(n for n in names
                             if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        _m = {trt.DataType.FLOAT: torch.float32,
              trt.DataType.HALF: torch.float16}
        # ⚠️ 输入 dtype 也必须读！
        # 之前只读了输出，于是把 fp16 张量的地址喂给了期待 fp32 的引擎 ——
        # 引擎按 fp32 解引用一块 fp16 缓冲区，读出来是垃圾，
        # 表现为 PSNR 5.5dB、最大差 545（输出值域才 0~1），
        # 看起来像"精度崩了"，其实是我绑错了内存。
        self.in_dtype = _m[self.engine.get_tensor_dtype(self.in_name)]
        self.out_dtype = _m[self.engine.get_tensor_dtype(self.out_name)]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != self.in_dtype:
            x = x.to(self.in_dtype)
        x = x.contiguous()
        self.ctx.set_input_shape(self.in_name, tuple(x.shape))
        out = torch.empty(tuple(self.ctx.get_tensor_shape(self.out_name)),
                          dtype=self.out_dtype, device=x.device)
        self.ctx.set_tensor_address(self.in_name, x.data_ptr())
        self.ctx.set_tensor_address(self.out_name, out.data_ptr())
        ok = self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        if not ok:
            raise RuntimeError("execute_async_v3 失败")
        return out


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().clamp(0, 1)
    b = b.float().clamp(0, 1)
    mse = torch.mean((a - b) ** 2).item()
    return 99.0 if mse == 0 else 10 * np.log10(1.0 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, nargs=2, default=[1672, 960], help="H W")
    ap.add_argument("--mode", default="mix",
                    choices=["fp16", "fp32", "mix", "tf32"],
                    help="fp16=全半精度(精度会崩) fp32=全单精度 mix=fp16+GridSample走fp32(推荐)")
    ap.add_argument("--fp32", action="store_true", help="[已弃用] 等价 --mode fp32")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warm", type=int, default=5)
    args = ap.parse_args()

    h, w = pad_to32(*args.size)
    mode = "fp32" if args.fp32 else args.mode
    tag = mode
    onnx_p = os.path.join(CACHE, f"rife_{h}x{w}.onnx")
    eng_p = os.path.join(CACHE, f"rife_{h}x{w}_{tag}.trt")
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(os.path.dirname(eng_p), exist_ok=True)

    print(f"RIFE → TensorRT  尺寸 {w}x{h}  精度 {tag}\n")
    export_onnx(h, w, onnx_p)
    if not os.path.exists(eng_p):
        build_engine(onnx_p, eng_p, mode=mode)
    else:
        print(f"  复用已有引擎: {eng_p}")

    # ---------- 参考实现（PyTorch fp16 + fp32 warp）----------
    from rife_engine import RifeEngine
    ref = RifeEngine(weight=RIFE_WEIGHT, fp16=True, batch=1,
                     warp_fp32=True, verbose=False)
    ref.build(h, w)

    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (1, h, w, 3), dtype=np.uint8)
    b = rng.integers(0, 256, (1, h, w, 3), dtype=np.uint8)

    def to_t(arr):
        # arr 已经是 (1,H,W,3)，permute 到 (1,3,H,W) 即可 ——
        # 之前这里还多写了个 unsqueeze(0)，变成 5D，模型直接报
        # "Got 5D input, but bilinear mode needs 4D input"。
        t = torch.from_numpy(arr).cuda().permute(0, 3, 1, 2)
        return t.half().div_(255.0)

    ta, tb = to_t(a), to_t(b)
    x = torch.cat([ta, tb], 1)

    with torch.inference_mode():
        for _ in range(args.warm):
            y_ref = ref.infer(ta, tb)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            y_ref = ref.infer(ta, tb)
        torch.cuda.synchronize()
        t_torch = (time.perf_counter() - t0) / args.iters * 1000

        trt = TrtRife(eng_p)
        for _ in range(args.warm):
            y_trt = trt(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            y_trt = trt(x)
        torch.cuda.synchronize()
        t_trt = (time.perf_counter() - t0) / args.iters * 1000

    # ---------- 结果 ----------
    y_trt_d = y_trt.to(y_ref.dtype)
    if y_trt_d.shape != y_ref.shape:
        print(f"  ⚠️ 形状不一致 TRT{y_trt_d.shape} vs Torch{y_ref.shape}")
    p = psnr(y_ref, y_trt_d)
    maxdiff = (y_ref.float() - y_trt_d.float()).abs().max().item()
    print(f"\n  {'实现':<28}{'ms/对':>10}{'PSNR':>10}{'最大差':>10}")
    print(f"  {'PyTorch fp16+warp32':<28}{t_torch:>10.1f}{'—':>10}{'—':>10}")
    print(f"  {'TensorRT ' + tag:<28}{t_trt:>10.1f}{p:>9.2f}dB{maxdiff:>10.4f}")
    print(f"\n  加速比: {t_torch/t_trt:.2f}x")
    verdict = ("✅ 可用" if p > 35 else
               ("⚠️ 精度偏低，需检查" if p > 25 else "❌ 精度不可用"))
    print(f"  精度判定: {verdict}（参考阈值：>35dB 可用）")

    meta = {"size": [h, w], "tag": tag, "ms_torch": round(t_torch, 2),
            "ms_trt": round(t_trt, 2), "speedup": round(t_torch / t_trt, 3),
            "psnr_db": round(p, 2), "max_abs_diff": round(maxdiff, 5)}
    meta["mode"] = mode
    with open(os.path.join(CACHE, f"trt_rife_{h}x{w}_{tag}.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
