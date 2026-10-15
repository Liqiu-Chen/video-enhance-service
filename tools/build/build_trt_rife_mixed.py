"""RIFE 混合精度引擎：卷积走 fp16，只有 grid_sample（warp）走 fp32。

为什么要这么做
--------------
实测 RIFE 的两种引擎各有硬伤：

  TRT-fp16 : 2.47x 快，但只有 46.59 dB（最大差 63/255）
  TRT-fp32 : 62.10 dB（最大差 1）但比 PyTorch 还慢（0.88x）

根因：warp 把坐标归一化到 [-1,1]，高分辨率下相邻像素间距可能小于 fp16 在
1.0 附近的精度（~1e-3），采样位置被量化。其余全是普通卷积，用 fp16 完全够。

为什么不能直接设 layer.precision
---------------------------------
`layer.precision` / `set_output_type` 自 TRT 10.12 起**已废弃**
（官方改为 "strong typing"），强行设置会把精度传播搞坏 —— 实测输出直接 NaN。

正确做法（本脚本）
------------------
  1. 把整图转成 fp16（权重+张量），但保留 I/O 为 fp32；
  2. 在每个 GridSample 前后**显式插入 Cast**：输入升 fp32、输出降回 fp16；
  3. 用 `NetworkDefinitionCreationFlag.STRONGLY_TYPED` 建网络，
     并**不加** FP16 flag —— 这样 TRT 严格按图里的类型执行，不自行改精度。

    python build_trt_rife_mixed.py --size 1672 960
"""
from __future__ import annotations

import argparse
import os
import sys

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


def make_mixed_onnx(src_onnx: str, dst_onnx: str, verbose: bool = True) -> str:
    """fp32 ONNX → 全 fp16 + GridSample 段 fp32 的混合图。"""
    import onnx
    from onnx import helper, TensorProto
    from onnxruntime.transformers.float16 import convert_float_to_float16

    m = onnx.load(src_onnx)
    # 先整体转 fp16（keep_io_types 保留输入输出为 fp32，省得调用方还要改）
    m16 = convert_float_to_float16(m, keep_io_types=True)

    # 找出所有 GridSample，在它前面插 Cast→fp32，后面插 Cast→fp16
    g = m16.graph
    nodes = list(g.node)
    inserts = []                       # (index, [nodes])
    n_casts = 0
    for i, node in enumerate(nodes):
        # GridSample：warp 的坐标归一化到 [-1,1]，fp16 会把采样位置量化。
        # Resize  ：上采样光流/掩码时也有半像素对齐与插值精度问题，
        #           实测一起拉回 fp32 后 PSNR 还能再涨一截。
        # 其余（Conv/PRelu/Add 等）留在 fp16，速度几乎不受影响。
        if node.op_type not in ("GridSample", "Resize"):
            continue
        pre = []
        # 输入侧：为每个数据输入插一个 Cast→fp32。
        # ⚠️ 必须跳过**可选的空输入**：ONNX 的 Resize 有 roi/scales/sizes
        # 三个可选输入，不填时就是空字符串 ""。给空名插 Cast 会直接报
        # `INVALID_NODE: Invalid Node`。这类输入本来就是常量（形状/尺度），
        # 也不该参与精度转换。
        for j, inp in enumerate(node.input):
            if not inp:
                continue
            casted = f"gs_{i}_in{j}_fp32"
            pre.append(helper.make_node(
                "Cast", [inp], [casted], to=TensorProto.FLOAT,
                name=f"Cast_fp32_{i}_{j}"))
            node.input[j] = casted
        # 输出侧：★ 不能直接改 node.output[0] 的名字 ★
        # 那样下游引用旧名的节点会全部断链（第一次就是这么挂的）。
        # 正确做法：让 GridSample 输出到一个**中间名**，再用 Cast 把它转回
        # **原来的名字** —— 这样下游完全不用动，图的拓扑也保持完整。
        orig_out = node.output[0]
        mid = f"gs_{i}_mid_fp32"
        node.output[0] = mid
        post = helper.make_node(
            "Cast", [mid], [orig_out], to=TensorProto.FLOAT16,
            name=f"Cast_fp16_{i}")
        inserts.append((i, pre + [post]))
        n_casts += 1

    # 从后往前插，避免索引错位
    for idx, new_nodes in sorted(inserts, key=lambda x: -x[0]):
        for k, nd in enumerate(new_nodes):
            g.node.insert(idx + k, nd)

    if verbose:
        print(f"  已把 {n_casts} 个 GridSample 段拉回 fp32（插入 "
              f"{sum(len(x[1]) for x in inserts)} 个 Cast 节点）")
        from collections import Counter
        c = Counter(nd.op_type for nd in g.node)
        print(f"  算子统计: " + ", ".join(f"{k}×{v}" for k, v in c.most_common(8)))
    onnx.save(m16, dst_onnx)
    return dst_onnx


def build_strongly_typed(onnx_path: str, engine_path: str,
                         workspace_gb: float = 6.0, verbose: bool = True):
    """用 STRONGLY_TYPED 建网络：严格按图里的类型，不自动降精度。"""
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) | \
            (1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("  解析错误:", parser.get_error(i))
            raise RuntimeError("ONNX 解析失败")
    if verbose:
        print(f"  网络: {network.num_layers} 层（strongly typed，不自动降精度）")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                 int(workspace_gb * 2**30))
    # 注意：**不加** BuilderFlag.FP16 —— 类型由图决定
    config.clear_flag(trt.BuilderFlag.TF32)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("引擎构建失败")
    with open(engine_path, "wb") as f:
        f.write(serialized)
    if verbose:
        print(f"  引擎 {os.path.getsize(engine_path)/2**20:.1f} MB → "
              f"{os.path.basename(engine_path)}")
    return engine_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, nargs=2, default=[1672, 960], help="H W")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    h = (args.size[0] + 31) // 32 * 32
    w = (args.size[1] + 31) // 32 * 32

    fp32_onnx = os.path.join(CACHE, f"rife_{h}x{w}.onnx")
    mix_onnx = os.path.join(CACHE, f"rife_{h}x{w}_mixed.onnx")
    eng = os.path.join(CACHE, f"rife_{h}x{w}_strongmix.trt")
    if not os.path.exists(fp32_onnx):
        from rife_engine import export_rife_onnx
        export_rife_onnx(RIFE_WEIGHT, fp32_onnx, (h, w), fp16=False)

    print(f"RIFE 混合精度（strong typing）  {w}x{h}\n")
    if not os.path.exists(mix_onnx):
        make_mixed_onnx(fp32_onnx, mix_onnx)
    else:
        print(f"  复用已有混合 ONNX")
    if not os.path.exists(eng):
        build_strongly_typed(mix_onnx, eng)

    # ---------- 基准 ----------
    from rife_engine import RifeEngine
    from build_trt_rife import TrtRife, psnr
    from bench_trt_vsr import make_frames

    fr = make_frames(h, w, 2)
    a = torch.from_numpy(fr[0:1]).cuda().permute(0, 3, 1, 2)
    b = torch.from_numpy(fr[1:2]).cuda().permute(0, 3, 1, 2)
    a32, b32 = a.float().div_(255.0), b.float().div_(255.0)
    a16, b16 = a32.half(), b32.half()

    ref32 = RifeEngine(weight=RIFE_WEIGHT, fp16=False, batch=1, verbose=False)
    ref32.build(h, w)
    with torch.inference_mode():
        y32 = ref32.infer(a32, b32)          # fp32 真值
        del ref32
        import gc; gc.collect(); torch.cuda.empty_cache()

    t = TrtRife(eng)
    x = torch.cat([a16, b16], 1)
    with torch.inference_mode():
        for _ in range(5):
            y = t(x)
        torch.cuda.synchronize()
        import time
        t0 = time.perf_counter()
        for _ in range(args.iters):
            y = t(x)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / args.iters * 1000

    yf = y.to(torch.float32)
    if yf.shape != y32.shape:
        print(f"  ⚠️ 形状不一致 {tuple(yf.shape)} vs {tuple(y32.shape)}")
        return 1
    p = psnr(y32, yf)
    md = (y32.float() - yf).abs().max().item()
    print(f"\n  {'实现':<26}{'ms/对':>10}{'PSNR(vs fp32真值)':>20}{'最大差':>10}")
    print(f"  {'TRT-fp16（原）':<26}{'10.7':>10}{'46.59':>19}dB{'63':>10}")
    print(f"  {'TRT-fp32':<26}{'30.2':>10}{'62.10':>19}dB{'1':>10}")
    print(f"  {'TRT-strongmix（本方案）':<26}{ms:>10.1f}{p:>19.2f}dB{md:>10.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
