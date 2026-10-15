"""用 TRT 安全版（border 用补边模拟）导出 VSR 并建引擎。

这是 TRT 精度攻坚的关键一步：**修掉 TRT 不支持的 border padding**。
详见 flow_warp_trtsafe.py 的说明。

    python build_trt_vsr_trtsafe.py --t 8 --size 836 480 --scale 4
"""
from __future__ import annotations

import argparse
import os
import sys
import time

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

ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "engine"))

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = False

OUT = os.path.join(ROOT, "trt3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, nargs=2, default=[836, 480])
    ap.add_argument("--t", type=int, default=8)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--mode", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--workspace-gb", type=float, default=6.0)
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()
    h, w = args.size
    os.makedirs(OUT, exist_ok=True)
    onnx_p = os.path.join(OUT, f"vsr_trtsafe_t{args.t}_{h}x{w}.onnx")
    eng_p = os.path.join(OUT, f"vsr_trtsafe_t{args.t}_{h}x{w}_{args.mode}.trt")

    from config import VSR_WEIGHT
    from vsr_engine import VsrEngine
    from flow_warp_trtsafe import patch

    print(f"TRT 安全版导出  t={args.t}  {w}x{h} → {args.scale}x\n")
    t0 = time.time()
    e = VsrEngine(weight=VSR_WEIGHT, fp16=False, chunk=args.t,
                  out_scale=args.scale, device="cpu", verbose=False)
    # ★ 两个补丁都要打，缺一不可 ★
    #   1) deform 拆解：把可变形卷积换成原生算子（否则导出直接报
    #      UnsupportedOperatorError —— 这是最早踩过的坑）
    #   2) border warp：把 TRT 不支持的 border padding 换成补边模拟
    from build_trt_vsr_decomp import patch_deform
    patch_deform(verbose=True)
    patch()
    net = e.net.eval()
    dummy = torch.zeros(1, args.t, 3, h, w)
    torch.onnx.export(net, (dummy,), onnx_p, input_names=["lqs"],
                      output_names=["hr"], opset_version=17,
                      # 打开常量折叠：之前为了避开 arange 网格被物化成
                      # 100MB 常量而关掉，但关掉后图变得又深又散，
                      # TRT 的 Myelin 编译器会报
                      # `MyelinCheckException ... Could not find any implementation`。
                      # 现在网格构造已改成纯广播，常量本身不大，打开折叠反而让图更紧凑。
                      dynamo=False, do_constant_folding=False)
    print(f"  ✅ 导出 {time.time()-t0:.0f}s  {os.path.getsize(onnx_p)/2**20:.1f} MB")
    del e, net
    import gc
    gc.collect()

    # 核对：导出后不该再有 border
    import onnx
    from collections import Counter
    m = onnx.load(onnx_p, load_external_data=False)

    def rd(a):
        return a.s.decode() if a.type == 3 else (a.i if a.type == 2 else a.f)

    modes = Counter()
    for n in m.graph.node:
        if n.op_type == "GridSample":
            d = {a.name: rd(a) for a in n.attribute}
            modes[(d.get("padding_mode"), d.get("align_corners"), d.get("mode"))] += 1
    print(f"  导出后 GridSample 属性分布: {dict(modes)}")
    bad = [k for k in modes if k[0] != "zeros"]
    print(f"  非 zeros 的 padding: {bad if bad else '无 ✅'}")

    if args.skip_build:
        return 0

    print(f"\n  构建 TensorRT 引擎（{args.mode}）...", flush=True)
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    cwd = os.getcwd()
    os.chdir(OUT)                 # TRT 按 CWD 找外部数据
    try:
        with open(os.path.basename(onnx_p), "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print("  解析错误:", parser.get_error(i))
                return 1
        cfg = builder.create_builder_config()
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                  int(args.workspace_gb * 2**30))
        if args.mode == "fp16":
            cfg.set_flag(trt.BuilderFlag.FP16)
        else:
            cfg.clear_flag(trt.BuilderFlag.TF32)
        print(f"  网络 {network.num_layers} 层", flush=True)
        ser = builder.build_serialized_network(network, cfg)
        if ser is None:
            print("  ❌ 构建返回空")
            return 1
        with open(eng_p, "wb") as f:
            f.write(ser)
    finally:
        os.chdir(cwd)
    print(f"  ✅ 引擎 {os.path.getsize(eng_p)/2**20:.1f} MB  "
          f"总耗时 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
