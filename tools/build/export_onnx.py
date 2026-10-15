"""把 RIFE 导出成固定 shape 的 ONNX（附实测结论：**本机上是负优化**）。

    python export_onnx.py --size 5120 2880            # 导出 fp16 + fp32 两份
    python export_onnx.py --size 5120 2880 --bench    # 导出并实测速度

结论先写在前面：在 2880×5120 上 ONNX Runtime(CUDA EP) 要 1646 ms/对，
而 PyTorch fp16+fp32warp 只要 288 ms/对 —— **慢 3.3 倍**，而且 fp16 化之后
精度也崩（PSNR 40 dB）。所以正式链路不用 ONNX，这个脚本保留是为了可复现地
证明这个结论，以及供别的硬件环境复核。

BasicVSR++ 无法导出：它依赖 `torchvision.ops.deform_conv2d`（可变形卷积），
ONNX 没有对应的标准算子。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

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

sys.path.insert(0, os.path.join(HERE, "engine"))

from config import CACHE, RIFE_WEIGHT  # noqa: E402
from rife_engine import export_rife_onnx  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, nargs=2, default=[5120, 2880], help="H W")
    ap.add_argument("--fp16", action="store_true", help="同时导出 fp16 版本")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()
    h, w = args.size
    os.makedirs(CACHE, exist_ok=True)

    out32 = os.path.join(CACHE, f"rife_{h}x{w}.onnx")
    print("导出 fp32 ...")
    export_rife_onnx(RIFE_WEIGHT, out32, (h, w), fp16=False)
    for p in (out32,):
        print(f"  {p}  {os.path.getsize(p)/1048576:.1f} MB")
    if args.fp16:
        out16 = os.path.join(CACHE, f"rife_{h}x{w}_fp16.onnx")
        print("导出 fp16 ...")
        export_rife_onnx(RIFE_WEIGHT, out16, (h, w), fp16=True)
        print(f"  {out16}  {os.path.getsize(out16)/1048576:.1f} MB")

    if args.bench:
        import torch
        from rife_engine import RifeEngine
        dev = "cuda"
        a = torch.rand(1, 3, h, w, device=dev)
        b = torch.roll(a, shifts=(5, 9), dims=(2, 3))
        for name, kw in (("torch fp32", dict(backend="torch", fp16=False)),
                         ("torch fp16+fp32warp", dict(backend="torch", fp16=True,
                                                      warp_fp32=True)),
                         ("onnx fp32", dict(backend="onnx", fp16=False)),
                         ("onnx fp16", dict(backend="onnx", fp16=True))):
            try:
                e = RifeEngine(weight=RIFE_WEIGHT, verbose=False, **kw)
                e.build(h, w)
                ins = (a.to(e.dtype), b.to(e.dtype))
                for _ in range(2):
                    e.infer(*ins)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(args.iters):
                    e.infer(*ins)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / args.iters
                print(f"  {name:24s} {dt*1000:8.1f} ms/对")
                del e
                torch.cuda.empty_cache()
            except Exception as ex:                            # noqa: BLE001
                print(f"  {name:24s} 失败: {type(ex).__name__}: {str(ex)[:90]}")


if __name__ == "__main__":
    main()
