"""验证：把 BasicVSR++ 里的可变形卷积换成「原生算子拆解」后，整网输出是否一致。

这是通往 TensorRT 的关键一步 —— 拆解后的图只含 grid_sample / conv / mul，
全部是 TensorRT 原生算子，不需要编译任何 plugin。

    python vsr_decompose_test.py --size 836 480 --t 4
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

from config import VSR_WEIGHT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, nargs=2, default=[836, 480], help="输入 H W")
    ap.add_argument("--t", type=int, default=4, help="时序长度")
    ap.add_argument("--scale", type=int, default=4)
    args = ap.parse_args()
    h, w = args.size

    # TF32 会让两边的卷积精度不同，比较时必须关掉，否则比出的是噪声不是差异
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    from vsr_engine import VsrEngine
    rng = np.random.default_rng(0)
    # 用真实感素材：低频随机图（纯噪声对这类网络没有意义）
    small = rng.random((h // 16, w // 16, 3)).astype(np.float32)
    t = torch.from_numpy(small).permute(2, 0, 1)[None]
    big = torch.nn.functional.interpolate(t, size=(h, w), mode="bicubic",
                                          align_corners=False)[0]
    big = (big.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    frames = np.stack([np.roll(big, (i, i * 2), axis=(0, 1)).copy()
                       for i in range(args.t)])

    print(f"输入 {args.t} 帧 {w}x{h}  out_scale={args.scale}\n")

    # ---- 1. 原实现 ----
    eng = VsrEngine(weight=VSR_WEIGHT, fp16=False, chunk=args.t,
                    out_scale=args.scale, verbose=False)
    with torch.inference_mode():
        ref = eng.upscale_chunk(frames)
    print(f"  原实现输出 {ref.shape}  峰值 {ref.mean():.4f}")

    # ---- 2. 换成拆解实现 ----
    import vsr.basicvsr_plusplus_net as bnet
    from deform_decompose import deform_conv2d_decomposed

    calls = [0]

    def patched(x, offset, mask, weight, bias=None, stride=1, padding=0,
                dilation=1, groups=1, deform_groups=1):
        calls[0] += 1
        if groups != 1:
            raise NotImplementedError("拆解版暂不支持 groups != 1")
        return deform_conv2d_decomposed(
            x, offset, mask, weight, bias,
            stride=_pair(stride), padding=_pair(padding),
            dilation=_pair(dilation), deform_groups=deform_groups)

    def _pair(v):
        return (v, v) if isinstance(v, int) else tuple(v)

    bnet.modulated_deform_conv2d = patched
    eng2 = VsrEngine(weight=VSR_WEIGHT, fp16=False, chunk=args.t,
                     out_scale=args.scale, verbose=False)
    with torch.inference_mode():
        got = eng2.upscale_chunk(frames)
    print(f"  拆解版输出 {got.shape}  峰值 {got.mean():.4f}   "
          f"（替换了 {calls[0]} 次可变形卷积）")

    # ---- 3. 比对 ----
    a = ref.astype(np.float64)
    b = got.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    psnr = 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    md = np.abs(a - b).max()
    print(f"\n  PSNR {psnr:.2f} dB   最大差 {md:.0f}/255   平均差 "
          f"{np.abs(a-b).mean():.3f}")
    if psnr > 45:
        print("  ✅ 整网输出一致 —— 可以走 ONNX → TensorRT 这条路")
    elif psnr > 35:
        print("  ⚠️ 基本一致，但有可见差异，需要继续查")
    else:
        print("  ❌ 差异过大")


if __name__ == "__main__":
    main()
