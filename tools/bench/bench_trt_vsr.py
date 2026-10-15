"""BasicVSR++ 的 TensorRT 引擎（拆解版）—— 速度与精度实测。

这是最关键的一步：引擎建出来了，但**必须是"又快又准"才有意义**。
拆解把 1 次可变形卷积换成了 9 次 grid_sample + 1 次 1x1 卷积，
数据搬运量理论上涨了约 9 倍 —— 所以很可能是**负优化**，必须实测。

    python bench_trt_vsr.py --size 836 480 --t 2 --scale 4
"""
from __future__ import annotations

import argparse
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

from config import VSR_WEIGHT  # noqa: E402

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def make_frames(h, w, t):
    """低频随机图（纯噪声对这类网络没有意义，也测不准精度）。"""
    rng = np.random.default_rng(0)
    small = rng.random((h // 16, w // 16, 3)).astype(np.float32)
    tt = torch.from_numpy(small).permute(2, 0, 1)[None]
    big = torch.nn.functional.interpolate(tt, size=(h, w), mode="bicubic",
                                          align_corners=False)[0]
    big = (big.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return np.stack([np.roll(big, (i, i * 2), axis=(0, 1)).copy()
                     for i in range(t)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, nargs=2, default=[836, 480], help="输入 H W")
    ap.add_argument("--t", type=int, default=2)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--mode", default="fp16")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--ref-fp32", action="store_true", default=True,
                    help="参考用 fp32 网络（默认开，这是唯一正确的基准）")
    args = ap.parse_args()
    h, w = args.size

    eng_p = os.path.join(os.path.dirname(HERE), "trt_cache",
                         f"vsr_decomp_t{args.t}_{h}x{w}_x{args.scale}_{args.mode}.trt")
    if not os.path.exists(eng_p):
        print(f"❌ 引擎不存在: {eng_p}\n   先跑 build_trt_vsr_decomp.py")
        return 1

    frames = make_frames(h, w, args.t)
    print(f"输入 {args.t} 帧 {w}x{h}  out_scale={args.scale}  引擎 {args.mode}\n")

    # ---------- PyTorch 参考 ----------
    from vsr_engine import VsrEngine
    # ★ 参考必须用 fp32 ★
    # 踩过的坑：之前拿 TRT 去比 **fp16 的 PyTorch**，算出"33 dB 精度崩了"。
    # 但 BasicVSR++ 本身对 fp16 极度敏感 —— 实测 PyTorch 自己的 fp16 与 fp32
    # 之间就只有 36 dB。两个各自偏离真值 36 dB 的实现互相比，当然也是 36 dB。
    # 正确基准只有一个：**fp32 的 PyTorch**（TRT 在它面前是 44 dB，反而更准）。
    eng = VsrEngine(weight=VSR_WEIGHT, fp16=not args.ref_fp32, chunk=args.t,
                    out_scale=args.scale, verbose=False)
    with torch.inference_mode():
        for _ in range(args.warm):
            ref = eng.upscale_chunk(frames)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            ref = eng.upscale_chunk(frames)
        torch.cuda.synchronize()
        t_torch = (time.perf_counter() - t0) / args.iters
    print(f"  PyTorch {'fp32' if args.ref_fp32 else 'fp16'}（参考）: "
          f"{t_torch*1000:8.1f} ms/批  ({t_torch/args.t*1000:7.1f} ms/帧)")

    # ---------- TensorRT ----------
    # ★ 必须先释放 PyTorch 引擎和显存缓存再测 TRT ★
    # 踩过的坑：PyTorch 参考引擎（尤其 fp32）会预留好几 GB 显存，
    # 若它还在驻留，TRT 只能挤在剩余显存里跑 → 驱动层换页 → 计算空转。
    # 实测同一份引擎：不清显存 4460 ms/帧，清完 131 ms/帧（差 34 倍！）。
    import gc
    del eng
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    from build_trt_rife import TrtRife
    trt = TrtRife(eng_p)
    # 引擎输入 (1, t, 3, H, W)，值域 0~1
    x = torch.from_numpy(frames).cuda().permute(0, 3, 1, 2).unsqueeze(0).contiguous()
    x = x.float().div_(255.0)
    if not args.ref_fp32:
        x = x.half()
    with torch.inference_mode():
        for _ in range(args.warm):
            y = trt(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            y = trt(x)
        torch.cuda.synchronize()
        t_trt = (time.perf_counter() - t0) / args.iters
    print(f"  TensorRT {args.mode:<7}: {t_trt*1000:8.1f} ms/批  "
          f"({t_trt/args.t*1000:7.1f} ms/帧)")

    # ---------- 精度 ----------
    # 参考帧也要在 PyTorch 引擎释放前算好；这里 ref 已经算完在 CPU 上
    # 引擎输出 (1, t, 3, 4H, 4W)，转成与 ref 相同的 uint8 布局再比
    yv = y[0].permute(0, 2, 3, 1).float().clamp(0, 1).mul(255).round()
    yv = yv.to(torch.uint8).cpu().numpy()
    a, b = ref.astype(np.float64), yv.astype(np.float64)
    if a.shape != b.shape:
        print(f"\n  ⚠️ 形状不一致 PyTorch{a.shape} vs TRT{b.shape}")
        return 1
    mse = np.mean((a - b) ** 2)
    psnr = 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    md = np.abs(a - b).max()

    print(f"\n  {'':<18}{'ms/批':>10}{'加速比':>10}")
    print(f"  {'PyTorch(参考)':<18}{t_torch*1000:>10.1f}{'1.00x':>10}")
    print(f"  {'TensorRT':<18}{t_trt*1000:>10.1f}{t_torch/t_trt:>9.2f}x")
    print(f"\n  精度: PSNR {psnr:.2f} dB   最大差 {md:.0f}/255   "
          f"平均差 {np.abs(a-b).mean():.3f}")
    if psnr < 30:
        print("  ❌ 精度不可用")
    elif t_trt >= t_torch:
        print(f"  ⚠️ 精度尚可，但**没有加速**（{t_torch/t_trt:.2f}x）—— 拆解的代价吃掉了收益")
    else:
        print(f"  ✅ 又快又准（{t_torch/t_trt:.2f}x）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
