"""按**真实需求尺寸**做基准：720x1280 → 1440x2560。

720p 直接 ×4 得到 2880x5120，已经超过 2K，视频流里基本用不到。
真正的需求是 480p→1080p 这个区间，对应本项目就是：
    720x1280 先降 2 倍 → 360x640 → 4x 超分 → 1440x2560

尺寸小了 4 倍，之前的结论（合批没收益、分块更慢、VSR 显存爆炸）**都可能变**，
所以全部重测。

    python bench_target.py                 # 720x1280 → 1440x2560（2x 模式）
    python bench_target.py --direct 640 360  # 直接给 360x640 输入跑 4x
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

sys.path.insert(0, os.path.join(HERE, "engine"))
from config import RESR_WEIGHT, VSR_WEIGHT  # noqa: E402


def frames(n, h, w):
    rng = np.random.default_rng(7)
    small = rng.random((max(8, h // 16), max(8, w // 16), 3)).astype(np.float32)
    t = torch.from_numpy(small).permute(2, 0, 1)[None]
    big = torch.nn.functional.interpolate(t, size=(h, w), mode="bicubic",
                                          align_corners=False)[0]
    big = (big.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return np.stack([np.roll(big, (i % 3, (i % 3) * 2), axis=(0, 1)).copy()
                     for i in range(n)])


def bench(eng, fr, warm=2, iters=3):
    with torch.inference_mode():
        for _ in range(warm):
            eng.upscale_chunk(fr)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(iters):
            out = eng.upscale_chunk(fr)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
    return dt, out, torch.cuda.max_memory_allocated() / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, nargs=2, default=[1280, 720], help="输入 H W")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--out-scale", type=int, default=2)
    args = ap.parse_args()
    h, w = args.size
    oh, ow = (h // 2) * 4, (w // 2) * 4 if args.out_scale == 2 else (h * 4, w * 4)
    fr = frames(args.frames, h, w)
    print(f"目标尺寸基准：{args.frames} 帧 {w}x{h} → {ow}x{oh}（out_scale={args.out_scale}）")
    print(f"（对照：720p 直接 ×4 是 2880x5120，超出实际需求）\n")

    from realesrgan_engine import RealEsrganEngine
    print("  ── Real-ESRGAN 单帧 SR ──")
    print(f"  {'配置':<32}{'ms/帧':>9}{'峰值GiB':>9}")
    for tag, kw in [
        ("batch=1", dict(batch=1)),
        ("batch=2", dict(batch=2)),
        ("batch=4", dict(batch=4)),
        ("batch=8", dict(batch=8)),
        ("batch=16", dict(batch=16)),
        ("batch=4 tile=256", dict(batch=4, tile=256)),
        ("batch=8 tile=256", dict(batch=8, tile=256)),
    ]:
        try:
            eng = RealEsrganEngine(weight=RESR_WEIGHT, fp16=True, out_scale=args.out_scale,
                                   verbose=False, **kw)
            dt, out, peak = bench(eng, fr)
            print(f"  {tag:<32}{dt/args.frames*1000:>9.1f}{peak:>9.2f}", flush=True)
            del eng
            torch.cuda.empty_cache()
        except Exception as e:                              # noqa: BLE001
            print(f"  {tag:<32} 失败: {type(e).__name__}: {str(e)[:50]}", flush=True)
            torch.cuda.empty_cache()

    print("\n  ── BasicVSR++（对照）──")
    from vsr_engine import VsrEngine
    print(f"  {'配置':<32}{'ms/帧':>9}{'峰值GiB':>9}")
    for tag, kw in [
        ("chunk=4", dict(chunk=4)),
        ("chunk=8", dict(chunk=8)),
        ("chunk=16", dict(chunk=16)),
        ("chunk=32", dict(chunk=32)),
    ]:
        try:
            eng = VsrEngine(weight=VSR_WEIGHT, fp16=True, out_scale=args.out_scale,
                            verbose=False, **kw)
            dt, out, peak = bench(eng, fr)
            print(f"  {tag:<32}{dt/args.frames*1000:>9.1f}{peak:>9.2f}", flush=True)
            del eng
            torch.cuda.empty_cache()
        except Exception as e:                              # noqa: BLE001
            print(f"  {tag:<32} 失败: {type(e).__name__}: {str(e)[:50]}", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
