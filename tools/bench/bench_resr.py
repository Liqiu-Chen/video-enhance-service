"""Real-ESRGAN 单帧超分的批量/分块基准。

    python bench_resr.py --size 1280 720 --frames 8

关注三件事：ms/帧、显存预留（**必须看 reserved**，不是 allocated）、以及
与原来 BasicVSR++ 的对比。
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
from config import VSR_WEIGHT  # noqa: E402


def real_frames(n, h, w):
    rng = np.random.default_rng(7)
    small = rng.random((max(8, h // 16), max(8, w // 16), 3)).astype(np.float32)
    t = torch.from_numpy(small).permute(2, 0, 1)[None]
    big = torch.nn.functional.interpolate(t, size=(h, w), mode="bicubic",
                                          align_corners=False)[0]
    big = (big.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return np.stack([np.roll(big, (i % 3, (i % 3) * 2), axis=(0, 1)).copy()
                     for i in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, nargs=2, default=[1280, 720])
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--iters", type=int, default=2)
    args = ap.parse_args()
    h, w = args.size
    fr = real_frames(args.frames, h, w)
    print(f"Real-ESRGAN 基准：{args.frames} 帧 {w}x{h} → {w*4}x{h*4}")
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    print(f"  {'配置':<34}{'ms/帧':>9}{'预留GiB':>10}{'输出':>14}")

    from realesrgan_engine import RealEsrganEngine
    from config import BASE
    W4 = os.path.join(BASE, "超分模型", "RealESRGAN", "weights",
                      "RealESRGAN_x4plus.pth")

    cases = [
        ("不分块 batch=1", dict(batch=1, tile=0)),
        ("不分块 batch=2", dict(batch=2, tile=0)),
        ("不分块 batch=4", dict(batch=4, tile=0)),
        ("分块 256 pad32 batch=4", dict(batch=4, tile=256)),
        ("分块 256 pad32 batch=8", dict(batch=8, tile=256)),
        ("分块 384 pad32 batch=4", dict(batch=4, tile=384)),
    ]
    for tag, kw in cases:
        try:
            eng = RealEsrganEngine(weight=W4, fp16=True, verbose=False, **kw)
            with torch.inference_mode():
                for _ in range(args.warm):
                    eng.upscale_chunk(fr)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(args.iters):
                    out = eng.upscale_chunk(fr)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / args.iters
            per = dt / args.frames * 1000
            resv = torch.cuda.memory_reserved() / 2**30
            print(f"  {tag:<34}{per:>9.1f}{resv:>10.2f}{str(out.shape[1:]):>14}",
                  flush=True)
            del eng
            torch.cuda.empty_cache()
        except Exception as e:                              # noqa: BLE001
            print(f"  {tag:<34} 失败: {type(e).__name__}: {str(e)[:60]}", flush=True)
            torch.cuda.empty_cache()

    # 对比：BasicVSR++（同样 4 帧一批）
    try:
        from vsr_engine import VsrEngine
        eng = VsrEngine(weight=VSR_WEIGHT, fp16=True, chunk=4, verbose=False)
        with torch.inference_mode():
            eng.upscale_chunk(fr[:4])
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            eng.upscale_chunk(fr[:4])
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        print(f"\n  对照 BasicVSR++（段长4，含 1.5x overlap）: "
              f"{dt/4*1000:.1f} ms/帧   预留 {torch.cuda.memory_reserved()/2**30:.2f} GiB")
    except Exception as e:                                  # noqa: BLE001
        print(f"\n  VSR 对照失败: {type(e).__name__}: {str(e)[:60]}")


if __name__ == "__main__":
    main()
