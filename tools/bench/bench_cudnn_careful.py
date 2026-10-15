"""可靠地测 cuDNN benchmark 开/关。

前两次测出来的差距自相矛盾（2% vs 2 倍），原因是 `benchmark=True` 会为新 shape
做**一次性算法试探**，而之前只预热 1 次，试探开销会漏进计时。
这里每个配置先跑 5 次预热把试探彻底摊掉，再计时 5 次，并重复两轮看稳定性。

    python bench_cudnn_careful.py --size 1280 720
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
    """真实素材。**不能用 zeros**：全零会触发 mirror-extended 捷径，跳过正向光流。"""
    rng = np.random.default_rng(7)
    small = rng.random((max(8, h // 16), max(8, w // 16), 3)).astype(np.float32)
    t = torch.from_numpy(small).permute(2, 0, 1)[None]
    big = torch.nn.functional.interpolate(t, size=(h, w), mode="bicubic",
                                          align_corners=False)[0]
    big = (big.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return np.stack([np.roll(big, (i % 3, (i % 3) * 2), axis=(0, 1)).copy()
                     for i in range(n)])


def run(tag, h, w, chunk, benchmark, tf32, warm=5, iters=5):
    from vsr_engine import VsrEngine
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.allow_tf32 = tf32
    eng = VsrEngine(weight=VSR_WEIGHT, fp16=True, chunk=chunk, overlap=2,
                    verbose=False)
    fr = real_frames(chunk, h, w)
    with torch.inference_mode():
        for _ in range(warm):
            eng.upscale_chunk(fr)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            eng.upscale_chunk(fr)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
    per = dt / chunk * 1000
    print(f"  {tag:<40}{per:>9.1f} ms/帧   {dt*1000:>8.0f} ms/段", flush=True)
    del eng
    torch.cuda.empty_cache()
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, nargs=2, default=[1280, 720])
    ap.add_argument("--chunk", type=int, default=8)
    args = ap.parse_args()
    h, w = args.size
    print(f"cuDNN 配置可靠测量：输入 {w}x{h}，段长 {args.chunk}")
    print(f"每个配置预热 5 次 + 计时 5 次，跑两轮\n")

    for rnd in (1, 2):
        print(f"第 {rnd} 轮：")
        run("benchmark=True", h, w, args.chunk, True, True)
        run("benchmark=False", h, w, args.chunk, False, True)
        print(flush=True)


if __name__ == "__main__":
    main()
