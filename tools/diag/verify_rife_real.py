"""用**真实连续帧**验收 RIFE 各引擎，并判断大误差是否构成可见伪影。

为什么要重做：
  之前测最大差用的是 `make_frames`（np.roll 造帧），**接缝处有硬跳变** ——
  光流在那种地方本来就算不准，会人为放大误差。必须换成 liuqiang.mp4 的
  真实连续帧，并且让输入走一遍真实的 VSR，才是流水线里真正的数据分布。

为什么不能只看"最大差"：
  2000 万个像素里有 1 个差 183/255，和有一片 5000 像素的差 183/255，
  是完全不同性质的问题 —— 前者肉眼不可见，后者是明显的块状伪影。
  所以这里额外做**连通域分析**：看超线像素是散点还是成片，最大一片有多大。

    python verify_rife_real.py --n 6
"""
from __future__ import annotations

import argparse
import gc
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

from config import RIFE_WEIGHT, VSR_WEIGHT  # noqa: E402

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def cluster_stats(mask: np.ndarray):
    """超线像素是散点还是成片？返回最大连通块大小（用简单 flood fill）。"""
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    if n == 0:
        return 0, 0
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return n, int(sizes.max())


def render(seq, sr_tag, rife, h, w):
    """把一个 RIFE 引擎跑完整段，返回 (N-1,H,W,3) uint8。"""
    from pipeline import Pipeline
    p = Pipeline.__new__(Pipeline)
    p.rife = rife
    p.verbose = False
    return p._interp_pairs(seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="源帧数")
    ap.add_argument("--clip", default=os.path.join(HERE, "..", "liuqiang.mp4"))
    args = ap.parse_args()

    # ---------- 真实连续帧 → 走一遍 VSR 得到真实的插帧输入 ----------
    from bench_pipe_vs_iso import decode_frames
    from vsr_engine import VsrEngine

    src = decode_frames(args.clip, args.n)
    n, H, W = src.shape[0], src.shape[1], src.shape[2]
    v = VsrEngine(weight=VSR_WEIGHT, fp16=False, chunk=n, out_scale=4,
                  verbose=False)
    with torch.inference_mode():
        up = v.upscale_chunk(src)          # 真实的高分帧（fp32 VSR）
    del v
    gc.collect()
    torch.cuda.empty_cache()
    h, w = up.shape[1], up.shape[2]
    hp, wp = (h + 31) // 32 * 32, (w + 31) // 32 * 32
    print(f"源 {n} 帧 {W}x{H} → VSR 4x → {w}x{h}（补到 {wp}x{hp}）\n")

    # ---------- 参考：PyTorch fp32（数值真值）----------
    from rife_engine import RifeEngine
    r0 = RifeEngine(weight=RIFE_WEIGHT, fp16=False, batch=1, verbose=False)
    ref = render(up, "fp32", r0, h, w)
    del r0
    gc.collect()
    torch.cuda.empty_cache()

    print(f"  {'实现':<22}{'ms/对':>8}{'PSNR':>9}{'最大差':>8}"
          f"{'>2/255':>9}{'超线块数':>10}{'最大块':>8}")
    rows = []
    def tick():
        torch.cuda.synchronize()
        return time.perf_counter()

    for tag, kw in (("PyTorch fp16", dict(verbose=False)),
                    ("TRT-fp16", dict(backend="trt", trt_mode="fp16", verbose=False)),
                    ("TRT-warp+resize", dict(backend="trt", trt_mode="warp+resize", verbose=False))):
        gc.collect()
        torch.cuda.empty_cache()
        try:
            r = RifeEngine(weight=RIFE_WEIGHT, fp16=True, batch=1, **kw)
            r.build(hp, wp)
            render(up, tag, r, h, w)                     # 预热
            torch.cuda.synchronize()
            t = tick()
            out = render(up, tag, r, h, w)
            ms = (tick() - t) / (n - 1) * 1000
        except Exception as e:                           # noqa: BLE001
            print(f"  {tag:<22} 失败: {type(e).__name__} {str(e)[:50]}")
            continue
        d = np.abs(ref.astype(np.float64) - out.astype(np.float64))
        big = (d > 2).any(-1)                            # 任一通道超线
        nblk, mx = cluster_stats(big)
        print(f"  {tag:<22}{ms:>8.1f}{psnr(ref, out):>8.2f}dB{d.max():>8.1f}"
              f"{big.mean()*100:>8.3f}%{nblk:>10}{mx:>8}")
        rows.append((tag, d, big))
        del r, out, big
        gc.collect()
        torch.cuda.empty_cache()

    # ---------- 大误差在哪 ----------
    for tag, d, big in rows:
        if not big.any():
            continue
        idx = np.argwhere(big)
        hs = h
        dist_edge = np.minimum(idx[:, 1], hs - 1 - idx[:, 1])
        print(f"\n  【{tag}】超线像素 {len(idx)} 个")
        print(f"    距上下边界 50%分位 {np.percentile(dist_edge,50):.0f}px "
              f"90% {np.percentile(dist_edge,90):.0f}px")
        print(f"    最大单点差 {d.max():.1f}；误差 >20 的有 {(d>20).sum()} 个像素"
              f"（占 {(d>20).mean()*100:.5f}%）")


if __name__ == "__main__":
    main()
