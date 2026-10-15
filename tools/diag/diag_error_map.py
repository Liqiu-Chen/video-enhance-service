"""定位精度误差的**空间分布** —— 最大差到底出在哪里。

背景：
  TRT 的两个引擎 PSNR 都还行（VSR 44dB / RIFE 48dB），但**最大差高达 101/255**。
  PSNR 是全局平均，掩盖了局部；最大差才是"看得见"的那一部分。
  用户的诉求是「最大差控制在 2/255 以内」，所以必须先知道大误差长什么样。

猜测（待验证）：
  1. 只在**图像边框**（grid_sample 的越界语义与 PyTorch 不同）
  2. 只在**极少数孤立像素**（某个算子的边界条件）
  3. 弥散分布（那就真是精度问题了）

    python diag_error_map.py --size 1672 960
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

from config import RIFE_WEIGHT  # noqa: E402
from bench_trt_vsr import make_frames  # noqa: E402

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def analyze(tag: str, ref: np.ndarray, got: np.ndarray):
    """ref/got: (N,H,W,3)，值域 0-255 的 float。"""
    d = np.abs(ref - got)
    H, W = d.shape[1], d.shape[2]
    # 去掉最外圈 8 像素后的误差
    m = 8
    interior = d[:, m:H - m, m:W - m]
    border = np.concatenate([
        d[:, :m].ravel(), d[:, H - m:].ravel(),
        d[:, :, :m].ravel(), d[:, :, W - m:].ravel()])
    print(f"\n  【{tag}】")
    print(f"    整体   最大差 {d.max():7.2f}   平均 {d.mean():6.3f}")
    print(f"    内部   最大差 {interior.max():7.2f}   平均 {interior.mean():6.3f}")
    print(f"    边框   最大差 {border.max():7.2f}   平均 {border.mean():6.3f}")

    # 大误差像素占比
    for thr in (0.5, 1, 2, 5, 20):
        pct = (d > thr).mean() * 100
        print(f"    >{thr:>4}/255 的像素占比 {pct:8.4f}%", end="")
        if thr == 2:
            print("   ← 目标线")
        else:
            print()
    # 大误差的位置画像
    big = d > 2
    if big.any():
        idx = np.argwhere(big)
        cols = idx[:, 3]
        print(f"    超线像素通道分布 R/G/B: "
              f"{(cols==0).sum()}/{(cols==1).sum()}/{(cols==2).sum()}")
    return d.max(), d.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, nargs=2, default=[1672, 960])
    args = ap.parse_args()
    h, w = args.size

    frames = make_frames(h, w, 4)
    pairs = []
    for i in range(3):
        a = torch.from_numpy(frames[i:i + 1]).cuda().permute(0, 3, 1, 2)
        b = torch.from_numpy(frames[i + 1:i + 2]).cuda().permute(0, 3, 1, 2)
        pairs.append((a.float().div_(255.0), b.float().div_(255.0)))

    from rife_engine import RifeEngine
    e = RifeEngine(weight=RIFE_WEIGHT, fp16=False, batch=1, verbose=False)
    e.build(h, w)
    with torch.inference_mode():
        ref = torch.cat([e.infer(a, b) for a, b in pairs], 0)
    del e
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    # 注意转成 NHWC 再分析（torch 是 NCHW，直接 numpy 出来是 (N,C,H,W)）
    R = (ref.float().clamp(0, 1) * 255).permute(0, 2, 3, 1).cpu().numpy()

    from build_trt_rife import TrtRife
    for tag, f in (("TRT-fp16", "rife_1696x960_fp16.trt"),
                   ("TRT-strongmix", "rife_1696x960_strongmix.trt")):
        p = os.path.join(HERE, "cache", f)
        if not os.path.exists(p):
            continue
        t = TrtRife(p)
        with torch.inference_mode():
            o = torch.cat([t(torch.cat([a.to(t.in_dtype), b.to(t.in_dtype)], 1))
                           for a, b in pairs], 0)
        O = (o.float().clamp(0, 1) * 255).permute(0, 2, 3, 1).cpu().numpy()
        analyze(tag, R, O)
        del t, o, O
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
