"""Q1：流水线 vs 隔离 —— 哪种跑法性能更高。

两种跑法的本质差别：

  流水线：VSR 与 RIFE 交替处理微批，两个模型**同时常驻显存**。
          好处是解码/编码能和 GPU 计算重叠；坏处是显存争抢、
          每个微批都有固定开销（Python 循环、H2D/D2H 往返）。

  隔离：  先把所有帧过 VSR，再统一过 RIFE，任一时刻只有一个模型在算。
          好处是显存独占、可以用更激进的分块；坏处是 I/O 无法与计算重叠，
          而且中间帧必须暂存在内存/磁盘里。

判定标准：同样的帧数、同样的分辨率、同样的参数，比总墙钟时间。

    python bench_pipe_vs_iso.py --clip ../liuqiang.mp4 --frames 240 --scale 2
"""
from __future__ import annotations

import argparse
import os
import subprocess
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

from config import RIFE_WEIGHT, VSR_WEIGHT, ffmpeg  # noqa: E402


def decode_frames(clip: str, n: int) -> np.ndarray:
    """解出前 n 帧为 (n,H,W,3) uint8。"""
    from pipeline import color_report
    meta = color_report(ffmpeg(), clip)
    H, W = int(meta["height"]), int(meta["width"])
    cmd = [ffmpeg(), "-v", "error", "-i", clip, "-frames:v", str(n),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.run(cmd, capture_output=True)
    buf = p.stdout
    fsz = H * W * 3
    got = len(buf) // fsz
    arr = np.frombuffer(buf[:got * fsz], np.uint8).reshape(got, H, W, 3)
    print(f"  解码 {got} 帧  {W}x{H}", flush=True)
    return arr


def gpu_peak_gb() -> float:
    return torch.cuda.max_memory_allocated() / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=os.path.join(HERE, "..", "liuqiang.mp4"))
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--rife-backend", default="torch",
                    choices=["torch", "trt", "onnx"])
    args = ap.parse_args()

    src = decode_frames(args.clip, args.frames)
    N, H, W = len(src), src.shape[1], src.shape[2]
    out_h = (H // 2) * 4 if args.scale == 2 else H * 4
    out_w = (W // 2) * 4 if args.scale == 2 else W * 4
    print(f"\n素材 {N} 帧 {W}x{H} → 输出 {out_w}x{out_h}（out_scale={args.scale}）")
    print(f"段长 {args.chunk}  微批 {args.micro_batch}\n")

    from vsr_engine import VsrEngine
    from rife_engine import RifeEngine

    vsr = VsrEngine(weight=VSR_WEIGHT, fp16=True, chunk=args.chunk,
                    out_scale=args.scale, verbose=False)
    rife = RifeEngine(weight=RIFE_WEIGHT, fp16=True, batch=1,
                      backend=args.rife_backend, verbose=False)

    # ---------- 预热（把 cuDNN 首次开销排除掉，否则比的不是稳态）----------
    print("预热中 ...", flush=True)
    vsr.warmup(H, W)
    ph = (out_h + rife.PAD - 1) // rife.PAD * rife.PAD
    pw = (out_w + rife.PAD - 1) // rife.PAD * rife.PAD
    rife.build(ph, pw)
    torch.cuda.empty_cache()

    # ---------- A. 隔离：VSR 全跑完，再 RIFE 全跑完 ----------
    print("\n[A] 隔离跑法", flush=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    ups = []
    for i in range(0, N, args.micro_batch):
        ups.append(vsr.upscale_chunk(src[i:i + args.micro_batch]))
    up = np.concatenate(ups, 0)
    torch.cuda.synchronize()
    t_vsr = time.perf_counter() - t0
    peak_vsr = gpu_peak_gb()
    print(f"  VSR  : {t_vsr:7.1f}s  ({t_vsr/N*1000:6.1f} ms/帧)  "
          f"峰值 {peak_vsr:.2f} GiB", flush=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    # 直接复用流水线自己的插帧实现，保证「隔离」与「流水线」两边
    # 喂给 RIFE 的张量、补齐、裁剪逻辑完全一致，比出来的差才是真的差。
    from pipeline import Pipeline as _P
    _tmp = _P(vsr, rife, micro_batch=args.micro_batch, verbose=False)
    mids = []
    for i in range(0, N - 1, args.micro_batch):
        seq = up[i:i + args.micro_batch + 1]
        if len(seq) < 2:
            break
        mids.append(_tmp._interp_pairs(seq))
    torch.cuda.synchronize()
    t_rife = time.perf_counter() - t0
    peak_rife = gpu_peak_gb()
    print(f"  RIFE : {t_rife:7.1f}s  ({t_rife/N*1000:6.1f} ms/帧)  "
          f"峰值 {peak_rife:.2f} GiB", flush=True)
    iso_total = t_vsr + t_rife
    print(f"  ── 隔离合计 {iso_total:.1f}s  "
          f"(峰值显存取两者较大：{max(peak_vsr, peak_rife):.2f} GiB)", flush=True)

    # 释放中间结果再测流水线，避免内存压力影响公平性
    del ups, up, mids
    torch.cuda.empty_cache()

    # ---------- B. 流水线：交替处理 ----------
    print("\n[B] 流水线跑法", flush=True)
    from pipeline import Pipeline
    torch.cuda.empty_cache()
    # ⚠️ 必须让流水线也只跑 N 帧。
    # 之前这里直接跑整个素材，结果拿 240 帧（隔离）对比 1158 帧（流水线），
    # 算出"流水线慢 4.9 倍"的假象。先截一段等长的临时素材。
    tmp_clip = os.path.join(HERE, "..", "work", "_pvi_clip.mp4")
    os.makedirs(os.path.dirname(tmp_clip), exist_ok=True)
    subprocess.run(
        [ffmpeg(), "-y", "-v", "error", "-i", args.clip, "-frames:v", str(N),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "14",
         "-pix_fmt", "yuv420p", "-an", tmp_clip], check=True)
    pipe = Pipeline(vsr, rife, micro_batch=args.micro_batch, verbose=False)
    out_path = os.path.join(HERE, "..", "work", "pipe_vs_iso.mp4")
    t0 = time.perf_counter()
    rep = pipe.run(tmp_clip, out_path, interp=True, realtime=True, crf=16,
                   preset="veryfast")
    pipe_total = time.perf_counter() - t0
    st = rep.get("stages_s", {})
    n_done = rep.get("frames_in", N)
    print(f"  端到端 {pipe_total:.1f}s / {n_done} 帧 = "
          f"{pipe_total/n_done*1000:.1f} ms/帧   "
          f"(VSR {st.get('vsr_s')}s + RIFE {st.get('rife_s')}s + I/O "
          f"{st.get('wait_decode_s', 0) + st.get('wait_encode_s', 0):.1f}s)", flush=True)

    # ---------- 结论（统一换算成每帧，帧数一致才可比）----------
    iso_per = iso_total / N * 1000
    pipe_per = pipe_total / n_done * 1000
    rel = pipe_per / iso_per
    print("\n" + "=" * 66)
    print(f"  {'跑法':<14}{'ms/帧':>10}{'总耗时':>12}{'相对':>10}")
    print(f"  {'隔离':<14}{iso_per:>10.1f}{iso_total:>11.1f}s{'1.00x':>10}")
    verdict = "流水线更快" if rel < 0.97 else ("隔离更快" if rel > 1.03 else "基本持平")
    print(f"  {'流水线':<14}{pipe_per:>10.1f}{pipe_total:>11.1f}s"
          f"{1/rel:>9.2f}x   ← {verdict}")
    print("=" * 66)
    if rel < 1.03:
        print("  → 流水线不慢于隔离，且解码/编码与计算重叠，**保持流水线**")
    else:
        print(f"  → 隔离快 {rel:.2f}x，但需要暂存中间帧；内存/磁盘够可考虑分阶段")


if __name__ == "__main__":
    main()
