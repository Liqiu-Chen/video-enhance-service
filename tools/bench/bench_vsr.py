"""VSR(BasicVSR++) 各配置的基准测试。

**每个配置跑在独立子进程里并带超时**：5K 尺寸下 chunk 稍大就会撑爆显存，
一旦落到共享内存会慢上百倍、把基准拖死（RIFE 那边已经踩过）。

  python bench_vsr.py                    # 默认 720x1280 -> 2880x5120（真实负载）
  python bench_vsr.py --size 360 640     # 小尺寸快速迭代
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

sys.path.insert(0, os.path.join(HERE, "engine"))
sys.path.insert(0, os.path.join(HERE, "超分模型", "VSR", "src"))

from config import VSR_WEIGHT  # noqa: E402

CASES = [
    ("fp32 chunk8 基线", dict(fp16=False, chunk=8)),
    ("fp16 chunk4", dict(fp16=True, chunk=4)),
    ("fp16 chunk8", dict(fp16=True, chunk=8)),
    ("fp16 chunk16", dict(fp16=True, chunk=16)),
    ("fp16 chunk32", dict(fp16=True, chunk=32)),
    ("fp16 chunk8 +graph", dict(fp16=True, chunk=8, use_cuda_graph=True)),
    ("fp16 chunk16 +graph", dict(fp16=True, chunk=16, use_cuda_graph=True)),
]


def make_frames(n, h, w):
    """平滑图像 + 缓慢平移，模拟真实视频。"""
    rng = np.random.default_rng(20261011)
    small = rng.random((max(8, h // 16), max(8, w // 16), 3)).astype(np.float32)
    t = torch.from_numpy(small).permute(2, 0, 1)[None]
    big = torch.nn.functional.interpolate(t, size=(h, w), mode="bicubic",
                                          align_corners=False)[0]
    big = (big.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return [np.roll(big, (i % 5, (i % 5) * 2), axis=(0, 1)).copy() for i in range(n)]


def bench_engine(eng, frames, iters=2):
    eng.upscale_chunk(frames[:2])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = eng.upscale_chunk(frames)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / (iters * len(frames))
    return dt, torch.cuda.max_memory_allocated() / 2**30, out


def psnr(a_u8, b_u8):
    a = a_u8.astype(np.float32)
    b = b_u8.astype(np.float32)
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def run_one(i, h, w, n_frames, iters, ref_path=None):
    from vsr_engine import VsrEngine
    name, kw = CASES[i]
    frames = np.stack(make_frames(n_frames, h, w))
    eng = VsrEngine(weight=VSR_WEIGHT, verbose=False, **kw)
    torch.cuda.empty_cache()
    dt, peak, out = bench_engine(eng, frames, iters=iters)
    if ref_path and os.path.exists(ref_path):
        ps = psnr(out, np.load(ref_path))
    else:
        ps = float("inf")
        np.save(os.path.join(HERE, "..", "work", "_vsr_ref.npy"), out)
    print(f"RESULT\t{name}\t{dt*1000:.1f}\t{peak:.2f}\t{ps:.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, nargs=2, default=[1280, 720], help="输入 H W")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--only", type=int, default=-1)
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()
    h, w = args.size
    ref = os.path.join(HERE, "..", "work", "_vsr_ref.npy")

    if args.only >= 0:
        run_one(args.only, h, w, args.frames, args.iters, ref)
        return

    if os.path.exists(ref):
        os.remove(ref)
    print(f"VSR 基准 输入 {w}x{h} -> 输出 {w*4}x{h*4}  帧数 {args.frames} "
          f"迭代 {args.iters}  每配置超时 {args.timeout}s")
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB\n")
    rows = []
    for i, (name, _) in enumerate(CASES):
        t0 = time.time()
        try:
            p = subprocess.run(
                [sys.executable, "-u", os.path.abspath(__file__),
                 "--size", str(h), str(w), "--frames", str(args.frames),
                 "--iters", str(args.iters), "--only", str(i)],
                capture_output=True, text=True, timeout=args.timeout)
            line = [x for x in p.stdout.splitlines() if x.startswith("RESULT")]
            if line:
                f = line[-1].split("\t")
                dt, peak, ps = float(f[2]), float(f[3]), float(f[4])
                rows.append((name, dt, peak, ps))
                print(f"  {name:24s} {dt:8.1f} ms/帧   显存 {peak:5.2f} GiB   "
                      f"与基线 PSNR {ps:6.2f} dB", flush=True)
            else:
                tail = (p.stderr or p.stdout).strip().splitlines()[-1:] or ["无输出"]
                print(f"  {name:24s} 失败: {tail[0][:110]}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"  {name:24s} 超时 {args.timeout}s", flush=True)
        print(f"      ({time.time()-t0:.0f}s)", flush=True)

    if rows:
        base = rows[0][1]
        print("\n汇总（每帧输出耗时，越小越好）")
        for name, dt, peak, ps in sorted(rows, key=lambda r: r[1]):
            print(f"  {name:24s} {dt:7.1f} ms/帧   加速 {base/dt:5.2f}x   "
                  f"显存 {peak:5.2f} GiB   PSNR {ps:6.2f} dB")


if __name__ == "__main__":
    main()
