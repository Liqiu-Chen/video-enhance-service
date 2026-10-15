"""新旧链路对照基准：同一段素材，分别测「旧实现」与「新引擎」的耗时。

旧实现 = 现在的 `run_example.py` 走法：
    ffmpeg 解码 → PNG 落盘 → 读回 numpy → 模型(no_grad, fp16/fp32) → 再写 PNG 落盘
新引擎 = 本目录 `engine/` 下的实现：
    全程内存直传、inference_mode、fp16 + fp32 warp、按 batch 喂

  python bench_e2e.py --clip ../work/clip32.mp4 --frames 32
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import numpy as np

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
sys.path.insert(0, os.path.join(HERE, "..", "超分模型", "VSR", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "插帧模型", "RIFE", "tools"))

from config import RIFE_WEIGHT, VSR_WEIGHT, ffmpeg  # noqa: E402

WORK = os.path.join(HERE, "..", "work", "bench_e2e")


def decode(clip, n):
    p = subprocess.run([ffmpeg(), "-v", "error", "-i", clip, "-frames:v", str(n),
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       capture_output=True)
    # frombuffer 出来的是只读数组，复制一份免得后面 torch.from_numpy 报警
    a = np.frombuffer(p.stdout, np.uint8).copy()
    side = 720 * 1280 * 3
    k = len(a) // side
    return a[:k * side].reshape(k, 1280, 720, 3)


def png_write(arrs, d, level=1):
    from PIL import Image
    os.makedirs(d, exist_ok=True)
    for i, a in enumerate(arrs):
        Image.fromarray(a).save(os.path.join(d, f"{i:06d}.png"), compress_level=level)


def png_read(d):
    from PIL import Image
    fs = sorted(f for f in os.listdir(d) if f.endswith(".png"))
    return np.stack([np.asarray(Image.open(os.path.join(d, f)).convert("RGB"),
                                np.uint8) for f in fs])


def old_vsr(frames_u8):
    from vsr.runner import VSRRunner
    din, dout = os.path.join(WORK, "in"), os.path.join(WORK, "out")
    t0 = time.perf_counter()
    png_write(frames_u8, din)
    ro = png_read(din)
    t_io = time.perf_counter() - t0

    eng = VSRRunner(VSR_WEIGHT, variant="basicvsr_plusplus", fp16=True, max_seq_len=4)

    def save(i, a):
        from PIL import Image
        os.makedirs(dout, exist_ok=True)
        Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)).save(
            os.path.join(dout, f"{i:06d}.png"), compress_level=1)

    t1 = time.perf_counter()
    eng.upscale(list(ro.astype(np.float32) / 255.0), on_frame=save)
    t_inf = time.perf_counter() - t1
    return t_inf, t_io


def new_vsr(frames_u8, chunk=4):
    import torch
    from vsr_engine import VsrEngine
    eng = VsrEngine(weight=VSR_WEIGHT, fp16=True, chunk=chunk, verbose=False)
    eng.upscale_chunk(frames_u8[:2])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = eng.upscale_chunk(frames_u8)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, out


def old_rife(up_u8):
    """旧实现：RIFE 逐对调用 + PNG 往返。fp32（旧链路 fp16 有 dtype bug）。"""
    from vfi import RifeRunner
    din = os.path.join(WORK, "rin")
    t0 = time.perf_counter()
    png_write(up_u8, din)
    arrs = png_read(din)
    t_io = time.perf_counter() - t0
    ro = arrs.astype(np.float32) / 255.0

    r = RifeRunner(RIFE_WEIGHT, fp16=False)
    t1 = time.perf_counter()
    for i in range(len(ro) - 1):
        r.mid(ro[i], ro[i + 1])
    t_inf = time.perf_counter() - t1
    return t_inf, t_io


def new_rife(up_u8):
    import torch
    from rife_engine import RifeEngine
    from pipeline import Pipeline
    eng = RifeEngine(weight=RIFE_WEIGHT, fp16=True, batch=2, warp_fp32=True,
                     use_cuda_graph=True, verbose=False)
    h, w = up_u8.shape[1], up_u8.shape[2]
    ph = (h + eng.PAD - 1) // eng.PAD * eng.PAD
    pw = (w + eng.PAD - 1) // eng.PAD * eng.PAD
    eng.build(ph, pw)
    p = Pipeline(None, eng, verbose=False)
    p._interp_pairs(up_u8[:3])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = p._interp_pairs(up_u8)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=os.path.join(HERE, "..", "work", "clip32.mp4"))
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--skip-old", action="store_true")
    ap.add_argument("--reuse-vsr", action="store_true",
                    help="复用上次的 VSR 输出（跳过 VSR 阶段，只补测 RIFE）")
    args = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)

    frames = decode(args.clip, args.frames)
    n = len(frames)
    print(f"素材 {n} 帧  {frames.shape[2]}x{frames.shape[1]}  uint8\n", flush=True)
    up_path = os.path.join(WORK, "up.npy")

    res = {}
    if args.reuse_vsr and os.path.exists(up_path):
        up = np.load(up_path)
        print(f"复用已有 VSR 输出 {up.shape}\n", flush=True)
    else:
        if not args.skip_old:
            print("=== VSR ×4 旧实现（PNG 往返 + no_grad + fp16 chunk4）===", flush=True)
            ti, tio = old_vsr(frames)
            res["vsr_old"] = ti + tio
            print(f"  推理 {ti/n*1000:7.0f} ms/帧 | PNG 往返 {tio/n*1000:6.0f} ms/帧 | "
                  f"合计 {(ti+tio)/n*1000:7.0f} ms/帧", flush=True)

        print("\n=== VSR ×4 新引擎（内存直传 + inference_mode + fp16）===", flush=True)
        t, up = new_vsr(frames)
        res["vsr_new"] = t
        np.save(up_path, up)
        print(f"  合计 {t/n*1000:7.0f} ms/帧  →  输出 {up.shape}", flush=True)
        if "vsr_old" in res:
            print(f"  加速 {res['vsr_old']/t:.2f}x", flush=True)

    if not args.skip_old:
        print("\n=== RIFE ×2 旧实现（PNG 往返 + 逐对 fp32）===", flush=True)
        ti, tio = old_rife(up)
        res["rife_old"] = ti + tio
        print(f"  推理 {ti/(n-1)*1000:7.0f} ms/对 | PNG 往返 {tio/n*1000:6.0f} ms/帧 | "
              f"合计 {(ti+tio)/n*1000:7.0f} ms/帧", flush=True)

    print("\n=== RIFE ×2 新引擎（batch + fp16 + fp32 warp + CUDA Graph）===", flush=True)
    t, mid = new_rife(up)
    res["rife_new"] = t
    print(f"  合计 {t/n*1000:7.0f} ms/帧（{mid.shape[0]} 个中间帧）", flush=True)
    if "rife_old" in res:
        print(f"  加速 {res['rife_old']/t:.2f}x", flush=True)

    if "vsr_old" in res and "rife_old" in res:
        o, nw = res["vsr_old"] + res["rife_old"], res["vsr_new"] + res["rife_new"]
        print(f"\n端到端（两级串联，每帧源帧）: 旧 {o/n*1000:.0f} ms  新 {nw/n*1000:.0f} ms"
              f"  总加速 {o/nw:.2f}x")


if __name__ == "__main__":
    main()
