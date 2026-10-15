"""定位「流水线比隔离慢 4.9 倍」的根因。

上一轮实测：同样 240 帧、同样段长/微批，隔离 24.4s，流水线 119.1s。
两个模型在流水线里都慢 4~5 倍，而 I/O 等待只有 0.1s。

候选原因逐个排查：
  A 基线       ：VSR 连续调用                      —— 隔离跑法
  B 交替       ：VSR/RIFE 交替                      —— 查"两模型交替"的影响
  C 编码线程   ：VSR 连续 + 后台线程持续编码         —— 查 ffmpeg/写线程的影响
  D 空转线程   ：VSR 连续 + 后台线程空转            —— 查纯线程/GIL 的影响
  E 解码线程   ：VSR 连续 + 后台线程持续解码         —— 查读线程的影响

    python bench_interleave.py --frames 120
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
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


def decode(clip, n):
    from pipeline import color_report
    m = color_report(ffmpeg(), clip)
    H, W = int(m["height"]), int(m["width"])
    p = subprocess.run([ffmpeg(), "-v", "error", "-i", clip, "-frames:v", str(n),
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       capture_output=True)
    fsz = H * W * 3
    got = len(p.stdout) // fsz
    return np.frombuffer(p.stdout[:got * fsz], np.uint8).reshape(got, H, W, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=os.path.join(HERE, "..", "liuqiang.mp4"))
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=8)
    args = ap.parse_args()

    src = decode(args.clip, args.frames)
    N, H, W = len(src), src.shape[1], src.shape[2]
    MB = args.micro_batch
    print(f"素材 {N} 帧 {W}x{H}  段长 {args.chunk}  微批 {MB}\n")

    from vsr_engine import VsrEngine
    from rife_engine import RifeEngine
    vsr = VsrEngine(weight=VSR_WEIGHT, fp16=True, chunk=args.chunk,
                    out_scale=2, verbose=False)
    rife = RifeEngine(weight=RIFE_WEIGHT, fp16=True, batch=1, verbose=False)
    vsr.warmup(H, W)
    oh, ow = (H // 2) * 4, (W // 2) * 4
    rife.build((oh + 31) // 32 * 32, (ow + 31) // 32 * 32)
    torch.cuda.empty_cache()

    def timed(fn, tag, note=""):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"  {tag:<26}{dt:7.2f}s{'':4}{dt/N*1000:7.1f} ms/帧   {note}",
              flush=True)
        return dt

    print(f"  {'实验':<24}{'总耗时':>9}{'':4}{'每帧':>12}")
    results = {}

    # ---- A 基线 ----
    def run_a():
        for i in range(0, N, MB):
            vsr.upscale_chunk(src[i:i + MB])
    results["A 基线（VSR 连续）"] = timed(run_a, "A 基线（VSR 连续）")

    # ---- B 交替 VSR/RIFE ----
    # 复用流水线自己的 _interp_pairs：它内部会按 rife.batch 切片、补齐、裁剪，
    # 手写这些逻辑容易和真实链路不一致（前两次就踩了 32 倍数补齐的坑）。
    from pipeline import Pipeline
    _p = Pipeline(vsr, rife, micro_batch=MB, verbose=False)

    def run_b():
        for i in range(0, N, MB):
            up = vsr.upscale_chunk(src[i:i + MB])
            if len(up) >= 2:
                _p._interp_pairs(up)
    results["B VSR/RIFE 交替"] = timed(run_b, "B VSR/RIFE 交替",
                                     "同批内先 VSR 再 RIFE")

    # ---- C 后台编码线程 ----
    stop = threading.Event()
    enc_path = os.path.join(HERE, "..", "work", "_enc_probe.mp4")
    os.makedirs(os.path.dirname(enc_path), exist_ok=True)
    oh_, ow_ = (H // 2) * 4, (W // 2) * 4

    def encoder_worker():
        """模拟流水线的写线程：把 4x 帧不断喂给 ffmpeg。"""
        cmd = [ffmpeg(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt",
               "rgb24", "-s", f"{ow_}x{oh_}", "-framerate", "120", "-i", "-",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
               "-pix_fmt", "yuv420p", enc_path]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             bufsize=1 << 24)
        blank = np.zeros((oh_, ow_, 3), np.uint8).tobytes()
        try:
            while not stop.is_set():
                p.stdin.write(blank)
        except Exception:                                       # noqa: BLE001
            pass
        try:
            p.stdin.close()
        except Exception:                                       # noqa: BLE001
            pass
        p.wait()

    th = threading.Thread(target=encoder_worker, daemon=True)
    th.start()
    time.sleep(1.0)
    results["C VSR + 后台编码线程"] = timed(run_a, "C VSR + 后台编码线程",
                                        "模拟写线程")
    stop.set()
    time.sleep(1.5)

    # ---- D 空转线程 ----
    stop2 = threading.Event()
    def idle_worker():
        while not stop2.is_set():
            time.sleep(0.05)
    for _ in range(3):
        threading.Thread(target=idle_worker, daemon=True).start()
    results["D VSR + 3 个空转线程"] = timed(run_a, "D VSR + 3 个空转线程")

    # ---- E 解码线程（持续读管道）----
    stop3 = threading.Event()
    q = queue.Queue(maxsize=6)
    def reader_worker():
        cmd = [ffmpeg(), "-v", "error", "-i", args.clip, "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-"]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL)
        fsz = H * W * 3
        try:
            while not stop3.is_set():
                buf = p.stdout.read(fsz)
                if len(buf) < fsz:
                    break
                try:
                    q.put(np.frombuffer(buf, np.uint8).reshape(H, W, 3), timeout=0.5)
                except queue.Full:
                    continue
        except Exception:                                       # noqa: BLE001
            pass
        p.kill()
    threading.Thread(target=reader_worker, daemon=True).start()
    time.sleep(0.5)
    # 让队列被消费掉，否则读线程会一直阻塞在 put（不构成 CPU 压力）
    def drain_q():
        while not stop3.is_set():
            try:
                q.get(timeout=0.2)
            except queue.Empty:
                continue
    threading.Thread(target=drain_q, daemon=True).start()
    results["E VSR + 后台解码线程"] = timed(run_a, "E VSR + 后台解码线程",
                                        "模拟读线程")
    stop2.set(); stop3.set()
    time.sleep(0.5)

    # ---- 结论 ----
    base = results["A 基线（VSR 连续）"]
    print("\n" + "=" * 62)
    print("  相对基线的倍数（>1.3x 即为显著拖慢）")
    for k, v in results.items():
        if k == "A 基线（VSR 连续）":
            continue
        print(f"    {k:<28}{v/base:6.2f}x")
    print("=" * 62)


if __name__ == "__main__":
    main()
