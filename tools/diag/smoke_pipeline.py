"""流水线冒烟测试：直接调用 Pipeline 跑一小段，检查产物规格是否正确。

  python smoke_pipeline.py --clip ../work/clip32.mp4 --out ../work/pipe_out.mp4
"""
from __future__ import annotations

import argparse
import os
import sys
import time

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=os.path.join(HERE, "..", "work", "clip32.mp4"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "work", "pipe_out.mp4"))
    ap.add_argument("--no-interp", action="store_true")
    ap.add_argument("--slowmo", action="store_true")
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--scale", type=int, default=4, choices=[2, 4],
                    help="输出倍率：4=直接4x超分；2=先降采样2倍再4x超分")
    ap.add_argument("--downsample", default="linear",
                    choices=["nearest", "linear", "area", "bicubic"])
    ap.add_argument("--segments", type=int, default=1, help="一次前向并行几段")
    ap.add_argument("--sr-backend", default="resr", choices=["resr", "vsr"])
    ap.add_argument("--resr-batch", type=int, default=2)
    ap.add_argument("--resr-tile", type=int, default=0)
    ap.add_argument("--rife-batch", type=int, default=1)
    ap.add_argument("--rife-backend", default="torch",
                    choices=["torch", "trt", "onnx"])
    ap.add_argument("--rife-graph", action="store_true")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--stacks", type=int, default=0,
                    help=">0 时每 N 秒 dump 一次线程栈（排查卡死用）")
    args = ap.parse_args()
    if args.stacks:
        import faulthandler
        faulthandler.dump_traceback_later(args.stacks, repeat=True)

    from vsr_engine import VsrEngine
    from rife_engine import RifeEngine
    from pipeline import Pipeline, Progress
    from config import RESR_WEIGHT

    print("构建引擎 ...", flush=True)
    if args.sr_backend == "resr":
        from realesrgan_engine import RealEsrganEngine
        vsr = RealEsrganEngine(weight=RESR_WEIGHT, fp16=True,
                               batch=args.resr_batch, tile=args.resr_tile,
                               out_scale=args.scale,
                               downsample=args.downsample, verbose=False)
    else:
        vsr = VsrEngine(weight=VSR_WEIGHT, fp16=True, chunk=args.chunk,
                        out_scale=args.scale, downsample=args.downsample,
                        segments=args.segments, verbose=False)
    rife = None
    if not args.no_interp:
        rife = RifeEngine(weight=RIFE_WEIGHT, fp16=True, batch=args.rife_batch,
                          backend=args.rife_backend,
                          warp_fp32=True,
                          use_cuda_graph=args.rife_graph, verbose=False)
    pipe = Pipeline(vsr, rife, micro_batch=args.micro_batch)

    pr = Progress()
    t0 = time.time()
    rep = pipe.run(args.clip, args.out, interp=not args.no_interp,
                   realtime=not args.slowmo, crf=args.crf, preset="medium",
                   progress=pr)
    print(f"\n总耗时 {time.time()-t0:.1f}s")
    for k, v in rep.items():
        print(f"  {k}: {v}")

    # 用 ffmpeg 核对产物规格
    import subprocess
    p = subprocess.run([ffmpeg(), "-hide_banner", "-i", args.out],
                       capture_output=True, text=True, errors="replace")
    for ln in p.stderr.splitlines():
        if "Duration" in ln or "Stream #" in ln:
            print(" ", ln.strip())


if __name__ == "__main__":
    main()
