"""逐个模块解除 fp16：找出 VSR 里真正怕 fp16 的那一块。

思路（比挂 hook 省内存，也不会段错误）：
  整网保持 fp16，**每次只把一个子模块换成 fp32**（带自动的喂入转 fp32、
  输出转回 fp16 的包装），看 PSNR 能恢复多少。
  恢复最多的那个模块就是元凶。

BasicVSR++ 的一级子模块：
  feat_extract / spynet / deform_align / backbone / reconstruction
  / upsample1 / upsample2 / conv_hr / conv_last

    python diag_unhalf_bisect.py --size 836 480 --t 8
"""
from __future__ import annotations

import argparse
import gc
import os
import sys

import numpy as np
import torch
import torch.nn as nn

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

from config import VSR_WEIGHT  # noqa: E402

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = False


class CastWrap(nn.Module):
    """把内层模块跑在 fp32：入口转 float，出口转回 half。"""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner.float()

    def forward(self, *a, **kw):
        fa = tuple(x.float() if torch.is_tensor(x) else x for x in a)
        fk = {k: (v.float() if torch.is_tensor(v) else v) for k, v in kw.items()}
        out = self.inner(*fa, **fk)
        if torch.is_tensor(out):
            return out.half()
        if isinstance(out, (tuple, list)):
            return type(out)(o.half() if torch.is_tensor(o) else o for o in out)
        return out


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, nargs=2, default=[836, 480])
    ap.add_argument("--t", type=int, default=8)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--clip", default=os.path.join(HERE, "..", "liuqiang.mp4"))
    args = ap.parse_args()
    h, w = args.size

    from bench_pipe_vs_iso import decode_frames
    src = decode_frames(args.clip, args.t)
    print(f"真实连续帧 {args.t} 帧 {w}x{h} → {args.scale}x\n")

    from vsr_engine import VsrEngine
    e32 = VsrEngine(weight=VSR_WEIGHT, fp16=False, chunk=args.t,
                    out_scale=args.scale, verbose=False)
    with torch.inference_mode():
        ref = e32.upscale_chunk(src)
    del e32
    gc.collect()
    torch.cuda.empty_cache()

    e16 = VsrEngine(weight=VSR_WEIGHT, fp16=True, chunk=args.t,
                    out_scale=args.scale, verbose=False)
    net = e16.net
    orig = {n: m for n, m in net.named_children()}
    print(f"  {'保持 fp16、只把该模块换回 fp32':<34}{'PSNR':>9}{'最大差':>9}")

    with torch.inference_mode():
        base = e16.upscale_chunk(src)
    print(f"  {'（全部 fp16 的基线）':<34}{psnr(ref,base):>8.2f}dB"
          f"{np.abs(ref.astype(int)-base.astype(int)).max():>9}")
    del base
    gc.collect()
    torch.cuda.empty_cache()

    rows = []
    for name in list(orig):
        gc.collect()
        torch.cuda.empty_cache()
        # 换掉该模块 → 跑 → 换回
        setattr(net, name, CastWrap(orig[name]))
        try:
            with torch.inference_mode():
                out = e16.upscale_chunk(src)
            p = psnr(ref, out)
            mx = np.abs(ref.astype(int) - out.astype(int)).max()
        except Exception as e:                                   # noqa: BLE001
            p, mx = -1, -1
            print(f"  {name:<34} 失败 {type(e).__name__}: {str(e)[:60]}")
        finally:
            setattr(net, name, orig[name])
        if p > 0:
            print(f"  {name:<34}{p:>8.2f}dB{mx:>9}")
            rows.append((name, p, mx))
        out = None
        gc.collect()
        torch.cuda.empty_cache()

    rows.sort(key=lambda r: -r[1])
    print(f"\n  ── 换成 fp32 后提升最大的模块 ──")
    for name, p, mx in rows[:5]:
        print(f"     {name:<28} {p:.2f} dB")


if __name__ == "__main__":
    main()
