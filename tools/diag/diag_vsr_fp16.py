"""查 VSR 的 fp16 到底哪里出了问题。

用户质疑得对：**优秀的卷积网络切 fp16 不该掉到 28 dB**。
正常应该是 50~60 dB。28 dB + 最大差 113 说明有真 bug，不是"精度损失"。

头号怀疑：`VsrEngine(fp16=True)` 用的是 `net = net.half()` —— **整网粗暴转 fp16**。
而正确的做法是 `torch.autocast`（混合精度）：它会把数值敏感的算子
（softmax / norm / 累加 / 相关运算）自动留在 fp32。

BasicVSR++ 里有 **SpyNet 光流估计**，其相关层是点积求和 —— fp16 下极易溢出。

    python diag_vsr_fp16.py --size 836 480 --t 4
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

from config import VSR_WEIGHT  # noqa: E402

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = False


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def stats(tag, ref, got):
    d = np.abs(ref.astype(np.float64) - got.astype(np.float64))
    q = np.percentile(d, [50, 90, 99, 99.9])
    print(f"  {tag:<34}{psnr(ref,got):>8.2f}dB{d.max():>8.1f}"
          f"{q[0]:>7.1f}{q[1]:>7.1f}{q[2]:>7.1f}{q[3]:>8.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, nargs=2, default=[836, 480])
    ap.add_argument("--t", type=int, default=4)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--clip", default=os.path.join(HERE, "..", "liuqiang.mp4"))
    args = ap.parse_args()
    h, w = args.size

    from bench_pipe_vs_iso import decode_frames
    src = decode_frames(args.clip, args.t)
    print(f"真实连续帧 {args.t} 帧 {w}x{h} → {args.scale}x\n")

    from vsr_engine import VsrEngine
    from vsr.basicvsr_plusplus_net import BasicVSRPlusPlusNet  # noqa: F401

    # ---------- 参考：fp32 ----------
    e32 = VsrEngine(weight=VSR_WEIGHT, fp16=False, chunk=args.t,
                    out_scale=args.scale, verbose=False)
    e32.warmup(h, w)
    with torch.inference_mode():
        ref = e32.upscale_chunk(src)
    net32 = e32.net            # 直接拿到网络，后面自己喂
    print("  [参考 = fp32 引擎输出]\n")
    print(f"  {'实现':<34}{'PSNR':>8}{'最大':>8}{' 50%':>7}{' 90%':>7}"
          f"{' 99%':>7}{'99.9%':>8}")

    # ---------- 预处理（与引擎内部一致）----------
    t = torch.from_numpy(np.ascontiguousarray(src)).cuda()
    t = t.permute(0, 3, 1, 2)[None].float().div_(255.0)   # (1,t,3,H,W)

    def to_np(y):
        y = y[0].permute(0, 2, 3, 1).float().clamp(0, 1)
        return y.mul(255).round().to(torch.uint8).cpu().numpy()

    with torch.inference_mode():
        # A. 现方案：net.half() + 输入 half
        net16 = VsrEngine(weight=VSR_WEIGHT, fp16=True, chunk=args.t,
                          out_scale=args.scale, verbose=False)
        net16.warmup(h, w)
        y_a = to_np(net16.net(t.half()))
        stats("A. net.half()（现方案）", ref, y_a)

        # B. 混合精度：权重保持 fp32，前向套 autocast
        y_b = to_np(net32(t))
        #   ^ net32 是 fp16=False 建的，权重本来就是 fp32；先跑一次纯 fp32 对照
        stats("B. 纯 fp32（自检，应≈0）", ref, y_b)

        with torch.autocast("cuda", dtype=torch.float16):
            y_c = to_np(net32(t))
        stats("C. autocast 混合精度", ref, y_c)

        # D. 只把 SpyNet 留 fp32，其余 fp16
        #    SpyNet 的相关层是点积求和，fp16 极易溢出 —— 单独验证
        net16.net.spynet.float()
        y_d = to_np(net16.net(t.half()))
        stats("D. 仅 SpyNet 回 fp32，其余 fp16", ref, y_d)

        # E. 只把 deform_align 留 fp32
        net16.net.spynet.half()
        net16.net.deform_align.float()
        y_e = to_np(net16.net(t.half()))
        stats("E. 仅 deform_align 回 fp32", ref, y_e)

        # F. 两者都留 fp32
        net16.net.spynet.float()
        y_f = to_np(net16.net(t.half()))
        stats("F. SpyNet + deform_align 均 fp32", ref, y_f)


if __name__ == "__main__":
    main()
