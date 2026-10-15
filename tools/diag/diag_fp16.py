"""定位 RIFE fp16 数值误差出在哪一层。

逐层对比 fp32 / fp16 前向的中间量（flow / mask / warp 结果），
找出误差从哪一步开始爆掉。
"""
from __future__ import annotations

import os
import sys

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

from config import RIFE_SRC, RIFE_WEIGHT  # noqa: E402

sys.path.insert(0, RIFE_SRC)

from rife_ms.IFNet_HDv3 import IFNet  # noqa: E402
from rife_ms import warplayer  # noqa: E402


def load(dtype):
    net = IFNet()
    sd = torch.load(RIFE_WEIGHT, map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in sd.items() if "module." in k}
    net.load_state_dict(sd)
    return net.eval().cuda().to(dtype)


def make_pair(h, w, dev):
    small = torch.rand(1, 3, h // 16, w // 16, device=dev)
    base = torch.nn.functional.interpolate(small, size=(h, w), mode="bicubic",
                                           align_corners=False).clamp_(0, 1)
    return base, torch.roll(base, shifts=(5, 9), dims=(2, 3))


def rel(a, b):
    a = a.float()
    b = b.float()
    denom = b.abs().mean().clamp_min(1e-6)
    return float((a - b).abs().mean() / denom), float((a - b).abs().max())


def main():
    for h, w in ((512, 288), (2048, 1152), (5120, 2880)):
        a, b = make_pair(h, w, "cpu")
        x = torch.cat([a, b], 1).cuda()
        print(f"\n=== {w}x{h} ===")
        for warp32 in (False, True):
            warplayer.FORCE_FP32_WARP = warp32
            n32 = load(torch.float32)
            n16 = load(torch.float16)
            with torch.inference_mode():
                f32, m32, mg32 = n32(x, scale_list=[4.0, 2.0, 1.0])
                f16, m16, mg16 = n16(x.half(), scale_list=[4.0, 2.0, 1.0])
            tag = "warp fp32" if warp32 else "warp fp16"
            rm, xm = rel(mg16[2], mg32[2])
            rm0, xm0 = rel(mg16[0], mg32[0])
            print(f"  [{tag}] flow[0..2] 相对误差:",
                  " ".join(f"{rel(f16[i], f32[i])[0]:.2e}" for i in range(3)))
            print(f"  [{tag}] mask(末层) 相对误差 {rel(m16, m32)[0]:.2e} | "
                  f"warped@scale8 {rm0:.2e} | warped@full {rm:.2e}")
            out32 = mg32[2].clamp(0, 1)
            out16 = mg16[2].float().clamp(0, 1)
            mse = float(((out32 - out16) ** 2).mean())
            ps = 10 * torch.log10(torch.tensor(1.0 / max(mse, 1e-12))).item()
            print(f"  [{tag}] 最终输出 PSNR {ps:.2f} dB   "
                  f"最大像素差 {(out32-out16).abs().max():.4f}")


if __name__ == "__main__":
    main()
