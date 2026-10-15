"""拆解 TensorRT 的误差来源：到底差在哪儿。

三方对照，两两比 PSNR：
  1. PyTorch fp32              （数值金标准）
  2. PyTorch fp16 + fp32 warp  （当前生产配置）
  3. TensorRT fp32 / fp16

若 TRT-fp32 vs Torch-fp32 就已经差很多 → 是**图/算子语义**不一致（比如
Resize 的半像素对齐），跟精度无关，再怎么调精度也没用。
若 TRT-fp32 能对上、只有 fp16 差 → 才是精度问题，可以考虑混合精度。

    python diag_trt_rife.py --size 1672 960
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

from config import CACHE, RIFE_WEIGHT  # noqa: E402
from build_trt_rife import TrtRife, pad_to32, psnr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, nargs=2, default=[1672, 960])
    ap.add_argument("--clip", default=os.path.join(HERE, "..", "liuqiang.mp4"))
    ap.add_argument("--noise", action="store_true",
                    help="用随机噪声（病态输入，仅供对照）")
    args = ap.parse_args()
    h, w = pad_to32(*args.size)

    # ⚠️ 素材选择很关键。
    # 之前用**纯随机噪声**，结果连 PyTorch 自己的 fp16 都只有 21 dB —— 噪声是最
    # 病态的输入：RIFE 对随机噪声的输出本身就是"垃圾进垃圾出"，任何数值差异
    # 都会被放大。真实视频帧才有参考价值，所以默认用真帧。
    if args.noise:
        rng = np.random.default_rng(0)
        a8 = rng.integers(0, 256, (1, h, w, 3), dtype=np.uint8)
        b8 = rng.integers(0, 256, (1, h, w, 3), dtype=np.uint8)
        src_tag = "随机噪声（病态，仅对照）"
    else:
        from bench_pipe_vs_iso import decode_frames
        print("解码真实视频帧 ...", flush=True)
        # 先按原始尺寸取 2 帧，再双三次放大到目标尺寸
        raw = decode_frames(args.clip, 2)
        t = torch.from_numpy(raw).permute(0, 3, 1, 2).float() / 255.0
        t = torch.nn.functional.interpolate(t, size=(h, w), mode="bicubic",
                                            align_corners=False)
        arr = (t.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
        a8, b8 = arr[0:1], arr[1:2]
        src_tag = f"真实视频帧（{os.path.basename(args.clip)}）"

    def prep(arr, dt):
        t = torch.from_numpy(arr).cuda().permute(0, 3, 1, 2)
        t = t.float().div_(255.0)
        return t.to(dt)

    print(f"素材: {src_tag}   尺寸 {w}x{h}\n")
    from rife_engine import RifeEngine

    outs = {}
    with torch.inference_mode():
        # 1) PyTorch fp32
        e32 = RifeEngine(weight=RIFE_WEIGHT, fp16=False, batch=1,
                         warp_fp32=False, verbose=False)
        e32.build(h, w)
        outs["torch-fp32"] = e32.infer(prep(a8, torch.float32),
                                       prep(b8, torch.float32))
        # 2) PyTorch fp16 + fp32 warp（现行生产配置）
        e16 = RifeEngine(weight=RIFE_WEIGHT, fp16=True, batch=1,
                         warp_fp32=True, verbose=False)
        e16.build(h, w)
        outs["torch-fp16w32"] = e16.infer(prep(a8, torch.float16),
                                          prep(b8, torch.float16))
        # 3) TensorRT
        x16 = torch.cat([prep(a8, torch.float16), prep(b8, torch.float16)], 1)
        x32 = torch.cat([prep(a8, torch.float32), prep(b8, torch.float32)], 1)
        for tag in ("fp32", "fp16"):
            p = os.path.join(CACHE, f"rife_{h}x{w}_{tag}.trt")
            if os.path.exists(p):
                t = TrtRife(p)
                outs[f"trt-{tag}"] = t(x32 if t.in_dtype == torch.float32 else x16)

    keys = list(outs)
    print(f"  {'对比':<34}{'PSNR':>10}{'最大差':>10}")
    base = outs["torch-fp32"]
    for k in keys:
        y = outs[k]
        if k == "torch-fp32":
            continue
        p = psnr(base, y.to(base.dtype))
        md = (base.float() - y.float()).abs().max().item()
        print(f"  {('torch-fp32  vs  ' + k):<34}{p:>9.2f}dB{md:>10.4f}")

    # 结论
    print()
    trt32 = outs.get("trt-fp32")
    if trt32 is not None:
        p32 = psnr(base, trt32.to(base.dtype))
        if p32 < 45:
            print("  ⚠️ TRT-fp32 与 Torch-fp32 都对不上 → **不是精度问题**，")
            print("     而是算子语义/图结构不一致（重点怀疑 Resize 的半像素对齐、")
            print("     或 GridSample 的 padding/align_corners 语义）。")
            print("     这种情况靠调精度解决不了，要改导出图本身。")
        else:
            print("  ✅ TRT-fp32 能对上 Torch-fp32 → 误差来自 **fp16 精度**，")
            print("     可以尝试混合精度（只把 warp 相关的层留在 fp32）。")


if __name__ == "__main__":
    main()
