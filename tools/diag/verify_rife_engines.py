"""RIFE 各引擎的最终验收：速度 + 精度 + **取整后是否与 fp32 完全一致**。

为什么加"取整一致率"这个指标：
  视频链路最终会落到 uint8。如果取整后逐像素完全相同，那这个引擎对这个
  画面就是**无损的** —— 比 PSNR 更能说明问题（PSNR 48dB 听着一般，
  但如果取整后 100% 一致，实际就是没差别）。
"""
from __future__ import annotations

import gc
import os
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

from config import RIFE_WEIGHT  # noqa: E402
from bench_trt_vsr import make_frames  # noqa: E402

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def psnr_np(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main():
    h, w = 1696, 960
    frames = make_frames(h, w, 6)
    pairs = []
    for i in range(len(frames) - 1):
        a = torch.from_numpy(frames[i:i + 1]).cuda().permute(0, 3, 1, 2)
        b = torch.from_numpy(frames[i + 1:i + 2]).cuda().permute(0, 3, 1, 2)
        pairs.append((a.float().div_(255.0), b.float().div_(255.0)))

    # ---- fp32 真值 ----
    from rife_engine import RifeEngine
    e = RifeEngine(weight=RIFE_WEIGHT, fp16=False, batch=1, verbose=False)
    e.build(h, w)
    with torch.inference_mode():
        ref = [e.infer(a, b) for a, b in pairs]
    del e
    gc.collect()
    torch.cuda.empty_cache()
    R = torch.cat(ref, 0).float()
    del ref
    gc.collect()

    # ---- PyTorch fp16+warp32（现行生产配置）用于测速 ----
    e16 = RifeEngine(weight=RIFE_WEIGHT, fp16=True, batch=1, warp_fp32=True,
                     verbose=False)
    e16.build(h, w)
    a0, b0 = pairs[0][0].half(), pairs[0][1].half()
    with torch.inference_mode():
        for _ in range(5):
            e16.infer(a0, b0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            e16.infer(a0, b0)
        torch.cuda.synchronize()
        ms_torch = (time.perf_counter() - t0) / 20 * 1000
        outs16 = [e16.infer(a.half(), b.half()).float() for a, b in pairs]
    del e16
    gc.collect()
    torch.cuda.empty_cache()

    print(f"  {'引擎':<26}{'ms/对':>8}{'加速比':>8}{'PSNR':>9}{'最大差':>8}"
          f"{'取整一致':>10}")

    idx16 = torch.cat(outs16, 0)
    del outs16
    gc.collect()

    rows = [("PyTorch fp16+warp32", ms_torch, idx16)]
    from build_trt_rife import TrtRife
    for tag, f in (("TRT-fp16", "rife_1696x960_fp16.trt"),
                   ("TRT-strongmix（本方案）", "rife_1696x960_strongmix.trt")):
        p = os.path.join(HERE, "cache", f)
        if not os.path.exists(p):
            continue
        t = TrtRife(p)
        with torch.inference_mode():
            x = torch.cat([a0.to(t.in_dtype), b0.to(t.in_dtype)], 1)
            for _ in range(5):
                t(x)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                t(x)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / 20 * 1000
            o = torch.cat([t(torch.cat([a.to(t.in_dtype), b.to(t.in_dtype)], 1))
                           .float() for a, b in pairs], 0)
        rows.append((tag, ms, o))
        del t, o
        gc.collect()
        torch.cuda.empty_cache()

    Ru = (R.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    for tag, ms, o in rows:
        Ou = (o.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        Rf = R.cpu().numpy()
        Of = o.cpu().numpy()
        ident = float((Ru == Ou).mean() * 100)
        print(f"  {tag:<26}{ms:>8.1f}{ms_torch/ms:>7.2f}x"
              f"{psnr_np(Ru, Ou):>8.2f}dB"
              f"{np.abs(Rf-Of).max()*255:>8.2f}{ident:>9.2f}%")
        del o, Ou, Of
        gc.collect()
        torch.cuda.empty_cache()
    print("\n  注：'取整一致' = 落到 uint8 后逐像素完全相同的比例，100% 即对该画面无损")


if __name__ == "__main__":
    main()
