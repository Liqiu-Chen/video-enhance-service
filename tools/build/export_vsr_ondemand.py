"""验证「dummy 用全零导致导出缺分支」这个猜想。

背景：
  BasicVSRPlusPlusNet 里有个**实例标志** `is_mirror_extended`，
  由 `check_if_mirror_extended(lqs)` 在 forward 时设置：

      if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
          self.is_mirror_extended = True        # 只置 True，永不重置

  而 `compute_flow` 里：

      if self.is_mirror_extended:
          flows_forward = None                  # ← 跳过正向光流计算

  导出时如果 dummy 是 `torch.zeros(...)`，则 norm(0)==0 成立 →
  标志被置 True → **导出的图里没有正向光流分支**，是个残缺的网络。

本脚本用**真实帧**做 dummy 重新导出，并用 ONNX Runtime 对比精度。

    python export_vsr_ondemand.py --t 2 --size 836 480
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

ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "engine"))

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, nargs=2, default=[836, 480])
    ap.add_argument("--t", type=int, default=2)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--zeros", action="store_true", help="用全零 dummy（复现旧 bug）")
    args = ap.parse_args()
    h, w = args.size
    D = os.path.join(ROOT, "trt4")
    os.makedirs(D, exist_ok=True)
    tag = "zeros" if args.zeros else "real"
    onnx_p = os.path.join(D, f"vsr_{tag}_t{args.t}.onnx")

    from config import VSR_WEIGHT
    from vsr_engine import VsrEngine
    from build_trt_vsr_decomp import patch_deform
    from bench_pipe_vs_iso import decode_frames

    src = decode_frames(os.path.join(ROOT, "liuqiang.mp4"), args.t)
    real = torch.from_numpy(np.ascontiguousarray(src)).permute(0, 3, 1, 2)[None]
    real = real.float().div_(255.0)
    # ★ 必须缩放到导出尺寸 ★（第一次忘了，导致 ORT 报输入维度不符）
    if real.shape[-2:] != (h, w):
        real = torch.nn.functional.interpolate(real[0], size=(h, w),
                                               mode="bilinear",
                                               align_corners=False)[None]

    e = VsrEngine(weight=VSR_WEIGHT, fp16=False, chunk=args.t, out_scale=args.scale,
                  device="cpu", verbose=False)
    patch_deform(verbose=False)
    net = e.net.eval()

    if args.zeros:
        dummy = torch.zeros(1, args.t, 3, h, w)
    else:
        dummy = real
    torch.onnx.export(net, (dummy,), onnx_p, input_names=["lqs"],
                      output_names=["hr"], opset_version=17, dynamo=False,
                      do_constant_folding=False)
    flag = getattr(net, "is_mirror_extended", None)
    print(f"  [{tag}] 导出完成  is_mirror_extended={flag}  "
          f"({onnx_p}  {os.path.getsize(onnx_p)/2**20:.1f} MB)")

    # PyTorch 参考
    with torch.inference_mode():
        ref = e.net(real)[0].permute(0, 2, 3, 1).clamp(0, 1).mul(255).round()
        ref = ref.to(torch.uint8).numpy()
    del e, net
    import gc
    gc.collect()

    # ORT
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    cwd = os.getcwd()
    os.chdir(D)
    try:
        o = ort.InferenceSession(onnx_p, so,
                                 providers=["CPUExecutionProvider"]).run(
            None, {"lqs": real.numpy().astype(np.float32)})[0]
    finally:
        os.chdir(cwd)
    # 输出是 (1, t, 3, 4H, 4W) —— 5D，转成 (t, 4H, 4W, 3) 再比
    o = np.transpose(o[0], (0, 2, 3, 1))
    o = np.round(np.clip(o, 0, 1) * 255).astype(np.uint8)
    d = np.abs(ref.astype(np.float64) - o.astype(np.float64))
    mse = np.mean(d ** 2)
    p = 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    print(f"  ORT vs PyTorch: PSNR {p:.2f} dB  最大差 {d.max():.1f}  "
          f">2/255 {(d>2).mean()*100:.3f}%")
    print(f"  {'✅ 导出忠实（图完整）' if p > 70 else '❌ 导出丢精度（图可能残缺）'}")


if __name__ == "__main__":
    main()
