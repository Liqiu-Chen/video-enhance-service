"""定位 VSR 拆解版精度下降的真正来源（逐帧 + ONNX 对照）。

之前的假设「grid_sample 边界语义不同」并没有被证实 —— 单元测试误差只有 3e-6。
所以先用两个决定性实验分清责任：

  实验 1：ONNX Runtime 的输出 vs PyTorch 原实现
          - 若一致  → 导出/拆解没问题，问题在 TensorRT 那一侧
          - 若不一致 → 导出或拆解本身有问题
  实验 2：逐帧 PSNR
          - 若首帧正确、误差随后帧递增 → 时序递归把误差放大了
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
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def make_frames(h, w, t):
    rng = np.random.default_rng(0)
    small = rng.random((h // 16, w // 16, 3)).astype(np.float32)
    tt = torch.from_numpy(small).permute(2, 0, 1)[None]
    big = torch.nn.functional.interpolate(tt, size=(h, w), mode="bicubic",
                                          align_corners=False)[0]
    big = (big.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return np.stack([np.roll(big, (i, i * 2), axis=(0, 1)).copy()
                     for i in range(t)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, nargs=2, default=[836, 480])
    ap.add_argument("--t", type=int, default=8)
    ap.add_argument("--scale", type=int, default=4)
    args = ap.parse_args()
    h, w = args.size
    frames = make_frames(h, w, args.t)
    print(f"输入 {args.t} 帧 {w}x{h}  out_scale={args.scale}\n")

    from vsr_engine import VsrEngine
    # ---------- 参考：原实现 ----------
    e1 = VsrEngine(weight=VSR_WEIGHT, fp16=True, chunk=args.t, out_scale=args.scale,
                   verbose=False)
    x = torch.from_numpy(frames).cuda().permute(0, 3, 1, 2)[None]
    x = x.float().div_(255.0).half()
    with torch.inference_mode():
        ref = e1.net(x)[0].permute(0, 2, 3, 1).float().clamp(0, 1)
        ref = ref.mul(255).round().to(torch.uint8).cpu().numpy()

    # ---------- 拆解版（PyTorch 内）----------
    import vsr.basicvsr_plusplus_net as bnet
    from deform_decompose import deform_conv2d_decomposed

    def _pair(v):
        return (v, v) if isinstance(v, int) else tuple(v)

    orig = bnet.modulated_deform_conv2d

    def patched(a, offset, mask, weight, bias=None, stride=1, padding=0,
                dilation=1, groups=1, deform_groups=1):
        return deform_conv2d_decomposed(
            a, offset, mask, weight, bias, stride=_pair(stride),
            padding=_pair(padding), dilation=_pair(dilation),
            deform_groups=deform_groups)
    bnet.modulated_deform_conv2d = patched
    e2 = VsrEngine(weight=VSR_WEIGHT, fp16=True, chunk=args.t, out_scale=args.scale,
                   verbose=False)
    with torch.inference_mode():
        dec = e2.net(x)[0].permute(0, 2, 3, 1).float().clamp(0, 1)
        dec = dec.mul(255).round().to(torch.uint8).cpu().numpy()
    bnet.modulated_deform_conv2d = orig

    print("  实验 2：逐帧 PSNR（原实现 vs 拆解版，同在 PyTorch fp16 内）")
    print(f"    {'帧':<6}{'PSNR':>10}{'最大差':>9}{'平均差':>10}")
    for i in range(args.t):
        d = np.abs(ref[i].astype(np.float64) - dec[i].astype(np.float64))
        print(f"    #{i:<5}{psnr(ref[i], dec[i]):>9.2f}dB{d.max():>9.0f}{d.mean():>10.4f}")
    d = np.abs(ref.astype(np.float64) - dec.astype(np.float64))
    print(f"    {'整体':<6}{psnr(ref, dec):>9.2f}dB{d.max():>9.0f}{d.mean():>10.4f}")

    # ---------- 实验 1：ONNX Runtime ----------
    onnx_p = os.path.join(os.path.dirname(HERE), "trt_cache",
                          f"vsr_decomp_t{args.t}_{h}x{w}_x{args.scale}.onnx")
    if os.path.exists(onnx_p):
        print(f"\n  实验 1：ONNX Runtime vs PyTorch 原实现")
        try:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            cwd0 = os.getcwd()
            os.chdir(os.path.dirname(onnx_p))       # 外部数据按 CWD 解析
            try:
                sess = ort.InferenceSession(onnx_p, so,
                                            providers=["CUDAExecutionProvider"])
                out = sess.run(None, {"lqs": x.cpu().numpy().astype(np.float32)})[0]
            finally:
                os.chdir(cwd0)
            o = np.transpose(out[0], (1, 2, 0))
            o = np.clip(o, 0, 1) * 255
            o = np.round(o).astype(np.uint8)
            print(f"    整体 PSNR {psnr(ref, o):.2f} dB")
            print(f"    逐帧: " + "  ".join(
                f"#{i}:{psnr(ref[i], o[i]):.1f}" for i in range(args.t)))
        except Exception as e:                                   # noqa: BLE001
            print(f"    （ORT 对照跳过：{type(e).__name__}: {str(e)[:150]}）")
    else:
        print(f"\n  实验 1 跳过：找不到 {onnx_p}")


if __name__ == "__main__":
    main()
