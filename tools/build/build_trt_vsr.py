"""尝试把 BasicVSR++（VSR）导出 ONNX / 编译 TensorRT，并如实报告卡在哪。

为什么大概率走不通：
  BasicVSR++ 的核心是 `SecondOrderDeformableAlignment`，内部用
  `torchvision.ops.deform_conv2d(..., mask=...)` —— **带掩码的可变形卷积**。
  - ONNX 标准算子到 opset 28 也只有 `DeformConv`（无 mask），
    没有"带掩码"的版本；
  - PyTorch 的 ONNX 导出器对 `torchvision.ops.deform_conv2d` 没有注册符号
    （symbolic），所以根本导不出图；
  - TensorRT 也没有原生的 DeformConv，社区要靠自己编译 plugin。

  所以这个脚本的作用是**用证据确认结论**，而不是指望它能成功。
  如果真导出来了（比如未来 PyTorch/TensorRT 支持了），脚本会自动继续建 TRT 引擎。

    python build_trt_vsr.py --size 60 107        # 输入 (1, t=60, 3, H, W) 的 H W
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

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

from config import CACHE, VSR_WEIGHT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=int, default=2, help="段长（t 维）")
    ap.add_argument("--size", type=int, nargs=2, default=[836, 480], help="输入 H W")
    ap.add_argument("--out", default=None)
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()
    h, w = args.size
    out = args.out or os.path.join(CACHE, f"vsr_t{args.seg}.onnx")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print(f"BasicVSR++ → ONNX 可行性验证")
    print(f"  输入 (1, {args.seg}, 3, {h}, {w})  opset {args.opset}\n")

    from vsr_engine import VsrEngine
    eng = VsrEngine(weight=VSR_WEIGHT, fp16=False, chunk=args.seg,
                    out_scale=4, device="cpu", verbose=False)
    net = eng.net.eval()
    dummy = torch.zeros(1, args.seg, 3, h, w)

    # 先看网络里到底有哪些算子会被 trace 到
    print("  尝试 torch.onnx.export ...", flush=True)
    try:
        torch.onnx.export(
            net, (dummy,), out, input_names=["lqs"], output_names=["hr"],
            opset_version=args.opset, dynamo=False, do_constant_folding=True)
        print(f"  ✅ 导出成功 → {out}  {os.path.getsize(out)/2**20:.1f} MB")
        ok = True
    except Exception as e:                                       # noqa: BLE001
        ok = False
        print(f"  ❌ 导出失败：{type(e).__name__}: {str(e)[:400]}")
        # 只在需要细节时才打完整栈
        if os.environ.get("VSR_TRACE"):
            traceback.print_exc()

    if not ok:
        print("\n" + "=" * 66)
        print("  结论：BasicVSR++ 无法导出 ONNX → 也就无法编译 TensorRT 引擎。")
        print("  根因：SecondOrderDeformableAlignment 用的『带掩码可变形卷积』")
        print("        torchvision.ops.deform_conv2d(mask=...) 没有 ONNX 算子，")
        print("        PyTorch 导出器也没有为它注册 symbolic。")
        print("  TensorRT 同样没有原生 DeformConv，需要自行编译 plugin（成本高、")
        print("        且要保证 plugin 与 TRT 版本的 ABI 兼容）。")
        print("=" * 66)
        # 把结论落盘，免得以后又有人来试一遍
        import json
        with open(os.path.join(CACHE, "trt_vsr_feasibility.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"exportable": False,
                       "reason": "modulated deformable conv (torchvision.ops."
                                 "deform_conv2d with mask) has no ONNX op "
                                 "and no torch exporter symbolic",
                       "opset_tried": args.opset,
                       "input": [1, args.seg, 3, h, w]}, f,
                      ensure_ascii=False, indent=2)
        return 1

    # 万一成功，继续建 TRT
    print("\n  继续构建 TensorRT 引擎 ...")
    from build_trt_rife import build_engine
    eng_path = out.replace(".onnx", ".fp16.trt")
    build_engine(out, eng_path, fp16=True)
    print("  ✅ 完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
