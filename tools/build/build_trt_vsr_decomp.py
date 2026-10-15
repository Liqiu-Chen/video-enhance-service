"""把 BasicVSR++（可变形卷积已换成原生算子）导出 ONNX 并编译 TensorRT。

前置条件：`vsr_decompose_test.py` 已验证拆解后整网输出一致（78 dB）。

    python build_trt_vsr_decomp.py --size 836 480 --t 2
"""
from __future__ import annotations

import argparse
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

from config import CACHE, VSR_WEIGHT  # noqa: E402
from bench_pipe_vs_iso import decode_frames  # noqa: E402

# 导出必须关 TF32：一是数值要可复现，二是 TF32 会让"拆解 vs 原实现"的
# 对照凭空多出 3e-4 的误差（已实测）。
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = False


def patch_deform(verbose: bool = True):
    """把 BasicVSR++ 里的可变形卷积换成原生算子拆解版。"""
    import vsr.basicvsr_plusplus_net as bnet
    from deform_decompose import deform_conv2d_decomposed

    def _pair(v):
        return (v, v) if isinstance(v, int) else tuple(v)

    n = [0]

    def patched(x, offset, mask, weight, bias=None, stride=1, padding=0,
                dilation=1, groups=1, deform_groups=1):
        n[0] += 1
        assert groups == 1, "拆解版暂不支持 groups != 1"
        return deform_conv2d_decomposed(
            x, offset, mask, weight, bias,
            stride=_pair(stride), padding=_pair(padding),
            dilation=_pair(dilation), deform_groups=deform_groups)

    bnet.modulated_deform_conv2d = patched
    if verbose:
        print("  已打补丁：可变形卷积 → grid_sample + mul + 1x1conv")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, nargs=2, default=[836, 480], help="输入 H W")
    ap.add_argument("--t", type=int, default=2, help="时序长度（会展开进图，别太大）")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--mode", default="fp16", choices=["fp16", "fp32", "mix"])
    ap.add_argument("--no-trt", action="store_true", help="只导出 ONNX，不建引擎")
    ap.add_argument("--outdir", default=None,
                    help="输出目录（必须纯 ASCII，默认 <项目>/trt_cache）")
    args = ap.parse_args()
    h, w = args.size
    tag = f"t{args.t}_{h}x{w}_x{args.scale}"
    # ⚠️ 必须用**纯 ASCII 路径**。
    # 放在 `推理服务/cache/` 下时导出会报
    #   `Could not open file or directory: .../cache/_Constant_18_attr__value`
    # —— PyTorch 的 ONNX 导出器在处理中间常量文件时不支持路径里的非 ASCII 字符。
    outdir = args.outdir or os.path.join(os.path.dirname(HERE), "trt_cache")
    os.makedirs(outdir, exist_ok=True)
    onnx_p = os.path.join(outdir, f"vsr_decomp_{tag}.onnx")
    eng_p = os.path.join(outdir, f"vsr_decomp_{tag}_{args.mode}.trt")

    print(f"BasicVSR++（拆解版）→ ONNX → TensorRT")
    print(f"  输入 (1, {args.t}, 3, {h}, {w})  out_scale={args.scale}  opset={args.opset}\n")

    calls = None
    from vsr_engine import VsrEngine
    # 先建引擎：VsrEngine 加载时才会把 VSR 源码目录加进 sys.path，
    # 之后 `import vsr.basicvsr_plusplus_net` 才找得到。
    eng = VsrEngine(weight=VSR_WEIGHT, fp16=False, chunk=args.t,
                    out_scale=args.scale, device="cpu", verbose=False)
    calls = patch_deform()
    net = eng.net.eval()
    # ★★★ 绝不能用 torch.zeros 做 dummy ★★★
    # BasicVSRPlusPlusNet 里有个实例标志 `is_mirror_extended`，由
    # `check_if_mirror_extended(lqs)` 在 forward 时设置：
    #     if torch.norm(lqs_1 - lqs_2.flip(1)) == 0: self.is_mirror_extended = True
    # **全零输入时该条件恒成立**，而 compute_flow 里：
    #     if self.is_mirror_extended: flows_forward = None    # 跳过正向光流！
    # 于是导出的图**缺少正向光流分支**，是个结构残缺的网络。
    # 实测（ORT vs PyTorch）：
    #     全零 dummy : 55.89 dB / 最大差 32   ← 图残缺
    #     真实帧 dummy: 98.64 dB / 最大差  1   ← 完全忠实
    # 这正是把 TRT 精度一直卡在 31 dB 的元凶。
    _s = decode_frames(os.path.join(HERE, "..", "liuqiang.mp4"), args.t)
    _r = torch.from_numpy(np.ascontiguousarray(_s)).permute(0, 3, 1, 2)[None]
    _r = _r.float().div_(255.0)
    if _r.shape[-2:] != (h, w):
        _r = torch.nn.functional.interpolate(_r[0], size=(h, w), mode="bilinear",
                                             align_corners=False)[None]
    dummy = _r
    print(f"  dummy 用真实帧（非 zero），形状 {tuple(dummy.shape)}")

    # ---- 导出 ----
    print("  导出 ONNX（会展开 t 帧的循环，可能较慢）...", flush=True)
    t0 = time.time()
    try:
        with torch.inference_mode():
            torch.onnx.export(
                net, (dummy,), onnx_p, input_names=["lqs"], output_names=["hr"],
                # ⚠️ 不要开常量折叠。
                # 拆解版里的采样网格是由 arange + 广播拼出来的**常量**，
                # 一折叠就会被物化成 14~100 MB 的巨型常量塞进 ONNX：
                # 实测外部数据因此膨胀到 1.4 GB（模型本体才 29 MB），
                # 而且 TensorRT 还要按 CWD 去逐个找这些外部文件，很容易失败。
                # 关掉折叠后，网格在运行时用 arange 现算，图小、外部数据也小。
                opset_version=args.opset, dynamo=False, do_constant_folding=False)
    except Exception as e:                                       # noqa: BLE001
        print(f"  ❌ 导出失败：{type(e).__name__}: {str(e)[:500]}")
        import traceback
        if os.environ.get("VSR_TRACE"):
            traceback.print_exc()
        return 1
    dt = time.time() - t0
    print(f"  ✅ 导出成功 {dt:.1f}s  {os.path.getsize(onnx_p)/2**20:.1f} MB  "
          f"（替换了 {calls[0]} 次可变形卷积）")

    # 顺带报一下图里有哪些算子，确认没有自定义算子残留
    try:
        import onnx
        m = onnx.load(onnx_p)
        from collections import Counter
        cnt = Counter(n.op_type for n in m.graph.node)
        bad = [k for k in cnt if k.lower() in
               ("deformconv", "modulateddeformconv2d", "customop")]
        top = ", ".join(f"{k}×{v}" for k, v in cnt.most_common(10))
        print(f"  算子统计: {top}")
        print(f"  自定义算子残留: {bad if bad else '无 ✅'}")
    except Exception as e:                                       # noqa: BLE001
        print(f"  （算子统计跳过：{e}）")

    if args.no_trt:
        return 0

    # ---- 建 TensorRT 引擎 ----
    print(f"\n  构建 TensorRT 引擎（{args.mode}）...", flush=True)
    from build_trt_rife import build_engine
    # TensorRT 的 ONNX parser 按**当前工作目录**解析外部数据文件，
    # 不按 onnx 文件所在目录，所以必须切到该目录再解析。
    cwd0 = os.getcwd()
    os.chdir(outdir)
    try:
        # fp32 在 t=8 这种大图上，单个 tactic 就要 9+ GB 工作区，
        # 默认 6 GB 会让 TRT 直接报 "insufficient memory ... Could not find
        # any implementation"。给到 12 GB 才够（16 GB 卡的可用余量）。
        build_engine(os.path.basename(onnx_p), eng_p, mode=args.mode,
                     workspace_gb=12.0)
    except Exception as e:                                       # noqa: BLE001
        print(f"  ❌ 建引擎失败：{type(e).__name__}: {str(e)[:400]}")
        return 1
    finally:
        os.chdir(cwd0)
    print("  ✅ 完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
