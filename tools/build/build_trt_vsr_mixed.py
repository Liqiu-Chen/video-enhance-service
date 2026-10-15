"""给 VSR 的拆解版 ONNX 做混合精度手术：GridSample / Resize 拉回 fp32。

为什么这样能修好 TRT 的精度
--------------------------
PyTorch 的 `autocast` 有一条重要规则：**自定义/未注册的算子默认按 fp32 跑**。
`F.grid_sample`、`F.interpolate`、`torchvision.ops.deform_conv2d` 都在 autocast
的 fp32 名单里 —— 所以 PyTorch 走 autocast 时精度能到 66.77 dB。

TensorRT 的 `BuilderFlag.FP16` 是**整体降精度**，等价于旧的 `net.half()`
（实测 30 dB）。要做的是把 autocast 的选择**显式写进 ONNX 图**：
  1. 整图转 fp16（权重+张量）
  2. 在每个 GridSample / Resize 前后插 Cast：输入升 fp32、输出降回 fp16
  3. 用 STRONGLY_TYPED 建网络且**不加** FP16 flag → TRT 严格按图里的类型执行

注意：拆解版里可变形卷积已被换成 grid_sample，所以敏感算子天然覆盖到了。

    python build_trt_vsr_mixed.py --t 8 --size 836 480 --scale 4
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

ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "engine"))

TRT_DIR = os.path.join(ROOT, "trt_cache")


def make_mixed(src_onnx: str, dst_onnx: str, ops=("GridSample", "Resize"),
               verbose: bool = True) -> str:
    """fp32 ONNX → 全 fp16 + 指定算子段 fp32 的混合图。"""
    import onnx
    from onnx import helper, TensorProto
    from onnxruntime.transformers.float16 import convert_float_to_float16

    t0 = time.time()
    m = onnx.load(src_onnx, load_external_data=True)
    if verbose:
        print(f"  已载入 ONNX（含外部数据）{time.time()-t0:.0f}s", flush=True)
    m16 = convert_float_to_float16(m, keep_io_types=True)
    del m
    if verbose:
        print(f"  已转 fp16 {time.time()-t0:.0f}s", flush=True)

    g = m16.graph
    nodes = list(g.node)
    inserts = []
    n_hit = {}
    for i, nd in enumerate(nodes):
        if nd.op_type not in ops:
            continue
        n_hit[nd.op_type] = n_hit.get(nd.op_type, 0) + 1
        pre = []
        # 输入侧：★ 只转「数据输入」 ★
        # Resize 的 roi/scales/sizes 是 INT/尺度类，转 fp32 会让 TRT 报
        # `ValueType::kINT` 断言失败；Gather/Slice 的 indices 同理。
        # GridSample 只有 X/grid 两个浮点输入，全转即可。
        if nd.op_type in ("Resize", "Upsample"):
            _idxs = [0]
        elif nd.op_type in ("Gather", "Slice"):
            _idxs = [0]
        else:
            _idxs = [j for j, x in enumerate(nd.input) if x]
        for j in _idxs:
            inp = nd.input[j]
            c = f"vmp_{i}_{j}"
            pre.append(helper.make_node("Cast", [inp], [c],
                                        to=TensorProto.FLOAT, name=f"cf32_{i}_{j}"))
            nd.input[j] = c
        # 输出侧：算子写到中间名，再用 Cast 转回**原名字**，
        # 这样下游引用旧名的节点完全不用动（否则会断链）。
        oo = nd.output[0]
        mid = f"vmp_m_{i}"
        nd.output[0] = mid
        post = helper.make_node("Cast", [mid], [oo],
                                to=TensorProto.FLOAT16, name=f"cf16_{i}")
        inserts.append((i, pre + [post]))

    for idx, ns in sorted(inserts, key=lambda x: -x[0]):
        for k, nn in enumerate(ns):
            g.node.insert(idx + k, nn)
    if verbose:
        print(f"  已把 {n_hit} 拉回 fp32（插 {sum(len(x[1]) for x in inserts)} 个 Cast）"
              f"  {time.time()-t0:.0f}s", flush=True)
    onnx.save(m16, dst_onnx)
    if verbose:
        print(f"  已保存 {os.path.getsize(dst_onnx)/2**20:.1f} MB  {time.time()-t0:.0f}s")
    return dst_onnx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, nargs=2, default=[836, 480], help="输入 H W")
    ap.add_argument("--t", type=int, default=8)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--ops", default="GridSample,Resize")
    ap.add_argument("--workspace-gb", type=float, default=6.0)
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()
    h, w = args.size
    tag = f"t{args.t}_{h}x{w}_x{args.scale}"
    src = os.path.join(TRT_DIR, f"vsr_decomp_{tag}.onnx")
    if not os.path.exists(src):
        print(f"❌ 找不到 {src}\n   先跑 build_trt_vsr_decomp.py --no-trt")
        return 1
    suffix = "_".join(sorted(args.ops))
    mix = os.path.join(TRT_DIR, f"vsr_mix_{tag}.onnx")
    eng = os.path.join(TRT_DIR, f"vsr_mix_{tag}_{suffix}.trt")

    print(f"VSR 混合精度引擎  {tag}  敏感算子 = {args.ops}\n")
    if not os.path.exists(mix):
        make_mixed(src, mix, ops=tuple(args.ops.split(",")))
    else:
        print(f"  复用已有混合 ONNX")

    if args.skip_build:
        return 0

    print(f"\n  构建 TensorRT 引擎（strongly typed）...", flush=True)
    from build_trt_rife_mixed import build_strongly_typed
    os.chdir(TRT_DIR)          # TRT 按 CWD 找外部数据
    try:
        build_strongly_typed(os.path.basename(mix), eng,
                             workspace_gb=args.workspace_gb)
    except Exception as e:                                       # noqa: BLE001
        print(f"  ❌ 失败：{type(e).__name__}: {str(e)[:300]}")
        return 1
    print("  ✅ 完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
