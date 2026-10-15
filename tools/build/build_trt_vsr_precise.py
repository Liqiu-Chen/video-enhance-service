"""VSR 的 TRT 精度修复：精确锁定「可变形卷积拆解」整段并拉回 fp32。

为什么之前的尝试失败
------------------
1. 按"权重是 1×1"识别 → 找到 0 个。拆解里的权重是 `view→permute→reshape`
   **算出来的**，不是 initializer，静态识别不到。
2. 改成从 GridSample 做图遍历 → 标记过量，走进了 SpyNet 的索引计算链，
   `Gather` 的 indices 被转成 float → TRT 报 `indices must be Int64/Int32`。

这次的做法
---------
**回溯识别**：对每个 Conv，从它的输入沿「生产者」链向上走，如果在有限跳数内
碰到 GridSample，那它就是拆解出来的 1×1 卷积。这条链是
    GridSample*N → Concat/Stack → Mul(mask) → Reshape → Conv(1×1)
所以顺带就能把 mask 的 Mul 一起圈进来。

**类型安全**：只给**浮点**输入插 Cast。按算子逐条给规则：
  Gather/Slice/Resize 的 index/scale/size 输入绝不转；
  Shape/Constant/Cast 本身不参与。

    python build_trt_vsr_precise.py --t 8 --size 836 480 --scale 4
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

TRT_DIR = os.path.join(ROOT, "trt2")

# 每个算子**可以安全转 fp32** 的输入下标。
# None 表示"所有输入"。索引/尺度类输入必须排除，否则 TRT 会报类型错误。
CASTABLE = {
    "Gather": [0],            # 1 是 indices
    "Slice": [0],             # 1..4 是 starts/ends/axes/steps
    "Resize": [0],            # 1..3 是 roi/scales/sizes
    "Constant": [],
    "Shape": [],
    "Cast": [],
    "CastLike": [],
    "ConstantOfShape": [],
    "Range": [],
}
# 这些算子不参与转换（纯形状/索引运算）
NEVER = {"Constant", "Shape", "ConstantOfShape", "Range", "Cast", "CastLike"}


def build_producer(g):
    prod = {}
    for nd in g.node:
        for o in nd.output:
            if o:
                prod[o] = nd
    return prod


def upstream_has(prod, tensor, want, max_hops=14, skip_conv=False):
    """从 tensor 沿生产者链向上找 op_type == want。"""
    frontier, seen, hops = [tensor], set(), 0
    while frontier and hops <= max_hops:
        nxt = []
        for t in frontier:
            if t in seen:
                continue
            seen.add(t)
            nd = prod.get(t)
            if nd is None:
                continue
            if nd.op_type == want:
                return True
            # 遇到 3x3 主干卷积就不要再往上穿了（避免误判）
            if skip_conv and nd.op_type == "Conv":
                continue
            nxt.extend([i for i in nd.input if i])
        frontier = nxt
        hops += 1
    return False


def is_decomp_conv(prod, conv_node):
    """拆解出的 1x1 卷积：其数据输入的上游存在 GridSample。"""
    if not conv_node.input:
        return False
    return upstream_has(prod, conv_node.input[0], "GridSample")


def make_precise(src_onnx: str, dst_onnx: str, verbose: bool = True) -> dict:
    import onnx
    from onnx import helper, TensorProto
    from onnxruntime.transformers.float16 import convert_float_to_float16

    t0 = time.time()
    m = onnx.load(src_onnx, load_external_data=True)
    prod0 = build_producer(m.graph)

    # ---- 选出要拉回 fp32 的张量 ----
    targets = set()      # 输出张量名
    kinds = {}
    # 1) 所有 GridSample
    for nd in m.graph.node:
        if nd.op_type == "GridSample":
            targets.add(nd.output[0])
            kinds["GridSample"] = kinds.get("GridSample", 0) + 1
    # 2) 拆解出的 1x1 卷积（回溯识别）
    n_conv = 0
    for nd in m.graph.node:
        if nd.op_type == "Conv" and is_decomp_conv(prod0, nd):
            targets.add(nd.output[0])
            n_conv += 1
    if n_conv:
        kinds["Conv(拆解)"] = n_conv
    # 3) 给这些 Conv 供数的 Mul（mask 相乘）
    n_mul = 0
    for nd in m.graph.node:
        if nd.op_type == "Mul" and any(o in targets for o in nd.output):
            continue
        if nd.op_type != "Mul":
            continue
        for o in nd.output:
            # 该 Mul 的消费者里有没有拆解 Conv
            for nd2 in m.graph.node:
                if nd2.op_type == "Conv" and o in nd2.input and \
                        nd2.output and nd2.output[0] in targets:
                    targets.add(o)
                    n_mul += 1
                    break
    if n_mul:
        kinds["Mul(mask)"] = n_mul
    if verbose:
        print(f"  锁定目标：{kinds}  共 {len(targets)} 个张量  {time.time()-t0:.0f}s")

    # ---- 转 fp16 并插 Cast ----
    m16 = convert_float_to_float16(m, keep_io_types=True)
    del m
    g = m16.graph
    nodes = list(g.node)
    ins = []
    hit = {}
    for i, nd in enumerate(nodes):
        if nd.op_type in NEVER:
            continue
        if not nd.output or nd.output[0] not in targets:
            continue
        allowed = CASTABLE.get(nd.op_type)
        if allowed is None:
            idxs = [j for j, x in enumerate(nd.input) if x]
        else:
            idxs = [j for j in allowed if j < len(nd.input) and nd.input[j]]
        if not idxs:
            continue
        hit[nd.op_type] = hit.get(nd.op_type, 0) + 1
        pre = []
        for j in idxs:
            c = f"pv_{i}_{j}"
            pre.append(helper.make_node("Cast", [nd.input[j]], [c],
                                        to=TensorProto.FLOAT, name=f"pv_f_{i}_{j}"))
            nd.input[j] = c
        oo = nd.output[0]
        mid = f"pv_m_{i}"
        nd.output[0] = mid
        pre.append(helper.make_node("Cast", [mid], [oo],
                                    to=TensorProto.FLOAT16, name=f"pv_g_{i}"))
        ins.append((i, pre))
    for idx, ns in sorted(ins, key=lambda x: -x[0]):
        for k, nn in enumerate(ns):
            g.node.insert(idx + k, nn)
    if verbose:
        print(f"  实际改写：{hit}  共插 {sum(len(x[1]) for x in ins)} 个 Cast"
              f"  {time.time()-t0:.0f}s")
    onnx.save(m16, dst_onnx)
    if verbose:
        print(f"  已保存 {os.path.getsize(dst_onnx)/2**20:.1f} MB")
    return hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, nargs=2, default=[836, 480])
    ap.add_argument("--t", type=int, default=8)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--workspace-gb", type=float, default=6.0)
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()
    h, w = args.size
    tag = f"t{args.t}_{h}x{w}_x{args.scale}"
    src = os.path.join(TRT_DIR, f"vsr_decomp_{tag}.onnx")
    if not os.path.exists(src):
        print(f"❌ 找不到 {src}")
        return 1
    mix = os.path.join(TRT_DIR, f"vsr_precise_{tag}.onnx")
    eng = os.path.join(TRT_DIR, f"vsr_precise_{tag}.trt")

    print(f"VSR 精确选取版混合精度  {tag}\n")
    make_precise(src, mix)
    if args.skip_build:
        return 0

    print(f"\n  构建 TensorRT 引擎（strongly typed）...", flush=True)
    from build_trt_rife_mixed import build_strongly_typed
    os.chdir(TRT_DIR)
    try:
        build_strongly_typed(os.path.basename(mix), eng,
                             workspace_gb=args.workspace_gb)
    except Exception as e:                                       # noqa: BLE001
        print(f"  ❌ {type(e).__name__}: {str(e)[:300]}")
        return 1
    print("  ✅ 完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
