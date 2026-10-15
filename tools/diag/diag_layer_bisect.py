"""逐层定位：fp16 到底从哪一层开始崩。

思路：给 fp32 与 fp16 两个网络挂同样的 forward hook，喂**同一批输入**，
逐模块比较输出。找出**相对误差第一次暴涨**的那一层 —— 那就是元凶。

BasicVSR++ 的模块会被反复调用（时序循环），所以按 (模块名, 第几次调用) 配对。

    python diag_layer_bisect.py --size 836 480 --t 4
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


def collect(net, x, tag):
    """跑一次前向，收集每个模块每次调用的输出（转 fp32 存 CPU）。"""
    rec = {}
    handles = []

    def mk(name, counter):
        def hook(mod, inp, out):
            if isinstance(out, (tuple, list)):
                out = out[0]
            if not torch.is_tensor(out):
                return
            k = (name, counter[0])
            counter[0] += 1
            rec[k] = out.detach().float().cpu()
        return hook

    counts = {}
    for name, mod in net.named_modules():
        if name == "":
            continue
        # ⚠️ 只记录**浅层**模块。
        # 之前记录全部模块 + 4x 输出尺寸 → 激活值总和不小于几十 GB → 段错误。
        # 顶层模块（spynet / deform_align / backbone / reconstruction）足以定位问题。
        if name.count(".") > 1:
            continue
        cnt = [0]
        counts[name] = cnt
        handles.append(mod.register_forward_hook(mk(name, cnt)))
    with torch.inference_mode():
        net(x)
    for h in handles:
        h.remove()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, nargs=2, default=[270, 480])
    ap.add_argument("--t", type=int, default=4)
    ap.add_argument("--clip", default=os.path.join(HERE, "..", "liuqiang.mp4"))
    ap.add_argument("--top", type=int, default=18),
    args = ap.parse_args()
    h, w = args.size

    from bench_pipe_vs_iso import decode_frames
    src = decode_frames(args.clip, args.t)
    t = torch.from_numpy(np.ascontiguousarray(src)).cuda()
    t = t.permute(0, 3, 1, 2)[None].float().div_(255.0)     # (1,t,3,H,W)

    from vsr_engine import VsrEngine
    e32 = VsrEngine(weight=VSR_WEIGHT, fp16=False, chunk=args.t, out_scale=4,
                    verbose=False)
    e16 = VsrEngine(weight=VSR_WEIGHT, fp16=True, chunk=args.t, out_scale=4,
                    verbose=False)

    print(f"输入 (1,{args.t},3,{h},{w})\n收集各层输出 ...", flush=True)
    a = collect(e32.net, t, "fp32")
    del e32
    import gc as _gc
    _gc.collect()
    torch.cuda.empty_cache()
    b = collect(e16.net, t.half(), "fp16")
    print(f"  fp32 记录 {len(a)} 条，fp16 记录 {len(b)} 条，"
          f"可配对 {len(set(a) & set(b))} 条\n")

    rows = []
    for k in sorted(set(a) & set(b), key=lambda x: (x[0], x[1])):
        ra, rb = a[k], b[k]
        if ra.shape != rb.shape or ra.numel() == 0:
            continue
        diff = (ra - rb).abs()
        scale = ra.abs().max().item()
        rel = diff.max().item() / scale if scale > 0 else 0.0
        rows.append((k[0], k[1], rel, diff.max().item(), scale))

    # 按相对误差排序
    rows.sort(key=lambda r: -r[2])
    print(f"  {'模块':<46}{'第n次':>5}{'相对误差':>12}{'最大差':>10}{'量级':>10}")
    for name, idx, rel, mx, sc in rows[:args.top]:
        short = name if len(name) <= 44 else name[:20] + "…" + name[-22:]
        print(f"  {short:<46}{idx:>5}{rel:>12.3e}{mx:>10.4f}{sc:>10.3f}")

    print("\n  ── 只写真经出问题的区域（相对误差 > 1e-2）──")
    bad = [r for r in rows if r[2] > 1e-2]
    if not bad:
        print("     无明显发散层（全部 < 1e-2）")
    else:
        seen = set()
        for name, idx, rel, mx, sc in bad[:args.top]:
            root = name.split(".")[0]
            if root in seen:
                continue
            seen.add(root)
            print(f"     {root:<24} 例: {name}  相对误差 {rel:.2e}")


if __name__ == "__main__":
    main()
