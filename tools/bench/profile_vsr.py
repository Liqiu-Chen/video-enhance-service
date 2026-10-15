"""VSR 算子级性能剖析：找出 600 ms/帧到底花在哪。

不再靠猜。torch.profiler 抓一次真实前向，按**算子自耗时**排序，
并统计每个算子的 dtype —— fp16 模型里混进 fp32 算子是最常见的隐形慢因。

    python profile_vsr.py                       # 720x1280 输入，段长 8
    python profile_vsr.py --size 360 640
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

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

from config import VSR_WEIGHT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, nargs=2, default=[1280, 720], help="输入 H W")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--fp16", action="store_true", default=True)
    args = ap.parse_args()
    h, w = args.size

    from vsr_engine import VsrEngine
    eng = VsrEngine(weight=VSR_WEIGHT, fp16=args.fp16, chunk=args.chunk,
                    overlap=2, verbose=False)
    x = torch.zeros(1, args.chunk + 2, 3, h, w, device=eng.device)
    x = x.half() if args.fp16 else x.float()

    with torch.inference_mode():
        for _ in range(2):                       # 预热 + cudnn 选型
            eng.net(x)
    torch.cuda.synchronize()

    from torch.profiler import profile, ProfilerActivity
    with torch.inference_mode():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=False, profile_memory=False) as prof:
            for _ in range(3):
                eng.net(x)
            torch.cuda.synchronize()

    evs = prof.key_averages()
    # torch 2.9 用 self_device_time_total；老版本叫 self_cuda_time_total。
    # 不能写成 `a or b`：某些纯 CPU 事件上 device time 是 0，会去取不存在的属性。
    field = "self_device_time_total"
    if not hasattr(evs[0], field):
        field = "self_cuda_time_total"
    rows = []
    for e in evs:
        cuda = getattr(e, field, 0) or 0
        if cuda <= 0:
            continue
        rows.append((e.key, cuda / 3000.0, e.count / 3, e.self_cpu_time_total / 3000.0))
    rows.sort(key=lambda r: -r[1])
    total = sum(r[1] for r in rows)

    print(f"\n输入 {w}x{h}  段长 {args.chunk}  总 CUDA 时间 {total:.1f} ms/次前向")
    print(f"{'算子':<44}{'ms/次':>9}{'占比':>8}{'调用次/前向':>12}{'CPU ms':>9}")
    print("-" * 84)
    for name, ms, cnt, cpu in rows[:25]:
        print(f"{name[:43]:<44}{ms:>9.2f}{ms/total*100:>7.1f}%{cnt:>12.0f}{cpu:>9.2f}")

    # 按类别聚合，看是哪一大块在吃时间
    print("\n按类别聚合：")
    cats = defaultdict(float)
    for name, ms, cnt, cpu in rows:
        n = name.lower()
        if "deform" in n or "conv2d_backward" in n:
            k = "可变形卷积 deform_conv"
        elif "conv" in n or "cudnn" in n:
            k = "普通卷积"
        elif "grid" in n or "upsample" in n or "interpolate" in n or "resize" in n:
            k = "采样/插值 grid_sample·resize"
        elif "prelu" in n or "relu" in n or "leaky" in n or "sigmoid" in n:
            k = "激活"
        elif "elementwise" in n or "add" in n or "mul" in n or "cat" in n:
            k = "逐元素/拼接"
        elif "copy" in n or "contiguous" in n or "to" in n or "cast" in n:
            k = "拷贝/类型转换"
        elif "norm" in n or "batch" in n:
            k = "归一化"
        else:
            k = "其他"
        cats[k] += ms
    for k, v in sorted(cats.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<28}{v:>9.2f} ms  {v/total*100:>5.1f}%")

    print("\n提示：若「拷贝/类型转换」占比高，说明有算子不支持 fp16 在偷偷升精度；")
    print("      若「普通卷积」占比高但 conv 数不多，说明单个 conv 的 kernel 选得不好。")


if __name__ == "__main__":
    main()
