"""RIFE 各执行路径的基准测试 + 数值一致性校验。

**每个配置跑在独立子进程里，并带超时**：5K 尺寸下 batch 稍大就会撑爆显存，
一旦落到共享内存会慢上百倍，把整个基准拖死（第一次跑就卡了 12 分钟没出结果）。
隔离开之后，坏配置只会自己被 kill，其余照常出数。

  python bench_rife.py                    # 驱动：逐个配置起子进程
  python bench_rife.py --only 3           # 只跑第 3 个配置（子进程内部用）
"""
from __future__ import annotations

import argparse
import os
import subprocess
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

sys.path.insert(0, os.path.join(HERE, "engine"))

CASES = [
    ("torch fp32 b1 基线", dict(backend="torch", fp16=False, batch=1)),
    ("torch fp16 纯fp16 warp16", dict(backend="torch", fp16=True, batch=1,
                                      warp_fp32=False)),
    ("torch fp16+fp32warp b1", dict(backend="torch", fp16=True, batch=1,
                                    warp_fp32=True)),
    ("torch fp16+fp32warp b2", dict(backend="torch", fp16=True, batch=2,
                                    warp_fp32=True)),
    ("torch fp16+fp32warp +graph", dict(backend="torch", fp16=True, batch=2,
                                        warp_fp32=True, use_cuda_graph=True)),
    ("onnx fp16 b1（对照）", dict(backend="onnx", fp16=True, batch=1)),
]


def make_input(batch, h, w, dtype, device):
    """低通图像 + 平移，模拟一对自然帧。

    ⚠️ **必须固定随机种子**：每个配置跑在独立子进程里，如果各自用默认随机源，
    生成的就是不同的输入，所谓"与基线 PSNR"就变成比较两张无关图像的结果
    （第一次跑就因为这个，所有配置都显示 ~9 dB，包括 fp32 自己）。
    纯白噪声也会让光流饱和、夸大 fp16 误差，所以这里用平滑图像。
    """
    import torch
    g = torch.Generator(device=device).manual_seed(20261011)
    small = torch.rand(batch, 3, max(8, h // 16), max(8, w // 16),
                       device=device, generator=g)
    base = torch.nn.functional.interpolate(small, size=(h, w), mode="bicubic",
                                           align_corners=False).clamp_(0, 1)
    shifted = torch.roll(base, shifts=(5, 9), dims=(2, 3))
    return base.to(dtype), shifted.to(dtype)


def psnr(x, y):
    import torch
    mse = float(((x.float() - y.float()) ** 2).mean())
    return float("inf") if mse == 0 else 10 * torch.log10(torch.tensor(1.0 / mse)).item()


def run_one(idx, h, w, iters, ref_path=None):
    import torch
    from rife_engine import RifeEngine
    from config import RIFE_WEIGHT

    name, kw = CASES[idx]
    eng = RifeEngine(weight=RIFE_WEIGHT, verbose=False, **kw)
    eng.build(h, w)
    a, b = make_input(eng.batch, h, w, eng.dtype, eng.device)
    for _ in range(2):
        eng.infer(a, b)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = eng.infer(a, b)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    peak = torch.cuda.max_memory_allocated() / 2**30
    o = out[:1].float().cpu()

    d = ps = ""
    if ref_path is None:
        torch.save(o, os.path.join(HERE, "..", "work", "_rife_ref.pt"))
        d, ps = 0.0, float("inf")
    else:
        ref = torch.load(ref_path)
        d = float((o - ref).abs().max())
        ps = psnr(o, ref)
    print(f"RESULT\t{name}\t{dt*1000:.1f}\t{dt/eng.batch*1000:.1f}\t{eng.batch}"
          f"\t{peak:.2f}\t{d:.2e}\t{ps:.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, nargs=2, default=[5120, 2880])
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--only", type=int, default=-1)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()
    h, w = args.size

    if args.only >= 0:
        ref = os.path.join(HERE, "..", "work", "_rife_ref.pt")
        run_one(args.only, h, w, args.iters, ref if os.path.exists(ref) else None)
        return

    ref = os.path.join(HERE, "..", "work", "_rife_ref.pt")
    if os.path.exists(ref):
        os.remove(ref)
    print(f"RIFE 基准 @ 输出 {w}x{h}  迭代 {args.iters}  每配置超时 {args.timeout}s\n")
    rows = []
    for i, (name, _) in enumerate(CASES):
        t0 = time.time()
        try:
            p = subprocess.run(
                [sys.executable, "-u", os.path.abspath(__file__),
                 "--size", str(h), str(w), "--iters", str(args.iters), "--only", str(i)],
                capture_output=True, text=True, timeout=args.timeout)
            line = [l for l in p.stdout.splitlines() if l.startswith("RESULT")]
            if line:
                f = line[-1].split("\t")
                ms, per, b, peak, d, ps = (float(f[2]), float(f[3]), int(f[4]),
                                           float(f[5]), float(f[6]), float(f[7]))
                rows.append((name, ms, per, b, peak, d, ps))
                print(f"  {name:24s} {ms:8.1f} ms/batch {per:7.1f} ms/对 "
                      f"b={b}  显存 {peak:5.2f} GiB  PSNR {ps:6.2f} dB", flush=True)
            else:
                tail = (p.stderr or p.stdout).strip().splitlines()[-1:] or ["无输出"]
                print(f"  {name:24s} 失败: {tail[0][:110]}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"  {name:24s} 超时 {args.timeout}s（大概率显存溢出落到共享内存）",
                  flush=True)
        print(f"      ({time.time()-t0:.0f}s)", flush=True)

    if rows:
        base = rows[0][2]
        print("\n汇总（每对帧耗时）")
        for name, ms, per, b, peak, d, ps in sorted(rows, key=lambda r: r[2]):
            print(f"  {name:24s} {per:7.1f} ms/对   加速 {base/per:5.2f}x   "
                  f"显存 {peak:5.2f} GiB   PSNR {ps:6.2f} dB")


if __name__ == "__main__":
    main()
