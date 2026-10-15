"""VSR 优化实验台：把「batch 有没有用」这类问题先用最小实验问清楚，再决定要不要写代码。

    python exp_vsr.py batch      # 多段并行：固定 n*t，只改 n，看总耗时变化
    python exp_vsr.py scale      # 2x 模式 vs 4x 模式
    python exp_vsr.py opt        # jit/compile/量化/剪枝

结论都会打印出来，负优化也照实记录。
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

sys.path.insert(0, os.path.join(HERE, "engine"))

from config import VSR_WEIGHT  # noqa: E402


def load_engine(chunk=8, fp16=True, **kw):
    from vsr_engine import VsrEngine
    return VsrEngine(weight=VSR_WEIGHT, fp16=fp16, chunk=chunk, verbose=False, **kw)


def bench_fn(fn, iters=3, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return dt, torch.cuda.max_memory_allocated() / 2**30


# ---------------------------------------------------------------- batch ---
def exp_batch(h=1280, w=720, t=8, fp16=True, ns=(1, 2, 4)):
    """**固定段长 t，只改一次前向里塞几段（n）**，看 ms/帧 怎么变。

    为什么不固定总帧数去扫 t：BasicVSR++ 的传播要给 4 个方向各留一份逐帧特征，
    显存随 t 线性涨。t=32 单段在 720x1280 输入下直接顶满 16 GiB（实测），
    根本不代表生产配置（真实段长是 8）。

    判读方法：
      如果 n 增大时 ms/帧 基本不变 → 这段计算**算力已饱和**，多段并行拿不到收益；
      如果 ms/帧 明显下降 → 之前是启动开销/占用率不足主导，值得做。
    """
    print(f"多段并行实验：输入 {w}x{h}，**段长固定 {t}**，fp16={fp16}")
    print(f"{'n(段)':>6} {'输入形状':>22} {'总耗时ms':>10} {'ms/帧':>8} {'ms/段':>8} {'显存GiB':>8}")
    eng = load_engine(chunk=t, fp16=fp16)
    x_cpu = torch.zeros(t, 3, h, w)
    for n in ns:
        x = x_cpu[None].repeat(n, 1, 1, 1, 1).to(eng.device)
        x = x.half() if fp16 else x.float()
        try:
            dt, peak = bench_fn(lambda: eng._forward(x), iters=3, warmup=1)
            print(f"{n:>6} {str((n, t, 3, h, w)):>22} {dt*1000:>10.1f} "
                  f"{dt/(n*t)*1000:>8.1f} {dt/n*1000:>8.1f} {peak:>8.2f}", flush=True)
        except torch.OutOfMemoryError:
            print(f"{n:>6} {str((n, t, 3, h, w)):>22} {'OOM（显存不够）':>10}", flush=True)
        del x
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- scale ---
def exp_scale(h=1280, w=720, t=8, fp16=True):
    """2x 模式：先 2x 下采样再 4x 超分。输出像素只有 4x 模式的 1/4。"""
    import cv2
    print(f"2x 模式实验：输入 {w}x{h}，段长 {t}")
    eng = load_engine(chunk=t, fp16=fp16)
    x = torch.zeros(1, t, 3, h, w, device=eng.device)
    x = x.half() if fp16 else x.float()

    dt4, peak4 = bench_fn(lambda: eng._forward(x))
    print(f"  4x 直出          {dt4*1000:7.1f} ms / {t} 帧  → "
          f"{dt4/t*1000:6.1f} ms/帧  显存 {peak4:.2f} GiB  输出 {4*h}x{4*w}")

    small = cv2.resize(np.zeros((h, w, 3), np.uint8), (w // 2, h // 2),
                       interpolation=cv2.INTER_LINEAR)
    xs = torch.from_numpy(small).permute(2, 0, 1)[None, None].repeat(1, t, 1, 1, 1)
    xs = xs.to(eng.device)
    xs = xs.half() if fp16 else xs.float()
    dt2, peak2 = bench_fn(lambda: eng._forward(xs))
    print(f"  2x 下采样+4x 超分 {dt2*1000:7.1f} ms / {t} 帧  → "
          f"{dt2/t*1000:6.1f} ms/帧  显存 {peak2:.2f} GiB  输出 {2*h}x{2*w}")
    print(f"  → 2x 模式比 4x 模式快 {dt4/dt2:.2f}x（输出像素少 4 倍）")


# ------------------------------------------------------------------ opt ---
def _psnr_u8(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def _real_frames(n, h, w):
    """用平滑图像+平移造素材（纯噪声不具代表性）。"""
    rng = np.random.default_rng(7)
    small = rng.random((max(8, h // 16), max(8, w // 16), 3)).astype(np.float32)
    t = torch.from_numpy(small).permute(2, 0, 1)[None]
    big = torch.nn.functional.interpolate(t, size=(h, w), mode="bicubic",
                                          align_corners=False)[0]
    big = (big.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return np.stack([np.roll(big, (i % 3, (i % 3) * 2), axis=(0, 1)).copy()
                     for i in range(n)])


def exp_opt(h=360, w=640, t=4, fp16=True):
    """整网融合 / 编译 / 量化 / 剪枝，逐项实测。

    默认用**小尺寸**跑（360x640 输入）：这些手段的效果与尺寸无关，
    小尺寸能快速试错；确认有用再用全尺寸复核。
    每项都同时给出**速度**和**相对基线的 PSNR**（质量掉了多少）。
    """
    import torch.nn as nn
    import torch.nn.utils.prune as prune

    print(f"优化方案实测：输入 {w}x{h}，段长 {t}，fp16={fp16}")
    eng = load_engine(chunk=t, fp16=fp16)
    net = eng.net
    frames = _real_frames(t, h, w)
    ref = eng.upscale_chunk(frames)                      # 基线产物（也是质量参照）
    print(f"  基线输出 {ref.shape}\n")

    def run(tag, model, cast=None):
        """用给定模型跑一次，返回 (ms/帧, PSNR)。"""
        def f():
            x = torch.from_numpy(frames).to(eng.device).permute(0, 3, 1, 2)
            x = x.unsqueeze(0).float().div_(255.0)
            x = x.half() if cast is None and fp16 else (
                x.float() if cast is None else cast(x))
            with torch.inference_mode():
                return model(x)
        try:
            f()
            dt, peak = bench_fn(f, iters=2, warmup=1)
            y = f()
            if isinstance(y, (tuple, list)):
                y = y[0]
            arr = (y[0].float().clamp(0, 1).mul(255).round()
                   .permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy())
            ps = _psnr_u8(arr, ref)
            print(f"  {tag:26s} {dt/t*1000:7.1f} ms/帧   PSNR {ps:6.2f} dB   "
                  f"显存 {peak:5.2f} GiB", flush=True)
            return dt / t
        except Exception as e:                            # noqa: BLE001
            print(f"  {tag:26s} 失败: {type(e).__name__}: {str(e)[:95]}", flush=True)
            return None
        finally:
            torch.cuda.empty_cache()

    base = run("基线（fp16 + inference_mode）", net)

    # ---- 1) Conv+BN 融合 ----
    n_bn = sum(1 for m in net.modules()
               if isinstance(m, nn.modules.batchnorm._BatchNorm))
    n_conv = sum(1 for m in net.modules() if isinstance(m, nn.Conv2d))
    print(f"\n  整网融合")
    print(f"    BN 层 {n_bn} 个 / Conv2d {n_conv} 个 → "
          f"{'模型里没有 BN，Conv+BN 融合**不适用**（没有可折叠的算子）' if n_bn == 0 else '可融合'}")

    # ---- 2) TorchScript ----
    print(f"\n  图编译")
    try:
        scripted = torch.jit.script(net)
        torch.jit.freeze(scripted)
        run("torch.jit.script + freeze", scripted)
    except Exception as e:                                # noqa: BLE001
        print(f"    torch.jit.script 失败: {type(e).__name__}: {str(e)[:90]}")

    try:
        import importlib.util
        if importlib.util.find_spec("triton") is None:
            print("    torch.compile 跳过：需要 triton，Windows 官方无轮子")
        else:
            run("torch.compile", torch.compile(net))
    except Exception as e:                                # noqa: BLE001
        print(f"    torch.compile 失败: {type(e).__name__}: {str(e)[:90]}")

    # ---- 3) 量化 ----
    print(f"\n  量化")
    try:
        from torch.ao.quantization import quantize_dynamic
        qnet = quantize_dynamic(net.cpu().float(), {nn.Conv2d, nn.Linear},
                                dtype=torch.qint8)
        n_q = sum(1 for m in qnet.modules() if "quantized" in type(m).__module__)
        print(f"    quantize_dynamic 实际替换了 {n_q} 个层")
        print("    → PyTorch 的动态量化只覆盖 Linear/LSTM，卷积网络几乎没被替换；")
        print("      而且它是 CPU 后端（fbgemm），GPU 上无效。详见文档结论。")
    except Exception as e:                                # noqa: BLE001
        print(f"    quantize_dynamic 失败: {type(e).__name__}: {str(e)[:90]}")
    finally:
        net.to(eng.device)
        if fp16:
            net.half()

    # ---- 4) 剪枝 ----
    print(f"\n  剪枝")
    try:
        targets = [(m, "weight") for m in net.modules()
                   if isinstance(m, (nn.Conv2d,))]
        prune.global_unstructured(targets, pruning_method=prune.L1Unstructured,
                                  amount=0.3)
        nz = sum(float(m.weight.ne(0).sum()) for m, _ in targets)
        tot = sum(m.weight.numel() for m, _ in targets)
        print(f"    非结构化剪枝 30%：权重稀疏度 {(1-nz/tot)*100:.1f}%")
        run("非结构化剪枝 30%", net)
        print("    → 未结构化剪枝产生的是**稀疏但不规则**的权重，稠密卷积核照样算 0，"
              "所以速度不会变（要提速得配稀疏 kernel，PyTorch 默认不吃）。")
        for m, _ in targets:
            prune.remove(m, "weight")
    except Exception as e:                                # noqa: BLE001
        print(f"    剪枝失败: {type(e).__name__}: {str(e)[:90]}")
    finally:
        for m in net.modules():
            if hasattr(m, "weight_orig"):
                try:
                    prune.remove(m, "weight")
                except Exception:                         # noqa: BLE001
                    pass


def exp_more(h=1280, w=720, fp16=True):
    """补充实验：jit.trace + 段长/重叠帧扫描。

    段长和 overlap 直接决定「每帧要算多少次」：
      段长 = step，overlap = 上下文帧数 → 每 step 帧要多算 overlap 帧
      所以 overlap=2/step=8 时，真实计算量是 10/8 = 1.25 倍。
    这里把代价和收益都量出来。
    """
    from vsr_engine import VsrEngine
    print(f"补充实验：输入 {w}x{h}，fp16={fp16}\n")

    # ---- jit.trace ----
    print("图编译（固定形状后 trace 变得可行，而 script 已失败）")
    eng = load_engine(chunk=8, fp16=fp16)
    frames = _real_frames(8, h, w)
    dt0, _ = bench_fn(lambda: eng.upscale_chunk(frames), iters=2, warmup=1)
    print(f"  基线 eager            {dt0/8*1000:7.1f} ms/帧")
    try:
        import copy
        m = copy.deepcopy(eng.net).eval()
        x = torch.zeros(1, 10, 3, h, w, device=eng.device)
        x = x.half() if fp16 else x.float()
        with torch.inference_mode():
            traced = torch.jit.trace(m, x, strict=False)
            traced = torch.jit.freeze(traced)
            traced(x)
        def f():
            with torch.inference_mode():
                return traced(x)
        dt, peak = bench_fn(f, iters=3, warmup=1)
        print(f"  jit.trace + freeze    {dt/8*1000:7.1f} ms/帧   "
              f"加速 {dt0/dt:.2f}x   显存 {peak:.2f} GiB")
    except Exception as e:                                  # noqa: BLE001
        print(f"  jit.trace 失败: {type(e).__name__}: {str(e)[:100]}")
    del eng
    torch.cuda.empty_cache()

    # ---- 段长扫描 ----
    print(f"\n段长扫描（overlap 固定 2）")
    print(f"  {'段长':>4} {'ms/帧':>9} {'相对':>7} {'显存GiB':>8}   实际计算量倍数")
    base = None
    for step in (4, 8, 16, 32):
        try:
            e = load_engine(chunk=step, fp16=fp16)
            fr = _real_frames(step, h, w)
            dt, peak = bench_fn(lambda: e.upscale_chunk(fr), iters=2, warmup=1)
            per = dt / step * 1000
            if base is None:
                base = per
            print(f"  {step:>4} {per:>9.1f} {per/base:>6.2f}x {peak:>8.2f}   "
                  f"{(step+2)/step:.2f}", flush=True)
            del e, fr
            torch.cuda.empty_cache()
        except Exception as ex:                             # noqa: BLE001
            print(f"  {step:>4} 失败: {type(ex).__name__}: {str(ex)[:70]}")
            torch.cuda.empty_cache()

    # ---- overlap 扫描 ----
    print(f"\n上下文帧扫描（段长固定 8）")
    print(f"  {'overlap':>8} {'ms/帧':>9} {'相对':>7}   实际计算量倍数")
    base = None
    for ov in (0, 1, 2, 4):
        try:
            e = VsrEngine(weight=VSR_WEIGHT, fp16=fp16, chunk=8, overlap=ov,
                          verbose=False)
            fr = _real_frames(8, h, w)
            dt, _ = bench_fn(lambda: e.upscale_chunk(fr), iters=2, warmup=1)
            per = dt / 8 * 1000
            if base is None:
                base = per
            print(f"  {ov:>8} {per:>9.1f} {per/base:>6.2f}x   {(8+ov)/8:.2f}",
                  flush=True)
            del e, fr
            torch.cuda.empty_cache()
        except Exception as ex:                             # noqa: BLE001
            print(f"  {ov:>8} 失败: {type(ex).__name__}: {str(ex)[:70]}")


def exp_cudnn(h=1280, w=720, t=8, fp16=True):
    """cuDNN 后端配置实验。

    profile 显示每 935 次卷积配了 **1848 次 nchw↔nhwc 布局转换**（占 GPU 时间 8%）
    —— 说明 cuDNN 给每个卷积都选了 NHWC 引擎，再前后各插一次转置。
    `benchmark=True` 时 PyTorch 会用 cudnnFind 挑"最快"引擎，而它只看卷积本身、
    不一定把转置算进去。所以这里直接对比 benchmark 开/关。
    """
    from vsr_engine import VsrEngine
    print(f"cuDNN 配置实验：输入 {w}x{h}，段长 {t}，fp16={fp16}\n")
    print(f"  {'配置':<38}{'ms/帧':>9}{'相对':>8}{'显存GiB':>9}")

    frames = _real_frames(t, h, w)
    base = None
    cases = [
        ("benchmark=True（当前默认）", dict(benchmark=True)),
        ("benchmark=False（用 cudnn 启发式）", dict(benchmark=False)),
        ("benchmark=True + allow_tf32", dict(benchmark=True, tf32=True)),
        ("benchmark=False + allow_tf32", dict(benchmark=False, tf32=True)),
    ]
    for tag, cfg in cases:
        try:
            torch.backends.cudnn.benchmark = cfg["benchmark"]
            torch.backends.cudnn.allow_tf32 = cfg.get("tf32", True)
            e = load_engine(chunk=t, fp16=fp16)
            dt, peak = bench_fn(lambda: e.upscale_chunk(frames), iters=2, warmup=1)
            per = dt / t * 1000
            if base is None:
                base = per
            print(f"  {tag:<38}{per:>9.1f}{per/base:>7.2f}x{peak:>9.2f}", flush=True)
            del e
            torch.cuda.empty_cache()
        except Exception as ex:                             # noqa: BLE001
            print(f"  {tag:<38} 失败: {type(ex).__name__}: {str(ex)[:60]}")
            torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True


def make_batched_upsample(net, group=2):
    """返回一个与 net.upsample **数学等价**、但按 `group` 帧合批的实现。

    原实现是 `for i in range(t)` 逐帧 batch=1 跑 reconstruction→upsample1→
    upsample2→conv_hr→conv_last，t 次小前向，GPU 利用率很差；而帧与帧之间
    完全独立（每帧用各自那份 feats，`feats[k].pop(0)` 顺序取用），
    所以可以直接合批 —— 这是纯执行层面的工程优化，不动任何参数或结构。

    显存是唯一约束：4x 分辨率下 64 通道的中间张量一帧就 ~1.9 GB，group 不能大。
    """
    def upsample(lqs, feats, group_=group):
        keys = [k for k in feats if k != "spatial"]
        t = lqs.size(1)
        outs = []
        for i0 in range(0, t, group_):
            i1 = min(i0 + group_, t)
            parts = [torch.cat(list(feats["spatial"][i0:i1]), 0)]
            for k in keys:
                parts.append(torch.cat(list(feats[k][i0:i1]), 0))
            hr = torch.cat(parts, dim=1)
            hr = net.reconstruction(hr)
            hr = net.lrelu(net.upsample1(hr))
            hr = net.lrelu(net.upsample2(hr))
            hr = net.lrelu(net.conv_hr(hr))
            hr = net.conv_last(hr)
            lo = lqs[:, i0:i1].reshape(-1, lqs.size(2), lqs.size(3), lqs.size(4))
            outs.append(hr + (net.img_upsample(lo) if net.is_low_res_input else lo))
        return torch.cat(outs, 0).view(lqs.size(0), t, *outs[0].shape[1:])
    return upsample


def exp_up(h=1280, w=720, t=10, fp16=True):
    """上采样阶段：逐帧循环 vs 按帧合批。"""
    from vsr_engine import VsrEngine
    print(f"upsample 合批实验：输入 {w}x{h}，序列长度 {t}，fp16={fp16}\n")
    eng = VsrEngine(weight=VSR_WEIGHT, fp16=fp16, chunk=t - 2, overlap=2,
                    verbose=False)
    net = eng.net
    rng = torch.Generator().manual_seed(11)
    small = torch.rand(1, 3, max(8, h // 16), max(8, w // 16), generator=rng)
    base = torch.nn.functional.interpolate(small, size=(h, w), mode="bicubic",
                                           align_corners=False).clamp_(0, 1)
    lqs = torch.stack([torch.roll(base[0], (i % 3, (i % 3) * 2), dims=(1, 2))
                       for i in range(t)]).unsqueeze(0).to(eng.device)
    lqs = lqs.half() if fp16 else lqs.float()

    with torch.inference_mode():
        n, tt, c, hh, ww = lqs.size()
        net.check_if_mirror_extended(lqs)          # forward 里会先调它设置标志位
        net.cpu_cache = tt > net.cpu_cache_length and lqs.is_cuda
        feats_ = net.feat_extract(lqs.view(-1, c, hh, ww))
        fh, fw = feats_.shape[2:]
        feats_ = feats_.view(n, tt, -1, fh, fw)
        fl_f, fl_b = net.compute_flow(lqs)

        def propagate():
            f = {"spatial": [feats_[:, i] for i in range(tt)]}
            for it in (1, 2):
                for d in ("backward", "forward"):
                    mod = f"{d}_{it}"
                    f[mod] = []
                    fl = fl_b if d == "backward" else (fl_f if fl_f is not None
                                                       else fl_b.flip(1))
                    f = net.propagate(f, fl, mod)
            return f

        base_feats = propagate()
        net(lqs)                                     # 预热 cudnn

        # ---- 基线：原逐帧实现 ----
        orig = type(net).upsample
        dt0, _ = bench_fn(
            lambda: net.upsample(lqs, {k: list(v) for k, v in base_feats.items()}),
            iters=2, warmup=1)
        with torch.inference_mode():
            ref = net.upsample(lqs, {k: list(v) for k, v in base_feats.items()})
        print(f"  {'原实现（逐帧循环）':<28}{dt0*1000:>9.1f} ms{dt0/tt*1000:>10.1f} ms/帧"
              f"{'1.00x':>8}")

        # ---- 合批 ----
        for g in (2, 3, 4):
            if g > tt:
                continue
            net.upsample = make_batched_upsample(net, group=g)
            try:
                dt, _ = bench_fn(
                    lambda: net.upsample(lqs, {k: list(v)
                                               for k, v in base_feats.items()}),
                    iters=2, warmup=1)
                with torch.inference_mode():
                    out = net.upsample(lqs, {k: list(v)
                                             for k, v in base_feats.items()})
                ps = _psnr_u8(
                    (out[0].float().clamp(0, 1).mul(255).round().permute(0, 2, 3, 1)
                     .to(torch.uint8).cpu().numpy()),
                    (ref[0].float().clamp(0, 1).mul(255).round().permute(0, 2, 3, 1)
                     .to(torch.uint8).cpu().numpy()))
                print(f"  {f'合批 group={g}':<28}{dt*1000:>9.1f} ms"
                      f"{dt/tt*1000:>10.1f} ms/帧{dt0/dt:>7.2f}x   PSNR {ps:.2f} dB",
                      flush=True)
            except Exception as e:                      # noqa: BLE001
                print(f"  合批 group={g} 失败: {type(e).__name__}: {str(e)[:70]}")
            finally:
                net.upsample = orig.__get__(net)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["batch", "scale", "opt", "more", "cudnn", "up"])
    ap.add_argument("--size", type=int, nargs=2, default=[1280, 720], help="输入 H W")
    ap.add_argument("--seg-len", type=int, default=8,
                    help="多段并行实验里的段长（生产默认 8）")
    args = ap.parse_args()
    h, w = args.size
    print(f"GPU: {torch.cuda.get_device_name(0)}\n", flush=True)
    if args.exp == "batch":
        exp_batch(h, w, args.seg_len)
    elif args.exp == "scale":
        exp_scale(h, w)
    elif args.exp == "more":
        exp_more(h, w)
    elif args.exp == "cudnn":
        exp_cudnn(h, w)
    elif args.exp == "up":
        exp_up(h, w, args.seg_len + 2)
    else:
        exp_opt(h, w)


if __name__ == "__main__":
    main()
