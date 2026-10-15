"""验证 RifeEngine 的 warp_fp32 开关是否真的生效。

三个配置各起一个子进程跑（同一进程里放三个 5K 模型会 OOM），
结果存成 .pt 再统一比对。
"""
import os
import subprocess
import sys

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

CFGS = [("fp32", dict(fp16=False)),
        ("fp16_warp32", dict(fp16=True, warp_fp32=True)),
        ("fp16_warp16", dict(fp16=True, warp_fp32=False))]


def run_one(i):
    sys.path.insert(0, os.path.join(HERE, "engine"))
    sys.path.insert(0, os.path.join(HERE, "..", "插帧模型", "RIFE", "src"))
    from rife_engine import RifeEngine
    import rife_ms.warplayer as wp

    h, w = 5120, 2880
    torch.manual_seed(0)
    small = torch.rand(1, 3, h // 16, w // 16, device="cuda")
    base = torch.nn.functional.interpolate(small, size=(h, w), mode="bicubic",
                                           align_corners=False).clamp_(0, 1)
    a, b = base, torch.roll(base, shifts=(5, 9), dims=(2, 3))

    name, kw = CFGS[i]
    e = RifeEngine(verbose=False, **kw)
    e.build(h, w)
    print(f"{name:12s} 引擎内 FORCE_FP32_WARP = {wp.FORCE_FP32_WARP}", flush=True)
    with torch.inference_mode():
        y = e.infer(a.to(e.dtype), b.to(e.dtype))
    torch.save(y[:1].float().clamp(0, 1).cpu(), os.path.join(HERE, "..", "work",
                                                             f"_warp_{name}.pt"))


def main():
    if len(sys.argv) > 1:
        run_one(int(sys.argv[1]))
        return
    for i in range(len(CFGS)):
        subprocess.run([sys.executable, "-u", os.path.abspath(__file__), str(i)],
                       check=True)
    ref = torch.load(os.path.join(HERE, "..", "work", "_warp_fp32.pt"))
    for name, _ in CFGS[1:]:
        o = torch.load(os.path.join(HERE, "..", "work", f"_warp_{name}.pt"))
        d = (o - ref).abs()
        mse = float((d ** 2).mean())
        ps = 10 * torch.log10(torch.tensor(1.0 / max(mse, 1e-12))).item()
        print(f"{name:12s} vs fp32:  PSNR {ps:6.2f} dB   最大像素差 {float(d.max()):.4f}")


if __name__ == "__main__":
    main()
