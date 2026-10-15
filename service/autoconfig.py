"""按输入/输出分辨率自动选择后端与并行度，避免爆显存并保持性能最优。

为什么需要它
------------
实测结论**强烈依赖分辨率**，同一份代码在两个尺寸上最优配置完全相反：

  输出 1440x2560（3.7 MP）：BasicVSR++ chunk=8 最快（172 ms/帧），只要 3.09 GiB
  输出 2880x5120（14.7 MP）：BasicVSR++ 要 15.26 GiB → **驱动换页，慢 4.8 倍**，
                             必须换 Real-ESRGAN（0.06 GiB）

手工调参很容易踩到"显存超限 → 换页 → 性能崩 4 倍（而且不报错、只是慢）"这个坑，
所以这里做成自动的。

策略：**先按实测表选一套保守配置 → 跑起来后核对真实峰值显存 → 超了就降级重试。**
单靠静态表不够，因为可用显存还会被桌面/其他进程占掉一部分。
"""
from __future__ import annotations

import torch

# 显存预算：16 GiB 卡留出余量给桌面、CUDA context、cudnn workspace。
# 超过这个值就有换页风险（实测 94% 占用时就开始了）。
VRAM_BUDGET_GB = 12.0

# 按「输出像素数」分档的实测配置表。
# 每档是一个**候选链**：从上往下试，第一个能跑且不超预算的就用。
# (backend, 参数字典)
_TIERS = [
    # ---- 小尺寸（<= ~4 MP，例如 1440x2560）----
    (4.0e6, [
        ("vsr", {"chunk": 8}),        # 实测 172 ms/帧、3.09 GiB —— 最快且带时序一致性
        ("vsr", {"chunk": 4}),        # 198 ms/帧、2.52 GiB
        ("resr", {"batch": 2}),
        ("resr", {"batch": 1}),
    ]),
    # ---- 中等尺寸（<= ~9 MP，例如 1920x3413）----
    (9.0e6, [
        ("vsr", {"chunk": 4}),
        ("resr", {"batch": 1}),
        ("resr", {"batch": 1, "tile": 512}),
    ]),
    # ---- 大尺寸（> 9 MP，例如 2880x5120 = 14.7 MP）----
    (float("inf"), [
        ("resr", {"batch": 1}),       # VSR 在这个尺寸要 15+ GiB，直接排除
        ("resr", {"batch": 1, "tile": 512}),
        ("resr", {"batch": 1, "tile": 256}),
    ]),
]


def estimate_vram_gb(backend: str, params: dict, in_h: int, in_w: int,
                     out_scale: int) -> float:
    """粗略估算峰值显存（GiB）。

    来自实测的几个锚点（fitting 出来的经验式，只用于**排序候选**，真正的
    准入判断靠跑起来后核对 `torch.cuda.max_memory_allocated`）：

      1440x2560 输出：VSR chunk8 → 3.09 GiB；RealESRGAN batch1 → 1.05 GiB
      2880x5120 输出：VSR chunk4 → 9.31 GiB（allocated 峰值）
    """
    out_px = (in_h * out_scale) * (in_w * out_scale)
    if backend == "vsr":
        # 逐帧特征：4 条传播路径 × (chunk+overlap) 帧 × 64ch @ 输入分辨率（fp16）
        chunk = int(params.get("chunk", 8))
        feat = 4 * (chunk + 2) * 64 * (in_h * in_w) * 2 / 2**30
        # 4x 上采样阶段的激活
        up = 64 * out_px * 2 * 3 / 2**30
        return feat + up
    # Real-ESRGAN：主干特征 + 4x 阶段的 64 通道张量
    batch = int(params.get("batch", 1))
    tile = int(params.get("tile", 0))
    area = (tile * tile) if tile > 0 else (in_h * in_w)
    backbone = batch * 64 * area * 2 * 6 / 2**30
    up = batch * 64 * area * 16 * 2 * 2 / 2**30
    return backbone + up


def candidates(in_h: int, in_w: int, out_scale: int = 4, backend: str | None = None):
    """按输出尺寸返回候选配置链（可指定只考虑某个后端）。"""
    out_px = (in_h * out_scale) * (in_w * out_scale)
    for limit, chain in _TIERS:
        if out_px <= limit:
            break
    out = []
    for be, params in chain:
        if backend and be != backend:
            continue
        p = dict(params)
        p.setdefault("tile", 0)
        out.append((be, p))
    return out


def describe(in_h: int, in_w: int, out_scale: int) -> str:
    out_px = (in_h * out_scale) * (in_w * out_scale)
    tag = "小" if out_px <= 4e6 else ("中" if out_px <= 9e6 else "大")
    lines = [f"输入 {in_w}x{in_h} → 输出 {in_w*out_scale}x{in_h*out_scale}"
             f"（{out_px/1e6:.1f} MP，{tag}尺寸档）"]
    for be, p in candidates(in_h, in_w, out_scale):
        est = estimate_vram_gb(be, p, in_h, in_w, out_scale)
        flag = "OK " if est <= VRAM_BUDGET_GB else "超预算"
        lines.append(f"  {flag} {be} {p}  预估 {est:.2f} GiB")
    return "\n".join(lines)


def fits(backend: str, params: dict, in_h: int, in_w: int, out_scale: int) -> bool:
    return estimate_vram_gb(backend, params, in_h, in_w, out_scale) <= VRAM_BUDGET_GB


def probe_ok(engine, in_h: int, in_w: int, budget_gb: float = VRAM_BUDGET_GB) -> bool:
    """跑一次预热，用**真实峰值显存**判断这套配置能不能用。

    静态估算不够准（桌面程序也会占显存），所以以实测为准：
    跑一次后看 `max_memory_allocated`，超预算就弃用这个候选。
    """
    import numpy as np
    try:
        rng = np.random.default_rng(0)
        dummy = rng.integers(0, 256, (2, in_h, in_w, 3), dtype=np.uint8)
        torch.cuda.reset_peak_memory_stats()
        engine.upscale_chunk(dummy)
        peak = torch.cuda.max_memory_allocated() / 2**30
        torch.cuda.empty_cache()
        return peak <= budget_gb, peak
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return False, float("inf")
