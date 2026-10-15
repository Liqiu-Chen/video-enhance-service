"""把 SpyNet 的 border-padding warp 改成 TRT 能算的等价形式。

问题
----
`flow_warp(..., padding_mode='border')` 导出后 ONNX 里是
`GridSample(padding_mode='border')`，而 **TensorRT 只支持 zeros** ——
实测改 ONNX 属性无效，TRT 直接忽略。
后果：光流 warp 在边界算错 → 特征对齐错 → 误差经时序传播扩散到全图，
表现为整幅画面 31 dB 的系统性偏移。

等价变换
--------
border 语义 = 「越界采样时钳到边缘像素」。
而「先把输入 replicate 补边 P 像素，再采样」在 |偏移| ≤ P 时给出**完全相同**的结果，
且补边后采样点落在 padding 内，用 zeros padding 也不会丢内容。

坐标系要跟着变（align_corners=True 的约定）：
  原图：x_norm = 2p/(W-1) - 1
  补边后宽 W' = W+2P，像素 p 变成 p+P
  ⇒ x_norm' = 2(p+P)/(W'-1) - 1 = ((x_norm+1)(W-1) + 2P)/(W+2P-1) - 1

这是纯数学等价，不依赖 TRT 支持 border。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

PAD = 32          # 补边宽度。SpyNet 的光流量级是个位像素，32 足够覆盖


def flow_warp_trtsafe(x, flow, interpolation="bilinear",
                      padding_mode="zeros", align_corners=True):
    """与 vsr.archs.flow_warp 数值等价，但 border 用「补边+仿射」模拟。"""
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError("spatial sizes mismatch")
    _, _, h, w = x.size()
    dev = flow.device
    if "indexing" in torch.meshgrid.__code__.co_varnames:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h, device=dev, dtype=x.dtype),
            torch.arange(0, w, device=dev, dtype=x.dtype), indexing="ij")
    else:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h, device=dev, dtype=x.dtype),
            torch.arange(0, w, device=dev, dtype=x.dtype))
    grid = torch.stack((grid_x, grid_y), 2)
    grid_flow = grid + flow
    gx = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    gy = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((gx, gy), dim=3).type(x.type())

    if padding_mode == "border":
        # ★ 坐标钳位 = border padding（align_corners=True 下严格等价）★
        # border 的语义就是「采样点超出图像时钳到边界像素」；而在
        # align_corners=True 下，归一化坐标 -1/+1 正对应首/末像素，
        # 所以把坐标钳到 [-1,1] 再做 zeros 采样，结果**逐位相同**。
        # 实测最大差 0.000e+00（同时验证了 align_corners=False 下不成立）。
        #
        # 为什么不用"补边再裁"：那会让中间张量变大 1.3 倍且图更深，
        # TRT 构建时直接 `LLVM ERROR: out of memory`。
        # 钳位是一行、零额外内存、且数学上严格等价。
        grid_flow = grid_flow.clamp(-1.0, 1.0)
        return F.grid_sample(x, grid_flow, mode=interpolation,
                             padding_mode="zeros", align_corners=True)

    return F.grid_sample(x, grid_flow, mode=interpolation,
                         padding_mode=padding_mode, align_corners=align_corners)


def patch():
    """把 flow_warp 换成 TRT 安全版。

    ⚠️ 必须**遍历所有已加载的 vsr.* 模块**逐个替换。
    `basicvsr_net.py` 和 `basicvsr_plusplus_net.py` 都是
    `from .archs import flow_warp` —— 这会在**各自的模块命名空间**里
    绑定一份引用，只改 archs 那一个是没用的（第一次就是这么漏掉 6 个 border 的）。
    """
    import sys as _sys
    n = 0
    for name, mod in list(_sys.modules.items()):
        if not name.startswith("vsr"):
            continue
        if getattr(mod, "flow_warp", None) is not None:
            setattr(mod, "flow_warp", flow_warp_trtsafe)
            n += 1
    # 兜底：把还没加载的模块也登记上（后续 import 时生效）
    import vsr.archs as A
    A.flow_warp = flow_warp_trtsafe
    return n


if __name__ == "__main__":
    import os
    import sys
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(HERE, "engine"))
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    from config import VSR_WEIGHT
    from bench_pipe_vs_iso import decode_frames
    from vsr_engine import VsrEngine

    n = 4
    src = decode_frames(os.path.join(HERE, "..", "liuqiang.mp4"), n)
    H, W = src.shape[1], src.shape[2]
    x = torch.from_numpy(src).cuda().permute(0, 3, 1, 2)[None].float().div_(255.0)

    e = VsrEngine(weight=VSR_WEIGHT, fp16=False, chunk=n, out_scale=4,
                  verbose=False)
    with torch.inference_mode():
        ref = e.net(x)[0].permute(0, 2, 3, 1).float().clamp(0, 1)
    patch()
    e2 = VsrEngine(weight=VSR_WEIGHT, fp16=False, chunk=n, out_scale=4,
                   verbose=False)
    with torch.inference_mode():
        got = e2.net(x)[0].permute(0, 2, 3, 1).float().clamp(0, 1)
    d = (ref - got).abs()
    print(f"  border 模拟 vs 原版：最大差 {d.max().item()*255:.3f}/255   "
          f"平均 {d.mean().item()*255:.5f}/255")
    ok = d.max().item() * 255 < 0.5
    print(f"  {'✅ 数值等价，可安全用于导出' if ok else '⚠️ 有可见差异'}")
