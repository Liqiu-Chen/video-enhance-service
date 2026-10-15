# The implementation here is modified based on ECCV2022-RIFE,
# originally MIT License, Copyright  (c)  Megvii  Inc.,
# and publicly available at https://github.com/megvii-research/ECCV2022-RIFE

import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
backwarp_tenGrid = {}

# 是否强制用 fp32 做网格采样（由引擎在模型跑 fp16 时打开）。
#
# 为什么需要：grid_sample 的坐标是归一化到 [-1,1] 的。在 2880 宽的画面上，
# 相邻像素的坐标间距只有 2/2879 ≈ 6.9e-4，而 fp16 在 1.0 附近的精度约 1e-3
# —— 也就是**坐标在 fp16 里根本区分不开相邻像素**，采样位置被量化，
# 画面直接被破坏（实测 PSNR 从 inf 掉到 9.2 dB）。
# 只把这一步放回 fp32 即可，卷积仍在 fp16 上跑，速度收益基本保留。
FORCE_FP32_WARP = False


def warp(tenInput, tenFlow):
    if FORCE_FP32_WARP and tenInput.dtype == torch.float16:
        return _warp_fp(tenInput.float(), tenFlow.float()).to(torch.float16)
    return _warp_fp(tenInput, tenFlow)


def _warp_fp(tenInput, tenFlow):
    # 网格缓存键必须带上 dtype：原来只按 (device, size) 缓存，网格固定是 fp32，
    # 一旦模型跑 fp16，`grid + tenFlow` 就会报
    #   expected scalar type Half but found Float
    # —— 这就是之前"RIFE 不能用 fp16"的直接原因。
    k = (str(tenFlow.device), str(tenFlow.size()), tenFlow.dtype)
    if k not in backwarp_tenGrid:
        dt, dev = tenFlow.dtype, tenFlow.device
        tenHorizontal = torch.linspace(
            -1.0, 1.0, tenFlow.shape[3], device=dev, dtype=dt).view(
                1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1,
                                                  tenFlow.shape[2], -1)
        tenVertical = torch.linspace(
            -1.0, 1.0, tenFlow.shape[2],
            device=dev, dtype=dt).view(1, 1, tenFlow.shape[2],
                                       1).expand(tenFlow.shape[0], -1, -1,
                                                 tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat([tenHorizontal, tenVertical],
                                        1).to(dev, dt)

    tenFlow = torch.cat(
        [
            tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),  # no qa
            tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)
        ],
        1)  # no qa

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(
        input=tenInput,
        grid=g,
        mode='bilinear',
        padding_mode='border',
        align_corners=True)
