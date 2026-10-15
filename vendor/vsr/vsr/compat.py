"""把官方代码依赖的 `mmcv.ops` 变形卷积换成 torchvision 原生实现。

为什么不直接装 mmcv：
  mmcv 的 wheel 不含 ops，源码安装要编译 CUDA 算子（还挑 torch 版本），
  在无编译环境的 Windows 上基本装不上。而 `mmcv-lite`（纯 Python 版）
  只提供 `mmcv.cnn` 等，没有 ops。

替代方案：
  **`torchvision.ops.deform_conv2d` 是 PyTorch 原生的可变形卷积实现**，
  与 mmcv 的 `modulated_deform_conv2d` 语义一致：
      input   (N, C, H, W)
      offset  (N, deform_groups*2*kh*kw, out_h, out_w)
      mask    (N, deform_groups*1*kh*kw, out_h, out_w)
      weight  (out_c, in_c//groups, kh, kw)
  参数顺序不同（mmcv 把 mask 放在第 3 位），下面按 mmcv 的签名封装。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.ops


def _pair(v):
    if isinstance(v, (tuple, list)):
        return (int(v[0]), int(v[1]))
    return (int(v), int(v))


def modulated_deform_conv2d(input, offset, mask, weight, bias=None,
                            stride=1, padding=0, dilation=1,
                            groups=1, deform_groups=1):
    """签名与 mmcv.ops.modulated_deform_conv2d 一致。"""
    return torchvision.ops.deform_conv2d(
        input, offset, weight, bias,
        stride=_pair(stride), padding=_pair(padding), dilation=_pair(dilation),
        mask=mask)


class ModulatedDeformConv2d(nn.Module):
    """mmcv.ops.ModulatedDeformConv2d 的最小等价实现（只保留被用到的部分）。

    官方 `SecondOrderDeformableAlignment` 继承它并覆写 forward，
    实际依赖的成员是：weight / bias / stride / padding / dilation /
    groups / deform_groups / out_channels。
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 deform_groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        kh, kw = self.kernel_size
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kh, kw))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, offset, mask):
        return modulated_deform_conv2d(
            x, offset, mask, self.weight, self.bias,
            self.stride, self.padding, self.dilation,
            self.groups, self.deform_groups)


def install():
    """把本模块伪装成 `mmcv.ops`，让官方源码的 import 直接可用。"""
    import sys
    import types
    mod = types.ModuleType("mmcv.ops")
    mod.ModulatedDeformConv2d = ModulatedDeformConv2d
    mod.modulated_deform_conv2d = modulated_deform_conv2d
    sys.modules["mmcv.ops"] = mod
