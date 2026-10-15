"""RRDBNet —— Real-ESRGAN / ESRGAN 的生成器网络（自足实现，只依赖 torch）。

为什么不用 pip 的 `basicsr`：
  * `basicsr` 的 setup.py 会编译自定义 CUDA 算子（DCN 等），在无编译环境的 Windows 上
    必然装不上；而其 `rrdbnet_arch.py` 本身只用到 `make_layer / default_init_weights /
    pixel_unshuffle` 三个纯 Python 工具，完全没必要为此引入整个包。
  * 本文件按官方 `BasicSR/basicsr/archs/rrdbnet_arch.py` 逐行复刻，
    并在权重加载时用 strict=True 校验，保证与官方 checkpoint 严格对应。

对应关系（已用 checkpoint 实测确认）：
  RealESRGAN_x4plus.pth  → scale=4, num_feat=64, num_block=23, num_grow_ch=32   16.70 M
  RealESRGAN_x2.pth      → scale=2, 同上配置（输入先 pixel_unshuffle ×2 → 12 通道）
  RealESRGAN_x8.pth      → scale=8, 比标准版多一级 conv_up3（×8 需要 3 次上采样）
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


# --------------------------------------------------------------------------- #
def make_layer(basic_block, num_basic_block, **kwarg):
    """顺序堆叠 num_basic_block 个 basic_block。"""
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


def pixel_unshuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    """像素解洗牌（PixelShuffle 的逆操作）。

    把 (B, C, H*scale, W*scale) 变成 (B, C*scale^2, H, W) —— 空间维缩到 1/scale，
    通道数乘 scale^2。RRDBNet 用它把 ×2 / ×1 的超分任务统一成 ×4 的主干。
    """
    b, c, hh, ww = x.size()
    out_c = c * (scale ** 2)
    out_h, out_w = hh // scale, ww // scale
    x = x.view(b, c, out_h, scale, out_w, scale)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(b, out_c, out_h, out_w)


def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """官方同名函数的简化版：kaiming 初始化 + 可选 bias 填充。

    推理场景其实用不到（权重会从 checkpoint 覆盖），保留是为了能原样复刻网络定义、
    也便于将来自己训练时复用。
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for m in module_list:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", **kwargs)
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()


# --------------------------------------------------------------------------- #
class ResidualDenseBlock(nn.Module):
    """Residual Dense Block（ESRGAN 的核心组件）。

    每一层卷积都把前面所有层的输出拼进来（dense connection），
    最后再乘 0.2 加回输入（residual scaling）。
    """

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block：三个 RDB 串联后再残差加回。"""

    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Real-ESRGAN 的生成器。

    Args:
        num_in_ch:  输入通道（RGB 为 3）
        num_out_ch: 输出通道
        scale:      放大倍数。2 时先 pixel_unshuffle ×2，1 时 ×4；
                    8 时多加一级 conv_up3（标准版只支持到 4）
        num_feat:   主干特征通道，默认 64
        num_block:  RRDB 块数，x4plus / x2 / x8 均为 23
        num_grow_ch: 每个 RDB 内部增长率，默认 32
    """

    def __init__(self, num_in_ch: int = 3, num_out_ch: int = 3, scale: int = 4,
                 num_feat: int = 64, num_block: int = 23, num_grow_ch: int = 32):
        super().__init__()
        self.scale = scale
        self.num_block = num_block
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # 上采样：标准版两级（×4）；×8 需要第三级
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        assert scale in (1, 2, 4, 8), f"不支持的放大倍数: {scale}"
        self.has_up3 = scale == 8
        if self.has_up3:
            self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x

        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        if self.has_up3:
            feat = self.lrelu(self.conv_up3(F.interpolate(feat, scale_factor=2, mode="nearest")))

        return self.conv_last(self.lrelu(self.conv_hr(feat)))

    # ----------------------------------------------------------------------- #
    @classmethod
    def from_pretrained(cls, weight: str, scale: int = 4, device="cpu",
                        fp16: bool = False):
        """按 checkpoint 里 conv_first 的输入通道自动推断 scale，并 strict 加载。"""
        sd = torch.load(weight, map_location="cpu")
        for k in ("params_ema", "params", "state_dict"):
            if isinstance(sd, dict) and k in sd:
                sd = sd[k]
                break
        in_ch = sd["conv_first.weight"].shape[1]
        if in_ch == 12:
            scale = 2
        elif in_ch == 48:
            scale = 1
        # ×8 的权重多一组 conv_up3
        has_up3 = any(k.startswith("conv_up3.") for k in sd)
        if has_up3:
            scale = 8
        net = cls(scale=scale)
        missing, unexpected = net.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"权重与网络不匹配：缺失 {list(missing)[:4]} 多余 {list(unexpected)[:4]}")
        net = net.eval().to(device)
        if fp16:
            net = net.half()
        return net, scale
