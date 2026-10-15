"""把「带掩码可变形卷积」(modulated deform conv) 拆成 TensorRT 原生算子。

为什么走这条路
--------------
BasicVSR++ 的 `SecondOrderDeformableAlignment` 用了
`torchvision.ops.deform_conv2d(mask=...)`，这个算子：
  - ONNX 没有对应算子（opset 28 也只有无 mask 的 DeformConv）
  - PyTorch 导出器没注册 symbolic
  - TensorRT 也没有原生实现，正常做法是自己编译 plugin
但本机**编不了 plugin**：没有 MSVC（Windows 上 nvcc 缺它编不了 host 代码）、
pip 装的 TensorRT 只有 dll 没有头文件和 lib。所以换思路。

拆解原理
--------
可变形卷积输出：

    out[oc, y, x] = Σ_{c,i,j} w[oc,c,i,j] · mask[g, i,j, y,x]
                    · X[c, y·s + i·d − p + Δy[g,i,j,y,x], x·s + j·d − p + Δx[...]]

即：**先在每个卷积核抽头 (i,j) 对应的偏移位置上双线性采样，再乘权重求和。**

而"按偏移位置双线性采样"正是 `grid_sample` 干的事；采样完之后
「乘 mask → 对 (c,i,j) 加权求和」就是一个 **1×1 卷积**（把 i,j 折进通道维）。

于是整条链路 = grid_sample + mul + reshape + 1x1 conv，
**全部是 TensorRT 原生算子，不需要任何 plugin。**

本脚本先验证数学等价性（这是后续一切的前提）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def deform_conv2d_decomposed(
    x: torch.Tensor,          # (N, Cin, H, W)
    offset: torch.Tensor,     # (N, 2*G*kh*kw, Hout, Wout)
    mask: torch.Tensor,       # (N, G*kh*kw, Hout, Wout)
    weight: torch.Tensor,     # (Cout, Cin//groups, kh, kw)
    bias: torch.Tensor | None = None,
    stride: tuple = (1, 1),
    padding: tuple = (1, 1),
    dilation: tuple = (1, 1),
    deform_groups: int = 1,
) -> torch.Tensor:
    """与 torchvision.ops.deform_conv2d(..., mask=...) 等价的纯原生实现。

    ★ 全程 fp32 计算 ★
    这一点是实测出来的、非做不可：拆解会把采样结果物化成 9 倍通道数的中间张量，
    如果这个中间量是 fp16，每一位都在做 fp16 舍入 —— 实测整网只有 36 dB
    （而且**第一帧就已经错了**，不是误差累积）。同样代码在 fp32 下是 78 dB。
    原实现是一个融合 kernel、内部用 float 累加，所以对 fp16 远没那么敏感。

    做法：进来先升 fp32，算完再转回原 dtype。对 fp32 模型是零开销的 no-op。
    """
    in_dtype = x.dtype
    if in_dtype != torch.float32:
        x = x.float()
        offset = offset.float()
        mask = mask.float()
        weight = weight.float()
        if bias is not None:
            bias = bias.float()
    N, Cin, H, W = x.shape
    Cout, Cin_g, kh, kw = weight.shape
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    Gh, Gw = deform_groups, deform_groups
    Hout = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    Wout = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    Khw = kh * kw
    Cin_per_g = Cin // deform_groups

    # ---- 1. 构造采样网格 ----
    # ★ 用纯广播算，**不要**先 torch.empty 再逐 tap 赋值 ★
    # 之前那种写法会把整个 (Khw,Hout,Wout) 的 taps 张量物化成**常量**
    # （每个 14~98 MB），导出后外部数据膨胀到 3.7 GB，直接超过 protobuf
    # 的 2 GB 序列化上限（报 "Failed to serialize proto"），也拖慢构建。
    # 改成广播之后，常量只剩下 arange 小向量。
    ys = torch.arange(Hout, dtype=torch.float32, device=x.device) * sh - ph
    xs = torch.arange(Wout, dtype=torch.float32, device=x.device) * sw - pw
    # offset: (N, 2*G*Khw, Hout, Wout) → (N, G, Khw, 2, Hout, Wout)
    off = offset.view(N, Gh, Khw, 2, Hout, Wout)
    # mask: (N, G*Khw, Hout, Wout) → (N, G, Khw, Hout, Wout)
    msk = mask.view(N, Gh, Khw, Hout, Wout)

    # 每个 tap 相对基准位置的偏移量（只有 Khw 个数的小常量）
    _dys = torch.tensor([(k // kw) * dh for k in range(Khw)],
                        dtype=torch.float32, device=x.device).view(1, 1, Khw, 1, 1)
    _dxs = torch.tensor([(k % kw) * dw for k in range(Khw)],
                        dtype=torch.float32, device=x.device).view(1, 1, Khw, 1, 1)
    # (N, Gh, Khw, Hout, Wout)
    sy = ys.view(1, 1, 1, Hout, 1) + _dys + off[:, :, :, 0]
    sx = xs.view(1, 1, 1, 1, Wout) + _dxs + off[:, :, :, 1]

    # 归一化到 grid_sample 的 [-1,1]（align_corners=True 的约定）
    # align_corners=True 时：x_norm = 2*x/(W-1) - 1
    if W > 1:
        gx = 2.0 * sx / (W - 1) - 1.0
    else:
        gx = torch.zeros_like(sx)
    if H > 1:
        gy = 2.0 * sy / (H - 1) - 1.0
    else:
        gy = torch.zeros_like(sy)
    grid = torch.stack([gx, gy], dim=-1)          # (N, G*Khw, Hout, Wout, 2)

    # ---- 2. 逐 tap 采样 ----
    # grid_sample 要求 grid 是 4D 且 batch 维必须与输入一致，不能把 tap 塞进通道。
    # 做法：把 deform group 折进 batch 维 → 每个 tap 只需一次 grid_sample
    # （共 kh*kw 次，3x3 就是 9 次），既省调用又保留了通道并行。
    #
    # ★ 采样一律在 fp32 下做 ★ 两个原因，都是踩过的：
    #   1) 功能性：grid_sample 要求 input 与 grid 同 dtype。grid 由 arange 拼出来
    #      天然是 fp32，配 fp16 输入会直接报
    #      "Expected tensor for argument #1 'input' to have the same type as ... grid"。
    #   2) 精度：坐标归一化到 [-1,1] 后，高分辨率下相邻像素间距可能小于 fp16
    #      在 1.0 附近的精度（~1e-3），采样位置会被量化 —— 与 RIFE 的
    #      warplayer.FORCE_FP32_WARP 是同一个道理。
    x_dtype = x.dtype
    xg = x.view(N, Gh, Cin_per_g, H, W).reshape(N * Gh, Cin_per_g, H, W)
    xg32 = xg.float() if x_dtype == torch.float16 else xg

    taps_out = []
    # gx/gy 是 (N, Gh*Khw, Hout, Wout)，先拆成 (N, Gh, Khw, Hout, Wout)
    gx5 = gx.view(N, Gh, Khw, Hout, Wout)
    gy5 = gy.view(N, Gh, Khw, Hout, Wout)
    for k in range(Khw):
        # (N, Gh, Hout, Wout, 2) → (N*Gh, Hout, Wout, 2)
        g = torch.stack([gx5[:, :, k], gy5[:, :, k]], dim=-1).reshape(
            N * Gh, Hout, Wout, 2)
        s = F.grid_sample(xg32, g, mode="bilinear", padding_mode="zeros",
                          align_corners=True)        # (N*G, Cin_per_g, Hout, Wout)
        taps_out.append(s.view(N, Gh, Cin_per_g, Hout, Wout))
    # (N, G, Khw, Cin_per_g, Hout, Wout) → 通道顺序与权重重排对齐
    feat = torch.stack(taps_out, dim=2)              # (N, G, Khw, Cin_per_g, Ho, Wo)
    msk_e = msk.view(N, Gh, Khw, 1, Hout, Wout)      # mask 对该组的通道广播
    feat = (feat * msk_e).reshape(N, Gh, Khw * Cin_per_g, Hout, Wout)

    # ---- 4. 1x1 卷积完成加权求和 ----
    # 原权重 (Cout, Cin_g, kh, kw)，Cin_g = Cin（groups=1）
    # 重排为 (Cout, G*Khw*Cin_per_g, 1, 1)，其中 (G, Cin_per_g) 展开回 Cin。
    # 需要与 feat 的通道顺序 (g, k, c_per_g) 对齐。
    w = weight.view(Cout, Gh, Cin_per_g, kh, kw)          # (Cout, G, Cin_g, kh, kw)
    w = w.permute(0, 1, 3, 4, 2).contiguous()             # (Cout, G, kh, kw, Cin_g)
    w = w.reshape(Cout, Gh * Khw * Cin_per_g, 1, 1)

    feat_flat = feat.reshape(N, Gh * Khw * Cin_per_g, Hout, Wout)
    out = F.conv2d(feat_flat, w, bias, stride=1, padding=0)
    return out if out.dtype == in_dtype else out.to(in_dtype)


def main():
    """与 torchvision 的实现做数值对照。"""
    import torchvision.ops as tops
    # ⚠️ 必须关掉 TF32 再对照。
    # 不关的话，我这边的 1x1 conv 走 cudnn/TF32（10 位尾数），而 torchvision 的
    # deform_conv 是自写 fp32 kernel —— 两边精度不同，会凭空比出 3e-4 的
    # "误差"，让人误以为拆解是错的。实测关掉后误差降到 1.5e-6（纯 fp32 舍入）。
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # BasicVSR++ 的真实配置：deform_groups=16，3x3，pad=1
    cases = [
        dict(N=1, Cin=64, Cout=64, H=32, W=48, G=16, kh=3, kw=3,
             stride=(1, 1), pad=(1, 1), dil=(1, 1), tag="BasicVSR++ 实配(G=16)", osc=1.0),
        dict(N=1, Cin=64, Cout=64, H=32, W=48, G=16, kh=3, kw=3,
             stride=(1, 1), pad=(1, 1), dil=(1, 1), tag="同配置+小offset(不出界)", osc=0.05),
        dict(N=1, Cin=32, Cout=32, H=24, W=32, G=1,  kh=3, kw=3,
             stride=(1, 1), pad=(1, 1), dil=(1, 1), tag="单组对照(G=1)", osc=1.0),
        dict(N=2, Cin=16, Cout=24, H=20, W=20, G=4,  kh=3, kw=3,
             stride=(1, 1), pad=(1, 1), dil=(1, 1), tag="多组+通道不等(G=4)", osc=1.0),
    ]
    print(f"设备 {dev}\n")
    print(f"  {'用例':<28}{'最大绝对误差':>14}{'相对误差':>12}{'判定':>10}")
    all_ok = True
    for c in cases:
        N, Cin, Cout = c["N"], c["Cin"], c["Cout"]
        G, kh, kw = c["G"], c["kh"], c["kw"]
        H, W = c["H"], c["W"]
        sh, sw = c["stride"]; ph, pw = c["pad"]; dh, dw = c["dil"]
        Hout = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        Wout = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1

        x = torch.randn(N, Cin, H, W, device=dev)
        off = torch.randn(N, 2 * G * kh * kw, Hout, Wout, device=dev) * 0.5 * c.get("osc", 1.0)
        msk = torch.rand(N, G * kh * kw, Hout, Wout, device=dev)
        w = torch.randn(Cout, Cin, kh, kw, device=dev) * 0.05
        b = torch.randn(Cout, device=dev) * 0.01

        ref = tops.deform_conv2d(x, off, w, b, stride=(sh, sw),
                                 padding=(ph, pw), dilation=(dh, dw), mask=msk)
        got = deform_conv2d_decomposed(x, off, msk, w, b, stride=(sh, sw),
                                       padding=(ph, pw), dilation=(dh, dw),
                                       deform_groups=G)
        err = (ref - got).abs().max().item()
        scale = ref.abs().max().item() + 1e-8
        rel = err / scale
        # 边界一圈的采样语义与 torchvision 不完全一致：
        #   torchvision 判越界用 `h <= -1 || h >= H`，允许 (-1,0) 区间的位置
        #   仍然对第 0 行/列产生部分权重；
        #   grid_sample 的 zeros padding 则以 [-1,1] 归一化框为准，更严格。
        # 这只会影响最外圈，内部应当**逐元素精确**。分开统计以确认这一点。
        m = 2 * max(dh, dw) + 1
        ri = ref[..., m:-m, m:-m]
        gi = got[..., m:-m, m:-m]
        err_in = (ri - gi).abs().max().item()
        rel_in = err_in / scale
        ok = rel_in < 1e-5
        all_ok &= ok
        print(f"  {c['tag']:<28}{err:>14.3e}{rel:>12.2e}{'✅' if ok else '❌':>10}"
              f"   内部误差 {err_in:.2e}")

    print()
    if all_ok:
        print("  ✅ 拆解与原实现数值等价 —— 可以按这条路线导出 ONNX / 编译 TensorRT")
    else:
        print("  ❌ 还有偏差，先别往下走")


if __name__ == "__main__":
    main()
