"""VSR（BasicVSR++ / BasicVSR / RealBasicVSR）推理后端。

和原始 `超分模型/VSR/src/vsr/runner.py` 的区别（性能相关）：

1. `torch.inference_mode()` 取代 `no_grad()`：省掉 autograd 的版本计数与视图追踪。
2. 全程在 **GPU 张量**上进出，不再 numpy ↔ torch 来回搬（原版每段都要
   `np.stack` → `torch.from_numpy` → `.cpu().numpy()`，在 5K 尺寸下这笔拷贝很贵）。
3. 输入输出用 **uint8**：单帧 2880x5120x3 float32 是 169 MB，uint8 只要 42 MB，
   显存和内存压力都降到 1/4，而 8bit 素材本来就只有 256 级。
4. `cudnn.benchmark = True`：固定 shape 下让 cudnn 选最快算法。
5. 固定分段长度后可以开 **CUDA Graph**，把一整段的几百个 kernel launch 压成一次。
6. `channels_last` 可选（对 conv 友好的内存布局）。

注意：BasicVSR++ 依赖 `torchvision.ops.deform_conv2d`（可变形卷积），
ONNX 里没有对应的标准算子，所以**这条链不做 ONNX 导出**，走 PyTorch + CUDA Graph。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

from config import VSR_SRC, VSR_WEIGHT

if VSR_SRC not in sys.path:
    sys.path.insert(0, VSR_SRC)

import vsr as _vsr_pkg                                     # noqa: E402,F401

MODELS = {
    "basicvsr": "BasicVSRNet",
    "basicvsr_plusplus": "BasicVSRPlusPlusNet",
    "real_basicvsr": "RealBasicVSRNet",
}
PRESETS = {
    "basicvsr": dict(mid_channels=64, num_blocks=30),
    "basicvsr_plusplus": dict(mid_channels=64, num_blocks=7, is_low_res_input=True),
    "real_basicvsr": dict(mid_channels=64, num_propagation_blocks=20,
                          num_cleaning_blocks=20, dynamic_refine_thres=255,
                          is_fix_cleaning=False, is_sequential_cleaning=False),
}


class VsrEngine:
    def __init__(self, weight: str = VSR_WEIGHT, variant: str = "basicvsr_plusplus",
                 fp16: bool = True, chunk: int = 8, overlap: int = 2,
                 use_cuda_graph: bool = False,
                 out_scale: int = 4, downsample: str = "area",
                 segments: int = 1, cudnn_benchmark: bool = False,
                 allow_tf32: bool = True,
                 device: str = "cuda", verbose: bool = True):
        """out_scale: 输出的放大倍数。

        官方权重只有 ×4 一档。要做 ×2 就**先把输入下采样 2 倍、再走 4x 超分**，
        输出正好是原图的 2 倍尺寸（不需要额外参数）。
        `downsample` 决定这个 2x 下采样用哪种简单算法：
          nearest  最近邻——最快，但会有锯齿
          linear   双线性
          area     区域平均（只对整数倍下采样有意义，抗锯齿最好）
          bicubic  双三次

        segments: 一次前向里并行算几段（每段长 `chunk` 帧）。>1 就是"多帧并行"，
        段之间互相独立，叠到 batch 维 n 上一次算完。

        ⚠️ **实测多段并行是负优化，默认必须保持 1。**
        640x360 输入、段长 8：

            n=1  120.3 ms/帧   2.80 GiB
            n=2  134.8 ms/帧   5.58 GiB   （慢 12%）
            n=4  148.9 ms/帧  11.14 GiB   （慢 24%）

        原因：这段计算本来就是**算力饱和**的，合批只会多占显存、加剧缓存压力。
        720x1280 输入下 n=2 需要的显存直接超过 16 GiB（n=1 已占 10.38 GiB）。
        想省时间应该走 out_scale=2（实测快 5.1x），而不是并行。
        """
        assert variant in MODELS, f"未知变体: {variant}"
        assert out_scale in (2, 4), "out_scale 只支持 2 或 4"
        self.variant = variant
        self.device = torch.device(device)
        self.fp16 = bool(fp16)
        self.chunk = int(chunk)
        self.overlap = int(overlap)
        self.use_cuda_graph = bool(use_cuda_graph)
        self.out_scale = int(out_scale)
        self.downsample = downsample
        self.segments = max(1, int(segments))
        self.verbose = verbose
        self._warned_graph = False
        self.scale = self.out_scale          # pipeline 用它算输出尺寸

        # cuDNN 后端配置：默认**关掉 benchmark**。
        #
        # 实测（720p 输入、段长 8，见 bench_cudnn_careful.py）：
        #   benchmark=True  稳态 879.5 ms/帧，但**首次要吞 9~15 s 算法试探** ——
        #                   预热不足时同一配置 10 次调用平均 1811 ms/帧（慢 2 倍）
        #   benchmark=False 稳态 897~906 ms/帧，没有试探开销
        # 稳态只差 ~2%，而试探开销会实打实砸在每个新分辨率的**第一个任务**上。
        # 服务场景任务普遍不长，所以选可预测的 False。
        # 想榨那 2%：cudnn_benchmark=True，并确保 warmup() 跑够次数。
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)

        cls = getattr(_vsr_pkg, MODELS[variant])
        net = cls(**PRESETS[variant])
        from vsr.checkpoint import load_checkpoint
        self.load_info = load_checkpoint(net, weight, strict=True)
        net = net.eval().to(self.device)
        # ★★★ 不要用 net.half() ！★★★
        # 这里以前是 `if self.fp16: net = net.half()` —— 把整个网络(含权重和
        # 所有中间激活)粗暴转 fp16。实测这是**严重错误**：
        #     net.half()       : PSNR 30.11 dB  最大差 102/255  ← 精度废掉
        #     autocast(推荐)   : PSNR 66.98 dB  最大差   1/255  ← 达标
        # 原因是 BasicVSR++ 里有对数值范围敏感的部分（光流相关、递归累加、
        # 可变形卷积的坐标与掩码），整网 fp16 会把这些量推到精度边缘。
        # 正确做法是 **autocast 混合精度**：卷积走 fp16，敏感算子自动留 fp32。
        # 代价只有 1.85x → 1.53x（相对 fp32），但换回 37 dB 精度。
        self._amp = bool(self.fp16)          # 混合精度开关
        self.net = net
        self.n_params = sum(p.numel() for p in net.parameters())
        self._warmed = None
        self._log(f"VSR {variant} 就绪 params={self.n_params/1e6:.2f}M "
                  f"fp16={self.fp16} chunk={self.chunk} "
                  f"out_scale={self.out_scale} segments={self.segments} "
                  f"cudnn.benchmark={cudnn_benchmark}")

    def _log(self, *a):
        if self.verbose:
            print("  [vsr]", *a, flush=True)

    # ------------------------------------------------------------------ #
    def _prep(self, x) -> torch.Tensor:
        """uint8 → 模型输入 float。

          (T,H,W,3)     → (1,T,3,H,W)
          (k,T,H,W,3)   → (k,T,3,H,W)     多段并行时的批量输入

        注意：**不要试图用 channels_last** —— 本模型的接口是 5 维 (N,T,C,H,W)，
        而 `contiguous(memory_format=channels_last)` 只接受 4 维张量，直接报
        "required rank 4 tensor to use channels_last format"。
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(np.ascontiguousarray(x))
        if x.dim() == 4:
            t = x.to(self.device, non_blocking=True).permute(0, 3, 1, 2).unsqueeze(0)
        else:
            t = x.to(self.device, non_blocking=True).permute(0, 1, 4, 2, 3)
        t = t.float().div_(255.0)
        if self.out_scale == 2:
            t = self._downsample2(t)
        # 混合精度下输入保持 fp32 —— autocast 会在每个算子处决定该用哪种，
        # 手动转 half 反而会把敏感算子也拖进 fp16。
        return t

    def _downsample2(self, t: torch.Tensor) -> torch.Tensor:
        """(1,T,3,H,W) → (1,T,3,H/2,W/2)。

        这是「2x 超分」的关键一步：官方权重只有 4x，先降 2 倍再 4x 超分，
        最终就是原图的 2 倍。用最朴素的插值算法即可，不需要额外参数。

        在**显存里**做（张量已经在 GPU 上、且是 float），比在 numpy 上先降采样
        再拷贝省一次往返；而且 F.interpolate 需要在 4 维上做，所以要 view 一下。
        """
        n, tt, c, h, w = t.shape
        oh, ow = h // 2, w // 2
        if self.downsample == "area":
            # 区域平均：等价于对 2x2 窗口求均值，抗锯齿最好
            out = torch.nn.functional.avg_pool2d(t.view(n * tt, c, h, w), 2, 2)
        else:
            mode = {"nearest": "nearest", "linear": "bilinear",
                    "bicubic": "bicubic"}.get(self.downsample, "bilinear")
            out = torch.nn.functional.interpolate(
                t.view(n * tt, c, h, w), size=(oh, ow), mode=mode,
                align_corners=(False if mode != "nearest" else None))
        return out.view(n, tt, c, oh, ow)

    def _forward(self, t: torch.Tensor) -> torch.Tensor:
        # ★ 混合精度走 autocast，不是把整网转 half ★
        # 见 _build_torch 里 net.half() 被移除的原因说明。
        with torch.inference_mode():
            if getattr(self, "_amp", False):
                with torch.autocast("cuda", dtype=torch.float16):
                    out = self.net(t)
            else:
                out = self.net(t)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

    # ------------------------------------------------------------------ #
    # CUDA Graph 在本模型上**不可用**：录制时直接 device-side assert
    # （BasicVSR++ 里的可变形卷积 + 逐帧递归，图形捕获不稳）。
    # 这里不再保留那段代码，`use_cuda_graph` 参数仍接受但只会打一条提示，
    # 免得调用方以为开了就生效。
    def _graph_unsupported(self):
        if self.use_cuda_graph and not self._warned_graph:
            self._warned_graph = True
            self._log("提示：VSR 的 CUDA Graph 实测不可用（录制时 device-side "
                      "assert），已忽略 use_cuda_graph=True")

    def _out_hw(self, h: int, w: int) -> tuple:
        """给定输入尺寸，算出实际输出尺寸。

        2x 模式是先降一半再 4x 超分，所以输出 = 4*(h//2)。输入是奇数时会和
        2h 差 1 像素，这里按模型的实际行为算，不靠假设。
        """
        if self.out_scale == 2:
            return (h // 2) * 4, (w // 2) * 4
        return h * 4, w * 4

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def warmup(self, h: int, w: int, iters: int = 3):
        """按实际尺寸把后端彻底跑热。

        **为什么必须做**：cuDNN 对每个新 shape 的首次调用要付一笔很大的初始化
        （kernel 加载 + 算法/workspace 建立）。实测首次调用要十几秒，
        而稳态只要 ~0.93 s。

        ⚠️ **预热输入不能用 `np.zeros`！**
        全零序列会被 `check_if_mirror_extended` 判定成"镜像扩展"，
        从而**跳过正向光流的计算**（flows_forward=None）——
        于是正向光流那条分支的 kernel 从没被初始化，白白拖到第一个真实微批
        才首次加载（实测那一批要 17 s）。这里用非零的随机图，
        保证两条光流分支都被跑到。
        """
        if self._warmed == (h, w, self.segments, self.out_scale):
            return
        import time
        t0 = time.time()
        n = max(2, self.chunk + self.overlap)
        rng = np.random.default_rng(0)
        dummy = rng.integers(0, 256, (n, h, w, 3), dtype=np.uint8)
        for _ in range(max(1, iters)):
            self.upscale_chunk(dummy)
        self._warmed = (h, w, self.segments, self.out_scale)
        self._log(f"已预热 {w}x{h} ×{iters}（{time.time()-t0:.1f}s，"
                  f"含 cuDNN 首次初始化）")

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def upscale_chunk(self, frames: np.ndarray) -> np.ndarray:
        """frames: (T,H,W,3) uint8 → (T, out_scale*H, out_scale*W, 3) uint8。

        内部把序列切成**固定长度 `chunk` 的段**顺序推进，每段带 `overlap` 帧上下文；
        `segments > 1` 时会把 `segments` 个连续段叠成一批一次前向算完
        （BasicVSR++ 的入参本身就带 batch 维 n，段与段之间互相独立，可以并行）。

        **不保留任何中间结果**：算完立刻转成 uint8 交给调用方。

        整个方法跑在 inference_mode 下 —— 模型前向产出的张量是"推理张量"，
        出了 inference_mode 再做原地操作（clamp_/mul_）会被 PyTorch 直接拒绝：
          Inplace update to inference tensor outside InferenceMode is not allowed
        """
        if frames is None or len(frames) == 0:
            return np.zeros((0, 0, 0, 3), np.uint8)
        self._graph_unsupported()

        T, h, w = len(frames), frames.shape[1], frames.shape[2]
        oh, ow = self._out_hw(h, w)
        step = self.chunk if self.chunk > 0 else T
        seg = max(1, int(self.segments))
        # 批次内每段的长度统一成 step+overlap（只有全片第一段没有上下文，
        # 用重复帧补到同一长度，结果裁掉），这样 cudnn 只需认一种形状。
        maxlen = step + self.overlap
        out = np.empty((T, oh, ow, 3), np.uint8)

        pos = 0
        while pos < T:
            plan = []
            for s in range(seg):
                bs = pos + s * step
                if bs >= T:
                    break
                be = min(bs + step, T)
                ctx = 0 if bs == 0 else min(self.overlap, bs)
                plan.append((bs, be, ctx))
            if not plan:
                break
            k = len(plan)
            buf = np.empty((k, maxlen, h, w, 3), np.uint8)
            for i, (bs, be, ctx) in enumerate(plan):
                src = frames[bs - ctx:be]
                n = len(src)
                buf[i, :n] = src
                if n < maxlen:
                    buf[i, n:] = src[-1]            # 尾部补重复帧，稍后裁掉
            y = self._forward(self._prep(buf))
            # 非原地运算：模型输出是 inference tensor
            y = y[:, :, :, :oh, :ow].float().clamp(0, 1).mul(255.0).round()
            # 5 维：(k, maxlen, 3, oh, ow) → (k, maxlen, oh, ow, 3)
            arr = y.permute(0, 1, 3, 4, 2).to(torch.uint8).cpu().numpy()
            for i, (bs, be, ctx) in enumerate(plan):
                out[bs:be] = arr[i, ctx:ctx + (be - bs)]
            pos = plan[-1][1]
            del buf, y, arr
        return out

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def upscale_roi(self, frame_u8: np.ndarray) -> np.ndarray:
        """单帧便捷接口（会丢失时序建模，仅用于调试/对照）。"""
        return self.upscale_chunk(frame_u8[None])[0]
