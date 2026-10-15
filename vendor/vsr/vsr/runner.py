"""VSR 推理封装（BasicVSR / BasicVSR++ / RealBasicVSR）。

三类模型都是**多帧输入、多帧输出**：
  输入  (N, T, 3, H, W)，值域 [0,1]
  输出  (N, T, 3, 4H, 4W)
内部靠光流在时间维做双向递归传播，所以**必须整段送进来**，不能逐帧调用。

显存与序列长度的关系
--------------------
传播本身是逐步的（显存 O(1) 与 T 无关），但每一帧的中间特征都要留到最后再上采样：
  单帧特征 (1, 64, H, W) 在 960×540 下是 132 MB，1080P 下是 530 MB。
  T 帧就是 T 倍 —— 1080P 送 30 帧会要 16 GB 以上。
因此本封装默认按 `max_seq_len` 分段处理，段间留 `overlap` 帧做上下文衔接
（重叠部分只取后半段结果贴回，避免拼接处突变）。
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

import vsr as _vsr_pkg                                    # noqa: F401  (触发垫片注入)

MODELS = {
    "basicvsr": "BasicVSRNet",
    "basicvsr_plusplus": "BasicVSRPlusPlusNet",
    "real_basicvsr": "RealBasicVSRNet",
}

# 官方配置里的结构超参（见 mmagic/.mim/configs/*/）
PRESETS = {
    "basicvsr": dict(mid_channels=64, num_blocks=30),
    "basicvsr_plusplus": dict(mid_channels=64, num_blocks=7, is_low_res_input=True),
    "real_basicvsr": dict(mid_channels=64, num_propagation_blocks=20,
                          num_cleaning_blocks=20, dynamic_refine_thres=255,
                          is_fix_cleaning=False, is_sequential_cleaning=False),
}


class VSRRunner:
    def __init__(self, weight: str, variant: str = "basicvsr_plusplus",
                 spynet_weight: str | None = None, device: str | None = None,
                 fp16: bool = False, max_seq_len: int = 0, overlap: int = 2):
        assert variant in MODELS, f"未知变体: {variant}"
        self.variant = variant
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.fp16 = fp16
        self.max_seq_len = int(max_seq_len)
        self.overlap = int(overlap)

        cls_name = MODELS[variant]
        net_cls = getattr(_vsr_pkg, cls_name)
        kwargs = dict(PRESETS[variant])
        if spynet_weight:
            kwargs["spynet_pretrained"] = spynet_weight
        self.net = net_cls(**kwargs)

        from vsr.checkpoint import load_checkpoint
        info = load_checkpoint(self.net, weight, strict=True)
        self.net = self.net.eval().to(self.device)
        if fp16:
            self.net = self.net.half()
        import torch as _t
        self.n_params = sum(p.numel() for p in self.net.parameters())
        self.load_info = info

    # ------------------------------------------------------------------ #
    def _run_once(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (1, T, 3, H, W)。"""
        with torch.no_grad():
            out = self.net(seq)
            if isinstance(out, (tuple, list)):
                out = out[0]
        return out

    def upscale(self, frames: list[np.ndarray],
                on_frame=None) -> list[np.ndarray]:
        """frames: HxWx3 float32 [0,1] 的列表 → 返回 4 倍尺寸的同格式列表。

        **务必用 on_frame 回调，不要靠返回值攒着。**
        单帧输出 2880x5120x3 float32 = 169 MB，791 帧全攒在内存里要 134 GB，
        实测跑到第 300 帧就 numpy._ArrayMemoryError 崩了。
        给了 on_frame 就逐帧交出去、不在内存里保留（返回值只占位）。
        """
        if not frames:
            return []
        T = len(frames)
        step = self.max_seq_len if self.max_seq_len > 0 else T
        if step <= 0:
            step = T

        results: list[np.ndarray | None] = [None] * T
        start = 0
        n_chunk = 0
        t_all = time.time()
        while start < T:
            end = min(start + step, T)
            # 往前多带 overlap 帧做上下文；只有第一段不需要
            ctx = 0 if start == 0 else min(self.overlap, start)
            lo, hi = start - ctx, end

            seq = np.stack(frames[lo:hi])                      # (t, H, W, 3)
            t = torch.from_numpy(np.ascontiguousarray(seq)).permute(0, 3, 1, 2)
            t = t.unsqueeze(0).to(self.device)
            t = t.half() if self.fp16 else t.float()

            out = self._run_once(t)[0].float().clamp_(0, 1)     # (t, 3, H', W')
            arr = out.permute(0, 2, 3, 1).cpu().numpy()
            arr = arr[ctx:]                                     # 丢掉上下文帧
            for i in range(arr.shape[0]):
                k = start + i
                if k >= T:
                    break
                a = np.ascontiguousarray(arr[i])
                if on_frame is not None:
                    on_frame(k, a)                   # 立刻交出去，不留在内存
                else:
                    results[k] = a

            # 关键：每段结束立刻释放显存。
            # 不释放的话，791 帧 = 198 个分段，碎片会持续累积，
            # 实测跑到约 300 帧时进程直接以 0xC0000005（访问违例）崩溃。
            del t, out, arr
            torch.cuda.empty_cache()

            n_chunk += 1
            if n_chunk % 10 == 0 or end >= T:
                el = time.time() - t_all
                done = min(end, T)
                print(f"        段 {n_chunk:>3}  已处理 {done}/{T} 帧  "
                      f"{el:.0f}s  {done/el:.2f} 帧/秒  "
                      f"显存 {torch.cuda.memory_allocated()/2**30:.1f}"
                      f"/{torch.cuda.max_memory_allocated()/2**30:.1f} GiB "
                      f"(当前/峰值)", flush=True)

            if end >= T:
                break
            start = end

        if on_frame is not None:
            return []                                # 已逐帧交出，不占内存
        return [r if r is not None else np.zeros_like(frames[0]) for r in results]

    # ------------------------------------------------------------------ #
    @property
    def scale(self) -> int:
        return 4
