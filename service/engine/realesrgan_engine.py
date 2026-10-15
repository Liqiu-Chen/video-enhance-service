"""单帧超分引擎（Real-ESRGAN x4plus），支持**跨帧合批**与**分块**。

为什么换掉 BasicVSR++：
  BasicVSR++ 是时序递归网络，要在 4 条传播路径（backward/forward × 2 轮）
  各留一份逐帧特征，720p→4x 下单是"显存预留"就 15.26 GiB（16 GiB 卡的 94%），
  稍微再加一个模型就超出物理显存 → 驱动换页 → 慢 4.8 倍。
  Real-ESRGAN 是**逐帧独立**的，没有时序状态，显存低一个量级，而且
  **帧与帧可以真正并行**（合批），这是 VSR 做不到的。

两种提速手段，可叠加：
  1. **跨帧合批**：一次前向吃 batch 帧，把 GPU 喂饱。
  2. **分块**：把每帧切成 tile，块间留 tile_pad 重叠避免接缝；显存由 tile 大小
     精确控制，与帧数解耦。

  两者结合的关键是：把 (帧 × 分块) 摊平成**同一个 batch 维**，
  既不浪费并行度，又能把峰值显存压到 `batch × tile_area`。

接口与 `VsrEngine.upscale_chunk` 保持一致（(T,H,W,3) uint8 → (T,sH,sW,3) uint8），
所以流水线可以无缝切换后端。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from config import BASE

_REALESRGAN_SRC = os.path.join(BASE, "超分模型", "RealESRGAN", "src")
# 独立仓库里模型源码放在 vendor/ 下
if not os.path.isdir(_REALESRGAN_SRC):
    _REALESRGAN_SRC = os.path.join(BASE, "vendor", "realesrgan")
if _REALESRGAN_SRC not in sys.path:
    sys.path.insert(0, _REALESRGAN_SRC)

from realesrgan.rrdbnet import RRDBNet  # noqa: E402


class RealEsrganEngine:
    def __init__(self, weight: str, fp16: bool = True, batch: int = 4,
                 tile: int = 0, tile_pad: int = 32, pre_pad: int = 0,
                 out_scale: int = 4, downsample: str = "area",
                 device: str = "cuda", verbose: bool = True):
        """tile=0 表示不分块（整帧一次前向）；>0 时按 tile×tile 分块。
        batch 是**一次前向的总图块数**（帧 × 分块摊平后），不是帧数。

        out_scale=2 时与 VSR 后端同样的做法：先把输入降 2 倍再 4x 超分，
        最终输出为原图的 2 倍（官方只有 x4 一档权重）。
        """
        assert out_scale in (2, 4), "out_scale 只支持 2 或 4"
        # 分块在**原图**上切，而 2x 模式是在块内再降采样，
        # 坐标对不上（tile 边界与降采样网格错位）—— 这个组合不支持。
        assert not (int(tile) > 0 and out_scale == 2), \
            "tile 与 out_scale=2 不能同时用（分块坐标与降采样网格错位）"
        self.device = torch.device(device)
        self.fp16 = bool(fp16)
        self.batch = max(1, int(batch))
        self.tile = int(tile)
        self.tile_pad = int(tile_pad)
        self.pre_pad = int(pre_pad)
        self.downsample = downsample
        self.verbose = verbose
        self.net, self.net_scale = RRDBNet.from_pretrained(
            weight, scale=4, device=self.device, fp16=self.fp16)
        self.out_scale = int(out_scale)
        # 对外报的倍率：2x 模式下最终就是 2（内部仍是 4x 模型）
        self.scale = self.out_scale
        self._warmed = None
        self._log(f"RealESRGAN 就绪 权重x{self.net_scale} 输出x{self.out_scale} "
                  f"fp16={self.fp16} batch={self.batch} "
                  f"tile={self.tile or '不分块'}")

    def _log(self, *a):
        if self.verbose:
            print("  [resr]", *a, flush=True)

    # ------------------------------------------------------------------ #
    def _out_hw(self, h: int, w: int) -> tuple:
        """输出尺寸。2x 模式是「先降一半再 4x」，所以是 4*(h//2)。"""
        if self.out_scale == 2:
            return (h // 2) * 4, (w // 2) * 4
        return h * 4, w * 4

    def _to_tensor(self, arr_u8: np.ndarray) -> torch.Tensor:
        """(B,H,W,3) uint8 → (B,3,h,w) float，值域 [0,1]。

        out_scale=2 时在这里先把输入降 2 倍（官方只有 x4 权重）。
        """
        t = torch.from_numpy(np.ascontiguousarray(arr_u8))
        t = t.to(self.device, non_blocking=True).permute(0, 3, 1, 2)
        t = t.float().div_(255.0)
        if self.out_scale == 2:
            b, c, h, w = t.shape
            if self.downsample == "area":
                t = torch.nn.functional.avg_pool2d(t, 2, 2)
            else:
                mode = {"nearest": "nearest", "linear": "bilinear",
                        "bicubic": "bicubic"}.get(self.downsample, "bilinear")
                t = torch.nn.functional.interpolate(
                    t, size=(h // 2, w // 2), mode=mode,
                    align_corners=(False if mode != "nearest" else None))
        return t.half() if self.fp16 else t

    def _from_tensor(self, y: torch.Tensor) -> np.ndarray:
        """(B,3,H,W) → (B,H,W,3) uint8。注意用**非原地**运算：
        模型输出是 inference tensor，出了 inference_mode 不能原地改。"""
        y = y.float().clamp(0, 1).mul(255.0).round().to(torch.uint8)
        return y.permute(0, 2, 3, 1).cpu().numpy()

    @torch.inference_mode()
    def infer(self, t: torch.Tensor) -> torch.Tensor:
        if self.pre_pad:
            t = F.pad(t, (0, self.pre_pad, 0, self.pre_pad), mode="reflect")
        y = self.net(t)
        if self.pre_pad:
            p = self.pre_pad * self.scale
            y = y[:, :, :-p, :-p]
        return y

    # ------------------------------------------------------------------ #
    def _tile_origins(self, h: int, w: int):
        """返回每块的核心区域 (y0,y1,x0,x1) 与带 padding 的输入区域。"""
        plan = []
        for y0 in range(0, h, self.tile):
            for x0 in range(0, w, self.tile):
                y1, x1 = min(y0 + self.tile, h), min(x0 + self.tile, w)
                iy0, ix0 = max(y0 - self.tile_pad, 0), max(x0 - self.tile_pad, 0)
                iy1, ix1 = min(y1 + self.tile_pad, h), min(x1 + self.tile_pad, w)
                plan.append((y0, y1, x0, x1, iy0, iy1, ix0, ix1))
        return plan

    @torch.inference_mode()
    def upscale_chunk(self, frames: np.ndarray) -> np.ndarray:
        """(T,H,W,3) uint8 → (T,scale*H,scale*W,3) uint8。

        把「帧 × 分块」摊平成同一个 batch 维再分批推理：
        既不浪费并行度，峰值显存也由 `batch × 单块面积` 决定、与总帧数无关。
        """
        if frames is None or len(frames) == 0:
            return np.zeros((0, 0, 0, 3), np.uint8)
        T, h, w = len(frames), frames.shape[1], frames.shape[2]
        s = self.net_scale                      # 模型自身的倍率（4）
        oh, ow = self._out_hw(h, w)
        out = np.empty((T, oh, ow, 3), np.uint8)

        if self.tile > 0:
            plan = self._tile_origins(h, w)
            jobs = [(i, blk) for i in range(T) for blk in plan]
            for i0 in range(0, len(jobs), self.batch):
                grp = jobs[i0:i0 + self.batch]
                patches = [frames[i, iy0:iy1, ix0:ix1]
                           for i, (_, _, _, _, iy0, iy1, ix0, ix1) in grp]
                # 边缘块的尺寸会比内部块小，必须先补到同一尺寸才能堆成 batch
                # （否则 np.stack 直接报 "all input arrays must have the same shape"）。
                # 补完推理后再按各自的真实尺寸裁回来。
                sizes = [p.shape[:2] for p in patches]
                mh = max(s[0] for s in sizes)
                mw = max(s[1] for s in sizes)
                if any(s != (mh, mw) for s in sizes):
                    patches = [np.pad(p, ((0, mh - p.shape[0]), (0, mw - p.shape[1]),
                                          (0, 0)), mode="reflect")
                               for p in patches]
                t = self._to_tensor(np.stack(patches))
                y = self.infer(t).float().clamp(0, 1).mul(255.0).round()
                y = y.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
                for k, (i, (y0, y1, x0, x1, iy0, iy1, ix0, ix1)) in enumerate(grp):
                    ty, tx = (y0 - iy0) * s, (x0 - ix0) * s
                    out[i, y0 * s:y1 * s, x0 * s:x1 * s] = \
                        y[k, ty:ty + (y1 - y0) * s, tx:tx + (x1 - x0) * s]
                del t, y
                torch.cuda.empty_cache()
            return out

        # 不分块：直接按帧分批
        for i0 in range(0, T, self.batch):
            blk = frames[i0:i0 + self.batch]
            t = self._to_tensor(blk)
            y = self._from_tensor(self.infer(t))
            out[i0:i0 + len(blk)] = y
            del t, y
            torch.cuda.empty_cache()
        return out

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def warmup(self, h: int, w: int, iters: int = 2):
        """预热。**输入必须非零**：全零序列在同类模型里可能走捷径分支，
        导致某些 kernel 没被初始化、拖到第一个真实任务才加载。"""
        if self._warmed == (h, w, self.batch, self.tile):
            return
        import time
        t0 = time.time()
        rng = np.random.default_rng(0)
        dummy = rng.integers(0, 256, (1, h, w, 3), dtype=np.uint8)
        for _ in range(max(1, iters)):
            self.upscale_chunk(dummy)
        self._warmed = (h, w, self.batch, self.tile)
        self._log(f"已预热 {w}x{h}（{time.time()-t0:.1f}s）")
