"""TRT 版 VSR 引擎：接口与 VsrEngine 完全一致，可直接替换进流水线。

关键约束：TensorRT 引擎是**固定 shape** 的 —— 编译时就把 t 帧写死进图里，
所以它只能吃恰好 t 帧。而流水线是按 chunk 喂的，长度会变。

处理办法：
  - 长度 < t ：在**尾部**重复最后一帧补齐到 t，跑完只取前 n 帧。
    之所以安全：BasicVSR++ 是因果的时序递归，第 i 帧的输出只依赖 0..i 帧，
    尾部补帧不会影响前面任何一帧的结果（PyTorch 引擎的 overlap 补齐同理）。
  - 长度 > t ：切成若干个 t 帧窗口分别跑。⚠️ 这会丢掉跨窗口的时序上下文，
    所以**推荐让流水线的 chunk 直接等于引擎的 t**（autoconfig 在
    960x1672 这一档选的就是 chunk=8）。

    from vsr_trt_engine import VsrTrtEngine
    e = VsrTrtEngine(t=8, out_scale=4)      # 自动找/建引擎
    hr = e.upscale_chunk(frames)            # 与 VsrEngine 同签名
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))          # 推理服务/engine
ROOT = os.path.dirname(os.path.dirname(HERE))              # 项目根
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from config import VSR_WEIGHT  # noqa: E402

# 引擎放在纯 ASCII 路径下（ONNX 导出器不支持非 ASCII 路径，见 build_trt_vsr_decomp.py）
# 引擎目录：现在放在 trt5（正确的图导出在这里）
TRT_DIR = os.path.join(ROOT, "trt5")


class VsrTrtEngine:
    """BasicVSR++ 的 TensorRT 后端（可变形卷积已拆成原生算子）。"""

    def __init__(self, t: int = 8, out_scale: int = 4, h: int | None = None,
                 w: int | None = None, mode: str = "tf32",
                 engine_path: str | None = None, verbose: bool = True):
        self.t = int(t)
        self.out_scale = int(out_scale)
        self.mode = mode
        self.verbose = verbose
        self.H = self.W = 0                     # 输入尺寸
        self.input_mode = "rgb"                 # 与 VsrEngine 对齐
        self.fp16 = True
        self.chunk = self.t
        self.overlap = 0
        self._engine_path = engine_path
        self._trt = None
        self._built = False

    # VsrEngine 有这个属性，autoconfig 会读，给个占位避免 AttributeError
    @property
    def net(self):
        return None

    @property
    def scale(self) -> int:
        """对外输出倍率。流水线靠它算输出尺寸，必须和 VsrEngine 语义一致。"""
        return self.out_scale

    @property
    def device(self):
        return torch.device("cuda")

    def _log(self, *a):
        if self.verbose:
            print("  [vsr-trt]", *a, flush=True)

    # ------------------------------------------------------------------ #
    def _ensure(self, h: int, w: int):
        """按**网络实际吃到的尺寸**找引擎。

        out_scale=2 时语义是「输入先降 2 倍 → 4x 超分」，所以网络吃的是
        (H//2, W//2)，引擎也得按那个尺寸编译。这一步与 VsrEngine 保持一致
        （用 avg_pool2d 做 2x2 箱式平均，抗锯齿比 bilinear 好且零成本）。
        """
        if self.out_scale == 2:
            h, w = h // 2, w // 2
        if self._built and (self.H, self.W) == (h, w):
            return
        # ★ 默认用 TF32 引擎 ★
        # 实测（真实帧、对 PyTorch fp32 真值、两边都不分块）：
        #     TF32 : 80.28 dB  最大差 1.0  1.86x  ← 精度最高且达标，选它
        #     fp32 : 72.22 dB  最大差 1.0  1.41x
        #     fp16 : 58.93 dB  最大差 21.0 3.97x  （快但超标）
        # 引擎目录见 build_trt_vsr_decomp.py；必须用**真实帧**做 dummy 导出，
        # 否则会因 is_mirror_extended 拿到残缺的图（精度只剩 31 dB）。
        path = self._engine_path or os.path.join(
            TRT_DIR, f"vsr_decomp_t{self.t}_{h}x{w}_x4_{self.mode}.trt")
        if not os.path.exists(path) and self.mode == "tf32":
            # TF32 引擎的命名与 build 脚本不同，兼容一下
            path = os.path.join(TRT_DIR, f"vsr_tf32_t{self.t}_{h}x{w}_x4.trt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"找不到 VSR 的 TRT 引擎：{path}\n"
                f"先构建：python build_trt_vsr_decomp.py --size {h} {w} "
                f"--t {self.t} --scale 4 --mode {self.mode}\n"
                f"（注意 --scale 固定 4：2x 输出是靠输入降采样实现的）")
        sys.path.insert(0, HERE)
        from build_trt_rife import TrtRife      # 通用 TRT 运行时封装
        self._trt = TrtRife(path)
        self.H, self.W = h, w
        self._built = True
        self._log(f"TensorRT 引擎就绪 {os.path.basename(path)} "
                  f"t={self.t} 网络输入 {w}x{h} in={self._trt.in_dtype}")

    def warmup(self, h: int, w: int):
        """与 VsrEngine.warmup 同签名：预跑一次，把首次开销摊掉。"""
        self._ensure(h, w)
        dummy = np.zeros((self.t, h, w, 3), np.uint8)
        self.upscale_chunk(dummy)

    # ------------------------------------------------------------------ #
    def upscale_chunk(self, frames: np.ndarray) -> np.ndarray:
        """frames: (n,H,W,3) uint8  →  返回 (n, out_h, out_w, 3) uint8。

        与 VsrEngine.upscale_chunk 行为对齐（含 out_scale 的 2x 降采样语义）。
        """
        n, h, w = frames.shape[0], frames.shape[1], frames.shape[2]

        # out_scale=2：先 2 倍降采样，再交给 4x 网络 —— 得到 2 倍输出。
        # 与 VsrEngine 同语义（avg_pool2d 箱式平均）。实测降采样开销
        # 0.02 ms/帧，占单帧的万分之一，可忽略。
        if self.out_scale == 2:
            t = torch.from_numpy(np.ascontiguousarray(frames))
            t = t.permute(0, 3, 1, 2).float()
            t = torch.nn.functional.avg_pool2d(t, 2, 2)
            frames = t.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8).numpy()
            h, w = frames.shape[1], frames.shape[2]

        self._ensure(frames.shape[1], frames.shape[2])
        assert (h, w) == (self.H, self.W), \
            f"尺寸变了 {(h,w)} != 已建引擎的 {(self.H,self.W)}，需要重建引擎"

        outs = []
        for s in range(0, n, self.t):
            chunk = frames[s:s + self.t]
            m = len(chunk)
            if m < self.t:
                # 尾部重复最后一帧补齐（因果递归 → 不影响前 m 帧结果）
                pad = np.repeat(chunk[-1:], self.t - m, axis=0)
                chunk = np.concatenate([chunk, pad], 0)
            x = torch.from_numpy(np.ascontiguousarray(chunk)).cuda()
            x = x.permute(0, 3, 1, 2)[None].float().div_(255.0)
            with torch.inference_mode():
                y = self._trt(x)
                # 注意：用**非原地**算子。inference tensor 不允许 inplace 修改
                # （会报 "Inplace update to inference tensor outside InferenceMode"）。
                y = y[0].permute(0, 2, 3, 1).float().clamp(0, 1).mul(255).round()
            outs.append(y[:m].to(torch.uint8).cpu().numpy())
        return np.concatenate(outs, 0)

    # ------------------------------------------------------------------ #
    def describe(self) -> str:
        return (f"VSR-TRT t={self.t} out_scale={self.out_scale} mode={self.mode} "
                f"({os.path.basename(self._engine_path or '')})")
