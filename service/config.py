"""路径与运行参数的统一配置。"""
from __future__ import annotations

import os

# 项目根目录：默认按本文件位置推导（本文件在 推理服务/ 下，故往上一级），
# 这样整个目录**搬到哪台机器都能跑**。需要时可用环境变量 FI_BASE 覆盖。
# （原来这里写死成 r"D:\project\FrameInterpolation"，打包发给别人解压就失效了。）
_HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("FI_BASE") or os.path.dirname(_HERE)

RIFE_DIR = os.path.join(BASE, "插帧模型", "RIFE")
VSR_DIR = os.path.join(BASE, "超分模型", "VSR")

RIFE_SRC = os.path.join(RIFE_DIR, "src")
# 独立仓库（本仓库）里模型源码放在 vendor/ 下，不是 插帧模型/RIFE/src
_VENDOR_RIFE = os.path.join(BASE, "vendor", "rife")
if not os.path.isdir(RIFE_SRC) and os.path.isdir(_VENDOR_RIFE):
    RIFE_DIR = _VENDOR_RIFE
    RIFE_SRC = _VENDOR_RIFE
RIFE_WEIGHT = os.path.join(RIFE_DIR, "weights", "flownet.pkl")

VSR_SRC = os.path.join(VSR_DIR, "src")
# 同理：仓库自带 vendor/vsr/（内含 vsr 包）
_VENDOR_VSR = os.path.join(BASE, "vendor", "vsr")
if not os.path.isdir(VSR_SRC) and os.path.isdir(_VENDOR_VSR):
    VSR_DIR = _VENDOR_VSR
    VSR_SRC = _VENDOR_VSR

RESR_DIR = os.path.join(BASE, "超分模型", "RealESRGAN")
RESR_WEIGHT = os.path.join(RESR_DIR, "weights", "RealESRGAN_x4plus.pth")
VSR_WEIGHT = os.path.join(VSR_DIR, "weights", "basicvsr_plusplus_reds4.pth")

# 模型产物（导出的 onnx / tensorrt 引擎）放这里
CACHE = os.path.join(BASE, "推理服务", "cache")

FFMPEG = None            # 延迟从 imageio_ffmpeg 取，避免 import 时的副作用


def ffmpeg() -> str:
    global FFMPEG
    if FFMPEG is None:
        import imageio_ffmpeg
        FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    return FFMPEG


def ensure_dirs() -> None:
    os.makedirs(CACHE, exist_ok=True)
