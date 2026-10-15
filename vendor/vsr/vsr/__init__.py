"""自足的 VSR 模型包（BasicVSR / BasicVSR++ / RealBasicVSR）。"""
from . import compat

compat.install()          # 必须先注入 mmcv.ops 垫片，再导入官方模型代码

from .basicvsr_net import BasicVSRNet, SPyNet                   # noqa: E402
from .basicvsr_plusplus_net import BasicVSRPlusPlusNet          # noqa: E402
from .real_basicvsr_net import RealBasicVSRNet                  # noqa: E402

__all__ = ["BasicVSRNet", "SPyNet", "BasicVSRPlusPlusNet", "RealBasicVSRNet"]
