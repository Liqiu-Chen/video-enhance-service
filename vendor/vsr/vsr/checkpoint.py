"""官方基于 mmengine 的 load_checkpoint 的轻量替代。

mmagic 的 checkpoint 有固定的封装格式，实测三种权重的结构是：

  basicvsr_reds4.pth          state_dict = { "generator.spynet.mean": ..., ... }        纯 generator
  basicvsr_plusplus_reds4.pth state_dict = { "step_counter": ..., "generator.xxx": ... } 混了 step_counter
  realbasicvsr_reds.pth       state_dict = { "generator.xxx", "discriminator.xxx",
                                             "perceptual_loss.xxx", "generator_ema.xxx" } 训练用全套

所以需要：① 取出 state_dict；② **剥掉 generator. 前缀**；③ 丢掉非 generator 的键。
本模块只做这三件事，保证能 strict 加载进推理用的 net。
"""
from __future__ import annotations

import torch


def load_checkpoint(model, filename, strict=True, logger=None, **kwargs):
    ckpt = torch.load(filename, map_location="cpu")
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model", "params_ema", "params"):
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break
    # 剥掉 generator. 前缀，并丢弃 discriminator / perceptual_loss / step_counter 等
    gkeys = [k for k in ckpt if k.startswith("generator.")]
    if gkeys:
        ckpt = {k[len("generator."):]: v for k, v in ckpt.items()
                if k.startswith("generator.")}

    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    # SPyNet 允许单独由 spynet_pretrained 提供，所以不算致命缺失
    hard_missing = [k for k in missing if not k.startswith("spynet.")]
    if strict and (hard_missing or unexpected):
        raise RuntimeError(
            f"权重与网络不匹配（{filename}）：\n"
            f"  缺失 {hard_missing[:6]}\n  多余 {list(unexpected)[:6]}")
    return {"missing": hard_missing, "missing_spynet": [k for k in missing
                                                        if k.startswith("spynet.")],
            "unexpected": list(unexpected)}
