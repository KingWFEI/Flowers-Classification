import torch
import torch.nn as nn
import torch.nn.functional as F

def register_last_conv_hook(model: nn.Module):
    """
    在 model 上注册 forward hook，抓取“最后一个卷积层”的输出特征图。
    保存到 model._snapmix_fmap（形状: [B, C, H, W]）
    """
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    assert last_conv is not None, "未找到卷积层，无法用于 SnapMix"

    def _hook(module, inp, out):
        # out: [B, C, H, W]
        model._snapmix_fmap = out

    handle = last_conv.register_forward_hook(_hook)
    model._snapmix_hook = handle
    return handle  # 可在退出时 .remove()
