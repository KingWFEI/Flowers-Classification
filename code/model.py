# -*- coding: utf-8 -*-
# model.py
import timm
import torch.nn as nn

def build_model(model_name: str = "tf_efficientnetv2_s",
                num_classes: int = 100,
                pretrained: bool = True,
                drop_rate: float = 0.2):
    """
    构建 timm 模型（默认 EfficientNetV2-S），自动替换分类头为 num_classes。
    """
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
    return model

def head_param_names_timm(model) -> set:
    """
    尽可能稳健地查找分类头参数名称（不同 timm 模型命名不同）。
    """
    names = set()
    for n, _ in model.named_parameters():
        if any(k in n.lower() for k in ["classifier", "head", "fc", "last_linear"]):
            names.add(n)
    # 兜底：若没识别到，直接把最后一层线性层的参数名加入
    if not names:
        last_lin = None
        for m in model.modules():
            if isinstance(m, nn.Linear):
                last_lin = m
        if last_lin is not None:
            for n, p in model.named_parameters():
                if p is last_lin.weight or p is last_lin.bias:
                    names.add(n)
    return names
