import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, Bottleneck
from torchvision import models

__all__ = [
    "GeM", "ECA", "ECABottleneck", "ECAResNet50",
    "build_model"
]

# ---------- GeM ----------
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * float(p))
        self.eps = float(eps)

    def forward(self, x):
        x = torch.clamp(x, min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1).pow(1.0 / self.p)
        return x

# ---------- ECA ----------
class ECA(nn.Module):
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        y = self.avg(x)                            # [B,C,1,1]
        y = y.squeeze(-1).transpose(-1, -2)        # [B,1,C]
        y = self.conv(y)
        y = self.act(y.transpose(-1, -2).unsqueeze(-1))  # [B,C,1,1]
        return x * y

# ---------- Bottleneck with ECA ----------
class ECABottleneck(Bottleneck):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # conv1/bn1, conv2/bn2, conv3/bn3, downsample
        self.eca = ECA(self.bn3.num_features, k_size=3)

    def forward(self, x):
        identity = x

        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)

        out = self.eca(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# ---------- ResNet50 + ECA + GeM + Dropout head ----------
class ECAResNet50(ResNet):
    def __init__(self, num_classes=100, dropout=0.2, use_gem=True):
        super().__init__(block=ECABottleneck, layers=[3, 4, 6, 3], num_classes=num_classes)
        if use_gem:
            self.avgpool = GeM(p=3.0)  # 替换 GAP
        in_f = self.fc.in_features
        self.fc = nn.Sequential(
            nn.Dropout(p=float(dropout)),
            nn.Linear(in_f, num_classes)
        )

    @torch.no_grad()
    def load_from_torchvision(self):
        """把 torchvision 的 resnet50 预训练权重迁移到当前模型（除新增层外）。"""
        try:
            from torchvision.models import resnet50, ResNet50_Weights
            tv = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        except Exception:
            tv = models.resnet50(pretrained=True)  # 兼容旧版 torchvision
        tv_sd = tv.state_dict()
        sd = self.state_dict()
        matched = 0
        for k, v in tv_sd.items():
            if k in sd and sd[k].shape == v.shape:
                sd[k].copy_(v); matched += 1
        self.load_state_dict(sd, strict=False)
        print(f"[ECAResNet50] Transferred {matched}/{len(sd)} tensors from torchvision resnet50.")
        return self

def build_model(
        num_classes: int = 100,
        backbone: str = "resnet50",
        use_eca: bool = True,
        use_gem: bool = True,
        dropout: float = 0.2,
        pretrained: bool = True
) -> nn.Module:
    """
    构建模型：
      - 默认：ResNet50 + ECA + GeM + Dropout
      - 兼容纯 ResNet50（use_eca=False/use_gem=False）
    """
    if backbone.lower() != "resnet50":
        raise ValueError("Only resnet50 backbone is supported in this reference implementation.")
    if use_eca:
        m = ECAResNet50(num_classes=num_classes, dropout=dropout, use_gem=use_gem)
        if pretrained:
            m.load_from_torchvision()
        return m
    else:
        try:
            from torchvision.models import resnet50, ResNet50_Weights
            m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        except Exception:
            m = models.resnet50(pretrained=pretrained)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(in_f, num_classes))
        return m