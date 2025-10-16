# ---------------- FADC 模块 ----------------
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import ResNet


class FADCConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=2, reduction=4, cutoff_ratio=0.2):
        super().__init__()
        self.low_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                  padding=kernel_size//2, bias=False)
        self.high_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, max(8, out_channels // reduction), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, out_channels // reduction), 2, 1, bias=True),
            nn.Sigmoid()
        )
        self.cutoff_ratio = float(cutoff_ratio)

    def forward(self, x):
        orig_dtype = x.dtype
        if x.dtype in [torch.float16, torch.bfloat16]:
            x = x.float()

        B, C, H, W = x.shape

        if H < 8 or W < 8:
            low_sp = self.low_conv(x)
            high_sp = self.high_conv(x)
        else:
            Xf = torch.fft.rfft2(x, dim=(-2, -1))
            _, _, Hf, Wf = Xf.shape

            fy = torch.linspace(0, 0.5, steps=Hf, device=x.device)
            fx = torch.linspace(0, 0.5, steps=Wf, device=x.device)
            yy = fy[:, None].repeat(1, Wf)
            xx = fx[None, :].repeat(Hf, 1)
            rad = torch.sqrt(xx**2 + yy**2)
            cutoff = self.cutoff_ratio * rad.max()

            low_mask = (rad <= cutoff).to(x.dtype)[None, None, :, :]
            Xf_low = Xf * low_mask
            Xf_high = Xf * (1.0 - low_mask)

            low_sp = torch.fft.irfft2(Xf_low, s=(H, W), dim=(-2, -1))
            high_sp = torch.fft.irfft2(Xf_high, s=(H, W), dim=(-2, -1))

            # 强制匹配尺寸
            if low_sp.shape[-2:] != (H, W):
                low_sp = F.interpolate(low_sp, size=(H, W), mode='bilinear', align_corners=False)
            if high_sp.shape[-2:] != (H, W):
                high_sp = F.interpolate(high_sp, size=(H, W), mode='bilinear', align_corners=False)

            low_sp = self.low_conv(low_sp)
            high_sp = self.high_conv(high_sp)

        out_stack = torch.stack([low_sp, high_sp], dim=1)
        gate_in = out_stack.sum(dim=1)
        w = self.gate(gate_in).view(B, 2, 1, 1, 1)
        fused = (out_stack * w).sum(dim=1)
        fused = self.bn(fused)
        fused = self.act(fused)

        if orig_dtype in [torch.float16, torch.bfloat16]:
            fused = fused.to(orig_dtype)
        return fused

# ---------------- FADC Bottleneck ----------------
class FADCBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        self.fadc = FADCConv2d(width, width, kernel_size=3, dilation=dilation)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

        # 如果 downsample 没提供，且通道/尺寸不匹配，则自动生成 1x1 下采样
        if downsample is None and (stride != 1 or inplanes != planes * self.expansion):
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * self.expansion)
            )
        else:
            self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 如果 stride > 1，需要对主分支下采样
        if self.stride > 1:
            out = F.interpolate(out, scale_factor=1/self.stride, mode='bilinear', align_corners=False)

        out = self.fadc(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class FADCResNet(ResNet):
    def __init__(self, num_classes=102):
        super().__init__(block=FADCBottleneck, layers=[3, 4, 6, 3], num_classes=num_classes)
