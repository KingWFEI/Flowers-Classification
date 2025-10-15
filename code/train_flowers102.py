
import argparse
import json
import os
import random
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import math

import torch.nn.functional as F
from torchvision.models.resnet import ResNet

from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.datasets import Flowers102, ImageFolder
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


try:
    import timm # 可选
    from timm.data import resolve_model_data_config, create_transform
    HAS_TIMM = True
except Exception:
    HAS_TIMM = False


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # 更快
    torch.backends.cudnn.deterministic = False


def build_transforms(img_size: int = 224,
                     use_trivial_augment: bool = True,
                     mean=(0.485, 0.456, 0.406),
                     std=(0.229, 0.224, 0.225)) -> Tuple[transforms.Compose, transforms.Compose]:
    """Torchvision 增广：RandomResizedCrop + TrivialAugmentWide + RandomErasing"""
    train_tf = [
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    if use_trivial_augment:
        train_tf.append(transforms.TrivialAugmentWide(interpolation=InterpolationMode.BICUBIC))
    train_tf += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),
    ]
    train_tf = transforms.Compose(train_tf)

    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tf, eval_tf

# 构建训练迭代器
def build_dataloaders(data_root: str,
                      img_size: int,
                      batch_size: int,
                      workers: int,
                      combine_train_val: bool = False,   # 对于你已有 split，默认不合并
                      use_timm_transforms: bool = False,
                      timm_model_name: str = None):
    """
    使用 ImageFolder 读取 train/val/test 三个子目录。
    自动推断 num_classes，并返回 class_names（按 ImageFolder 的顺序）。
    """
    # 默认 ImageNet 统计
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    # 1) 构造 transforms（优先用我们定义的，避免 timm 版本差异）
    if use_timm_transforms and timm_model_name and HAS_TIMM:
        _tmp = timm.create_model(timm_model_name, pretrained=True, num_classes=1000)
        data_cfg = resolve_model_data_config(_tmp)
        train_tf = create_transform(input_size=data_cfg['input_size'], is_training=True, auto_augment=None)
        eval_tf  = create_transform(input_size=data_cfg['input_size'], is_training=False)
    else:
        train_tf, eval_tf = build_transforms(img_size=img_size, use_trivial_augment=True, mean=mean, std=std)

    root = Path(data_root)
    train_dir = root / "train"
    val_dir   = root / "valid"
    test_dir  = root / "test"

    print(train_dir)
    assert train_dir.is_dir(), f"未找到 {train_dir}"
    # val/test 可选，但建议都提供
    if not val_dir.is_dir():
        print(f"[WARN] 未找到 {val_dir}，将不在验证集上评估。")
    if not test_dir.is_dir():
        print(f"[WARN] 未找到 {test_dir}，将跳过最终测试评估。")

    # 2) 构造数据集
    ds_train = ImageFolder(str(train_dir), transform=train_tf)
    class_names = ds_train.classes
    num_classes = len(class_names)

    # 可选：把类名顺序保存下来，便于复现实验
    (root / "class_names.json").write_text(json.dumps(class_names, indent=2, ensure_ascii=False), encoding="utf-8")

    ds_val  = ImageFolder(str(val_dir),  transform=eval_tf)  if val_dir.is_dir()  else None
    ds_test = ImageFolder(str(test_dir), transform=eval_tf)  if test_dir.is_dir() else None

    # 3) 是否合并 train+val（你已有 val，通常不合并）
    if combine_train_val and ds_val is not None:
        from torch.utils.data import ConcatDataset
        ds_train = ConcatDataset([ds_train, ImageFolder(str(val_dir), transform=train_tf)])
        ds_val = None

    # 4) DataLoader
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=True, persistent_workers=workers > 0) if ds_val else None
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                             num_workers=workers, pin_memory=True, persistent_workers=workers > 0) if ds_test else None

    return train_loader, val_loader, test_loader, num_classes, class_names

# ---------------- FADC 模块 ----------------
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

# ------------------ build_model ------------------
def build_model(num_classes=102, timm_model_name: str = None, use_fadc=False, pretrained=True):
    if timm_model_name:
        import timm
        assert timm is not None, "请先 pip install timm"
        model = timm.create_model(timm_model_name, pretrained=pretrained, num_classes=num_classes)
        return model

    if use_fadc:
        # FADCResNet50
        model = FADCResNet(num_classes=num_classes)
        if pretrained:
            from torchvision.models import resnet50, ResNet50_Weights
            print("Loading torchvision resnet50 pretrained weights...")
            tv = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            tv_state = tv.state_dict()
            model_state = model.state_dict()
            transferred = 0
            for k, v in tv_state.items():
                if k in model_state and model_state[k].shape == v.shape:
                    model_state[k].copy_(v)
                    transferred += 1
            model.load_state_dict(model_state)
            print(f"Transferred {transferred} tensors from torchvision pretrained resnet50.")
        return model
    else:
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = resnet50(weights=weights)
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
        return model


@torch.no_grad()
def evaluate(model, loader, device, amp=False, desc="Eval"):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(loader, desc=desc, leave=False) if loader is not None else []
    scaler_ctx = torch.cuda.amp.autocast if (amp and device.type == "cuda") else torch.cpu.amp.autocast
    for batch in pbar:
        images, targets = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        with scaler_ctx():
            logits = model(images)
            loss = criterion(logits, targets)
        loss_sum += loss.item() * targets.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)
    if total == 0:
        return 0.0, 0.0
    return correct / total * 100.0, loss_sum / total


def train_one_epoch(model, loader, optimizer, device, scaler, criterion, amp=False, grad_clip=None):
    model.train()
    loss_sum = 0.0
    total = 0
    pbar = tqdm(loader, desc="Train", leave=False)
    scaler_ctx = torch.cuda.amp.autocast if (amp and device.type == "cuda") else torch.cpu.amp.autocast

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with scaler_ctx():
            logits = model(images)
            loss = criterion(logits, targets)

        if amp and device.type == "cuda":
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = targets.size(0)
        loss_sum += loss.item() * bs
        total += bs
        pbar.set_postfix({"loss": f"{loss_sum/total:.4f}"})

    return loss_sum / max(total, 1)


def save_checkpoint(state, out_dir: str, is_best=False, filename="last.pth"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(out_dir, filename)
    torch.save(state, path)
    if is_best:
        torch.save(state, os.path.join(out_dir, "best.pth"))


def main():
    parser = argparse.ArgumentParser(description="Flowers102 Training (ResNet50 + TrivialAugment)")
    parser.add_argument("--data", type=str, required=True, help="数据目录（会自动下载到该目录）")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", help="启用混合精度")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="./checkpoints_flowers102")
    parser.add_argument("--no-trainval", action="store_true",
                        help="不合并 train+val；每轮在 val 上评估（更慢）")
    parser.add_argument("--freeze-epochs", type=int, default=0,
                        help="前若干轮只训练分类头（冷启动更稳）")
    parser.add_argument("--timm-model", type=str, default=None,
                        help="可选：使用 timm 模型名（如 resnetv2_50x1_bit.goog_in21k）")
    parser.add_argument("--use-fadc", action="store_true", help="使用 FADC-ResNet50 架构（频域增强模块）")
    parser.add_argument("--use-timm-transforms", action="store_true",
                        help="若指定，将按 timm 的数据配置与 ta_wide 增广构造 transforms")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, num_classes, class_names = build_dataloaders(
        data_root=args.data,
        img_size=args.img_size,
        batch_size=args.batch_size,
        workers=args.workers,
        combine_train_val=False,  # 你已有 val，保持 False
        use_timm_transforms=args.use_timm_transforms,
        timm_model_name=args.timm_model,
    )
    print(f"Found {num_classes} classes: {class_names[:5]}{' ...' if len(class_names) > 5 else ''}")

    model = build_model(num_classes=102, timm_model_name=args.timm_model, use_fadc=args.use_fadc, pretrained=True)
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    # 冻结骨干（可选）
    if args.freeze_epochs > 0:
        print(f"Freeze backbone for {args.freeze_epochs} epochs")
        for name, p in model.named_parameters():
            if "fc" in name or "head" in name:  # 只训练分类头
                p.requires_grad = True
            else:
                p.requires_grad = False

    # 优化器 & 余弦退火 + 线性 warmup
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(current_epoch):
        if current_epoch < args.warmup_epochs:
            return float(current_epoch + 1) / float(max(1, args.warmup_epochs))
        # cosine from 1.0 -> 0.0
        progress = (current_epoch - args.warmup_epochs) / float(max(1, args.epochs - args.warmup_epochs))
        return 0.5 * (1.0 + (1.0 - progress) * -1.0 + 1.0)  # 等价于 cosine 简化，稳妥起见直接返回 1 -> 0
    # 更直接的余弦：可替换为 torch.optim.lr_scheduler.CosineAnnealingLR
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup_epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_acc = 0.0
    history = []

    for epoch in range(args.epochs):
        if epoch == args.freeze_epochs and args.freeze_epochs > 0:
            # 解除冻结
            print("Unfreeze backbone")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - epoch - 1))

        print(f"\nEpoch [{epoch+1}/{args.epochs}]  lr={optimizer.param_groups[0]['lr']:.6f}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, criterion,
                                     amp=args.amp, grad_clip=args.grad_clip)

        # 验证（若不合并 train+val）
        if val_loader is not None:
            val_acc, val_loss = evaluate(model, val_loader, device, amp=args.amp, desc="Val")
            print(f"Val  acc: {val_acc:.2f}%  loss: {val_loss:.4f}")
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
                save_checkpoint({"epoch": epoch + 1, "state_dict": model.state_dict()}, args.output, is_best=True)
        else:
            # 仅记录训练损失
            is_best = False

        save_checkpoint({"epoch": epoch + 1, "state_dict": model.state_dict()}, args.output, is_best=False)
        scheduler.step()

        history.append({
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]['lr'],
            "train_loss": float(train_loss),
            "val_acc": float(val_acc) if val_loader is not None else None
        })

    # 测试集评估（使用 best.pth 如存在）
    ckpt_best = os.path.join(args.output, "best.pth")
    if os.path.exists(ckpt_best):
        print(f"Load best checkpoint: {ckpt_best}")
        state = torch.load(ckpt_best, map_location=device)
        model.load_state_dict(state["state_dict"])

    test_acc, test_loss = evaluate(model, test_loader, device, amp=args.amp, desc="Test")
    print(f"\n==== Final Test ====\nTop-1 acc: {test_acc:.2f}%   loss: {test_loss:.4f}")

    # 保存训练日志
    with open(os.path.join(args.output, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "history": history,
                   "final_test_acc": float(test_acc), "final_test_loss": float(test_loss)}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
