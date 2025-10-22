#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.models import resnet50, ResNet50_Weights

# ======== Dataset：递归遍历文件夹 =========
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".JPG", ".PNG"}

class ImageFolderFlat(Dataset):
    def __init__(self, root: str, transform=None, recursive: bool = True):
        self.root = Path(root)
        self.transform = transform
        it = self.root.rglob("*") if recursive else self.root.glob("*")
        self.paths = [p for p in it if p.is_file() and p.suffix in IMG_EXTS]
        self.paths.sort()

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        invalid = False
        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            # 坏图：用占位图，记录标志
            img = Image.fromarray(torch.zeros(3, 64, 64, dtype=torch.uint8).permute(1,2,0).numpy())
            invalid = True
        if self.transform: img = self.transform(img)
        return img, str(path), invalid

# ======== Transforms：与 eval 对齐 =========
def build_eval_tf(img_size: int, mean: Tuple[float, float, float], std: Tuple[float, float, float]):
    return transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

# ======== Model：与训练保持一致 =========
def build_model(num_classes: int, use_fadc: bool = False, pretrained: bool = True):
    if use_fadc:
        # 你的 FADCResNet，与训练一致
        from model import FADCResNet
        model = FADCResNet(num_classes=num_classes)
        if pretrained:
            # 可选：把 torchvision resnet50 权重部分迁移（与训练时逻辑一致）
            tv = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            tv_state = tv.state_dict()
            msd = model.state_dict()
            for k, v in tv_state.items():
                if k in msd and msd[k].shape == v.shape:
                    msd[k].copy_(v)
            model.load_state_dict(msd)
        return model
    else:
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = resnet50(weights=weights)
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
        return model

def _strip_module_prefix(sd):
    # 兼容 DataParallel 保存的 "module.xxx"
    if any(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd

def _strip_module_prefix(sd: dict):
    """兼容 DataParallel 保存的 'module.' 前缀"""
    if any(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd

def load_trained_model(
    ckpt_path: str,
    num_classes: int,
    use_fadc: bool = False,
    device:str = "cuda",
    init_with_pretrained_backbone: bool = False,
):
    """
    返回: model（to(device).eval()），并打印 missing/unexpected key 数量。
    - ckpt_path: 训练保存的 best.pth / last.pth
    - num_classes: 类别数（建议用 class_names.json 的长度）
    - use_fadc: 训练时是否用了 FADCResNet
    - init_with_pretrained_backbone: 是否先用 ImageNet 预训练初始化骨干（通常推理阶段 False 即可）
    """
    device = torch.device(device if isinstance(device, str) else device)

    # ① 实例化与训练时一致的结构
    model = build_model(
        num_classes=num_classes,
        use_fadc=use_fadc,
        pretrained=init_with_pretrained_backbone,  # 推理/继续训可设 False，避免多余初始化
    ).to(device)

    # ② 读取 checkpoint（兼容两种格式）
    ckpt_path = str(ckpt_path)
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(f"ckpt 不存在：{ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
    else:
        sd = state  # 可能直接就是 state_dict

    sd = _strip_module_prefix(sd)

    # ③ 加载（strict=False 更宽容：分类头维度不一致也能跳过）
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] strict=False  missing={len(missing)}  unexpected={len(unexpected)}")
    if missing:
        print("  missing keys (前10):", missing[:10])
    if unexpected:
        print("  unexpected keys (前10):", unexpected[:10])

    model.eval()
    return model


# ======== 推理主流程 =========
@torch.no_grad()
def predict(
    ckpt: str,
    image_dir: str,
    class_names: List[str],
    output_csv: str = "submission.csv",
    img_size: int = 288,
    batch_size: int = 64,
    workers: int = 0,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    device: str = None,
    use_fadc: bool = False,
):
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # 1) 数据
    tf = build_eval_tf(img_size, mean, std)
    ds = ImageFolderFlat(image_dir, transform=tf, recursive=True)
    if len(ds) == 0:
        raise RuntimeError(f"在 {image_dir} 下没有找到图片文件（支持：{sorted(IMG_EXTS)}）")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)

    # 2) 模型
    # num_classes = len(class_names)
    # model = build_model(num_classes=num_classes, use_fadc=use_fadc, pretrained=True)
    # model = model.to(dev).eval()

    with open("data/flowers_train_images/class_names.json", "r", encoding="utf-8") as f:
        class_names = json.load(f)

    num_classes = len(class_names)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_trained_model(
        ckpt_path="model/last.pth",  # 或 last.pth
        num_classes=num_classes,
        use_fadc=True,  # ← 训练时用了 FADC 就改为 True
        device=device,
        init_with_pretrained_backbone=False
    )

    # 3) 加载权重
    if ckpt and os.path.isfile(ckpt):
        state = torch.load(ckpt, map_location=dev)
        sd = state.get("state_dict", state)
        sd = _strip_module_prefix(sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[load] missing={len(missing)} unexpected={len(unexpected)} (strict=False)")
    else:
        print(f"[WARN] 未找到 ckpt：{ckpt}，将仅用 ImageNet 预训练权重进行推理。")

    # 4) 推理 & 收集 submission
    results = []  # (img_name, predicted_class, confidence)
    n_invalid = 0
    for imgs, paths, invalid_flags in loader:
        imgs = imgs.to(dev, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=(dev.type == "cuda")):
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
        top1_prob, top1_idx = probs.max(dim=1)

        for i in range(len(paths)):
            img_name = Path(paths[i]).name
            if bool(invalid_flags[i]):
                n_invalid += 1
                # 坏图：空预测或标 0.0 均可，按你需要改
                results.append((img_name, "", ""))
            else:
                cls = class_names[int(top1_idx[i])]
                conf = float(top1_prob[i].item())
                results.append((img_name, cls, f"{conf:.6f}"))

    # 5) 写出 CSV（三列：img_name, predicted_class, confidence）
    import csv
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["img_name", "predicted_class", "confidence"])
        # 可按文件名排序输出
        for row in sorted(results, key=lambda x: x[0]):
            writer.writerow(row)

    print(f"[done] 预测完成：{len(results)} 张，坏图 {n_invalid} 张。CSV 已保存到：{output_csv}")

# ======== CLI =========
def parse_triplet(s: str) -> Tuple[float, float, float]:
    xs = [float(t.strip()) for t in s.split(",")]
    assert len(xs) == 3
    return tuple(xs)  # type: ignore

def main():
    ap = argparse.ArgumentParser("Predict a folder and export submission.csv")
    ap.add_argument("--ckpt", type=str, required=True, help="训练保存的 best.pth / last.pth")
    ap.add_argument("--class-names", type=str, required=True,
                    help="训练时保存的 class_names.json 路径")
    ap.add_argument("--image-dir", type=str, required=True, help="要预测的图片根目录（递归）")
    ap.add_argument("--output", type=str, default="submission.csv")
    ap.add_argument("--img-size", type=int, default=288)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--mean", type=str, default="0.45134348,0.46730715,0.32222468", help="均值，逗号分隔")
    ap.add_argument("--std",  type=str, default="0.24617702,0.22343232,0.25126648", help="方差，逗号分隔")
    ap.add_argument("--device", type=str, default=None, help="cuda:0 / cpu，不填自动检测")
    ap.add_argument("--use-fadc", action="store_true", help="若训练时使用了 FADCResNet，请加此项")
    args = ap.parse_args()

    with open(args.class_names, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    assert isinstance(class_names, list) and len(class_names) > 0

    predict(
        ckpt=args.ckpt,
        image_dir=args.image_dir,
        class_names=class_names,
        output_csv=args.output,
        img_size=args.img_size,
        batch_size=args.batch_size,
        workers=args.workers,
        mean=parse_triplet(args.mean),
        std=parse_triplet(args.std),
        device=args.device,
        use_fadc=args.use_fadc,
    )

if __name__ == "__main__":
    main()
