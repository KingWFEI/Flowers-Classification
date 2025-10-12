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

# ---------------------- transforms（与训练 eval 对齐） ----------------------
def build_transforms(
    img_size: int = 288,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return eval_tf

# ---------------------- 简单的数据集：遍历一个文件夹 ----------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

class ImageFolderFlat(Dataset):
    def __init__(self, root: str, transform=None, recursive: bool = True):
        self.root = Path(root)
        self.transform = transform
        self.paths = []
        if recursive:
            it = self.root.rglob("*")
        else:
            it = self.root.glob("*")
        for p in it:
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                self.paths.append(p)
        self.paths.sort()

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            # 用一张全零图占位，后续会标记为 invalid
            img = Image.fromarray(torch.zeros(3, 64, 64, dtype=torch.uint8).permute(1,2,0).numpy())
            invalid = True
        else:
            invalid = False

        if self.transform: img = self.transform(img)
        return img, str(path), invalid

# ---------------------- 构建/加载模型 ----------------------
def build_model_for_infer(num_classes: int):
    """
    默认用 torchvision ResNet50 + ImageNet1K 预训练，再把 fc 改为 num_classes。
    若你训练时用的是这个结构，能 100% 对齐；如果你做了轻微结构改动，下面 load_state_dict(strict=False) 也能加载到绝大多数层。
    """
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, num_classes)
    return model

# ---------------------- 预测主流程 ----------------------
@torch.no_grad()
def predict(ckpt_path: str,
                   image_dir: str,
                   class_names: List[str],
                   output_csv: str,
                   img_size: int = 288,
                   batch_size: int = 64,
                   workers: int = 0,
                   topk: int = 5,
                   mean=(0.485,0.456,0.406),
                   std=(0.229,0.224,0.225),
                   device: str = None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # 1) transforms & dataset
    tf = build_transforms(img_size=img_size, mean=mean, std=std)
    ds = ImageFolderFlat(image_dir, transform=tf, recursive=True)
    if len(ds) == 0:
        raise RuntimeError(f"在 {image_dir} 下没有找到图片文件（支持扩展名：{sorted(list(IMG_EXTS))}）")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)

    # 2) model
    num_classes = len(class_names)
    model = build_model_for_infer(num_classes)
    model = model.to(device).eval()

    # 3) 加载权重（兼容 strict=False，避免你自定义过轻改时报错）
    if ckpt_path and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        sd = state.get("state_dict", state)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[load] missing={len(missing)} unexpected={len(unexpected)}  (strict=False)")
    else:
        print(f"[WARN] 未提供有效 ckpt（{ckpt_path}），将仅用 ImageNet 预训练权重进行推理。")

    # 4) 推理
    rows = []
    n_invalid = 0
    for imgs, paths, invalid_flags in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        probs = F.softmax(logits, dim=1)
        topk_probs, topk_idx = probs.topk(k=min(topk, num_classes), dim=1)

        for i in range(imgs.size(0)):
            path = paths[i]
            invalid = bool(invalid_flags[i])
            if invalid:
                n_invalid += 1
                rows.append({
                    "path": path,
                    "invalid_image": True,
                    "pred_top1": "",
                    "prob_top1": "",
                    "topk_labels": "",
                    "topk_probs": ""
                })
                continue

            idxs = topk_idx[i].tolist()
            probs_i = topk_probs[i].tolist()
            labels = [class_names[j] for j in idxs]
            rows.append({
                "path": path,
                "invalid_image": False,
                "pred_top1": labels[0],
                "prob_top1": f"{probs_i[0]:.6f}",
                "topk_labels": ",".join(labels),
                "topk_probs": ",".join(f"{p:.6f}" for p in probs_i),
            })

    # 5) 写出 CSV
    import csv
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["path","invalid_image","pred_top1","prob_top1","topk_labels","topk_probs"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[done] 预测完成，共 {len(rows)} 张，坏图 {n_invalid} 张。已保存：{output_csv}")

# ---------------------- CLI ----------------------
def main():
    ap = argparse.ArgumentParser("Predict a folder of images")
    ap.add_argument("--ckpt", type=str, required=True, help="训练输出的 best.pth / last.pth")
    ap.add_argument("--class-names", type=str, default=None,
                    help="训练时保存的 class_names.json（推荐）。")
    ap.add_argument("--classes", type=str, default=None,
                    help="若没有 json，可用逗号传入类名列表，例如：rose,tulip,orchid,...")
    ap.add_argument("--image-dir", type=str, required=True, help="要预测的图片文件夹（递归遍历）")
    ap.add_argument("--output", type=str, default="preds.csv")
    ap.add_argument("--img-size", type=int, default=288)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--mean", type=str, default="0.485,0.456,0.406")
    ap.add_argument("--std", type=str, default="0.229,0.224,0.225")
    ap.add_argument("--device", type=str, default=None, help="指定 cuda:0 / cpu，不填自动检测")
    args = ap.parse_args()

    def parse_triplet(s):
        xs = [float(t.strip()) for t in s.split(",")]
        assert len(xs) == 3
        return tuple(xs)

    # 类名来源：优先 json，其次 --classes
    if args.class_names:
        with open(args.class_names, "r", encoding="utf-8") as f:
            class_names = json.load(f)
        assert isinstance(class_names, list) and len(class_names) > 0, "class_names.json 内容不合法"
    elif args.classes:
        class_names = [t.strip() for t in args.classes.split(",") if t.strip()]
    else:
        raise SystemExit("请提供 --class-names 或 --classes（逗号分隔的类别名）")

    predict(
        ckpt_path=args.ckpt,
        image_dir=args.image_dir,
        class_names=class_names,
        output_csv=args.output,
        img_size=args.img_size,
        batch_size=args.batch_size,
        workers=args.workers,
        topk=args.topk,
        mean=parse_triplet(args.mean),
        std=parse_triplet(args.std),
        device=args.device,
    )

if __name__ == "__main__":
    main()
