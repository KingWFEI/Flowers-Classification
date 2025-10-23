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

from model import build_model

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


def _strip_module_prefix(sd: dict):
    """去掉 DDP 的 'module.' 前缀"""
    if not isinstance(sd, dict):
        return sd
    return { (k[7:] if isinstance(k, str) and k.startswith("module.") else k): v for k, v in sd.items() }

def load_trained_model(
    ckpt_path: str,
    num_classes: int,
    device: str = "cuda",
    *,
    # 与 build_model 对齐的构建参数
    backbone: str = "resnet50",
    use_eca: bool = True,
    use_gem: bool = True,
    dropout: float = 0.2,
    init_with_pretrained_backbone: bool = False,  # 是否先用 ImageNet 初始化骨干
    strict: bool = False,                         # 加载权重时是否严格
    show_examples: int = 10,                      # 打印 missing/unexpected 的前 N 个
):
    """
    返回: 已加载权重并 .eval() 的模型（位于指定 device）。
    - 确保构建参数（use_eca/use_gem/backbone 等）与训练时保持一致。
    - strict=False 可在分类头或小结构不一致时尽量加载。
    """
    device = torch.device(device if isinstance(device, str) else device)

    # ① 先实例化与训练时一致的结构
    model = build_model(
        num_classes=num_classes,
        backbone=backbone,
        use_eca=use_eca,
        use_gem=use_gem,
        dropout=dropout,
        pretrained=init_with_pretrained_backbone,  # 推理阶段通常 False 即可
    )

    # ② 读取 checkpoint（兼容两种格式）
    ckpt_path = str(ckpt_path)
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"ckpt 不存在：{ckpt_path}")

    # torch>=2.0 支持 weights_only=True，更安全；低版本回退
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
        meta = state.get("meta", {})
    else:
        sd = state
        meta = {}

    sd = _strip_module_prefix(sd)

    # （可选）做个简单一致性检查，防止结构不一致
    ckpt_flags = {k: meta.get(k) for k in ["use_eca", "use_gem", "backbone", "num_classes"] if k in meta}
    if ckpt_flags:
        warn = []
        if "use_eca" in ckpt_flags and bool(ckpt_flags["use_eca"]) != bool(use_eca):
            warn.append(f"use_eca: ckpt={ckpt_flags['use_eca']} now={use_eca}")
        if "use_gem" in ckpt_flags and bool(ckpt_flags["use_gem"]) != bool(use_gem):
            warn.append(f"use_gem: ckpt={ckpt_flags['use_gem']} now={use_gem}")
        if "backbone" in ckpt_flags and str(ckpt_flags["backbone"]).lower() != backbone.lower():
            warn.append(f"backbone: ckpt={ckpt_flags['backbone']} now={backbone}")
        if warn:
            print("[warn] 架构开关与 ckpt 元信息不一致：", "; ".join(warn))

    # ③ 加载（strict 可配置；分类头维度不一致时建议 False）
    incompatible = model.load_state_dict(sd, strict=strict)
    try:
        # PyTorch 2.0+ 返回 IncompatibleKeys
        missing, unexpected = incompatible.missing_keys, incompatible.unexpected_keys
    except AttributeError:
        # 旧版本 load_state_dict 返回 (missing, unexpected)
        missing, unexpected = incompatible

    print(f"[load] strict={strict}  missing={len(missing)}  unexpected={len(unexpected)}")
    if missing:
        print("  missing keys (前{}): {}".format(show_examples, missing[:show_examples]))
    if unexpected:
        print("  unexpected keys (前{}): {}".format(show_examples, unexpected[:show_examples]))

    # ④ 放到目标设备并 eval
    model = model.to(device)
    model.eval()
    return model


# ======== 推理主流程 =========
@torch.no_grad()
def predict(
    ckpt: str,
    image_dir: str,
    class_names: list,
    output_csv: str,
    img_size: int,
    batch_size: int,
    workers: int,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    device: str,
    backbone: str,
    use_eca: bool,
    use_gem: bool,
    dropout: float,
    init_with_pretrained_backbone: bool,
    strict: bool,
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
        ckpt_path=ckpt,
        num_classes=num_classes,
        device=device,
        backbone=backbone,
        use_eca=use_eca,  # 训练时是否用了 ECA
        use_gem=use_gem,  # 训练时是否用了 GeM
        dropout=dropout,  # 与训练时一致
        init_with_pretrained_backbone=init_with_pretrained_backbone,  # 推理通常 False
        strict=strict,  # 分类头不同时更宽容
        show_examples=10,
    )


    # 推理 & 收集 submission
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
    ap.add_argument("--class-names", type=str, required=True, help="训练时保存的 class_names.json 路径")
    ap.add_argument("--image-dir", type=str, required=True, help="要预测的图片根目录（递归）")
    ap.add_argument("--output", type=str, default="results/submission.csv")
    ap.add_argument("--img-size", type=int, default=288)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--mean", type=str, default="0.45134348,0.46730715,0.32222468", help="均值，逗号分隔")
    ap.add_argument("--std",  type=str, default="0.24617702,0.22343232,0.25126648", help="方差，逗号分隔")
    ap.add_argument("--device", type=str, default=None, help="cuda:0 / cpu，不填自动检测")

    # === 新参数：与 build_model 对齐 ===
    ap.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50"],
                    help="特征骨干；当前实现仅支持 resnet50")
    ap.add_argument("--no-eca", action="store_true", help="关闭 ECA（训练时未用 ECA 则需要关闭）")
    ap.add_argument("--no-gem", action="store_true", help="关闭 GeM（训练时未用 GeM 则需要关闭）")
    ap.add_argument("--dropout", type=float, default=0.2, help="分类头 Dropout")
    ap.add_argument("--init-pretrained", action="store_true",
                    help="先用 ImageNet 预训练初始化骨干（推理通常不需要）")
    ap.add_argument("--strict", action="store_true",
                    help="load_state_dict(strict=True)。若分类头/结构不一致会报错")

    args = ap.parse_args()

    # 设备自动检测
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 读取类别名
    with open(args.class_names, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    assert isinstance(class_names, list) and len(class_names) > 0

    # 传参给 predict（建议在 predict 内部调用你前面写好的 load_trained_model）
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
        device=device,
        backbone=args.backbone,
        use_eca=(not args.no_eca),
        use_gem=(not args.no_gem),
        dropout=float(args.dropout),
        init_with_pretrained_backbone=args.init_pretrained,
        strict=args.strict,
    )

if __name__ == "__main__":
    main()
