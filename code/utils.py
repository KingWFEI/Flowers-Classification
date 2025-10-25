# -*- coding: utf-8 -*-
# utils.py
import csv, json, math, random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms

# ---------- CSV / Dataset ----------
class CsvDataset(Dataset):
    def __init__(self, rows: List[Dict], img_dir: Path, transform=None):
        self.rows = rows
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        path = self.img_dir / r["filename"]
        if not path.is_file():
            for ext in [".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff",".JPG",".PNG"]:
                p2 = self.img_dir / (r["filename"] + ext)
                if p2.is_file():
                    path = p2; break
        img = Image.open(path).convert("RGB")
        target = int(r["category_idx"])
        if self.transform: img = self.transform(img)
        return img, target

def read_csv(csv_path: Path) -> List[Dict]:
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("filename") or row.get("image") or row.get("file") or row.get("name")
            category = row.get("category_id") or row.get("label") or row.get("class_id") or row.get("category")
            zh = row.get("chinese_name") or row.get("chinese") or ""
            en = row.get("english_name") or ""
            if filename is None or category is None:
                raise ValueError(f"CSV 缺少 filename 或 category_id：{row}")
            rows.append({
                "filename": str(filename).strip(),
                "category_raw": str(category).strip(),
                "chinese": str(zh).strip(),
                "english_name": str(en).strip(),
            })
    if not rows:
        raise RuntimeError(f"CSV 为空：{csv_path}")
    return rows

def build_label_mapping(rows_a: List[Dict], rows_b: Optional[List[Dict]] = None) -> Tuple[Dict[str, int], List[str]]:
    cats = [r["category_raw"] for r in rows_a]
    if rows_b is not None:
        cats += [r["category_raw"] for r in rows_b]
    uniq = sorted(set(cats), key=lambda x: str(x))
    cat2idx = {c:i for i,c in enumerate(uniq)}
    class_names = []
    for c in uniq:
        sample = next((r for r in rows_a if r["category_raw"] == c), None)
        if sample is None and rows_b is not None:
            sample = next(r for r in rows_b if r["category_raw"] == c)
        zh = (sample.get("chinese") or "").strip()
        en = (sample.get("english_name") or "").strip()
        name = (zh + " " + en).strip() if (zh or en) else str(c)
        class_names.append(name)
    for r in rows_a: r["category_idx"] = cat2idx[r["category_raw"]]
    if rows_b is not None:
        for r in rows_b: r["category_idx"] = cat2idx[r["category_raw"]]
    return cat2idx, class_names

def stratified_split(rows: List[Dict], val_ratio: float, seed: int):
    rng = random.Random(seed)
    by_cls: Dict[int, List[Dict]] = {}
    for r in rows: by_cls.setdefault(r["category_idx"], []).append(r)
    train_rows, val_rows = [], []
    for _, items in by_cls.items():
        rng.shuffle(items)
        n = len(items)
        k = max(1, int(round(n * val_ratio)))
        val_rows.extend(items[:k]); train_rows.extend(items[k:])
    return train_rows, val_rows

def filter_broken(rows: List[Dict], img_dir: Path) -> List[Dict]:
    keep, drop = [], 0
    for r in rows:
        p = img_dir / r["filename"]
        if not p.is_file():
            ok = False
            for ext in [".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff",".JPG",".PNG"]:
                if (img_dir / (r["filename"] + ext)).is_file():
                    ok = True; break
            if not ok: drop += 1; continue
        keep.append(r)
    print(f"[filter] kept {len(keep)} rows, dropped {drop} broken images.")
    return keep

# ---------- Transforms ----------
def build_transforms(img_size=224, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225),
                     train=True, weak=False):
    from torchvision import transforms
    if train:
        tf = [
            transforms.RandomResizedCrop(img_size, scale=(0.7,1.0)),
            transforms.RandomHorizontalFlip(),
        ]
        if not weak:
            try:
                from torchvision.transforms import TrivialAugmentWide
                tf += [TrivialAugmentWide()]
            except Exception:
                tf += [transforms.ColorJitter(0.2,0.2,0.2,0.05)]
        tf += [transforms.ToTensor(), transforms.Normalize(mean, std)]
        if not weak:
            tf += [transforms.RandomErasing(p=0.15, scale=(0.02,0.2), ratio=(0.3,3.3), value='random')]
        return transforms.Compose(tf)
    else:
        return transforms.Compose([
            transforms.Resize(int(img_size*256/224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

# ---------- Mixup / CutMix ----------
def apply_mixup(images, targets, alpha=0.2):
    if alpha <= 0: return images, None, None
    B = images.size(0)
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    perm = torch.randperm(B, device=images.device)
    y1, y2 = targets, targets[perm]
    mixed = lam * images + (1. - lam) * images[perm]
    lam_i = images.new_full((B,), lam); lam_j = 1. - lam_i
    return mixed, (y1, y2, lam_i, lam_j), "mixup"

def apply_cutmix(images, targets, alpha=1.0):
    if alpha <= 0: return images, None, None
    B, C, H, W = images.size()
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    perm = torch.randperm(B, device=images.device)
    y1, y2 = targets, targets[perm]
    cx = torch.randint(0, W, (1,), device=images.device).item()
    cy = torch.randint(0, H, (1,), device=images.device).item()
    cut_w = int(W * math.sqrt(1 - lam)); cut_h = int(H * math.sqrt(1 - lam))
    x1 = max(cx - cut_w // 2, 0); x2 = min(cx + cut_w // 2, W)
    y1b = max(cy - cut_h // 2, 0); y2b = min(cy + cut_h // 2, H)
    images[:, :, y1b:y2b, x1:x2] = images[perm, :, y1b:y2b, x1:x2]
    lam_adj = 1. - ((x2 - x1) * (y2b - y1b)) / float(W * H)
    lam_i = images.new_full((B,), lam_adj); lam_j = 1. - lam_i
    return images, (y1, y2, lam_i, lam_j), "cutmix"

def weighted_ce(logits: torch.Tensor, target_pack, base_ce: nn.Module):
    if isinstance(target_pack, torch.Tensor):
        return base_ce(logits, target_pack)
    if isinstance(target_pack, (list, tuple)) and len(target_pack) >= 4:
        y1, y2, lam_i, lam_j = target_pack
        if getattr(base_ce, "reduction", "mean") != "none":
            base_ce = nn.CrossEntropyLoss(
                label_smoothing=getattr(base_ce, "label_smoothing", 0.0),
                reduction="none"
            )
        loss1 = base_ce(logits, y1)
        loss2 = base_ce(logits, y2)
        return (lam_i * loss1 + lam_j * loss2).mean()
    raise ValueError("weighted_ce: invalid target_pack")

# ---------- Evaluate ----------
@torch.no_grad()
def evaluate(model, loader, device, amp=False, use_tta=False, num_classes=100):
    model.eval()
    total, loss_sum = 0, 0.0
    correct1, correct5 = 0, 0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    criterion = nn.CrossEntropyLoss()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if use_tta:
            with torch.autocast(device_type="cuda", enabled=(amp and device.type=="cuda")):
                logits1 = model(images)
                logits2 = model(torch.flip(images, dims=[3]))
                logits = 0.5*(logits1 + logits2)
                loss = criterion(logits, targets)
        else:
            with torch.autocast(device_type="cuda", enabled=(amp and device.type=="cuda")):
                logits = model(images); loss = criterion(logits, targets)
        loss_sum += loss.item() * targets.size(0)
        total += targets.size(0)
        pred1 = logits.argmax(dim=1)
        correct1 += (pred1 == targets).sum().item()
        k = min(5, logits.size(1))
        top5 = logits.topk(k=k, dim=1).indices
        correct5 += top5.eq(targets.view(-1,1)).any(dim=1).float().sum().item()
        for t, p in zip(targets.view(-1).cpu(), pred1.view(-1).cpu()):
            cm[t, p] += 1
    tp = cm.diag().float()
    fp = cm.sum(dim=0).float() - tp
    fn = cm.sum(dim=1).float() - tp
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    # 只对“出现过的类别”计算宏 F1，避免小批次被未出现类别拉低
    present = (cm.sum(dim=1) > 0)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    macro_f1 = f1[present].mean().item() * 100.0 if present.any() else 0.0
    acc1 = 100.0 * correct1 / max(1, total)
    acc5 = 100.0 * correct5 / max(1, total)
    avg_loss = loss_sum / max(1, total)
    return acc1, acc5, macro_f1, avg_loss
