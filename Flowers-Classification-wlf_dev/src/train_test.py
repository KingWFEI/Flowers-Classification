# -*- coding: utf-8 -*-
# train_test.py (with vis_metrics visualization)
# CSV -> DataLoader, ResNet50(+ECA/GeM), Mixup/CutMix or SnapMix, EMA, 按 0.7/0.2/0.1 选最优
import os
import csv
import json
import math
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许加载尾部缺字节图片
from tqdm import tqdm

# 允许从 ./code 导入 model.py
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "code"))
from model import build_model  # noqa

# 从你的 utils.py 使用 SnapMix & hook
from utils import register_last_conv_hook, snapmix  # noqa

# ------------------------------
# 可视化（tool/vis_metrics.py）适配层
# ------------------------------
# 允许从 ./tool 导入 vis_metrics.py；若不存在则回退到内置可视化
sys.path.append(os.path.join(os.path.dirname(__file__), "tools"))
try:
    import importlib
    _vm = importlib.import_module("vis_metrics")  # 你项目中的可视化脚本
    _VM_VISUALIZER_CLS = getattr(_vm, "Visualizer", None)
except Exception as _e:
    _vm = None
    _VM_VISUALIZER_CLS = None


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _json_dumps(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


class _FallbackViz:
    """
    当 tool/vis_metrics.py 不可用时使用的兜底可视化：
    - 保存 history.json（原逻辑已做）
    - 画 metrics 曲线到 metrics.png
    - 画混淆矩阵到 cm_epochXX.png
    """
    def __init__(self, out_dir: str, class_names: Optional[List[str]] = None):
        self.out_dir = Path(out_dir)
        self.class_names = class_names or []
        _ensure_dir(self.out_dir)

    def _plot_history(self, history: List[Dict]):
        if not history:
            return
        try:
            import matplotlib.pyplot as plt
            xs = [h["epoch"] for h in history]
            tr_loss = [h["train_loss"] for h in history]
            va_loss = [h["val_loss"] for h in history]
            tr_acc  = [h["train_acc"] for h in history]
            va_acc1 = [h["val_acc1"] for h in history]
            va_f1   = [h["val_macro_f1"] for h in history]

            # 损失
            plt.figure()
            plt.plot(xs, tr_loss, label="train_loss")
            plt.plot(xs, va_loss, label="val_loss")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.out_dir / "metrics_loss.png")
            plt.close()

            # 准确率/F1
            plt.figure()
            plt.plot(xs, tr_acc, label="train_acc(%)")
            plt.plot(xs, va_acc1, label="val_acc1(%)")
            plt.plot(xs, va_f1, label="val_macro_f1(%)")
            plt.xlabel("epoch")
            plt.ylabel("metric(%)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.out_dir / "metrics_acc.png")
            plt.close()
        except Exception as e:
            print(f"[vis:fallback] plot history failed: {e}")

    def _plot_cm(self, cm_tensor: torch.Tensor, epoch: int):
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            cm = cm_tensor.detach().cpu().numpy().astype(float)
            cm_sum = cm.sum(axis=1, keepdims=True) + 1e-9
            cm_norm = cm / cm_sum
            plt.figure(figsize=(6, 5))
            plt.imshow(cm_norm, interpolation='nearest')
            plt.title(f"Confusion Matrix (epoch {epoch})")
            plt.colorbar()
            tick_marks = range(len(self.class_names)) if self.class_names else range(cm.shape[0])
            plt.xticks(tick_marks, self.class_names if self.class_names else tick_marks, rotation=90)
            plt.yticks(tick_marks, self.class_names if self.class_names else tick_marks)
            # 文本
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    val = cm_norm[i, j]
                    if val > 0.001:
                        plt.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=6)
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            plt.savefig(self.out_dir / f"cm_epoch{epoch:03d}.png")
            plt.close()
        except Exception as e:
            print(f"[vis:fallback] plot cm failed: {e}")

    # 统一接口（和自定义 vis_metrics 可能不同，这里只需在训练时调用）
    def log_epoch(self, record: Dict, cm: Optional[torch.Tensor] = None, history: Optional[List[Dict]] = None):
        if history is not None:
            self._plot_history(history)
        if cm is not None:
            self._plot_cm(cm, epoch=int(record.get("epoch", 0)))


class _VMAdapter:
    def __init__(self, out_dir: str, class_names: Optional[List[str]] = None):
        self.out_dir = out_dir
        self.class_names = class_names or []
        self._fallback = _FallbackViz(out_dir=out_dir, class_names=self.class_names)
        if _VM_VISUALIZER_CLS is not None:
            try:
                self._inst = _VM_VISUALIZER_CLS(out_dir=out_dir, class_names=self.class_names)
                print("[vis] vis_metrics.Visualizer initialized.")
            except Exception as e:
                print(f"[vis] Visualizer init failed: {e}")
                self._inst = None
        else:
            self._inst = None

    def _call(self, name: str, *args, **kwargs) -> bool:
        target = self._inst if self._inst is not None else _vm
        if target is None:
            return False
        fn = getattr(target, name, None)
        if callable(fn):
            try:
                fn(*args, **kwargs)
                return True
            except Exception as e:
                print(f"[vis] call {name} failed: {e}")
        return False

    def log_epoch(self, record: Dict, cm: Optional[torch.Tensor] = None, history: Optional[List[Dict]] = None):
        # 1) 首选标准接口
        if self._call("log_epoch", record=record, cm=cm, history=history): return
        if self._call("update", record=record, cm=cm, history=history): return
        if self._call("add_record", record=record): pass

        # 2) 兼容你的 tools：plot_curves + （如果有的话）plot_cm
        ok = False
        if history is not None:
            ok = self._call("plot_curves", history=history, out_dir=self.out_dir) \
                 or self._call("plot_history", history=history, out_dir=self.out_dir) \
                 or self._call("save_curves", history=history, out_dir=self.out_dir)

        if cm is not None:
            ok = self._call("plot_cm", cm=cm, classes=self.class_names, out_dir=self.out_dir) or ok

        # 3) 仍然没画？回退内置可视化
        if not ok:
            self._fallback.log_epoch(record=record, cm=cm, history=history)


# ------------------------------
# Utils
# ------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def json_dumps(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def parse_triplet(s: str) -> Tuple[float, float, float]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"需要 3 个逗号分隔的数，比如 0.4513,0.4673,0.3222；收到：{s}")
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except Exception:
        raise argparse.ArgumentTypeError(f"解析失败：{s}")


# ------------------------------
# Dataset & CSV helpers
# ------------------------------
class FlowerCsvDataset(Dataset):
    """
    读取 CSV；字段至少包含：
      - filename: 图片文件名
      - category: 类别（字符串或数字均可）
    可选：
      - chinese, english_name：不影响训练
    """
    def __init__(self, rows: List[Dict], img_dir: Path, transform=None, num_classes: Optional[int] = None):
        self.rows = rows
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        path = self.img_dir / r["filename"]
        if not path.is_file():  # 兼容 CSV 不含扩展名
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".JPG", ".PNG"]:
                p2 = self.img_dir / (r["filename"] + ext)
                if p2.is_file():
                    path = p2
                    break
        try:
            with Image.open(path) as im:
                im.load()
                img = im.convert("RGB")
        except Exception as e:
            print(f"[WARN] fail to load {path}: {e}; resampling another image.")
            return self[random.randrange(0, len(self))]

        target = int(r["category_idx"])
        if self.transform:
            img = self.transform(img)
        return img, target


def _read_csv(csv_path: Path) -> List[Dict]:
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = (
                    row.get("filename") or row.get("image") or row.get("file") or
                    row.get("name") or row.get("path") or row.get("filepath")
            )
            category = (
                    row.get("category") or row.get("label") or row.get("class_id") or
                    row.get("category_id") or row.get("cat_id") or row.get("cls")
            )
            chinese  = row.get("chinese") or row.get("chinese_name") or row.get("category_chinese") or ""
            english  = row.get("english_name") or row.get("category_english_name") or ""

            if filename is None or category is None:
                raise ValueError(
                    f"CSV 缺少 filename 或 category 字段：{row}\n"
                    f"该行包含的键：{list(row.keys())}"
                )

            rows.append({
                "filename": str(filename).strip(),
                "category_raw": str(category).strip(),
                "chinese": str(chinese).strip(),
                "english_name": str(english).strip(),
            })
    if len(rows) == 0:
        raise RuntimeError(f"CSV 为空：{csv_path}")
    return rows


def _build_label_mapping(rows_a: List[Dict], rows_b: Optional[List[Dict]] = None) -> Tuple[Dict[str, int], List[str]]:
    cats = [r["category_raw"] for r in rows_a]
    if rows_b is not None:
        cats += [r["category_raw"] for r in rows_b]
    uniq = sorted(set(cats), key=lambda x: (str(x)))
    cat2idx = {c: i for i, c in enumerate(uniq)}

    class_names: List[str] = []
    for c in uniq:
        sample = next((r for r in rows_a if r["category_raw"] == c), None)
        if sample is None and rows_b is not None:
            sample = next(r for r in rows_b if r["category_raw"] == c)
        zh = (sample.get("chinese") or "").strip()
        en = (sample.get("english_name") or "").strip()
        name = (zh + " " + en).strip() if (zh or en) else str(c)
        class_names.append(name)

    for r in rows_a:
        r["category_idx"] = cat2idx[r["category_raw"]]
    if rows_b is not None:
        for r in rows_b:
            r["category_idx"] = cat2idx[r["category_raw"]]
    return cat2idx, class_names


def _stratified_split(rows: List[Dict], val_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    rng = random.Random(seed)
    by_class: Dict[int, List[Dict]] = {}
    for r in rows:
        by_class.setdefault(r["category_idx"], []).append(r)
    train_rows, val_rows = [], []
    for _, items in by_class.items():
        rng.shuffle(items)
        n = len(items)
        k = max(1, int(round(n * val_ratio)))
        val_rows.extend(items[:k])
        train_rows.extend(items[k:])
    return train_rows, val_rows


def filter_broken(rows: List[Dict], img_dir: Path) -> List[Dict]:
    keep, drop = [], 0
    for r in rows:
        p = img_dir / r["filename"]
        if not p.is_file():
            ok = False
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".JPG", ".PNG"]:
                p2 = img_dir / (r["filename"] + ext)
                if p2.is_file():
                    p = p2
                    ok = True
                    break
            if not ok:
                drop += 1
                continue
        try:
            with Image.open(p) as im:
                im.verify()
            keep.append(r)
        except Exception:
            drop += 1
    print(f"[filter] kept {len(keep)} rows, dropped {drop} broken images.")
    return keep


# ------------------------------
# Augmentations
# ------------------------------

def build_transforms(img_size: int, train: bool, lite: bool = False,
                     mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    tf = []
    if train and not lite:
        tf += [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
        ]
        try:
            from torchvision.transforms import TrivialAugmentWide
            tf += [TrivialAugmentWide()]
        except Exception:
            tf += [transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)]
    else:
        tf += [transforms.Resize(int(img_size * 1.14)), transforms.CenterCrop(img_size)]
    tf += [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    if train and not lite:
        tf += [transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3))]
    return transforms.Compose(tf)


# ------------------------------
# Mixup/CutMix helpers（保留，用于非 SnapMix 模式）
# ------------------------------

def apply_mixup(images, targets, alpha=0.2):
    if alpha <= 0:
        return images, None, None
    B = images.size(0)
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    perm = torch.randperm(B, device=images.device)
    y1, y2 = targets, targets[perm]
    mixed = lam * images + (1. - lam) * images[perm]
    lam_i = images.new_full((B,), lam)
    lam_j = 1. - lam_i
    return mixed, (y1, y2, lam_i, lam_j), "mixup"


def apply_cutmix(images, targets, alpha=1.0):
    if alpha <= 0:
        return images, None, None
    B, C, H, W = images.size()
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    perm = torch.randperm(B, device=images.device)
    y1, y2 = targets, targets[perm]

    cx = torch.randint(0, W, (1,), device=images.device).item()
    cy = torch.randint(0, H, (1,), device=images.device).item()
    cut_w = int(W * math.sqrt(1 - lam)); cut_h = int(H * math.sqrt(1 - lam))
    x1 = max(cx - cut_w // 2, 0); x2 = min(cx + cut_w // 2, W)
    y1c = max(cy - cut_h // 2, 0); y2c = min(cy + cut_h // 2, H)
    images[:, :, y1c:y2c, x1:x2] = images[perm, :, y1c:y2c, x1:x2]
    lam_adj = 1. - ((x2 - x1) * (y2c - y1c)) / float(W * H)
    lam_i = images.new_full((B,), lam_adj)
    lam_j = 1. - lam_i
    return images, (y1, y2, lam_i, lam_j), "cutmix"


def weighted_ce(logits: torch.Tensor, target_pack, base_ce: nn.Module):
    # 传进来的是普通 targets 张量（形状 [B]），直接普通 CE
    if isinstance(target_pack, torch.Tensor):
        return base_ce(logits, target_pack)

    # 传进来的是 (y1, y2, lam_i, lam_j)
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

    # None 或其它非法情形交给外层处理
    raise ValueError("weighted_ce: invalid target_pack")


def mix_prob(epoch, total_epochs, p_start=0.5, p_end=0.1):
    t = epoch / max(1, total_epochs - 1)
    return p_start + (p_end - p_start) * t


# ------------------------------
# EMA
# ------------------------------
class ModelEMA:
    def __init__(self, model: nn.Module, decay=0.9998):
        import copy
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = float(decay)

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k], alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])


# ------------------------------
# Evaluate (Top1/Top5/Macro-F1) + optional TTA
# ------------------------------
@torch.no_grad()
def tta_logits(model, images, amp=False):
    outs = [model(images)]
    outs.append(model(torch.flip(images, dims=[3])))  # hflip
    return torch.stack(outs, dim=0).mean(dim=0)


@torch.no_grad()
def evaluate(model, loader, device, amp=False, use_tta=False, desc="Eval", num_classes: int = 100,
            return_cm: bool = False):
    model.eval()
    total, loss_sum = 0, 0.0
    correct1, correct5 = 0, 0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(loader, desc=desc, leave=False) if loader is not None else []
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        try:
            from torch.cuda.amp import autocast
            ctx = autocast(enabled=(amp and device.type == "cuda"))
        except Exception:
            ctx = torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda"))
        with ctx:
            logits = tta_logits(model, images, amp=amp) if use_tta else model(images)
            loss = criterion(logits, targets)

        loss_sum += loss.item() * targets.size(0)
        total += targets.size(0)

        pred1 = logits.argmax(dim=1)
        correct1 += (pred1 == targets).sum().item()

        k = min(5, logits.size(1))
        top5 = logits.topk(k=k, dim=1).indices
        correct5 += top5.eq(targets.view(-1, 1)).any(dim=1).float().sum().item()

        for t, p in zip(targets.view(-1).cpu(), pred1.view(-1).cpu()):
            cm[t, p] += 1

    tp = cm.diag().float()
    fp = cm.sum(dim=0).float() - tp
    fn = cm.sum(dim=1).float() - tp
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    macro_f1 = f1.mean().item() * 100.0

    acc1 = 100.0 * correct1 / max(1, total)
    acc5 = 100.0 * correct5 / max(1, total)
    avg_loss = loss_sum / max(1, total)

    if return_cm:
        return acc1, acc5, macro_f1, avg_loss, cm
    return acc1, acc5, macro_f1, avg_loss


# ------------------------------
# Train one epoch
# ------------------------------

def train_one_epoch(model, loader, optimizer, device, scaler, criterion, epoch, total_epochs,
                    amp=False, grad_clip=None, mixup_alpha=0.2, cutmix_alpha=1.0, ema: Optional[ModelEMA] = None,
                    use_snapmix=False, snapmix_alpha=5.0, snapmix_prob=0.5, cls_head: Optional[nn.Module]=None):
    model.train()
    loss_sum, total, correct = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Train[{epoch}/{total_epochs}]", leave=False)

    p_mix = mix_prob(epoch - 1, total_epochs)  # 1-based epoch to 0-based index
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 选择增强策略
        target_pack = None
        if use_snapmix:
            images, target_pack, tag = snapmix(images, targets, model, cls_head,
                                               alpha=snapmix_alpha, prob=snapmix_prob)
            if tag is None:
                target_pack = None
        else:
            if torch.rand(1).item() < p_mix:
                images, target_pack, _ = apply_mixup(images, targets, alpha=mixup_alpha)
            if target_pack is None and torch.rand(1).item() < p_mix * 0.6:
                images, target_pack, _ = apply_cutmix(images, targets, alpha=cutmix_alpha)

        optimizer.zero_grad(set_to_none=True)

        try:
            from torch.cuda.amp import autocast, GradScaler
            ctx = autocast(enabled=(amp and device.type == "cuda"))
        except Exception:
            ctx = torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda"))

        with ctx:
            logits = model(images)
            if target_pack is None:
                # 本 batch 没有做 mix（或 SnapMix 未触发），走普通 CE
                loss = criterion(logits, targets)
            else:
                # 做了 SnapMix / Mixup / CutMix：用加权 CE
                loss = weighted_ce(logits, target_pack, base_ce=criterion)
            preds = logits.argmax(dim=1)

        if scaler is not None and amp and device.type == "cuda":
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

        if ema is not None:
            ema.update(model)

        bs = targets.size(0)
        loss_sum += loss.item() * bs
        correct += (preds == targets).sum().item()
        total += bs
        pbar.set_postfix({"loss": f"{loss_sum / total:.4f}", "acc(%)": f"{100.0 * correct / max(total, 1):.2f}"})

    train_loss = loss_sum / max(total, 1)
    train_acc  = 100.0 * correct / max(total, 1)
    return train_loss, train_acc


# ------------------------------
# Checkpoint helpers
# ------------------------------

def save_checkpoint(state, out_dir: str, is_best=False, filename="last.pth"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    torch.save(state, os.path.join(out_dir, filename))
    if is_best:
        torch.save(state, os.path.join(out_dir, "best.pth"))


# ------------------------------
# Main
# ------------------------------

def main():
    parser = argparse.ArgumentParser(description="Flower Classification (ResNet50 + ECA/GeM + SnapMix 可选)")
    parser.add_argument("--data-root", type=str, required=True, help="数据根目录（包含图像子目录）")
    parser.add_argument("--train-csv", type=str, required=True, help="训练 CSV")
    parser.add_argument("--val-csv", type=str, default="", help="验证 CSV（可选；不提供则从训练集划分）")
    parser.add_argument("--img-subdir", type=str, default="train_images", help="图片相对 data-root 的子目录")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--final-epochs", type=int, default=12, help="末段高分辨率微调的 epoch 数")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--final-img-size", type=int, default=299, help="末段微调时的分辨率")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", help="启用混合精度")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="./runs")
    parser.add_argument("--freeze-epochs", type=int, default=0, help="前若干轮只训练分类头")
    parser.add_argument("--use-balanced-sampler", action="store_true", help="不均衡数据友好")
    parser.add_argument("--no-ema", action="store_true", help="禁用 EMA")
    parser.add_argument("--use-tta", action="store_true", help="验证/提交时启用水平翻转 TTA")
    parser.add_argument("--no-eca", action="store_true", help="关闭 ECA（仅用于对比）")
    parser.add_argument("--no-gem", action="store_true", help="关闭 GeM（仅用于对比）")
    parser.add_argument("--resume", type=str, default="", help="恢复训练的权重路径（best/last）")
    # 自定义 Normalize 统计
    parser.add_argument("--dataset-mean", type=parse_triplet, default=(0.485, 0.456, 0.406))
    parser.add_argument("--dataset-std", type=parse_triplet, default=(0.229, 0.224, 0.225))
    # SnapMix 开关与超参
    parser.add_argument("--use-snapmix", action="store_true", help="使用 SnapMix（需要 utils.register_last_conv_hook）")
    parser.add_argument("--snapmix-alpha", type=float, default=5.0, help="SnapMix 的 Beta 分布超参")
    parser.add_argument("--snapmix-prob",  type=float, default=0.5, help="每个 batch 启用 SnapMix 的概率")
    # 可视化开关
    parser.add_argument("--no-vis", action="store_true", help="禁用 vis_metrics 可视化输出")

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root = Path(args.data_root)
    img_dir = data_root / args.img_subdir

    # 读取 CSV
    train_rows = _read_csv(Path(args.train_csv))
    val_rows = _read_csv(Path(args.val_csv)) if args.val_csv else None

    # 类别映射（统一 train/val）
    cat2idx, class_names = _build_label_mapping(train_rows, val_rows)
    num_classes = len(class_names)
    print(f"Num classes: {num_classes}")

    # 如果没有提供 val.csv，则从训练集分层划分
    if val_rows is None:
        train_rows, val_rows = _stratified_split(train_rows, val_ratio=0.1, seed=args.seed)

    # 预扫描剔除坏图
    train_rows = filter_broken(train_rows, img_dir)
    val_rows   = filter_broken(val_rows,   img_dir)

    # Transforms（主阶段，使用数据集统计）
    mean, std = args.dataset_mean, args.dataset_std
    train_tf = build_transforms(args.img_size, train=True, lite=False, mean=mean, std=std)
    val_tf   = build_transforms(args.img_size, train=False, lite=True, mean=mean, std=std)

    # Datasets / Loaders
    train_ds = FlowerCsvDataset(train_rows, img_dir, transform=train_tf, num_classes=num_classes)
    val_ds   = FlowerCsvDataset(val_rows,   img_dir, transform=val_tf,   num_classes=num_classes)

    if args.use_balanced_sampler:
        counts = Counter([r["category_idx"] for r in train_rows])
        weights = [1.0 / counts[r["category_idx"]] for r in train_rows]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.workers, pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, drop_last=False)

    # 模型
    model = build_model(
        num_classes=num_classes,
        backbone="resnet50",
        use_eca=not args.no_eca,
        use_gem=not args.no_gem,
        dropout=0.2,
        pretrained=True
    )

    # DP
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    # SnapMix 需要提前注册最后卷积层的 hook —— 对“当前这个 model 对象”注册即可
    if args.use_snapmix:
        register_last_conv_hook(model)

    # 冻结骨干（可选）
    if args.freeze_epochs > 0:
        print(f"Freeze backbone for {args.freeze_epochs} epochs")
        for name, p in model.named_parameters():
            if "fc" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False

    # 分组 LR：head 学得快一点
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and "fc" not in n]
    head_params     = [p for n, p in model.named_parameters() if p.requires_grad and "fc" in n]
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.25},
        {"params": head_params,     "lr": args.lr},
    ], weight_decay=args.weight_decay)

    # Scheduler（warmup + cosine）
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup_epochs))

    # AMP scaler
    try:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler(enabled=(args.amp and device.type == "cuda"))
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    # CE with label smoothing
    try:
        criterion = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
    except TypeError:
        criterion = nn.CrossEntropyLoss()

    # EMA
    ema = None if args.no_ema else ModelEMA(model, decay=0.9998)

    # Resume（可选）
    start_epoch = 1
    best_score = -1e9
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        try:
            model.load_state_dict(ckpt["state_dict"], strict=False)
        except Exception:
            model.load_state_dict(ckpt, strict=False)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from {args.resume}, start at epoch {start_epoch}")

    out_dir = Path(args.output)
    _ensure_dir(out_dir)
    history: List[Dict] = []

    # 可视化实例：优先使用项目内 vis_metrics，否则使用兜底
    if args.no_vis:
        vis = None
        print("[vis] disabled by --no-vis")
    else:
        if _vm is not None:
            vis = _VMAdapter(out_dir=str(out_dir), class_names=class_names)
        else:
            vis = _FallbackViz(out_dir=str(out_dir), class_names=class_names)
            print("[vis] vis_metrics not found, using fallback visualizer.")

    # 找到“最后一个 nn.Linear”作为 SnapMix 的分类头（兼容 Sequential 头）
    def get_last_linear(m: nn.Module) -> nn.Linear:
        core = m.module if isinstance(m, nn.DataParallel) else m
        last = None
        for mod in core.modules():
            if isinstance(mod, nn.Linear):
                last = mod
        assert last is not None, "未找到线性分类层（nn.Linear）。"
        return last

    cls_head = get_last_linear(model)

    # ---------- 训练主循环 ----------
    for epoch in range(start_epoch, args.epochs + 1):
        # 解冻
        if epoch == args.freeze_epochs + 1 and args.freeze_epochs > 0:
            print("Unfreeze backbone")
            for p in model.parameters():
                p.requires_grad = True
            backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
            head_params     = [p for n, p in model.named_parameters() if "fc" in n]
            optimizer = optim.AdamW([
                {"params": backbone_params, "lr": args.lr * 0.25},
                {"params": head_params,     "lr": args.lr},
            ], weight_decay=args.weight_decay)

        # Warmup
        if epoch <= args.warmup_epochs:
            warmup_factor = float(epoch) / float(max(1, args.warmup_epochs))
            for i, pg in enumerate(optimizer.param_groups):
                base_lr = args.lr if i == 1 else args.lr * 0.25  # 组1: backbone, 组2: head
                pg["lr"] = base_lr * warmup_factor

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, scaler, criterion,
            epoch=epoch, total_epochs=args.epochs,
            amp=args.amp, grad_clip=args.grad_clip,
            mixup_alpha=0.2, cutmix_alpha=1.0, ema=ema,
            use_snapmix=args.use_snapmix, snapmix_alpha=args.snapmix_alpha,
            snapmix_prob=args.snapmix_prob, cls_head=cls_head
        )

        # 验证（EMA 优先）
        eval_model = ema.ema if ema is not None else model
        val_out = evaluate(
            eval_model, val_loader, device, amp=args.amp, use_tta=args.use_tta,
            desc="Val", num_classes=num_classes, return_cm=True
        )
        val_acc1, val_acc5, val_f1, val_loss, cm = val_out

        score = 0.7 * val_acc1 + 0.2 * val_acc5 + 0.1 * val_f1
        is_best = score > best_score
        if is_best:
            best_score = score

        # 保存
        save_checkpoint({"epoch": epoch, "state_dict": eval_model.state_dict()}, str(out_dir), is_best=is_best)
        save_checkpoint({"epoch": epoch, "state_dict": model.state_dict()}, str(out_dir), is_best=False, filename="last_raw.pth")

        if epoch > args.warmup_epochs:
            cosine.step()

        rec = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]['lr']),
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc1": float(val_acc1),
            "val_acc5": float(val_acc5),
            "val_macro_f1": float(val_f1),
            "score": float(score),
        }
        history.append(rec)
        with open(out_dir / "history.json", "w", encoding="utf-8") as f:
            f.write(json_dumps(history))

        # 可视化：每个 epoch 结束后更新曲线与混淆矩阵
        if not args.no_vis and vis is not None:
            try:
                vis.log_epoch(record=rec, cm=cm, history=history)
            except Exception as e:
                print(f"[vis] log_epoch failed: {e}")

        print(f"[Epoch {epoch}] "
              f"train_acc={train_acc:.2f}  val_acc1={val_acc1:.2f}  val_acc5={val_acc5:.2f}  "
              f"val_macro_f1={val_f1:.2f}  score={score:.3f}  best={best_score:.3f}")

    # ---------- 末段高分辨率微调 ----------
    if args.final_epochs > 0 and args.final_img_size != args.img_size:
        print(f"\n==> High-res fine-tuning at {args.final_img_size} for {args.final_epochs} epochs (weak aug)")
        mean, std = args.dataset_mean, args.dataset_std
        train_ds.transform = build_transforms(args.final_img_size, train=True, lite=True, mean=mean, std=std)
        val_ds.transform   = build_transforms(args.final_img_size, train=False, lite=True, mean=mean, std=std)

        if args.use_balanced_sampler:
            counts = Counter([r["category_idx"] for r in train_rows])
            weights = [1.0 / counts[r["category_idx"]] for r in train_rows]
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                      num_workers=args.workers, pin_memory=True, drop_last=True)
        else:
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True, drop_last=False)

        # 降 LR
        for pg in optimizer.param_groups:
            pg["lr"] *= 0.2

        for k in range(1, args.final_epochs + 1):
            epoch = args.epochs + k
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, device, scaler, criterion,
                epoch=epoch, total_epochs=args.epochs + args.final_epochs,
                amp=args.amp, grad_clip=args.grad_clip,
                mixup_alpha=0.05, cutmix_alpha=0.3, ema=ema,
                use_snapmix=args.use_snapmix, snapmix_alpha=max(1.0, args.snapmix_alpha * 0.5),
                snapmix_prob=max(0.1, args.snapmix_prob * 0.5), cls_head=cls_head
            )
            val_out = evaluate(
                ema.ema if ema is not None else model, val_loader, device, amp=args.amp, use_tta=args.use_tta,
                desc="Val(hires)", num_classes=num_classes, return_cm=True
            )
            val_acc1, val_acc5, val_f1, val_loss, cm = val_out
            score = 0.7 * val_acc1 + 0.2 * val_acc5 + 0.1 * val_f1
            is_best = score > best_score
            if is_best:
                best_score = score
            save_checkpoint({"epoch": epoch, "state_dict": (ema.ema if ema is not None else model).state_dict()}, str(out_dir), is_best=is_best)
            save_checkpoint({"epoch": epoch, "state_dict": model.state_dict()}, str(out_dir), is_best=False, filename="last_raw.pth")

            rec = {
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]['lr']),
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc1": float(val_acc1),
                "val_acc5": float(val_acc5),
                "val_macro_f1": float(val_f1),
                "score": float(score),
            }
            history.append(rec)
            with open(out_dir / "history.json", "w", encoding="utf-8") as f:
                f.write(json_dumps(history))

            if not args.no_vis and vis is not None:
                try:
                    vis.log_epoch(record=rec, cm=cm, history=history)
                except Exception as e:
                    print(f"[vis] log_epoch(hires) failed: {e}")

            print(f"[HiRes Epoch {k}/{args.final_epochs}] "
                  f"val_acc1={val_acc1:.2f} val_acc5={val_acc5:.2f} val_macro_f1={val_f1:.2f} score={score:.3f} best={best_score:.3f}")

    print("\nTraining done.")
    print(f"Best weighted score: {best_score:.3f}")
    print(f"Artifacts saved in: {str(out_dir.resolve())}")


if __name__ == "__main__":
    main()
