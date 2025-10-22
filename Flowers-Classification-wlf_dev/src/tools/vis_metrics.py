# tools/vis_metrics.py
import os
import json
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def save_history_json(history: List[Dict], out_dir: str):
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def plot_curves(history: List[Dict], out_dir: str):
    """绘制 loss / acc / lr 曲线"""
    ensure_dir(out_dir)
    epochs = [h["epoch"] for h in history]
    train_loss = [h.get("train_loss") for h in history]
    train_acc  = [h.get("train_acc")  for h in history]
    val_loss   = [h.get("val_loss")   for h in history if h.get("val_loss") is not None]
    val_acc    = [h.get("val_acc")    for h in history if h.get("val_acc") is not None]
    lrs        = [h.get("lr")         for h in history]

    # 训练/验证 loss
    plt.figure(figsize=(6,4))
    plt.plot(epochs, train_loss, label="train_loss")
    if len(val_loss) == len(epochs):
        plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curves"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=150); plt.close()

    # 训练/验证 acc
    if any(a is not None for a in train_acc) or any(a is not None for a in val_acc):
        plt.figure(figsize=(6,4))
        if any(a is not None for a in train_acc):
            plt.plot(epochs, train_acc, label="train_acc")
        if len(val_acc) == len(epochs):
            plt.plot(epochs, val_acc, label="val_acc")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("Accuracy Curves"); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "acc_curves.png"), dpi=150); plt.close()

    # 学习率
    if all(l is not None for l in lrs):
        plt.figure(figsize=(6,4))
        plt.plot(epochs, lrs, label="lr")
        plt.xlabel("Epoch"); plt.ylabel("Learning Rate"); plt.title("LR Schedule")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "lr_curve.png"), dpi=150); plt.close()

def eval_confusion(model, loader, device, class_names: Optional[List[str]], out_dir: str, amp: bool = False):
    """在给定 loader 上评估并画混淆矩阵与每类准确率"""
    ensure_dir(out_dir)
    model.eval()
    all_preds, all_labels = [], []
    import torch
    from tqdm import tqdm

    with torch.no_grad():
        pbar = tqdm(loader, desc="Confusion", leave=False)
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=(amp and device.type=="cuda")):
                logits = model(images)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(targets.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    np.save(os.path.join(out_dir, "confusion_matrix.npy"), cm)

    plt.figure(figsize=(8,6))
    im = plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix"); plt.colorbar(im, fraction=0.046, pad=0.04)
    if class_names and len(class_names) <= 50:
        plt.xticks(range(len(class_names)), class_names, rotation=90, fontsize=6)
        plt.yticks(range(len(class_names)), class_names, fontsize=6)
    else:
        plt.xticks([]); plt.yticks([])
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=200); plt.close()

    # 每类准确率
    per_class_acc = (cm.diagonal() / cm.sum(1).clip(min=1)) * 100.0
    plt.figure(figsize=(8,4))
    plt.bar(np.arange(len(per_class_acc)), per_class_acc)
    plt.xlabel("Class Index"); plt.ylabel("Accuracy (%)"); plt.title("Per-class Accuracy")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "per_class_acc.png"), dpi=150); plt.close()

    # 文本报告
    if class_names and len(class_names) == cm.shape[0]:
        report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    else:
        report = classification_report(y_true, y_pred, digits=4)
    with open(os.path.join(out_dir, "class_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

def try_tensorboard_writer(log_dir: str):
    """可选：TensorBoard 记录器（如果用户安装了 tensorboardX 或 torch.utils.tensorboard）"""
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=log_dir)
    except Exception:
        try:
            from tensorboardX import SummaryWriter  # 备选
            writer = SummaryWriter(log_dir=log_dir)
        except Exception:
            writer = None
    return writer

def tb_log_scalars(writer, history: Dict):
    if writer is None: return
    step = history["epoch"]
    for k in ("train_loss","train_acc","val_loss","val_acc","lr"):
        v = history.get(k)
        if v is not None:
            writer.add_scalar(k, v, step)
