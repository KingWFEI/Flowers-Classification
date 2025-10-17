import argparse
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import build_dataloaders_from_csv
from model import FADCResNet
from utils import register_last_conv_hook
from utils import snapmix
import os, csv, random
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
from tools.vis_metrics import save_history_json, plot_curves, eval_confusion, try_tensorboard_writer, tb_log_scalars


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # 更快
    torch.backends.cudnn.deterministic = False

class FlowerCsvDataset(Dataset):
    """
    从 CSV 读取标注：
      必须字段：filename, category
      可选字段：chinese, english_name（仅用于记录类名，不影响训练）
    图片目录：img_dir / filename
    """
    def __init__(self, rows: List[Dict], img_dir: Path, transform=None):
        self.rows = rows
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        path = self.img_dir / r["filename"]
        # 保险处理：部分 CSV 可能不带扩展名，这里尝试常见后缀
        if not path.is_file():
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".JPG", ".PNG"]:
                if (self.img_dir / (r["filename"] + ext)).is_file():
                    path = self.img_dir / (r["filename"] + ext)
                    break
        img = Image.open(path).convert("RGB")
        target = int(r["category_idx"])  # 已在构建时映射成 [0..C-1]
        if self.transform: img = self.transform(img)
        return img, target


def _read_csv(csv_path: Path) -> List[Dict]:
    """
    读取 CSV，兼容带中文列名：filename, category, chinese, english_name
    """
    csv_path = Path(csv_path)
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # 字段名容错（用户截图里的列名示例）
        # 允许：category, label, class_id
        for row in reader:
            # 统一键名
            filename = row.get("filename") or row.get("image") or row.get("file") or row.get("name")
            category = row.get("category") or row.get("label") or row.get("class_id")
            chinese  = row.get("chinese") or row.get("chinese_name") or row.get("category_chinese") or ""
            english  = row.get("english_name") or row.get("category_english_name") or ""
            if filename is None or category is None:
                raise ValueError(f"CSV 缺少 filename 或 category 字段：{row}")
            rows.append({
                "filename": filename.strip(),
                "category_raw": str(category).strip(),
                "chinese": chinese,
                "english_name": english,
            })
    if len(rows) == 0:
        raise RuntimeError(f"CSV 为空：{csv_path}")
    return rows


def _build_label_mapping(rows: List[Dict]) -> Tuple[Dict[str, int], List[str]]:
    """
    把 CSV 里的原始类别（可能是字符串/非连续数字）映射到 [0..C-1]
    class_names：优先用 “中文+英文” 拼接；若没有就用原 category
    """
    cats = []
    for r in rows:
        cats.append(r["category_raw"])
    uniq = sorted(set(cats), key=lambda x: (str(x)))
    cat2idx = {c: i for i, c in enumerate(uniq)}

    # 生成类名列表
    class_names: List[str] = []
    for c in uniq:
        # 找到该类的第一条，用中文/英文组成名称
        sample = next(r for r in rows if r["category_raw"] == c)
        zh = (sample.get("chinese") or "").strip()
        en = (sample.get("english_name") or "").strip()
        if zh or en:
            name = f"{zh} {en}".strip()
        else:
            name = str(c)
        class_names.append(name)

    # 写回 idx
    for r in rows:
        r["category_idx"] = cat2idx[r["category_raw"]]
    return cat2idx, class_names


def _stratified_split(rows: List[Dict], val_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    """
    简易分层抽样：各类别按比例划分到 flowers_train_images/val
    """
    rng = random.Random(seed)
    by_class: Dict[int, List[Dict]] = {}
    for r in rows:
        by_class.setdefault(r["category_idx"], []).append(r)

    train_rows, val_rows = [], []
    for cls, items in by_class.items():
        rng.shuffle(items)
        n = len(items)
        k = max(1, int(round(n * val_ratio)))  # 每类至少 1 张进 val（若 n 很小）
        val_rows.extend(items[:k])
        train_rows.extend(items[k:])
    return train_rows, val_rows


def json_dumps(obj) -> str:
    import json
    return json.dumps(obj, indent=2, ensure_ascii=False)

# 损失函数
def ce_with_snapmix(logits: torch.Tensor, target_pack, base_ce):
    """
    target_pack: (y1, y2, lam_i, lam_j) 或 None
    base_ce: nn.CrossEntropyLoss(label_smoothing=...)
    """
    y1, y2, lam_i, lam_j = target_pack
    loss1 = base_ce(logits, y1)
    loss2 = base_ce(logits, y2)
    # 注意 lam_i/lam_j 逐样本；把它们扩展到 batch
    # CrossEntropyLoss 已做了 batch 归一，这里做逐样本权重需要使用 reduction='none'
    if getattr(base_ce, 'reduction', 'mean') != 'none':
        # 重新实例化一个 none 版
        base_ce_none = torch.nn.CrossEntropyLoss(label_smoothing=getattr(base_ce, 'label_smoothing', 0.0), reduction='none')
        loss1 = base_ce_none(logits, y1)
        loss2 = base_ce_none(logits, y2)
    loss = (lam_i * loss1 + lam_j * loss2).mean()
    return loss

# 从预训练模型构建自己的模型
def build_model(num_classes=100,use_fadc=False,pretrained=True):

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
    for batch in pbar:
        images, targets = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=(amp and device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, targets)
        loss_sum += loss.item() * targets.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)
    if total == 0:
        return 0.0, 0.0
    return correct / total * 100.0, loss_sum / total


def train_one_epoch(model, loader, cls_head,optimizer, device, scaler, criterion, amp=False, grad_clip=None):
    model.train()
    loss_sum, total, correct = 0.0, 0, 0
    pbar = tqdm(loader, desc="Train", leave=False)

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 先默认不启用 SnapMix
        target_pack = None
        # 以 0.5 概率启用 SnapMix
        images_sm, target_pack, tag = snapmix(images, targets, model, cls_head, alpha=5.0, prob=0.5)

        optimizer.zero_grad(set_to_none=True)



        with torch.amp.autocast(device_type="cuda", enabled=amp):
            logits = model(images_sm)
            if tag is not None:
                loss = ce_with_snapmix(logits, target_pack, base_ce=criterion)
                preds = logits.argmax(dim=1)
            else:
                loss = criterion(logits, targets)
                preds = logits.argmax(dim=1)
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
        correct += (preds == targets).sum().item()
        total += bs
        pbar.set_postfix({"loss": f"{loss_sum / total:.4f}", "acc(%)": f"{100.0 * correct / max(total, 1):.2f}"})
    train_loss = loss_sum / max(total, 1)
    train_acc  = 100.0 * correct / max(total, 1)
    return train_loss, train_acc

# 保存最佳模型
def save_checkpoint(state, out_dir: str, is_best=False, filename="last.pth"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(out_dir, filename)
    torch.save(state, path)
    if is_best:
        torch.save(state, os.path.join(out_dir, "best.pth"))


def main():
    parser = argparse.ArgumentParser(description="Flowers102 Training (ResNet50 + TrivialAugment)")
    # parser.add_argument("--data", type=str, required=True, help="数据目录（会自动下载到该目录）")
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
    parser.add_argument("--output", type=str, default="./model")
    parser.add_argument("--freeze-epochs", type=int, default=0,
                        help="前若干轮只训练分类头（冷启动更稳）")
    parser.add_argument("--use-fadc", action="store_true", help="使用 FADC-ResNet50 架构（频域增强模块）")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, num_classes, class_names = build_dataloaders_from_csv(
        data_root="data/flowers_train_images",
        csv_path="train_labels.csv",
        img_subdir="train_images",
        img_size=args.img_size,
        batch_size=args.batch_size,
        workers=args.workers,
        val_ratio=0.1,  # 从训练集中划 10% 做验证
        seed=args.seed
    )
    print(f"Found {num_classes} classes: {class_names[:5]}{' ...' if len(class_names) > 5 else ''}")

    model = build_model(num_classes=100,use_fadc=args.use_fadc, pretrained=True)
    # 用 forward hook 抓取最后一个卷积特征图
    register_last_conv_hook(model)

    # 把分类头拿出来传给 snapmix（一般是 model.fc 或 model.classifier）
    cls_head = model.module.fc if isinstance(model, torch.nn.DataParallel) else model.fc
    print('分类头：', cls_head)
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
    scaler = torch.amp.GradScaler(device="cuda", enabled=(args.amp and device.type == "cuda"))
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_acc = 0.0
    writer = try_tensorboard_writer(args.output)
    history = []

    # 开始训练
    for epoch in range(args.epochs):
        if epoch == args.freeze_epochs and args.freeze_epochs > 0:
            # 解除冻结
            print("Unfreeze backbone")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - epoch - 1))

        print(f"\nEpoch [{epoch+1}/{args.epochs}]  lr={optimizer.param_groups[0]['lr']:.6f}")
        train_loss,train_acc  = train_one_epoch(model, train_loader,cls_head, optimizer, device, scaler, criterion,
                                     amp=args.amp, grad_clip=args.grad_clip)

        # 验证
        val_acc, val_loss = evaluate(model, val_loader, device, amp=args.amp, desc="Val")
        print(f"Val  acc: {val_acc:.2f}%  loss: {val_loss:.4f}")
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            save_checkpoint({"epoch": epoch + 1, "state_dict": model.state_dict()}, args.output, is_best=True)

        save_checkpoint({"epoch": epoch + 1, "state_dict": model.state_dict()}, args.output, is_best=False)
        scheduler.step()

        rec = {
            "epoch": epoch + 1,
            "lr": float(optimizer.param_groups[0]['lr']),
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc)
        }
        history.append(rec)
        # 保存/绘图/TensorBoard
        save_history_json(history, args.output)
        plot_curves(history, args.output)
        tb_log_scalars(writer, rec)

    eval_confusion(model, val_loader, device, class_names, out_dir=args.output, amp=args.amp)
    print(f"可视化结果已保存到：{args.output}\n"
          f"- loss_curves.png / acc_curves.png / lr_curve.png\n"
          f"- confusion_matrix.png / per_class_acc.png / class_report.txt\n"
          f"- history.json")

if __name__ == "__main__":
    main()
