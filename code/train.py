# -*- coding: utf-8 -*-
# train.py
import os, json, argparse
from pathlib import Path
from typing import List, Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from model import build_model, head_param_names_timm
from utils import (read_csv, build_label_mapping, stratified_split, filter_broken,
                   CsvDataset, build_transforms, apply_mixup, apply_cutmix,
                   weighted_ce, evaluate)

def set_seed(seed: int = 42):
    import random
    random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def json_dumps(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)

def main():
    ap = argparse.ArgumentParser("Train EfficientNetV2-S (timm) on CSV")
    ap.add_argument("--data-root", type=str, required=True, help="根目录（包含图像子目录）")
    ap.add_argument("--train-csv", type=str, required=True, help="训练 CSV 路径")
    ap.add_argument("--img-subdir", type=str, default="train", help="图片子目录名")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--final-epochs", type=int, default=12)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--final-img-size", type=int, default=299)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--freeze-epochs", type=int, default=2)
    ap.add_argument("--use-balanced-sampler", action="store_true")
    ap.add_argument("--use-tta", action="store_true")
    ap.add_argument("--model-name", type=str, default="tf_efficientnetv2_s")
    ap.add_argument("--drop-rate", type=float, default=0.2)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = Path(__file__).resolve().parents[1]          # submission/
    model_dir = root_dir / "model"                           # ../model
    results_dir = root_dir / "results"                       # ../results
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)
    img_dir = data_root / args.img_subdir

    # 1) 读 CSV & 标签映射（分层划分 10% 验证）
    all_rows = read_csv(Path(args.train_csv))
    cat2idx, class_names = build_label_mapping(all_rows, None)
    train_rows, val_rows = stratified_split(all_rows, val_ratio=0.1, seed=args.seed)
    train_rows = filter_broken(train_rows, img_dir)
    val_rows   = filter_broken(val_rows,   img_dir)

    # 保存标签映射供预测用
    idx2cat = {int(v): str(k) for k, v in cat2idx.items()}
    (model_dir / "idx2cat.json").write_text(json_dumps(idx2cat), encoding="utf-8")
    (model_dir / "class_names.json").write_text(json_dumps(class_names), encoding="utf-8")

    # 2) 数据
    mean, std = (0.485,0.456,0.406), (0.229,0.224,0.225)
    train_tf = build_transforms(img_size=args.img_size, train=True,  weak=False, mean=mean, std=std)
    val_tf   = build_transforms(img_size=args.img_size, train=False, weak=True,  mean=mean, std=std)
    train_ds = CsvDataset(train_rows, img_dir, transform=train_tf)
    val_ds   = CsvDataset(val_rows,   img_dir, transform=val_tf)

    if args.use_balanced_samplER if False else args.use_balanced_sampler:  # typo guard
        from collections import Counter
        counts = Counter([r["category_idx"] for r in train_rows])
        weights = [1.0 / counts[r["category_idx"]] for r in train_rows]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=4, pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, drop_last=False)

    num_classes = len(class_names)

    # 3) 模型
    model = build_model(model_name=args.model_name, num_classes=num_classes,
                        pretrained=True, drop_rate=args.drop_rate)
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    # 4) 冻结骨干
    head_names = head_param_names_timm(model.module if isinstance(model, nn.DataParallel) else model)
    if args.freeze_epochs > 0:
        print(f"Freeze backbone for first {args.freeze_epochs} epochs")
        for n, p in model.named_parameters():
            p.requires_grad = (n in head_names)

    # 5) 优化器（分组 LR：头部更快）
    backbone_params = [p for n,p in model.named_parameters() if p.requires_grad and (n not in head_names)]
    head_params     = [p for n,p in model.named_parameters() if p.requires_grad and (n in head_names)]
    if len(backbone_params) == 0:
        optimizer = optim.AdamW([{"params": head_params, "lr": args.lr}], weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW([
            {"params": backbone_params, "lr": args.lr*0.25},
            {"params": head_params,     "lr": args.lr},
        ], weight_decay=args.weight_decay)

    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup_epochs))
    try:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler(enabled=(args.amp and device.type=="cuda"))
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type=="cuda"))

    try:
        criterion = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
    except TypeError:
        criterion = nn.CrossEntropyLoss()

    best_score = -1e9
    history: List[Dict] = []

    def train_one_epoch(epoch, total_epochs, train_loader):
        model.train()
        loss_sum, total, correct = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Train[{epoch}/{total_epochs}]", leave=False)
        # 动态减弱 mix 概率
        p_mix = 0.5 + (0.1 - 0.5) * (epoch-1)/max(1, total_epochs-1)
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            target_pack = targets
            import torch as T
            if T.rand(1).item() < p_mix:
                images, target_pack, _ = apply_mixup(images, targets, alpha=0.2)
            if isinstance(target_pack, T.Tensor) and T.rand(1).item() < p_mix*0.6:
                images, target_pack, _ = apply_cutmix(images, targets, alpha=1.0)

            optimizer.zero_grad(set_to_none=True)
            with T.autocast(device_type="cuda", enabled=(args.amp and device.type=="cuda")):
                logits = model(images)
                loss = weighted_ce(logits, target_pack, base_ce=criterion)
                preds = logits.argmax(dim=1)
            if scaler is not None and args.amp and device.type == "cuda":
                scaler.scale(loss).backward()
                if args.grad_clip is not None:
                    scaler.unscale_(optimizer)
                    T.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                if args.grad_clip is not None:
                    T.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            bs = targets.size(0)
            loss_sum += loss.item() * bs
            correct += (preds == targets).sum().item()
            total += bs
            pbar.set_postfix({"loss": f"{loss_sum/max(total,1):.4f}", "acc(%)": f"{100.0*correct/max(total,1):.2f}"})
        return loss_sum/max(total,1), 100.0*correct/max(total,1)

    # ---------- 主训练 ----------
    for epoch in range(1, args.epochs + 1):
        # 解冻
        if epoch == args.freeze_epochs + 1 and args.freeze_epochs > 0:
            print("Unfreeze backbone now")
            for p in model.parameters(): p.requires_grad = True
            backbone_params = [p for n,p in model.named_parameters() if (n not in head_names)]
            head_params     = [p for n,p in model.named_parameters() if (n in head_names)]
            optimizer = optim.AdamW([
                {"params": backbone_params, "lr": args.lr*0.25},
                {"params": head_params,     "lr": args.lr},
            ], weight_decay=args.weight_decay)

        # Warmup
        if epoch <= args.warmup_epochs:
            warm = float(epoch) / float(max(1, args.warmup_epochs))
            for i, pg in enumerate(optimizer.param_groups):
                base = args.lr if len(optimizer.param_groups)==1 or i==1 else args.lr*0.25
                pg["lr"] = base * warm

        train_loss, train_acc = train_one_epoch(epoch, args.epochs, train_loader)
        val_acc1, val_acc5, val_f1, val_loss = evaluate(model, val_loader, device, amp=args.amp, use_tta=args.use_tta, num_classes=num_classes)
        score = 0.7*val_acc1 + 0.2*val_acc5 + 0.1*val_f1
        is_best = score > best_score
        if is_best: best_score = score

        # 保存 last/best
        torch.save({"epoch": epoch, "state_dict": model.state_dict()},
                   model_dir / "last_model.pth")
        if is_best:
            torch.save({"epoch": epoch, "state_dict": model.state_dict()},
                       model_dir / "best_model.pth")

        if epoch > args.warmup_epochs: cosine.step()

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
            "best": float(best_score),
        }
        history.append(rec)
        (results_dir / "history.json").write_text(json_dumps(history), encoding="utf-8")
        print(f"[{epoch:03d}] train_acc={train_acc:.2f}  val@1={val_acc1:.2f}  val@5={val_acc5:.2f}  "
              f"f1={val_f1:.2f}  score={score:.3f}  best={best_score:.3f}")

    # ---------- 末段高分辨率微调 ----------
    if args.final_epochs > 0 and args.final_img_size != args.img_size:
        print(f"\n==> Hi-Res finetune at {args.final_img_size} for {args.final_epochs} epochs (weak aug)")
        train_loader = DataLoader(CsvDataset(train_rows, img_dir, transform=build_transforms(args.final_img_size, train=True, weak=True, mean=mean, std=std)),
                                  batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        val_loader   = DataLoader(CsvDataset(val_rows,   img_dir, transform=build_transforms(args.final_img_size, train=False, mean=mean, std=std)),
                                  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        # 降 LR
        for pg in optimizer.param_groups: pg["lr"] *= 0.2

        for k in range(1, args.final_epochs+1):
            epoch = args.epochs + k
            train_loss, train_acc = train_one_epoch(epoch, args.epochs+args.final_epochs, train_loader)
            val_acc1, val_acc5, val_f1, val_loss = evaluate(model, val_loader, device, amp=args.amp, use_tta=args.use_tta, num_classes=num_classes)
            score = 0.7*val_acc1 + 0.2*val_acc5 + 0.1*val_f1
            is_best = score > best_score
            if is_best: best_score = score
            torch.save({"epoch": epoch, "state_dict": model.state_dict()},
                       model_dir / "last_model.pth")
            if is_best:
                torch.save({"epoch": epoch, "state_dict": model.state_dict()},
                           model_dir / "best_model.pth")

            rec = {
                "epoch": epoch, "lr": float(optimizer.param_groups[0]['lr']),
                "train_loss": float(train_loss), "train_acc": float(train_acc),
                "val_loss": float(val_loss), "val_acc1": float(val_acc1),
                "val_acc5": float(val_acc5), "val_macro_f1": float(val_f1),
                "score": float(score), "best": float(best_score),
            }
            history.append(rec)
            (results_dir / "history.json").write_text(json_dumps(history), encoding="utf-8")
            print(f"[HiRes {k}/{args.final_epochs}] val@1={val_acc1:.2f} val@5={val_acc5:.2f} f1={val_f1:.2f} "
                  f"score={score:.3f} best={best_score:.3f}")

    print("\nTraining done.")
    print(f"Best weighted score = {best_score:.3f}")
    print(f"Model saved to: {str((model_dir/'best_model.pth').resolve())}")

if __name__ == "__main__":
    main()
