#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image

from src.utils import build_transforms

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)


def parse_mean_std(s: str):
    # "0.485,0.456,0.406" -> tuple(float)
    parts = [float(x.strip()) for x in s.split(",")]
    assert len(parts) == 3, "mean/std 需为3个数，用逗号分隔"
    return tuple(parts)


def denormalize(t: torch.Tensor, mean, std):
    """
    t: [B,3,H,W] 或 [3,H,W]，值域在归一化后
    返回 0~1 范围 tensor 以便可视化/保存
    """
    if t.dim() == 3:
        t = t.unsqueeze(0)
    mean = torch.tensor(mean, device=t.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=t.device).view(1, 3, 1, 1)
    out = t * std + mean
    out = torch.clamp(out, 0.0, 1.0)
    return out.squeeze(0) if out.size(0) == 1 else out


def visualize_train_augs(
    dataset_root: Path,
    split: str,
    save_path: Path,
    img_size: int,
    mean,
    std,
    n_images: int = 8,
    n_augs: int = 6,
    seed: int = 42,
    use_trivial_augment: bool = True,
    show: bool = False
):
    """
    针对训练增强：对每张原图重复应用 n_augs 次，拼成网格并保存
    """
    set_seed(seed)
    # 读取原图（不加任何 transform）
    ds = ImageFolder(str(dataset_root / split), transform=None)
    indices = random.sample(range(len(ds)), k=min(n_images, len(ds)))
    train_tf, _ = build_transforms(img_size=img_size, use_trivial_augment=use_trivial_augment, mean=mean, std=std)

    # 收集每张图的 n_augs 次增强结果
    rows = []
    for idx in indices:
        pil_img, label = ds[idx]
        row_tensors = []
        for _ in range(n_augs):
            x = train_tf(pil_img)                  # [3,H,W], 已经是归一化后的
            x_vis = denormalize(x, mean, std)      # 可视化还原 0~1
            row_tensors.append(x_vis)
        row_grid = make_grid(row_tensors, nrow=n_augs, padding=2)
        rows.append(row_grid)

    # 把每行再叠成一个大网格
    grid = torch.stack(rows, dim=0)  # [R,3,H,W]
    # 垂直拼接（简单做法：再次 make_grid，nrow=1）
    big_grid = make_grid(list(grid), nrow=1, padding=4)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(big_grid, str(save_path))
    print(f"[train_augs] 保存增强预览 -> {save_path}")

    if show and HAS_PLT:
        plt.figure(figsize=(n_augs * 2.5, n_images * 2.5))
        plt.axis('off'); plt.imshow(big_grid.permute(1, 2, 0).cpu())
        plt.tight_layout(); plt.show()
    elif show and not HAS_PLT:
        print("[WARN] 未检测到 matplotlib，已保存文件但无法弹窗显示。")


def visualize_eval_preview(
    dataset_root: Path,
    split: str,
    save_path: Path,
    img_size: int,
    mean,
    std,
    n_images: int = 12,
    seed: int = 42,
    show: bool = False
):
    """
    针对验证/测试预处理：展示 eval_tf 后的图像（单视图），拼成网格并保存
    """
    set_seed(seed)
    ds = ImageFolder(str(dataset_root / split), transform=None)
    indices = random.sample(range(len(ds)), k=min(n_images, len(ds)))
    _, eval_tf = build_transforms(img_size=img_size, use_trivial_augment=False, mean=mean, std=std)

    imgs = []
    for idx in indices:
        pil_img, _ = ds[idx]
        x = eval_tf(pil_img)
        x_vis = denormalize(x, mean, std)
        imgs.append(x_vis)

    grid = make_grid(imgs, nrow=6, padding=2)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, str(save_path))
    print(f"[eval_preview] 保存验证/测试预处理预览 -> {save_path}")

    if show and HAS_PLT:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.axis('off'); plt.imshow(grid.permute(1, 2, 0).cpu())
        plt.tight_layout(); plt.show()
    elif show and not HAS_PLT:
        print("[WARN] 未检测到 matplotlib，已保存文件但无法弹窗显示。")


def main():
    parser = argparse.ArgumentParser(description="Preview flower data augmentations")
    parser.add_argument("--data", type=str, required=True,
                        help="数据根目录（包含 flowers_train_images/val/test 子目录的 ImageFolder 结构）")
    parser.add_argument("--split-flowers_train_images", type=str, default="flowers_train_images", help="训练集子目录名")
    parser.add_argument("--split-eval", type=str, default="val", help="验证/测试子目录名（用于 eval 预览）")
    parser.add_argument("--img-size", type=int, default=288)
    parser.add_argument("--mean", type=str, default="0.485,0.456,0.406",
                        help="均值，逗号分隔，如 0.485,0.456,0.406（若你有数据集统计，替换为统计值）")
    parser.add_argument("--std", type=str, default="0.229,0.224,0.225",
                        help="方差，逗号分隔，如 0.229,0.224,0.225")
    parser.add_argument("--n-images", type=int, default=8, help="训练增强中展示多少行（张原图）")
    parser.add_argument("--n-augs", type=int, default=6, help="每张原图重复增强多少次（列数）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-trivial", action="store_true", help="不要使用 TrivialAugmentWide（回退到 ColorJitter）")
    parser.add_argument("--out-dir", type=str, default="./aug_preview")
    parser.add_argument("--show", action="store_true", help="同时弹窗显示（需要 matplotlib）")
    args = parser.parse_args()

    mean = parse_mean_std(args.mean)
    std = parse_mean_std(args.std)
    root = Path(args.data)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 训练增强预览
    visualize_train_augs(
        dataset_root=root,
        split=args.split_train,
        save_path=out_dir / f"train_augs_grid_{args.img_size}.jpg",
        img_size=args.img_size,
        mean=mean,
        std=std,
        n_images=args.n_images,
        n_augs=args.n_augs,
        seed=args.seed,
        use_trivial_augment=not args.no_trivial,
        show=args.show
    )

    # 验证/测试预处理预览
    visualize_eval_preview(
        dataset_root=root,
        split=args.split_eval,
        save_path=out_dir / f"eval_preview_{args.img_size}.jpg",
        img_size=args.img_size,
        mean=mean,
        std=std,
        n_images=12,
        seed=args.seed,
        show=args.show
    )


if __name__ == "__main__":
    main()
