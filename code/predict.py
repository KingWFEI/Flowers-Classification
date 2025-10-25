# -*- coding: utf-8 -*-
# predict.py
import csv, json, argparse
from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from model import build_model

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
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, str(path.name)

def build_eval_tf(img_size: int, mean: Tuple[float,float,float], std: Tuple[float,float,float]):
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    return transforms.Compose([
        transforms.Resize(int(img_size*256/224), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def parse_triplet(s: str) -> Tuple[float, float, float]:
    xs = [float(t.strip()) for t in s.split(",")]
    assert len(xs) == 3
    return tuple(xs)  # type: ignore

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser("Generate submission.csv for a test folder")
    ap.add_argument("--test-dir", type=str, required=True, help="测试图片根目录（递归）")
    ap.add_argument("--ckpt", type=str, default=None, help="模型权重（默认 ../model/best_model.pth）")
    ap.add_argument("--model-name", type=str, default="tf_efficientnetv2_s")
    ap.add_argument("--img-size", type=int, default=299)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--mean", type=str, default="0.485,0.456,0.406")
    ap.add_argument("--std",  type=str, default="0.229,0.224,0.225")
    ap.add_argument("--output", type=str, default=None, help="输出 CSV（默认 ../results/submission.csv）")
    args = ap.parse_args()

    root_dir = Path(__file__).resolve().parents[1]  # submission/
    model_dir = root_dir / "model"
    results_dir = root_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.ckpt) if args.ckpt else (model_dir / "best_model.pth")
    out_csv   = Path(args.output) if args.output else (results_dir / "submission.csv")

    # 读取标签映射
    idx2cat_path = model_dir / "idx2cat.json"
    if not idx2cat_path.is_file():
        raise FileNotFoundError(f"缺少标签映射：{idx2cat_path}")
    idx2cat = json.loads(idx2cat_path.read_text(encoding="utf-8"))
    # 索引键可能是字符串，转为 int 键
    idx2cat = {int(k): v for k, v in idx2cat.items()}
    num_classes = len(idx2cat)

    # 模型
    model = build_model(args.model_name, num_classes=num_classes, pretrained=False, drop_rate=0.0)
    state = torch.load(ckpt_path, map_location="cpu")
    sd = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    # 去掉 DataParallel 前缀
    sd = { (k[7:] if k.startswith("module.") else k): v for k,v in sd.items() }
    model.load_state_dict(sd, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # 数据
    mean = parse_triplet(args.mean); std = parse_triplet(args.std)
    tf = build_eval_tf(args.img_size, mean, std)
    ds = ImageFolderFlat(args.test_dir, transform=tf, recursive=True)
    if len(ds) == 0:
        raise RuntimeError(f"在 {args.test_dir} 下没有找到图片（支持扩展名：{sorted(IMG_EXTS)}）")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 推理
    rows: List[tuple] = []
    for imgs, names in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=(device.type=="cuda")):
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
        top1 = probs.argmax(dim=1)
        for i, name in enumerate(names):
            idx = int(top1[i].item())
            cat = idx2cat[idx]
            rows.append((name, cat))

    # 写 CSV：filename,category_id
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["filename", "category_id"])
        for name, cat in sorted(rows, key=lambda x: x[0]):
            w.writerow([name, cat])

    print(f"Saved submission to: {str(out_csv.resolve())}")

if __name__ == "__main__":
    main()
