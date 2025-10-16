# tools/build_datasets_csv.py
import csv, random
from pathlib import Path
from typing import Dict, List, Tuple


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True


class FlowerCsvDataset(Dataset):
    """
    必要字段：filename, category（类别原始值，字符串或非连续数字都可）
    可选字段：chinese / english_name（仅用于生成可读类名）
    图片文件：img_dir / filename（若 CSV 不带扩展名，会自动尝试常见后缀）
    """
    def __init__(self, rows: List[Dict], img_dir: Path, transform=None):
        self.rows = rows
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        path = self.img_dir / r["filename"]
        if not path.is_file():  # 兼容 CSV 不含扩展名
            for ext in [".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff",".JPG",".PNG"]:
                p2 = self.img_dir / (r["filename"] + ext)
                if p2.is_file():
                    path = p2; break
        img = Image.open(path).convert("RGB")
        target = int(r["category_idx"])
        if self.transform: img = self.transform(img)
        return img, target


def _read_csv(csv_path: Path) -> List[Dict]:
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("filename") or row.get("image") or row.get("file") or row.get("name")
            category = row.get("category_id") or row.get("label") or row.get("class_id")
            chinese  = row.get("chinese") or row.get("category_chinese") or ""
            english  = row.get("english_name") or row.get("category_english_name") or ""
            if filename is None or category is None:
                raise ValueError(f"CSV 缺少 filename 或 category 字段：{row}")
            rows.append({
                "filename": str(filename).strip(),
                "category_raw": str(category).strip(),
                "chinese": (chinese or "").strip(),
                "english_name": (english or "").strip(),
            })
    if not rows:
        raise RuntimeError(f"CSV 为空：{csv_path}")
    return rows


def _build_label_mapping(rows: List[Dict]) -> Tuple[Dict[str,int], List[str]]:
    uniq = sorted({r["category_raw"] for r in rows}, key=lambda x: str(x))
    cat2idx = {c:i for i,c in enumerate(uniq)}
    # 生成可读类名：优先 中文+英文，否则用原 category
    class_names = []
    for c in uniq:
        sample = next(r for r in rows if r["category_raw"] == c)
        zh, en = sample.get("chinese",""), sample.get("english_name","")
        name = f"{zh} {en}".strip() if (zh or en) else str(c)
        class_names.append(name)
    # 写回 idx
    for r in rows: r["category_idx"] = cat2idx[r["category_raw"]]
    return cat2idx, class_names


def _stratified_split(rows: List[Dict], val_ratio: float, seed: int):
    rng = random.Random(seed)
    by_cls: Dict[int, List[Dict]] = {}
    for r in rows:
        by_cls.setdefault(r["category_idx"], []).append(r)
    train_rows, val_rows = [], []
    for cls, items in by_cls.items():
        rng.shuffle(items)
        n = len(items)
        k = max(1, int(round(n * val_ratio)))  # 每类至少1张进 val
        val_rows.extend(items[:k]); train_rows.extend(items[k:])
    return train_rows, val_rows


def build_dataloaders_from_csv(
    data_root: str = "data/flowers_train_images",
    csv_path: str = "train_labels.csv",
    img_subdir: str = "train_images",
    img_size: int = 288,
    batch_size: int = 32,
    workers: int = 0,
    val_ratio: float = 0.1,
    seed: int = 42,
    save_class_names_json: bool = True,
):
    """
    目录示例：
      data/flowers_train_images/
        ├─ train_labels.csv
        └─ train_images/   （全量训练图像）
    从训练集中分层抽样 val_ratio 比例作为验证集。
    """
    root = Path(data_root)
    csv_file = root / csv_path
    img_dir  = root / img_subdir
    assert csv_file.is_file(), f"未找到 CSV：{csv_file}"
    assert img_dir.is_dir(), f"未找到图像目录：{img_dir}"

    rows = _read_csv(csv_file)
    _, class_names = _build_label_mapping(rows)
    num_classes = len(class_names)

    train_rows, val_rows = _stratified_split(rows, val_ratio=val_ratio, seed=seed)

    mean, std = (0.485,0.456,0.406), (0.229,0.224,0.225)
    train_tf, eval_tf = build_transforms(img_size=img_size, use_trivial_augment=True, mean=mean, std=std)

    ds_train = FlowerCsvDataset(train_rows, img_dir=img_dir, transform=train_tf)
    ds_val   = FlowerCsvDataset(val_rows,   img_dir=img_dir, transform=eval_tf)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True, persistent_workers=(workers>0))
    val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=True, persistent_workers=False)

    if save_class_names_json:
        (root / "class_names.json").write_text(
            __import__("json").dumps(class_names, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    return train_loader, val_loader, None, num_classes, class_names


def build_transforms(
    img_size: int = 288,
    use_trivial_augment: bool = True,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    # 下面是可调强度的超参（按需修改）
    rrc_scale=(0.7, 1.0),          # RandomResizedCrop 的缩放范围
    hflip_p: float = 0.5,          # 水平翻转概率
    jitter_strength: float = 0.2,  # 颜色扰动强度（brightness/contrast/saturation）
    hue_max: float = 0.05,         # 花卉对色相敏感，建议 ≤ 0.05
    re_prob: float = 0.15,         # RandomErasing 概率
    re_scale=(0.02, 0.2),          # RandomErasing 遮挡面积比例
    re_ratio=(0.3, 3.3)            # RandomErasing 宽高比范围
):
    """
    返回:
        train_tf: 训练增强（RandomResizedCrop + HFlip + TrivialAugmentWide/ColorJitter + Normalize + RandomErasing）
        eval_tf : 验证/测试预处理（Resize + CenterCrop + Normalize）
    说明:
        - 对花卉类别，色相(hue)扰动不要太强；几何变换保持适中以保留形态细节。
        - MixUp / CutMix 如需启用，建议在训练循环里实现（不放在 transforms 里）。
    """
    # 训练增强
    train_list = [
        transforms.RandomResizedCrop(img_size, scale=rrc_scale, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=hflip_p),
    ]

    # 自动增强（若 torchvision 有 TrivialAugmentWide 就用；否则退回到轻度 ColorJitter）
    use_taw = use_trivial_augment and hasattr(transforms, "TrivialAugmentWide")
    if use_taw:
        train_list.append(transforms.TrivialAugmentWide(interpolation=InterpolationMode.BICUBIC))
    else:
        # 轻度颜色扰动，避免破坏花色
        train_list.append(transforms.ColorJitter(jitter_strength, jitter_strength, jitter_strength, hue_max))

    train_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=re_prob, scale=re_scale, ratio=re_ratio, value='random'),
    ]
    train_tf = transforms.Compose(train_list)

    # 验证/测试预处理（仅几何对齐 + 标准化）
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, eval_tf
