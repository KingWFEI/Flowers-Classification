import csv, random
from pathlib import Path
from typing import Dict, List, Tuple
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# 读取csv中分类标签
class FlowerCsvDataset(Dataset):
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

    # 更改为当前数据集的mean 和 std
    mean, std = (0.45134347677230835, 0.4673071503639221, 0.3222246766090393), (0.24617701768875122, 0.22343231737613678, 0.2512664794921875)
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


# SnapMix --------------------------------------------------------------


@torch.no_grad()
def _compute_cam_from_fmap_and_fc(fmap: torch.Tensor, fc_weights: torch.Tensor, labels: torch.Tensor):
    """
    fmap: [B, C, H, W]  —— 来自最后卷积层的特征图（forward hook 得到）
    fc_weights: [num_classes, C] —— 分类层 (nn.Linear) 的权重
    labels: [B] —— 每张图的标签
    返回:
        cams: [B, 1, H, W] —— 归一化后的 CAM
    """
    # B, C, H, W = fmap.shape
    # # 取每张图对应类别的权重 (B, C)
    # W_sel = fc_weights[labels]                    # [B, C]
    # # 按通道加权求和 -> (B, H, W)
    # # (B,C,H,W) 与 (B,C) 做逐样本通道加权
    # cams = torch.einsum('bchw,bc->bhw', fmap, W_sel)  # [B, H, W]
    # # ReLU & normalize 到 [0,1]
    # cams = torch.relu(cams)
    # cams = cams.view(B, -1)
    # cams -= cams.min(dim=1, keepdim=True)[0]
    # cams_den = cams.max(dim=1, keepdim=True)[0] + 1e-6
    # cams = (cams / cams_den).view(B, 1, H, W)
    B, C, H, W = fmap.shape
    W_sel = fc_weights[labels]                            # [B, C]
    cams = torch.einsum('bchw,bc->bhw', fmap, W_sel)     # [B, h, w]
    cams = torch.relu(cams)

    # 防止全零：加极小正数，然后按每个样本的最大值归一
    cams = cams + 1e-12
    maxv = cams.amax(dim=(1,2), keepdim=True)            # [B,1,1]
    cams = cams / maxv
    cams = cams.clamp_(0.0, 1.0).unsqueeze(1)            # [B,1,h,w]
    return cams

def _rand_bbox_from_cam(cam: torch.Tensor, lam: float):
    """
    cam: [1,H,W] 或 [H,W]  的 CAM (0~1)
    lam: Beta 采样出的面积系数
    返回: (x1,y1,x2,y2)
    """
    if cam.dim() == 3:
        cam = cam.squeeze(0)
    H, W = cam.shape[-2], cam.shape[-1]

    # 计算窗口大小
    cut_rat = math.sqrt(1. - float(lam))
    cut_w = max(1, int(W * cut_rat))
    cut_h = max(1, int(H * cut_rat))

    # 以 CAM 作为概率分布采样中心；若分布非法则回退到均匀分布
    cam_flat = cam.reshape(-1).float()
    cam_flat = torch.nan_to_num(cam_flat, nan=0.0, posinf=0.0, neginf=0.0)
    cam_flat.clamp_(min=0.0)

    s = cam_flat.sum()
    if (not torch.isfinite(s)) or s <= 1e-6:
        # 兜底：均匀随机一个像素
        idx = torch.randint(H * W, (1,), device=cam.device).item()
    else:
        prob = cam_flat / s
        idx = torch.multinomial(prob, 1).item()

    cy, cx = divmod(idx, W)
    x1 = max(cx - cut_w // 2, 0);  x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0);  y2 = min(cy + cut_h // 2, H)
    if x2 <= x1: x2 = min(x1 + 1, W)
    if y2 <= y1: y2 = min(y1 + 1, H)
    return int(x1), int(y1), int(x2), int(y2)

def snapmix(images: torch.Tensor,
            labels: torch.Tensor,
            model: nn.Module,
            cls_head: nn.Linear,
            alpha: float = 5.0,
            prob: float = 0.5):
    """
    SnapMix 主体：
      - 以 prob 概率启用
      - 生成基于 CAM 的贴合区域
      - 返回混合后的 images，以及“混合标签包” (y1, y2, lam1, lam2)
    说明：
      - cls_head: 模型的分类头 (nn.Linear)，用于读取权重计算 CAM
      - 混合标签使用两个 lambda（几何面积与 CAM 积分修正），更贴近论文
    """
    if random.random() > prob:
        return images, labels, None  # 不启用，返回原数据

    device = images.device
    B, C, H, W = images.shape

    # 先前向一次，触发 hook 得到 fmap；这里不需要梯度
    _ = model(images)
    assert hasattr(model, "_snapmix_fmap"), "未捕获到特征图，请先 register_last_conv_hook(model)"
    fmap = model._snapmix_fmap          # [B, C, h, w]
    cams = _compute_cam_from_fmap_and_fc(fmap, cls_head.weight, labels)  # [B,1,h,w]

    # 随机打乱索引，做配对
    index = torch.randperm(B, device=device)

    mixed = images.clone()
    y1 = labels
    y2 = labels[index]

    # 逐样本构造 CAM 引导的贴合框
    lam_list = []
    lam_cam_list = []
    for i in range(B):
        # 从 Beta 采 lam（面积比）
        lam = np.random.beta(alpha, alpha)
        # 在目标图 i 的 CAM 上采框（让目标保住关键区域）
        x1, y1b, x2, y2b = _rand_bbox_from_cam(cams[i], lam)
        # 把配对图 index[i] 在同一区域贴到 i 上
        mixed[i, :, y1b:y2b, x1:x2] = images[index[i], :, y1b:y2b, x1:x2]

        # 面积 lambda（几何）
        lam_geom = 1 - ((x2 - x1) * (y2b - y1b) / (H * W))
        lam_list.append(lam_geom)

        # CAM 加权的 lambda（对两张图分别按 CAM 积分修正）
        # 目标图 i 在保留区域上的 CAM 积分
        cam_i = F.interpolate(cams[i].unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)[0]
        cam_j = F.interpolate(cams[index[i]].unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)[0]
        total_i = cam_i.sum()
        total_j = cam_j.sum()
        keep_i = (cam_i.sum() - cam_i[y1b:y2b, x1:x2].sum()) / (total_i + 1e-6)
        paste_j = cam_j[y1b:y2b, x1:x2].sum() / (total_j + 1e-6)
        lam_cam_list.append((keep_i.item(), paste_j.item()))

    # 打包标签混合系数（两份 lambda）
    lam_tensor = torch.tensor(lam_list, dtype=torch.float32, device=device)              # 几何
    lam_keep = torch.tensor([p[0] for p in lam_cam_list], dtype=torch.float32, device=device)  # CAM 对 i
    lam_paste= torch.tensor([p[1] for p in lam_cam_list], dtype=torch.float32, device=device)  # CAM 对 j

    # 最终两侧权重（论文思路：几何 * CAM 修正；你也可以只用 CAM 比例）
    lam_i = torch.clamp(lam_tensor * lam_keep, 0.0, 1.0)
    lam_j = torch.clamp(1.0 - lam_i, 0.0, 1.0)

    # 返回“标签包”，供自定义 criterion 使用
    target_pack = (labels, labels[index], lam_i, lam_j)
    return mixed, target_pack, "snapmix"


# 保存最后一层特征图谱 --------------------------------------------------------------
def register_last_conv_hook(model: nn.Module):
    """
    在 model 上注册 forward hook，抓取“最后一个卷积层”的输出特征图。
    保存到 model._snapmix_fmap（形状: [B, C, H, W]）
    """
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    assert last_conv is not None, "未找到卷积层，无法用于 SnapMix"

    def _hook(module, inp, out):
        # out: [B, C, H, W]
        model._snapmix_fmap = out

    handle = last_conv.register_forward_hook(_hook)
    model._snapmix_hook = handle
    return handle
