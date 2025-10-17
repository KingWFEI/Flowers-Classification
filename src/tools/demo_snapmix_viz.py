# demo_snapmix_viz.py（或在 train.py 里加一段）
import torch
from pathlib import Path
from utils import build_dataloaders_from_csv, register_last_conv_hook
from model import FADCResNet
from torchvision.models import resnet50, ResNet50_Weights
from tools.vis_snapmix import snapmix_preview_once, save_snapmix_viz

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) 构造数据（与你训练一致）
train_loader, _, _, num_classes, class_names = build_dataloaders_from_csv(
    data_root="data/flowers_train_images",
    csv_path="train_labels.csv",
    img_subdir="train_images",
    img_size=288,
    batch_size=8,   # 小一点方便快速可视化
    workers=0,
    val_ratio=0.1,
)

# 2) 构造模型（与你训练一致），并注册 fmap hook
use_fadc = True   # ← 如果训练用的 FADC 就 True，否则 False
if use_fadc:
    from model import FADCResNet
    model = FADCResNet(num_classes=num_classes)
    # 可选：迁移部分 torchvision resnet50 权重（跟你训练初始化一致）
    tv = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    tv_state = tv.state_dict(); msd = model.state_dict()
    for k,v in tv_state.items():
        if k in msd and msd[k].shape == v.shape:
            msd[k].copy_(v)
    model.load_state_dict(msd)
else:
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    in_f = model.fc.in_features
    model.fc = torch.nn.Linear(in_f, num_classes)

register_last_conv_hook(model)      # 必须
cls_head = model.fc                 # 你的分类头
model = model.to(device).eval()

# 3) 抽一个 batch
images, labels = next(iter(train_loader))
images, labels = images.to(device), labels.to(device)

# 4) 生成可视化
mixed, viz_pack = snapmix_preview_once(
    model, cls_head, images, labels,
    mean=(0.45134348,0.46730715,0.32222468),      # ← 用你算的 mean/std
    std=(0.24617702,0.22343232,0.25126648),
    alpha=5.0, seed=123
)

# 5) 保存图 & 指标
save_snapmix_viz(
    images=images, mixed=mixed, viz_pack=viz_pack,
    out_dir="outputs/snapmix_viz",
    mean=(0.45134348,0.46730715,0.32222468),
    std=(0.24617702,0.22343232,0.25126648),
    class_names=class_names
)
