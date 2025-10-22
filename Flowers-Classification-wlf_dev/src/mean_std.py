import torch

from utils import _read_csv, _build_label_mapping, FlowerCsvDataset
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms

img_size = 288
tf = transforms.Compose([
    transforms.Resize(int(img_size*256/224)),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),    # 不要 Normalize、不要颜色增强
])

root = Path("../data/flowers_train_images")
rows = _read_csv(root / "train_labels.csv")
_ , _ = _build_label_mapping(rows)  # 给每行写入 category_idx
ds = FlowerCsvDataset(rows, img_dir=root / "train_images", transform=tf)
loader = DataLoader(ds, batch_size=64, num_workers=0, shuffle=False)

n = 0
mean = torch.zeros(3)
M2 = torch.zeros(3)
for imgs, _ in loader:
    b = imgs.size(0)
    n_new = n + b
    batch_mean = imgs.mean(dim=[0,2,3])
    mean_delta = batch_mean - mean
    mean += mean_delta * (b / n_new)
    batch_var = imgs.var(dim=[0,2,3], unbiased=False)
    M2 += batch_var * b + (mean_delta**2) * (n * b / n_new)
    n = n_new

var = M2 / n
std = torch.sqrt(var)
print("mean:", mean.tolist(), "std:", std.tolist())
