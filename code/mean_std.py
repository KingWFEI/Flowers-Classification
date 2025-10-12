"""
此代码用于计算图像数据集的均值和标准差，用于图像预处理中的归一化步骤。
"""
import torch, torchvision as tv
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np

root = "../data/flowers/flowers/train"  # 只用训练集
img_size = 288
tf = transforms.Compose([
    transforms.Resize(int(img_size*256/224)),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),  # 注意：统计时不要先 Normalize
])

ds = ImageFolder(root, transform=tf)
loader = DataLoader(ds, batch_size=64, num_workers=0, shuffle=False)

n = 0
mean = torch.zeros(3)
M2 = torch.zeros(3)  # online variance helper
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
