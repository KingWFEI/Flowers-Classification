import random
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

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
