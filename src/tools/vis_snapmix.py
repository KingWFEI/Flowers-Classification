# src/tools/vis_snapmix.py
import os, json, math, random
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# ---------- 你已有的：最后卷积层 fmap 由 register_last_conv_hook(model) 提供 ----------
# 这里直接用你传入的 model._snapmix_fmap

@torch.no_grad()
def compute_cam_from_fmap_and_fc(fmap: torch.Tensor, fc_weights: torch.Tensor, labels: torch.Tensor):
    """与你当前实现一致，输出 [B,1,h,w]，范围[0,1]"""
    B, C, H, W = fmap.shape
    W_sel = fc_weights[labels]                            # [B, C]
    cams = torch.einsum('bchw,bc->bhw', fmap, W_sel)     # [B, h, w]
    cams = torch.relu(cams) + 1e-12
    cams = cams / cams.amax(dim=(1,2), keepdim=True)
    return cams.unsqueeze(1).clamp_(0.0, 1.0)


def _to_numpy_image(t: torch.Tensor, mean, std):
    """denorm + to HWC uint8 numpy"""
    if t.ndim == 4: t = t[0]
    x = t.detach().cpu().float()
    m = torch.tensor(mean).view(3,1,1)
    s = torch.tensor(std).view(3,1,1)
    x = (x * s + m).clamp(0,1)
    x = (x.permute(1,2,0).numpy() * 255.0).round().astype(np.uint8)
    return x


def _heatmap_overlay(img_np: np.ndarray, cam_01: np.ndarray, alpha: float = 0.45):
    """
    img_np: HxWx3 uint8
    cam_01: HxW float in [0,1]
    return: overlaid uint8
    """
    cmap = plt.get_cmap('jet')
    cm = cmap(cam_01)[:, :, :3]  # HxWx3 float
    cm = (cm * 255).astype(np.uint8)
    out = (img_np.astype(np.float32) * (1 - alpha) + cm.astype(np.float32) * alpha).clip(0,255).astype(np.uint8)
    return out


def _draw_rect(img_np: np.ndarray, box: Tuple[int,int,int,int], color=(255, 0, 0), width=3):
    """在 numpy 图上画框（用 PIL）"""
    im = Image.fromarray(img_np)
    draw = ImageDraw.Draw(im)
    x1, y1, x2, y2 = box
    for w in range(width):
        draw.rectangle([x1-w, y1-w, x2+w, y2+w], outline=color, width=1)
    return np.asarray(im)


@torch.no_grad()
def snapmix_preview_once(
    model: nn.Module,
    cls_head: nn.Linear,
    images: torch.Tensor,            # [B,3,H,W] 归一化后的张量（与你训练时一致）
    labels: torch.Tensor,            # [B]
    mean=(0.485,0.456,0.406),
    std=(0.229,0.224,0.225),
    alpha: float = 5.0,
    seed: Optional[int] = None,
):
    """
    复现一轮 SnapMix，返回可视化与指标（不改变你的训练函数）。
    输出：
      mixed: [B,3,H,W]
      viz_pack: list[dict]，每个样本含：
        - idx, j_idx（配对索引）
        - box: (x1,y1,x2,y2)
        - lam_geom, lam_keep, lam_paste, lam_i, lam_j
        - cam_i/cam_j: [h,w] 的 CAM（0~1）
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    device = images.device
    B, _, H, W = images.shape

    # 前向一次得到 fmap
    _ = model(images)
    assert hasattr(model, "_snapmix_fmap"), "未捕获到特征图，请先 register_last_conv_hook(model)"
    fmap = model._snapmix_fmap                                # [B,C,h,w]
    cams = compute_cam_from_fmap_and_fc(fmap, cls_head.weight, labels).squeeze(1)  # [B,h,w]

    # 配对索引
    index = torch.randperm(B, device=device)

    mixed = images.clone()
    viz_pack = []

    # 逐样本
    for i in range(B):
        # 采样 lam 决定区域面积
        lam = np.random.beta(alpha, alpha)
        # 在目标 i 的 CAM 上采一个中心框（与训练一致）
        cam_i_small = cams[i]  # [h,w]
        h, w = cam_i_small.shape
        cut_rat = math.sqrt(1. - float(lam))
        cut_w = max(1, int(w * cut_rat))
        cut_h = max(1, int(h * cut_rat))

        # 用 CAM 概率采样中心
        cam_flat = cam_i_small.reshape(-1).float()
        cam_flat = torch.nan_to_num(cam_flat, nan=0.0, posinf=0.0, neginf=0.0).clamp_(min=0.0)
        s = cam_flat.sum()
        if (not torch.isfinite(s)) or s <= 1e-6:
            idx_flat = torch.randint(h*w, (1,), device=device).item()
        else:
            prob = cam_flat / s
            idx_flat = torch.multinomial(prob, 1).item()
        cy, cx = divmod(idx_flat, w)
        x1_s = max(cx - cut_w // 2, 0);  x2_s = min(cx + cut_w // 2, w)
        y1_s = max(cy - cut_h // 2, 0);  y2_s = min(cy + cut_h // 2, h)

        # 映射到原图分辨率
        # 上采样 CAM 到 (H,W)，便于计算加权比与画框
        cam_i = F.interpolate(cams[i].unsqueeze(0).unsqueeze(0), size=(H,W), mode='bilinear', align_corners=False).squeeze().cpu().numpy()
        cam_j = F.interpolate(cams[index[i]].unsqueeze(0).unsqueeze(0), size=(H,W), mode='bilinear', align_corners=False).squeeze().cpu().numpy()

        # 对应到原图坐标（按比例）
        x1 = int(round(x1_s / w * W)); x2 = int(round(x2_s / w * W))
        y1 = int(round(y1_s / h * H)); y2 = int(round(y2_s / h * H))
        if x2 <= x1: x2 = min(x1 + 1, W)
        if y2 <= y1: y2 = min(y1 + 1, H)

        # 贴图（用原图尺度）
        mixed[i, :, y1:y2, x1:x2] = images[index[i], :, y1:y2, x1:x2]

        # λ（几何 + CAM 修正）
        lam_geom = 1 - ((x2-x1) * (y2-y1) / (H*W))
        total_i = cam_i.sum() + 1e-6
        total_j = cam_j.sum() + 1e-6
        keep_i  = (cam_i.sum() - cam_i[y1:y2, x1:x2].sum()) / total_i
        paste_j = (cam_j[y1:y2, x1:x2].sum()) / total_j
        lam_i   = float(np.clip(lam_geom * keep_i, 0.0, 1.0))
        lam_j   = float(np.clip(1.0 - lam_i, 0.0, 1.0))

        viz_pack.append({
            "idx": int(i),
            "j_idx": int(index[i].item()),
            "box": (int(x1), int(y1), int(x2), int(y2)),
            "lam_geom": float(lam_geom),
            "lam_keep": float(keep_i),
            "lam_paste": float(paste_j),
            "lam_i": lam_i,
            "lam_j": lam_j,
            "cam_i": cam_i,  # HxW numpy
            "cam_j": cam_j,  # HxW numpy
        })

    return mixed, viz_pack


def save_snapmix_viz(
    images: torch.Tensor,               # 原图（归一化张量）
    mixed: torch.Tensor,                # SnapMix 后图（归一化张量）
    viz_pack: List[dict],
    out_dir: str,
    mean=(0.485,0.456,0.406),
    std=(0.229,0.224,0.225),
    class_names: Optional[List[str]] = None,
):
    """
    为每个样本输出一张 4×1 拼图：
      [ 原图A | CAM(A)+框 | 原图B(源) | 混合后 ]
    并保存每样本 JSON 指标与一个 batch 摘要。
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    B = images.size(0)

    # 统计
    lam_i_all = []
    box_area_all = []

    for rec in viz_pack:
        i   = rec["idx"]
        j   = rec["j_idx"]
        box = rec["box"]
        x1,y1,x2,y2 = box
        lam_i_all.append(rec["lam_i"])
        box_area_all.append((x2-x1)*(y2-y1))

        # 准备四张图
        A = _to_numpy_image(images[i], mean, std)
        Bimg = _to_numpy_image(images[j], mean, std)
        Mix = _to_numpy_image(mixed[i], mean, std)

        # 热力图叠加
        H, W, _ = A.shape
        camA = rec["cam_i"]
        if camA.shape != (H,W):
            camA = np.array(Image.fromarray((camA*255).astype(np.uint8)).resize((W,H), Image.BILINEAR)) / 255.0
        vis_cam = _heatmap_overlay(A, camA, alpha=0.45)
        vis_cam = _draw_rect(vis_cam, box, color=(255, 255, 0), width=3)

        # 原图B上画出“被贴走”的区域（可选）
        vis_B = _draw_rect(Bimg, box, color=(0, 255, 0), width=3)

        # 拼成一张宽图
        canvas = np.concatenate([A, vis_cam, vis_B, Mix], axis=1)
        Image.fromarray(canvas).save(os.path.join(out_dir, f"sample_{i:02d}_pair_{j:02d}.png"))

        # 保存 JSON 指标
        with open(os.path.join(out_dir, f"sample_{i:02d}_pair_{j:02d}.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)

    # 批量小结
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        if len(lam_i_all):
            f.write(f"Samples: {len(lam_i_all)}\n")
            f.write(f"lam_i  mean={np.mean(lam_i_all):.4f}  std={np.std(lam_i_all):.4f}  "
                    f"min={np.min(lam_i_all):.4f}  max={np.max(lam_i_all):.4f}\n")
            rel_area = np.array(box_area_all) / float(images.size(-1)*images.size(-2))
            f.write(f"box area ratio mean={rel_area.mean():.4f}  std={rel_area.std():.4f}\n")

    print(f"[viz] 保存到：{out_dir}")
