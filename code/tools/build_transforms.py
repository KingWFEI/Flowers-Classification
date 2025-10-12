from torchvision import transforms
from torchvision.transforms import InterpolationMode

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
