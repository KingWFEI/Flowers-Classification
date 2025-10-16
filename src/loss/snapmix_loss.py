import torch


def ce_with_snapmix(logits: torch.Tensor, target_pack, base_ce):
    """
    target_pack: (y1, y2, lam_i, lam_j) 或 None
    base_ce: nn.CrossEntropyLoss(label_smoothing=...)
    """
    y1, y2, lam_i, lam_j = target_pack
    loss1 = base_ce(logits, y1)
    loss2 = base_ce(logits, y2)
    # 注意 lam_i/lam_j 逐样本；把它们扩展到 batch
    # CrossEntropyLoss 已做了 batch 归一，这里做逐样本权重需要使用 reduction='none'
    if getattr(base_ce, 'reduction', 'mean') != 'none':
        # 重新实例化一个 none 版
        base_ce_none = torch.nn.CrossEntropyLoss(label_smoothing=getattr(base_ce, 'label_smoothing', 0.0), reduction='none')
        loss1 = base_ce_none(logits, y1)
        loss2 = base_ce_none(logits, y2)
    loss = (lam_i * loss1 + lam_j * loss2).mean()
    return loss
