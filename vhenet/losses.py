"""8_14 损失函数。

核心：per-virus softmax 排序损失。
同一病毒内的宿主 logits 做 softmax，正宿主目标按 importance score 加权，
迫使「该病毒的正宿主排在该病毒的负宿主前面」。
纯宿主偏置无法同时满足不同病毒相互冲突的宿主偏好 -> 必须用病毒序列信息。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def per_virus_softmax_loss(logits: torch.Tensor, labels: torch.Tensor,
                           weights: torch.Tensor, pos_clip: float = 20.0):
    """logits/labels/weights: [K]（同一病毒 K 个宿主行）。

    软目标 = 正样本 importance score 归一化；负样本目标 0。
    返回 -Σ t log softmax(z)。
    """
    pos_mask = labels > 0.5
    if pos_mask.sum() == 0:
        return torch.zeros((), device=logits.device)
    t = torch.zeros_like(logits)
    w_pos = weights[pos_mask].clamp(min=1.0, max=pos_clip)
    t[pos_mask] = w_pos / w_pos.sum()
    logp = F.log_softmax(logits, dim=0)
    return -(t * logp).sum()


def weighted_bce(logits: torch.Tensor, labels: torch.Tensor,
                 weights: torch.Tensor, clip: float = 5.0):
    """带样本权重的 BCE（用于绝对校准的辅助损失）。"""
    bce = F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none").squeeze(-1)
    w = weights.clamp(min=0.0, max=clip)
    return (bce * w).sum() / w.sum().clamp_min(1e-6)
