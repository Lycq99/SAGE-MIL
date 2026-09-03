from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 1.9, num_classes: int = 2):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.num_classes = int(num_classes)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.long().reshape(-1)
        if self.num_classes <= 2:
            logits = logits.reshape(-1)
            y = target.float()
            bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            p = torch.sigmoid(logits)
            pt = torch.where(y > 0.5, p, 1.0 - p)
            alpha_t = torch.where(y > 0.5, self.alpha, 1.0 - self.alpha)
            return (alpha_t * (1.0 - pt).pow(self.gamma) * bce).mean()
        ce = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt).pow(self.gamma) * ce).mean()


def gram_orthogonality(attention: torch.Tensor) -> torch.Tensor:
    """Encourage semantic attention heads to remain distinct."""
    if attention.ndim != 3:
        raise ValueError("attention must have shape [B,H,N]")
    w = F.normalize(attention, p=2, dim=-1)
    gram = torch.bmm(w, w.transpose(1, 2))
    eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype).unsqueeze(0)
    return (gram - eye).pow(2).sum(dim=(-2, -1)).mean()


def true_class_margin(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """True-class margin used by the feature-replacement constraint.

    Binary heads use one logit. This is treated as the symmetric two-class
    representation [-z/2, z/2], which gives margin (2y-1)z and is the scalar
    form of z_y - max_{c != y} z_c.
    """
    target = target.long().reshape(-1)
    if num_classes <= 2 and (logits.ndim == 1 or logits.shape[-1] == 1):
        z = logits.reshape(-1)
        return torch.where(target > 0, z, -z)

    if logits.ndim != 2 or logits.shape[-1] != num_classes:
        raise ValueError(f"expected logits [B,{num_classes}], got {tuple(logits.shape)}")
    true = logits.gather(1, target[:, None]).squeeze(1)
    mask = F.one_hot(target, num_classes=num_classes).bool()
    other = logits.masked_fill(mask, float("-inf")).max(dim=1).values
    return true - other


def verification_hinge(m_fact: torch.Tensor, m_cf: torch.Tensor, tau: torch.Tensor, gamma_margin: float) -> torch.Tensor:
    tau = tau.reshape(-1).to(m_fact)
    drop = m_fact - m_cf
    return F.relu(tau * float(gamma_margin) - drop).mean()


def linear_warmup_weight(target_weight: float, epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return float(target_weight)
    return float(target_weight) * min(1.0, max(0.0, float(epoch + 1) / float(warmup_epochs)))
