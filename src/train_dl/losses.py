from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedFocalLoss(nn.Module):
    """FL = −α_t · (1 − p_t)^γ · log(p_t); Lin et al. 2017."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('weight', weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size = logits.size(0)
        idx = torch.arange(batch_size)
        p_t = F.softmax(logits, dim=1)[idx, targets]
        log_p_t = F.log_softmax(logits, dim=1)[idx, targets]
        focal_loss = -((1.0 - p_t) ** self.gamma) * log_p_t
        if self.weight is not None:
            focal_loss = self.weight[targets] * focal_loss
        return focal_loss.mean()
