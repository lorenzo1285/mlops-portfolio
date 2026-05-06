"""PyTorch loss functions for DL training."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedFocalLoss(nn.Module):
    """Balanced Focal Loss for multi-class classification.
    
    Combines class weighting (α_t) with focal modulation ((1-p_t)^γ) to address
    class imbalance and focus on hard examples.
    
    Loss formula: FL = −α_t · (1 − p_t)^γ · log(p_t)
    
    Args:
        gamma: Focusing parameter. Higher values down-weight easy examples more.
               Default: 2.0 (standard focal loss)
        weight: Per-class weights (α_t). Shape: (n_classes,) or None.
                If None, uses equal weights.
    
    References:
        Lin et al. (2017). Focal Loss for Dense Object Detection.
        https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('weight', weight)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute balanced focal loss.
        
        Args:
            logits: Raw model outputs (before softmax). Shape: (batch_size, n_classes)
            targets: True class indices. Shape: (batch_size,)
        
        Returns:
            Scalar loss tensor (mean over batch).
        """
        # Compute softmax probabilities
        probs = F.softmax(logits, dim=1)  # (batch_size, n_classes)
        
        # Gather probabilities for true classes (p_t)
        batch_size = logits.size(0)
        p_t = probs[torch.arange(batch_size), targets]  # (batch_size,)
        
        # Compute focal modulation: (1 - p_t)^γ
        focal_modulation = (1.0 - p_t) ** self.gamma
        
        # Compute cross-entropy: -log(p_t)
        ce_loss = -torch.log(p_t + 1e-8)  # Add epsilon for numerical stability
        
        # Apply focal modulation
        focal_loss = focal_modulation * ce_loss
        
        # Apply class weights (α_t) if provided
        if self.weight is not None:
            alpha_t = self.weight[targets]  # (batch_size,)
            focal_loss = alpha_t * focal_loss
        
        # Return mean loss
        return focal_loss.mean()
