"""Imbalance-Aware Loss Functions for Industrial Surface Defect Segmentation.

Implements Dice Loss, Combo Loss (BCEWithLogits + Dice Loss), and Focal-Tversky Loss
designed specifically for micro-defect segmentation with severe class imbalance (<5% defect pixels).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice Loss operating directly on unnormalized raw logits."""

    def __init__(self, smooth: float = 1.0, eps: float = 1e-7) -> None:
        """Initialize DiceLoss.

        Args:
            smooth: Smoothing factor added to numerator and denominator to avoid division by zero.
            eps: Epsilon term for numerical stability.
        """
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate Dice Loss.

        Args:
            logits: Predicted logits tensor of shape (B, C, H, W) or (B, 1, H, W).
            targets: Binary ground-truth mask tensor of shape matching logits (0 or 1).

        Returns:
            torch.Tensor: Scalar Dice loss value.
        """
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1).float()

        intersection = (probs_flat * targets_flat).sum()
        cardinality = probs_flat.sum() + targets_flat.sum()

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth + self.eps)
        return 1.0 - dice_score


class ComboLoss(nn.Module):
    """Combination of Binary Cross Entropy with Logits and Soft Dice Loss.

    Formula:
        Loss_total = alpha * BCEWithLogitsLoss + beta * DiceLoss
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        pos_weight: Optional[float] = None,
        smooth: float = 1.0,
    ) -> None:
        """Initialize ComboLoss.

        Args:
            alpha: Weight multiplier for BCEWithLogits loss.
            beta: Weight multiplier for Soft Dice loss.
            pos_weight: Optional positive class weighting for BCE.
            smooth: Smoothing factor for Dice loss.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        pw_tensor = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted combo loss.

        Args:
            logits: Raw logits tensor of shape (B, 1, H, W).
            targets: Binary ground truth tensor of shape (B, 1, H, W).

        Returns:
            torch.Tensor: Weighted scalar combo loss.
        """
        targets = targets.float()
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return (self.alpha * bce) + (self.beta * dice)


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss for asymmetric penalty on False Negatives in micro-defects.

    The Tversky index TI is defined as:
        TI = (TP + smooth) / (TP + alpha * FN + beta * FP + smooth)
    Focal Tversky loss enhances focus on hard micro-defect samples:
        FTL = (1 - TI) ** (1 / gamma)
    """

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 1.33,
        smooth: float = 1.0,
        eps: float = 1e-7,
    ) -> None:
        """Initialize FocalTverskyLoss.

        Args:
            alpha: Weight allocated to False Negatives (higher alpha penalizes missed defects).
            beta: Weight allocated to False Positives (alpha + beta typically = 1.0).
            gamma: Focal exponent (gamma > 1 focuses on hard examples).
            smooth: Smoothing constant.
            eps: Epsilon for stability.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Focal Tversky Loss.

        Args:
            logits: Raw logits tensor (B, 1, H, W).
            targets: Binary ground truth (B, 1, H, W).

        Returns:
            torch.Tensor: Scalar Focal Tversky loss value.
        """
        probs = torch.sigmoid(logits)

        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1).float()

        # True Positives, False Positives, False Negatives
        tp = (probs_flat * targets_flat).sum()
        fp = (probs_flat * (1.0 - targets_flat)).sum()
        fn = ((1.0 - probs_flat) * targets_flat).sum()

        tversky = (tp + self.smooth) / (tp + (self.alpha * fn) + (self.beta * fp) + self.smooth + self.eps)
        focal_tversky = torch.pow((1.0 - tversky), (1.0 / self.gamma))
        return focal_tversky


def get_loss_function(config: Dict[str, Any]) -> nn.Module:
    """Build and return loss module based on configuration dictionary.

    Args:
        config: Central configuration dictionary under 'loss'.

    Returns:
        nn.Module: Configured loss function.
    """
    loss_cfg = config.get("loss", {})
    loss_type = loss_cfg.get("type", "combo").lower()

    if loss_type == "combo":
        alpha = float(loss_cfg.get("bce_weight", 0.5))
        beta = float(loss_cfg.get("dice_weight", 0.5))
        smooth = float(loss_cfg.get("smooth", 1.0))
        return ComboLoss(alpha=alpha, beta=beta, smooth=smooth)

    if loss_type == "focal_tversky":
        ft_cfg = loss_cfg.get("focal_tversky", {})
        alpha = float(ft_cfg.get("alpha", 0.7))
        beta = float(ft_cfg.get("beta", 0.3))
        gamma = float(ft_cfg.get("gamma", 1.33))
        smooth = float(loss_cfg.get("smooth", 1.0))
        return FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma, smooth=smooth)

    if loss_type == "dice":
        smooth = float(loss_cfg.get("smooth", 1.0))
        return DiceLoss(smooth=smooth)

    if loss_type == "bce":
        return nn.BCEWithLogitsLoss()

    raise ValueError(f"Unsupported loss type '{loss_type}'. Choose from 'combo', 'focal_tversky', 'dice', 'bce'.")
