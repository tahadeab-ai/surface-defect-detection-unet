"""Model Architecture Module for Industrial Surface Defect Semantic Segmentation.

Provides configurable U-Net builder using segmentation_models_pytorch (supporting
ResNet-34, EfficientNet-B2, etc.) along with a pure PyTorch baseline U-Net architecture.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Native Standard U-Net Architecture (Baseline from Scratch)
# -------------------------------------------------------------------------
class DoubleConv(nn.Module):
    """(Convolution => [BatchNorm] => ReLU) * 2 block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv with skip connection."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # Pad x1 if shapes differ slightly due to odd resolutions
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        if diff_y > 0 or diff_x > 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class StandardUNet(nn.Module):
    """Baseline Standard U-Net implemented purely in PyTorch (trained from scratch)."""

    def __init__(self, in_channels: int = 3, classes: int = 1, bilinear: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# -------------------------------------------------------------------------
# Configurable Model Factory
# -------------------------------------------------------------------------
def build_unet_model(
    encoder_name: Optional[str] = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
    activation: Optional[str] = None,
) -> nn.Module:
    """Factory function for building U-Net semantic segmentation models.

    Args:
        encoder_name: Backbone encoder name ('resnet34', 'efficientnet-b2', etc.),
                      or None for Standard U-Net baseline.
        encoder_weights: Pretrained weights name (e.g., 'imagenet') or None for scratch.
        in_channels: Number of input image channels (default 3 for RGB).
        classes: Number of output segmentation classes (1 for binary defect mask).
        activation: Optional activation ('sigmoid', 'softmax', or None for raw logits).

    Returns:
        nn.Module: PyTorch semantic segmentation model instance.
    """
    # 1. Baseline Standard U-Net requested
    if encoder_name is None or encoder_name.lower() in ("standard", "baseline", "scratch"):
        logger.info("Initializing Baseline Standard U-Net (from scratch)...")
        return StandardUNet(in_channels=in_channels, classes=classes)

    # 2. SMP Pretrained U-Net
    if HAS_SMP:
        logger.info(
            "Initializing SMP U-Net with backbone='%s', encoder_weights='%s', in_channels=%d, classes=%d...",
            encoder_name, encoder_weights, in_channels, classes
        )
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
        )
        return model

    # Fallback to standard U-Net if SMP is not available
    logger.warning("segmentation_models_pytorch not installed. Falling back to StandardUNet baseline.")
    return StandardUNet(in_channels=in_channels, classes=classes)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Calculate total and trainable parameters of a PyTorch model.

    Args:
        model: PyTorch model instance.

    Returns:
        Tuple[int, int]: (total_params, trainable_params).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_model_checkpoint(
    checkpoint_path: str,
    encoder_name: Optional[str] = "resnet34",
    in_channels: int = 3,
    classes: int = 1,
    device: str = "cpu",
) -> nn.Module:
    """Load model architecture and checkpoint state dict.

    Args:
        checkpoint_path: Path to .pth checkpoint file.
        encoder_name: Encoder name used during training.
        in_channels: Image channels.
        classes: Segmentation classes.
        device: Device to load model onto ('cpu' or 'cuda').

    Returns:
        nn.Module: Evaluation-ready model with loaded weights.
    """
    model = build_unet_model(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    logger.info("Successfully loaded checkpoint weights from %s onto %s", checkpoint_path, device)
    return model
