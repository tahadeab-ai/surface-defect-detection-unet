"""AMP-Enabled Training Engine for Industrial Surface Defect Semantic Segmentation.

Trains U-Net models with Automatic Mixed Precision (AMP), monitors foreground IoU and
Dice score, manages learning rate schedules, and saves best model checkpoints.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from src.dataset import (
    SurfaceDefectDataset,
    get_training_transforms,
    get_validation_transforms,
    prepare_dataset_splits,
)
from src.losses import get_loss_function
from src.model import build_unet_model, count_parameters

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> Tuple[float, float, float, float]:
    """Compute IoU (Jaccard), Dice (F1), Precision, and Recall on foreground defect pixels.

    Args:
        logits: Raw logits tensor of shape (B, 1, H, W).
        targets: Binary ground truth tensor of shape (B, 1, H, W).
        threshold: Classification probability threshold.
        eps: Epsilon term to prevent division by zero.

    Returns:
        Tuple[float, float, float, float]: (mean_iou, mean_dice, precision, recall).
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = targets.float()

    # Flatten tensors
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    tp = (preds_flat * targets_flat).sum().item()
    fp = (preds_flat * (1.0 - targets_flat)).sum().item()
    fn = ((1.0 - preds_flat) * targets_flat).sum().item()

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)

    return iou, dice, precision, recall


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[float, float, float]:
    """Execute one training epoch with Automatic Mixed Precision.

    Args:
        model: PyTorch segmentation model.
        loader: Training DataLoader.
        optimizer: AdamW optimizer.
        criterion: Configured loss function.
        scaler: PyTorch GradScaler.
        device: Target torch device.
        use_amp: Whether to enable AMP autocasting.

    Returns:
        Tuple[float, float, float]: (avg_loss, avg_iou, avg_dice).
    """
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0

    pbar = tqdm(loader, desc="Training Batch", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, masks)

        if use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_iou, batch_dice, _, _ = compute_batch_metrics(logits.detach(), masks)

        running_loss += loss.item()
        running_iou += batch_iou
        running_dice += batch_dice

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "iou": f"{batch_iou:.4f}",
            "dice": f"{batch_dice:.4f}",
        })

    num_batches = max(len(loader), 1)
    return (
        running_loss / num_batches,
        running_iou / num_batches,
        running_dice / num_batches,
    )


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[float, float, float, float, float]:
    """Evaluate model on validation or test dataset.

    Args:
        model: PyTorch segmentation model.
        loader: Validation/Test DataLoader.
        criterion: Configured loss function.
        device: Target torch device.
        use_amp: Whether to use AMP during evaluation.

    Returns:
        Tuple[float, float, float, float, float]: (avg_loss, avg_iou, avg_dice, precision, recall).
    """
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0
    running_prec = 0.0
    running_rec = 0.0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with autocast(enabled=use_amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, masks)

        iou, dice, prec, rec = compute_batch_metrics(logits, masks)

        running_loss += loss.item()
        running_iou += iou
        running_dice += dice
        running_prec += prec
        running_rec += rec

    num_batches = max(len(loader), 1)
    return (
        running_loss / num_batches,
        running_iou / num_batches,
        running_dice / num_batches,
        running_prec / num_batches,
        running_rec / num_batches,
    )


def train_pipeline(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Execute end-to-end model training and checkpointing pipeline.

    Args:
        config_path: Filepath to YAML configuration file.

    Returns:
        Dict[str, Any]: Training history dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. Device & Seeds
    seed = int(config.get("dataset", {}).get("random_seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    requested_device = config.get("training", {}).get("device", "cuda")
    device = torch.device(requested_device if torch.cuda.is_available() and requested_device == "cuda" else "cpu")
    logger.info("Operating on compute device: %s", device)

    # 2. Data Preparation & Transforms
    raw_dir = config.get("dataset", {}).get("raw_dir", "data/raw")
    processed_dir = config.get("dataset", {}).get("processed_dir", "data/processed")
    img_size = tuple(config.get("dataset", {}).get("image_size", [256, 256]))

    splits = prepare_dataset_splits(
        data_dir=raw_dir,
        output_dir=processed_dir,
        seed=seed,
    )

    train_tf = get_training_transforms(image_size=img_size)
    val_tf = get_validation_transforms(image_size=img_size)

    train_ds = SurfaceDefectDataset(splits["train"][0], splits["train"][1], transform=train_tf)
    val_ds = SurfaceDefectDataset(splits["val"][0], splits["val"][1], transform=val_tf)

    batch_size = int(config.get("training", {}).get("batch_size", 16))
    num_workers = int(config.get("training", {}).get("num_workers", 2 if os.name == "nt" else 4))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=(len(train_ds) > batch_size),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 3. Model Architecture
    model_cfg = config.get("model", {})
    encoder_name = model_cfg.get("encoder_name", "resnet34")
    encoder_weights = model_cfg.get("encoder_weights", "imagenet")
    in_channels = int(model_cfg.get("in_channels", 3))
    classes = int(model_cfg.get("classes", 1))

    model = build_unet_model(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    logger.info("Model Parameters: Total = %d, Trainable = %d", total_params, trainable_params)

    # 4. Loss, Optimizer, Scaler, Scheduler
    criterion = get_loss_function(config)
    lr = float(config.get("training", {}).get("optimizer", {}).get("lr", 1e-4))
    weight_decay = float(config.get("training", {}).get("optimizer", {}).get("weight_decay", 1e-5))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = int(config.get("training", {}).get("epochs", 30))
    sched_cfg = config.get("training", {}).get("scheduler", {})
    sched_name = sched_cfg.get("name", "CosineAnnealingLR")

    if sched_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=float(sched_cfg.get("min_lr", 1e-6))
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=float(sched_cfg.get("factor", 0.5)), patience=int(sched_cfg.get("patience", 3))
        )

    use_amp = bool(config.get("training", {}).get("amp", True)) and (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    # 5. Checkpoint Directory
    ckpt_dir = config.get("training", {}).get("checkpoint_dir", "checkpoints")
    ckpt_name = config.get("training", {}).get("checkpoint_name", "best_unet_resnet34.pth")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, ckpt_name)

    best_val_dice = -1.0
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_iou": [],
        "val_iou": [],
        "train_dice": [],
        "val_dice": [],
        "learning_rates": [],
    }

    # 6. Training Loop
    logger.info("Starting training loop for %d epochs...", epochs)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss, train_iou, train_dice = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, use_amp=use_amp
        )
        val_loss, val_iou, val_dice, val_prec, val_rec = validate(
            model, val_loader, criterion, device, use_amp=use_amp
        )

        current_lr = optimizer.param_groups[0]["lr"]
        if sched_name == "CosineAnnealingLR":
            scheduler.step()
        else:
            scheduler.step(val_dice)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_iou"].append(float(train_iou))
        history["val_iou"].append(float(val_iou))
        history["train_dice"].append(float(train_dice))
        history["val_dice"].append(float(val_dice))
        history["learning_rates"].append(float(current_lr))

        epoch_duration = time.time() - epoch_start
        logger.info(
            "Epoch [%02d/%02d] (%.1fs) | Train Loss: %.4f, IoU: %.4f, Dice: %.4f | "
            "Val Loss: %.4f, IoU: %.4f, Dice: %.4f, Prec: %.4f, Rec: %.4f | LR: %.2e",
            epoch, epochs, epoch_duration, train_loss, train_iou, train_dice,
            val_loss, val_iou, val_dice, val_prec, val_rec, current_lr
        )

        # Checkpointing best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_dice": best_val_dice,
                "val_iou": val_iou,
                "config": config,
            }, best_ckpt_path)
            logger.info("--> Checkpoint saved: New best Val Dice = %.4f to %s", best_val_dice, best_ckpt_path)

    total_time = time.time() - start_time
    logger.info("Training completed in %.2f minutes. Peak Val Dice: %.4f", total_time / 60.0, best_val_dice)

    # Save training history JSON
    history_path = os.path.join(ckpt_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info("Saved training history to %s", history_path)

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Industrial Surface Defect U-Net Training")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    train_pipeline(config_path=args.config)
