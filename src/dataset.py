"""Dataset Module for Industrial Surface Defect Semantic Segmentation.

Handles paired image and mask loading, CLAHE preprocessing for industrial lighting
normalization, synchronized spatial Albumentations transforms, deterministic rasterization
for RLE/XML bounding boxes, and dataset splitting.
"""

from __future__ import annotations

import glob
import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def rasterize_rle_mask(rle_string: str, height: int, width: int) -> np.ndarray:
    """Rasterize Run-Length Encoded (RLE) string into a binary 2D numpy mask.

    Args:
        rle_string: Pairs of start pixel (1-indexed) and run length separated by space.
        height: Target image height in pixels.
        width: Target image width in pixels.

    Returns:
        np.ndarray: Binary mask array of shape (height, width), uint8 (0 or 1).
    """
    mask = np.zeros(height * width, dtype=np.uint8)
    if not rle_string or not isinstance(rle_string, str) or rle_string.strip() == "":
        return mask.reshape((height, width), order="F")

    s = rle_string.strip().split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1

    return mask.reshape((height, width), order="F")


def rasterize_bbox_xml(xml_path: str, height: int, width: int) -> np.ndarray:
    """Rasterize Pascal VOC bounding box XML annotations into a binary 2D mask.

    Used as an automated fallback when pixel masks are not directly available.

    Args:
        xml_path: Filepath to Pascal VOC XML annotation file.
        height: Target image height in pixels.
        width: Target image width in pixels.

    Returns:
        np.ndarray: Binary mask array of shape (height, width), uint8 (0 or 1).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    if not os.path.exists(xml_path):
        return mask

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            if bndbox is not None:
                xmin = int(float(bndbox.find("xmin").text))  # type: ignore
                ymin = int(float(bndbox.find("ymin").text))  # type: ignore
                xmax = int(float(bndbox.find("xmax").text))  # type: ignore
                ymax = int(float(bndbox.find("ymax").text))  # type: ignore

                # Clamp to image boundaries
                xmin = max(0, min(xmin, width - 1))
                xmax = max(0, min(xmax, width - 1))
                ymin = max(0, min(ymin, height - 1))
                ymax = max(0, min(ymax, height - 1))

                mask[ymin:ymax + 1, xmin:xmax + 1] = 1
    except Exception as e:
        logger.warning("Failed to parse bounding box XML %s: %s", xml_path, e)

    return mask


def get_training_transforms(image_size: Tuple[int, int] = (256, 256)) -> Any:
    """Construct Albumentations augmentation pipeline for training with CLAHE and spatial transforms.

    Args:
        image_size: Target (height, width) tuple.

    Returns:
        Albumentations Compose transform.
    """
    if not HAS_ALBUMENTATIONS:
        raise ImportError("albumentations is required for transforms. Run: pip install albumentations")

    return A.Compose([
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(height=image_size[0], width=image_size[1], interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.10,
            rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT,
            p=0.5
        ),
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        ], p=0.3),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])


def get_validation_transforms(image_size: Tuple[int, int] = (256, 256)) -> Any:
    """Construct deterministic Albumentations preprocessing pipeline for validation and inference.

    Args:
        image_size: Target (height, width) tuple.

    Returns:
        Albumentations Compose transform.
    """
    if not HAS_ALBUMENTATIONS:
        raise ImportError("albumentations is required for transforms. Run: pip install albumentations")

    return A.Compose([
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(height=image_size[0], width=image_size[1], interpolation=cv2.INTER_LINEAR),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])


class SurfaceDefectDataset(Dataset):
    """PyTorch Dataset for Industrial Surface Defect Semantic Segmentation.

    Loads paired industrial surface images and binary defect masks (0=background, 1=defect),
    applies synchronized CLAHE and geometric augmentations, and returns normalized tensors.
    """

    def __init__(
        self,
        image_paths: List[str],
        mask_paths: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        is_test: bool = False,
    ) -> None:
        """Initialize the SurfaceDefectDataset.

        Args:
            image_paths: List of absolute or relative filepaths to industrial surface images.
            mask_paths: Optional paired list of ground-truth mask paths or XML paths.
            transform: Albumentations Compose pipeline.
            is_test: If True, operates in test mode without requiring ground-truth masks.
        """
        self.image_paths = sorted(image_paths)
        self.mask_paths = sorted(mask_paths) if mask_paths is not None else None
        self.transform = transform
        self.is_test = is_test

        if not self.is_test and self.mask_paths is not None:
            if len(self.image_paths) != len(self.mask_paths):
                raise ValueError(
                    f"Mismatch: found {len(self.image_paths)} images but {len(self.mask_paths)} masks."
                )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.image_paths[idx]

        # 1. Read Image in RGB
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image at path: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # 2. Read or Rasterize Mask
        if not self.is_test and self.mask_paths is not None:
            mask_path = self.mask_paths[idx]
            if mask_path.endswith(".xml"):
                mask = rasterize_bbox_xml(mask_path, height=h, width=w)
            else:
                raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if raw_mask is None:
                    mask = np.zeros((h, w), dtype=np.uint8)
                else:
                    # Deterministic binarization: foreground defect = 1, background = 0
                    mask = (raw_mask > 127).astype(np.uint8)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)

        # 3. Apply Synchronized Albumentations Transforms
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image_tensor = augmented["image"]
            mask_tensor = augmented["mask"]
            if not isinstance(mask_tensor, torch.Tensor):
                mask_tensor = torch.from_numpy(mask_tensor)
            mask_tensor = mask_tensor.unsqueeze(0).float()
        else:
            # Fallback manual tensor conversion
            image_norm = image.astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).float()
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": img_path,
        }


def prepare_dataset_splits(
    data_dir: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Tuple[List[str], List[str]]]:
    """Scan directory, match paired images and masks, and create deterministic split manifests.

    Supports both directory structure with 'images/' & 'masks/' subfolders, or flat directories
    matching prefixes.

    Args:
        data_dir: Source directory containing raw images and ground truth masks.
        output_dir: Destination directory to store processed split lists.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        seed: Random seed for deterministic reproducibility.

    Returns:
        Dict mapping 'train', 'val', 'test' to (image_paths, mask_paths) tuples.
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Search common image extensions
    valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif")
    all_images: List[str] = []

    img_subdir = os.path.join(data_dir, "images")
    search_root = img_subdir if os.path.isdir(img_subdir) else data_dir

    for ext in valid_exts:
        all_images.extend(glob.glob(os.path.join(search_root, "**", ext), recursive=True))

    # Exclude any files that are inside masks subdirectories
    all_images = [p for p in all_images if "mask" not in Path(p).parent.name.lower()]
    all_images.sort()

    paired_images: List[str] = []
    paired_masks: List[str] = []

    for img_path in all_images:
        stem = Path(img_path).stem
        parent = Path(img_path).parent

        # Look for corresponding mask in sibling 'masks' folder or with '_mask' suffix
        candidate_masks = [
            os.path.join(parent.parent, "masks", f"{stem}.png"),
            os.path.join(parent.parent, "masks", f"{stem}.bmp"),
            os.path.join(parent, f"{stem}_mask.png"),
            os.path.join(parent, f"{stem}_mask.bmp"),
            os.path.join(data_dir, "masks", f"{stem}.png"),
            os.path.join(data_dir, "masks", f"{stem}.bmp"),
            os.path.join(data_dir, "annotations", f"{stem}.xml"),
        ]

        found_mask = None
        for cand in candidate_masks:
            if os.path.exists(cand):
                found_mask = cand
                break

        if found_mask is not None:
            paired_images.append(img_path)
            paired_masks.append(found_mask)

    logger.info("Matched %d paired image-mask samples in %s", len(paired_images), data_dir)

    # If no paired masks found, synthesize verification sample pairs for testing
    if len(paired_images) == 0:
        logger.warning("No paired image-mask samples discovered. Synthesizing verification benchmark set.")
        synthetic_dir = os.path.join(output_dir, "synthetic_benchmark")
        os.makedirs(os.path.join(synthetic_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(synthetic_dir, "masks"), exist_ok=True)

        for i in range(40):
            # Generate realistic metallic texture with surface defect (scratches / pits)
            base_img = np.random.normal(140, 25, (256, 256, 3)).clip(0, 255).astype(np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)

            # Insert defect: scratch or pitting cluster
            if i % 2 == 0:
                # Scratch (curved polyline)
                pts = np.array([
                    [np.random.randint(20, 100), np.random.randint(20, 100)],
                    [np.random.randint(100, 180), np.random.randint(80, 180)],
                    [np.random.randint(180, 240), np.random.randint(160, 240)],
                ], dtype=np.int32)
                cv2.polylines(base_img, [pts], isClosed=False, color=(40, 40, 40), thickness=3)
                cv2.polylines(mask, [pts], isClosed=False, color=1, thickness=3)
            else:
                # Pitting / patch
                cx, cy = np.random.randint(60, 200), np.random.randint(60, 200)
                rx, ry = np.random.randint(10, 30), np.random.randint(8, 25)
                cv2.ellipse(base_img, (cx, cy), (rx, ry), np.random.randint(0, 180), 0, 360, (50, 45, 45), -1)
                cv2.ellipse(mask, (cx, cy), (rx, ry), np.random.randint(0, 180), 0, 360, 1, -1)

            img_p = os.path.join(synthetic_dir, "images", f"surface_sample_{i:04d}.png")
            msk_p = os.path.join(synthetic_dir, "masks", f"surface_sample_{i:04d}.png")
            cv2.imwrite(img_p, cv2.cvtColor(base_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(msk_p, mask * 255)
            paired_images.append(img_p)
            paired_masks.append(msk_p)

    num_total = len(paired_images)
    indices = np.random.permutation(num_total)

    train_end = int(train_ratio * num_total)
    val_end = train_end + int(val_ratio * num_total)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    splits: Dict[str, Tuple[List[str], List[str]]] = {
        "train": ([paired_images[i] for i in train_idx], [paired_masks[i] for i in train_idx]),
        "val": ([paired_images[i] for i in val_idx], [paired_masks[i] for i in val_idx]),
        "test": ([paired_images[i] for i in test_idx], [paired_masks[i] for i in test_idx]),
    }

    return splits
