# Google Colab Manual Execution Runbook (NVIDIA T4 GPU)

> **Document Type:** Manual Execution Guide  
> **Target Environment:** Google Colab (Standard GPU runtime — NVIDIA T4)  
> **Repository:** `https://github.com/tahadeab-ai/surface-defect-detection-unet.git`  
> **Objective:** Step-by-step manual execution of dataset acquisition, rasterization, baseline vs. production U-Net training, metric calculation, asset generation, Gradio launch, and GitHub release upload.

---

## Pre-Flight Checklist
- [ ] Open [Google Colab](https://colab.research.google.com).
- [ ] In the top menu, navigate to **Runtime → Change runtime type**.
- [ ] Select **T4 GPU** under Hardware accelerator, then click **Save**.
- [ ] Have your `kaggle.json` API token ready (obtainable from Kaggle: Profile → Settings → Create New Token).
- [ ] Have a GitHub Personal Access Token (`GH_TOKEN` or `GITHUB_TOKEN`) with `repo` scope ready if pushing the release from Colab.

---

## Cell 1 — GPU Verification & Hardware Check
Verify that an NVIDIA GPU accelerator is attached and PyTorch can access CUDA.

```python
!nvidia-smi

import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Device Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
```

- **Expected output:**
  ```text
  NVIDIA-SMI ... Driver Version: ... CUDA Version: ...
  Tesla T4, ~15 GB VRAM
  PyTorch Version: 2.x
  CUDA Available: True
  Device Name: Tesla T4
  Device Memory: ~14.75 GB
  ```
- **If this fails, try:**
  ```bash
  # Ensure GPU runtime is active
  !python -c "import torch; assert torch.cuda.is_available(), 'Switch runtime to GPU: Runtime -> Change runtime type'"
  ```

---

## Cell 2 — Install Core Dependencies
Install production segmentation, augmentation, metrics, and visualization libraries.

```python
!pip install -q segmentation-models-pytorch albumentations opencv-python-headless gradio torchmetrics pyyaml kaggle tqdm
```

- **Expected output:**
  ```text
  Successfully installed segmentation-models-pytorch-... albumentations-... gradio-... torchmetrics-...
  ```
- **If this fails, try:**
  ```bash
  !pip install --upgrade pip setuptools wheel
  ```

---

## Cell 3 — Clone the Repository into Colab
Clone the project repository and set it as the working directory.

```python
import os

if not os.path.exists("surface-defect-detection-unet"):
    !git clone https://github.com/tahadeab-ai/surface-defect-detection-unet.git

%cd surface-defect-detection-unet
!git status
```

- **Expected output:**
  ```text
  Cloning into 'surface-defect-detection-unet'...
  /content/surface-defect-detection-unet
  On branch main
  ```
- **If this fails, try:**
  ```bash
  !ls -la /content
  ```

---

## Cell 4 — Upload Kaggle Credentials & Set Permissions
Upload your `kaggle.json` API token to enable dataset download.

```python
from google.colab import files
import shutil
from pathlib import Path

kaggle_dir = Path.home() / ".kaggle"
kaggle_dir.mkdir(exist_ok=True)
kaggle_json = kaggle_dir / "kaggle.json"

if not kaggle_json.exists():
    print("Please upload your kaggle.json file:")
    uploaded = files.upload()
    for filename in uploaded.keys():
        if filename == "kaggle.json":
            shutil.move("kaggle.json", str(kaggle_json))
            os.chmod(str(kaggle_json), 0o600)
            print("kaggle.json configured successfully.")
            break
else:
    print("kaggle.json already present.")

!kaggle datasets list --max-size 1000000000 | head -n 5
```

- **Expected output:**
  ```text
  kaggle.json configured successfully.
  ref ... title ... size ...
  ```
- **If this fails, try:**
  ```bash
  !ls -l ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json
  ```

---

## Cell 5 — Download Real Industrial Segmentation Dataset
Download the Severstal Steel Defect Detection dataset (contains genuine RLE-encoded pixel masks) or NEU-Seg benchmark dataset.

```python
import os
from pathlib import Path

raw_dir = Path("data/raw")
raw_dir.mkdir(parents=True, exist_ok=True)

# Option A: NEU Surface Defect Segmentation Dataset
!kaggle datasets download -d alex000kim/neu-surface-defect-database -p data/raw --unzip

# Verify downloaded dataset structure
print("\n--- Downloaded files preview ---")
!ls -lh data/raw | head -n 15
```

- **Expected output:**
  ```text
  Downloading neu-surface-defect-database.zip to data/raw
  100%|...| ...
  Archive unzipped successfully.
  ```
- **If this fails, try:**
  ```bash
  # Alternate direct download if Kaggle API rate-limits:
  !kaggle datasets list -s "surface defect segmentation"
  ```

---

## Cell 6 — Inspect `src/dataset.py` API Signatures
Inspect the defined classes and functions in `src/dataset.py` to ensure interface alignment.

```python
!grep -n "^def \|^class " src/dataset.py
```

- **Expected output:**
  ```text
  33:def rasterize_rle_mask(rle_string: str, height: int, width: int) -> np.ndarray:
  57:def rasterize_bbox_xml(xml_path: str, height: int, width: int) -> np.ndarray:
  98:def get_training_transforms(image_size: Tuple[int, int] = (256, 256)) -> Any:
  133:def get_validation_transforms(image_size: Tuple[int, int] = (256, 256)) -> Any:
  157:class SurfaceDefectDataset(Dataset):
  241:def prepare_dataset_splits(...)
  ```
- **If this fails, try:**
  ```bash
  !head -n 50 src/dataset.py
  ```

---

## Cell 7 — Prepare and Rasterize Paired Images & Ground-Truth Masks
Run the dataset split and mask rasterization utility from `src/dataset.py` to generate paired manifests in `data/processed/`.

```python
import sys
sys.path.insert(0, ".")

from src.dataset import prepare_dataset_splits

splits = prepare_dataset_splits(
    data_dir="data/raw",
    output_dir="data/processed",
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
)

print(f"Train samples: {len(splits['train'][0])}")
print(f"Val samples:   {len(splits['val'][0])}")
print(f"Test samples:  {len(splits['test'][0])}")
```

- **Expected output:**
  ```text
  Train samples: _TBD_
  Val samples:   _TBD_
  Test samples:  _TBD_
  ```
- **If this fails, try:**
  ```bash
  !find data/raw -name "*.png" -o -name "*.jpg" -o -name "*.bmp" | wc -l
  ```

---

## Cell 8 — Sanity Check: Shapes & Value Ranges
Load one sample pair through `SurfaceDefectDataset` with augmentations, assert tensor contracts, and plot the augmented pair.

```python
import matplotlib.pyplot as plt
import numpy as np
from src.dataset import SurfaceDefectDataset, get_training_transforms

tf = get_training_transforms(image_size=(256, 256))
ds = SurfaceDefectDataset(splits["train"][0], splits["train"][1], transform=tf)
sample = ds[0]

img = sample["image"]
mask = sample["mask"]

print(f"Image tensor shape: {img.shape}, dtype: {img.dtype}")
print(f"Mask tensor shape:  {mask.shape}, dtype: {mask.dtype}")
print(f"Unique mask values: {torch.unique(mask).tolist()}")

assert img.shape == (3, 256, 256), f"Expected (3, 256, 256), got {img.shape}"
assert mask.shape == (1, 256, 256), f"Expected (1, 256, 256), got {mask.shape}"
assert set(torch.unique(mask).tolist()).issubset({0.0, 1.0}), f"Mask values must be in {{0, 1}}, got {torch.unique(mask)}"

# Visualize sample pair
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
# Denormalize image for display
img_disp = img.permute(1, 2, 0).cpu().numpy()
img_disp = (img_disp * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)

ax[0].imshow(img_disp)
ax[0].set_title("Augmented Surface Image")
ax[0].axis("off")

ax[1].imshow(mask.squeeze().cpu().numpy(), cmap="gray")
ax[1].set_title("Ground Truth Defect Mask")
ax[1].axis("off")
plt.tight_layout()
plt.show()
print("Sanity check passed successfully!")
```

- **Expected output:**
  ```text
  Image tensor shape: torch.Size([3, 256, 256]), dtype: torch.float32
  Mask tensor shape:  torch.Size([1, 256, 256]), dtype: torch.float32
  Unique mask values: [0.0, 1.0]
  Sanity check passed successfully!
  ```
- **If this fails, try:**
  ```bash
  !python -c "import cv2; img=cv2.imread('{splits[\"train\"][0][0]}'); print('Raw image shape:', None if img is None else img.shape)"
  ```

---

## Cell 9 — Train Production Model (ResNet-34 U-Net + Combo Loss)
Train the production model using Automatic Mixed Precision (AMP), AdamW optimizer, and CosineAnnealing scheduler.

```python
!python -m src.train --config configs/config.yaml
```

- **Expected output:**
  ```text
  Operating on compute device: cuda
  Model Parameters: Total = 24,435,041, Trainable = 24,435,041
  Starting training loop for 30 epochs...
  Epoch [01/30] | Train Loss: _TBD_, IoU: _TBD_, Dice: _TBD_ | Val Loss: _TBD_, IoU: _TBD_, Dice: _TBD_
  ...
  Epoch [30/30] | Train Loss: _TBD_, IoU: _TBD_, Dice: _TBD_ | Val Loss: _TBD_, IoU: _TBD_, Dice: _TBD_
  Checkpoint saved: New best Val Dice = _TBD_ to checkpoints/best_unet_resnet34.pth
  Saved training history to checkpoints/training_history.json
  ```
- **If this fails, try:**
  ```bash
  # Check GPU memory and reduce batch_size in configs/config.yaml if Out Of Memory (OOM) occurs:
  !sed -i 's/batch_size: 16/batch_size: 8/g' configs/config.yaml
  ```

---

## Cell 10 — Train Baseline Model (Scratch Standard U-Net + BCE Only)
Create a baseline configuration without pretrained weights and with BCE loss only, then execute baseline training.

```python
import yaml

with open("configs/config.yaml", "r") as f:
    base_cfg = yaml.safe_load(f)

base_cfg["model"]["encoder_name"] = None
base_cfg["model"]["encoder_weights"] = None
base_cfg["loss"]["type"] = "bce"
base_cfg["training"]["checkpoint_name"] = "baseline_unet_scratch.pth"

with open("configs/config_baseline.yaml", "w") as f:
    yaml.dump(base_cfg, f)

!python -m src.train --config configs/config_baseline.yaml
```

- **Expected output:**
  ```text
  Initializing Baseline Standard U-Net (from scratch)...
  Epoch [01/30] ... Epoch [30/30] ...
  Checkpoint saved to checkpoints/baseline_unet_scratch.pth
  ```
- **If this fails, try:**
  ```bash
  !cat configs/config_baseline.yaml | head -n 30
  ```

---

## Cell 11 — Compute Real Test Metrics & Latency
Run a test evaluation loop using `torchmetrics.classification.BinaryJaccardIndex` and `BinaryF1Score`, then measure GPU inference latency.

```python
import time
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score
from src.dataset import SurfaceDefectDataset, get_validation_transforms
from src.model import load_model_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_tf = get_validation_transforms(image_size=(256, 256))
test_ds = SurfaceDefectDataset(splits["test"][0], splits["test"][1], transform=val_tf)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

# Load production model
model = load_model_checkpoint(
    checkpoint_path="checkpoints/best_unet_resnet34.pth",
    encoder_name="resnet34",
    device=str(device),
)
model.eval()

jaccard_metric = BinaryJaccardIndex(threshold=0.50).to(device)
f1_metric = BinaryF1Score(threshold=0.50).to(device)

latencies = []

with torch.no_grad():
    for batch in test_loader:
        imgs = batch["image"].to(device)
        msks = batch["mask"].to(device)

        t0 = time.perf_counter()
        logits = model(imgs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

        probs = torch.sigmoid(logits)
        jaccard_metric.update(probs, msks.int())
        f1_metric.update(probs, msks.int())

measured_iou = jaccard_metric.compute().item()
measured_dice = f1_metric.compute().item()
avg_latency_ms = float(np.mean(latencies[5:]))  # exclude warmup passes

print("=" * 50)
print("REAL MEASURED BENCHMARKS (Unseen Test Set):")
print(f"Mean IoU (Jaccard Index): {measured_iou:.4f}")
print(f"Dice Score (F1):          {measured_dice:.4f}")
print(f"Mean Latency (T4 GPU):    {avg_latency_ms:.2f} ms")
print("=" * 50)
```

- **Expected output:**
  ```text
  ==================================================
  REAL MEASURED BENCHMARKS (Unseen Test Set):
  Mean IoU (Jaccard Index): <paste actual output here>
  Dice Score (F1):          <paste actual output here>
  Mean Latency (T4 GPU):    <paste actual output here> ms
  ==================================================
  ```
- **If this fails, try:**
  ```bash
  !ls -lh checkpoints/
  ```

---

## Cell 12 — Regenerate Visual Artifacts from Real Trained Model
Inspect `generate_visual_artifacts` signature and execute it to save genuine artifacts into `assets/`.

```python
import inspect
import json
from src.inference import DefectSegmenter, generate_visual_artifacts

print("generate_visual_artifacts signature:", inspect.signature(generate_visual_artifacts))

with open("checkpoints/training_history.json", "r") as f:
    real_history = json.load(f)

segmenter = DefectSegmenter(
    checkpoint_path="checkpoints/best_unet_resnet34.pth",
    encoder_name="resnet34",
    threshold=0.50,
)

test_pairs = list(zip(splits["test"][0], splits["test"][1]))

generated_files = generate_visual_artifacts(
    history=real_history,
    segmenter=segmenter,
    sample_pairs=test_pairs,
    assets_dir="assets",
)

print("\nReal visual artifacts generated successfully:")
for name, pth in generated_files.items():
    print(f" - {name}: {pth}")

!ls -lh assets/
```

- **Expected output:**
  ```text
  generate_visual_artifacts signature: (history=None, segmenter=None, sample_pairs=None, assets_dir='assets')
  Real visual artifacts generated successfully:
   - training_curves: assets/training_curves.png
   - pixel_confusion_matrix: assets/pixel_confusion_matrix.png
   - pr_curve: assets/pr_curve.png
   - prediction_grid: assets/prediction_grid.png
  ```
- **If this fails, try:**
  ```bash
  !python -c "import matplotlib; print(matplotlib.__version__)"
  ```

---

## Cell 13 — Launch Interactive Gradio UI
Launch the interactive Gradio inspection interface with a public temporary link (`share=True`).

```python
from app.app import demo

demo.launch(share=True, debug=False)
```

- **Expected output:**
  ```text
  Running on local URL:  http://127.0.0.1:7860
  Running on public URL: https://<random-subdomain>.gradio.live
  ```
- **If this fails, try:**
  ```bash
  !python -c "import gradio; print(gradio.__version__)"
  ```

---

## Cell 14 — Confirm Checkpoint & History Persistence
Verify that `checkpoints/best_unet_resnet34.pth` and `checkpoints/training_history.json` exist and are populated.

```python
import os

pth_path = "checkpoints/best_unet_resnet34.pth"
json_path = "checkpoints/training_history.json"

assert os.path.exists(pth_path), f"Missing {pth_path}"
assert os.path.exists(json_path), f"Missing {json_path}"

pth_size_mb = os.path.getsize(pth_path) / (1024 * 1024)
print(f"Model Checkpoint Size: {pth_size_mb:.2f} MB")

with open(json_path) as f:
    hist = json.load(f)
print(f"Recorded epochs in history: {len(hist.get('train_loss', []))}")
```

- **Expected output:**
  ```text
  Model Checkpoint Size: ~90-100 MB (real ResNet-34 U-Net weights)
  Recorded epochs in history: 30
  ```
- **If this fails, try:**
  ```bash
  !ls -lh checkpoints/
  ```

---

## Cell 15 — Publish Real Trained Model to NEW GitHub Release `v1.1.0`
Using your GitHub Personal Access Token (`GH_TOKEN`), publish the genuine model weights and evaluation plots to a new release (`v1.1.0`). **Do not reuse `v1.0.0`.**

```python
from google.colab import userdata

# Store GH_TOKEN in Google Colab Secrets (the key icon on the left sidebar)
try:
    gh_token = userdata.get('GH_TOKEN')
    os.environ['GH_TOKEN'] = gh_token
    print("GH_TOKEN loaded from Colab Secrets.")
except Exception:
    import getpass
    gh_token = getpass.getpass("Enter your GitHub Personal Access Token (GH_TOKEN): ")
    os.environ['GH_TOKEN'] = gh_token

!gh auth status

!gh release create v1.1.0 \
  checkpoints/best_unet_resnet34.pth \
  checkpoints/training_history.json \
  configs/config.yaml \
  assets/prediction_grid.png \
  assets/training_curves.png \
  assets/pixel_confusion_matrix.png \
  assets/pr_curve.png \
  --title "v1.1.0 — Real Trained ResNet-34 U-Net Weights & Benchmarks" \
  --notes "Genuine trained weights and real test benchmarks executed on Google Colab NVIDIA T4 GPU."

!gh release list
```

- **Expected output:**
  ```text
  Logged in to github.com as <username>
  https://github.com/tahadeab-ai/surface-defect-detection-unet/releases/tag/v1.1.0
  v1.1.0 ...
  ```
- **If this fails, try:**
  ```bash
  !echo $GH_TOKEN | gh auth login --with-token
  ```

---

## Cell 16 — Commit Measured Metrics & Assets Back to GitHub
Update `README.md` with the real measured IoU, Dice, and Latency numbers, replace placeholder figures, and push the final commit back to GitHub.

```python
!git config --global user.email "tahadeab.ai@gmail.com"
!git config --global user.name "tahadeab-ai"

# Update remote URL with token for authenticated push
!git remote set-url origin https://x-access-token:${GH_TOKEN}@github.com/tahadeab-ai/surface-defect-detection-unet.git

# Stage newly generated genuine assets
!git add assets/prediction_grid.png assets/training_curves.png assets/pixel_confusion_matrix.png assets/pr_curve.png
!git commit -m "feat(benchmark): publish real trained metrics and visual evaluation artifacts"
!git push origin main

!git log --oneline -3
```

- **Expected output:**
  ```text
  [main abc1234] feat(benchmark): publish real trained metrics and visual evaluation artifacts
  To https://github.com/tahadeab-ai/surface-defect-detection-unet.git
     ... -> main
  ```
- **If this fails, try:**
  ```bash
  !git status
  ```
