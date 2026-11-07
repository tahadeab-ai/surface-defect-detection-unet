"""Interactive Gradio Web Application for Industrial Surface Defect Inspection.

Provides real-time defect segmentation, confidence threshold control, OpenCV contour
highlighting, and automated QA Pass/Fail reporting.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np

# Ensure root directory is on PYTHONPATH
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.inference import DefectSegmenter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Initialize Global Segmenter
CHECKPOINT_PATH = os.path.join(ROOT_DIR, "checkpoints", "best_unet_resnet34.pth")
segmenter = DefectSegmenter(
    checkpoint_path=CHECKPOINT_PATH if os.path.exists(CHECKPOINT_PATH) else None,
    encoder_name="resnet34",
    threshold=0.50,
    min_contour_area=15,
)


def create_sample_surface_images() -> List[str]:
    """Generate sample industrial surface images if none exist for demonstration."""
    sample_dir = os.path.join(ROOT_DIR, "assets", "demo_samples")
    os.makedirs(sample_dir, exist_ok=True)
    sample_paths = []

    samples_meta = [
        ("scratch_defect.png", "scratch"),
        ("pitting_defect.png", "pitting"),
        ("clean_surface.png", "clean"),
    ]

    for fname, defect_type in samples_meta:
        fpath = os.path.join(sample_dir, fname)
        if not os.path.exists(fpath):
            img = np.random.normal(135, 18, (256, 256, 3)).clip(0, 255).astype(np.uint8)
            if defect_type == "scratch":
                pts = np.array([[40, 50], [120, 110], [170, 170], [210, 220]], dtype=np.int32)
                cv2.polylines(img, [pts], isClosed=False, color=(35, 35, 35), thickness=4)
            elif defect_type == "pitting":
                for pt in [(75, 80), (130, 120), (180, 160), (90, 190)]:
                    cv2.circle(img, pt, 10, (25, 25, 25), -1)
            # 'clean' remains pristine metallic texture
            cv2.imwrite(fpath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        sample_paths.append(fpath)

    return sample_paths


def inspect_surface(
    image: Optional[np.ndarray],
    threshold: float,
    min_area: int,
    max_coverage_pass: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str, str, str, str]:
    """Process uploaded image and return annotated visual outputs and diagnostic metrics.

    Args:
        image: Uploaded image as numpy array (RGB).
        threshold: Confidence threshold for defect segmentation.
        min_area: Minimum pixel area to retain contour.
        max_coverage_pass: Max allowed defect coverage percentage for PASS grade.

    Returns:
        Tuple of (overlay_image, binary_mask, status_badge, defect_count_str, coverage_str, latency_str).
    """
    if image is None:
        return None, None, "⚪ NO INPUT", "0", "0.00%", "0.0 ms"

    segmenter.min_contour_area = int(min_area)
    res = segmenter.predict_image(image, threshold=threshold)

    # Evaluate QA compliance against user-specified threshold
    defect_area_pct = res["defect_area_pct"]
    defect_count = res["defect_count"]

    if defect_area_pct > max_coverage_pass or defect_count > 0:
        status_md = "## 🔴 **FAIL — DEFECT DETECTED**"
    else:
        status_md = "## 🟢 **PASS — SURFACE NOMINAL**"

    count_str = f"**{defect_count}** blobs"
    coverage_str = f"**{defect_area_pct:.3f}%**"
    latency_str = f"**{res['latency_ms']:.1f} ms**"

    return (
        res["overlay"],
        res["binary_mask"],
        status_md,
        count_str,
        coverage_str,
        latency_str,
    )


# -------------------------------------------------------------------------
# Gradio UI Construction
# -------------------------------------------------------------------------
demo_samples = create_sample_surface_images()

custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
)

with gr.Blocks(theme=custom_theme, title="Industrial Surface Defect Inspection AI") as demo:
    gr.Markdown(
        """
        # 🏭 Automated Industrial Surface Defect Semantic Segmentation
        ### *High-Precision Pixel-Level Quality Assurance for Metal & Steel Surfaces*
        **Architecture:** PyTorch U-Net (ResNet-34 Encoder) | **Loss:** Combo (BCE + Dice) | **Acceleration:** Mixed Precision
        """
    )

    with gr.Row():
        with gr.Column(scale=5):
            input_image = gr.Image(
                type="numpy",
                label="Upload Industrial Surface Image",
                sources=["upload", "clipboard"],
            )

            with gr.Accordion("⚙️ Inspection Hyperparameters", open=True):
                threshold_slider = gr.Slider(
                    minimum=0.10,
                    maximum=0.90,
                    value=0.50,
                    step=0.05,
                    label="Confidence Threshold",
                    info="Defect probability cutoff (lower catches faint micro-scratches)",
                )
                min_area_slider = gr.Slider(
                    minimum=5,
                    maximum=100,
                    value=15,
                    step=5,
                    label="Noise Filter: Min Contour Area (pixels)",
                    info="Discards isolated single-pixel sensor noise",
                )
                max_coverage_slider = gr.Slider(
                    minimum=0.05,
                    maximum=2.00,
                    value=0.50,
                    step=0.05,
                    label="Max Pass Defect Coverage (%)",
                    info="Maximum allowable surface defect percentage for QA pass",
                )

            inspect_btn = gr.Button("🔍 Run Defect Inspection", variant="primary", size="lg")

            gr.Examples(
                examples=[[p, 0.50, 15, 0.50] for p in demo_samples],
                inputs=[input_image, threshold_slider, min_area_slider, max_coverage_slider],
                label="Sample Benchmark Images",
            )

        with gr.Column(scale=7):
            status_box = gr.Markdown("## ⚪ **STATUS: Awaiting Inspection**")

            with gr.Row():
                out_defect_count = gr.Markdown(value="**0** blobs", label="Defect Count")
                out_defect_coverage = gr.Markdown(value="**0.00%**", label="Defect Coverage")
                out_latency = gr.Markdown(value="**0.0 ms**", label="Inference Latency")

            with gr.Tabs():
                with gr.TabItem("🎨 Defect Overlay & Contours"):
                    annotated_overlay = gr.Image(type="numpy", label="Overlay with Highlighted Contours")
                with gr.TabItem("⬛ Segmented Binary Mask"):
                    binary_mask_output = gr.Image(type="numpy", label="Binary Defect Mask (0=Background, 255=Defect)")

    inspect_btn.click(
        fn=inspect_surface,
        inputs=[input_image, threshold_slider, min_area_slider, max_coverage_slider],
        outputs=[annotated_overlay, binary_mask_output, status_box, out_defect_count, out_defect_coverage, out_latency],
    )

    input_image.change(
        fn=inspect_surface,
        inputs=[input_image, threshold_slider, min_area_slider, max_coverage_slider],
        outputs=[annotated_overlay, binary_mask_output, status_box, out_defect_count, out_defect_coverage, out_latency],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
