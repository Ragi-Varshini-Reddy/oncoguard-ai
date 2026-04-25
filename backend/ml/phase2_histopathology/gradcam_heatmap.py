"""
Generates a combined diagnostic heatmap for histopathology images.

Method:
  1. Cancer-class Grad-CAM  — gradients from the cancer logit back through
     the last EfficientNet conv layer.
  2. Tissue masking         — blank / background regions suppressed via stain mask.
  3. Histology anomaly map  — dark-dense nuclei, hyperchromatic regions,
     texture complexity, irregular local edges.
  4. Blended evidence map   — 58 % Grad-CAM + 42 % anomaly signal.
  5. JET colourmap overlay  — returned as JPEG bytes; lesion contours in JSON.
"""

from __future__ import annotations

import io
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


# ─── Transforms ──────────────────────────────────────────────────────────────

_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# ─── Grad-CAM hook helpers ────────────────────────────────────────────────────

class _GradCAMHook:
    def __init__(self) -> None:
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

    def save_activations(self, _module: nn.Module, _inp: Any, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def save_gradients(self, _module: nn.Module, _inp: Any, output: tuple[torch.Tensor, ...]) -> None:
        self.gradients = output[0].detach()


def _compute_gradcam(model: nn.Module, input_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    """Return a (7×7) Grad-CAM heatmap normalised to [0, 1]."""
    hook = _GradCAMHook()

    # EfficientNet-B4: last conv block is features[-1]
    target_layer = model.features[-1]
    fwd_handle = target_layer.register_forward_hook(hook.save_activations)
    bwd_handle = target_layer.register_full_backward_hook(hook.save_gradients)

    try:
        input_tensor = input_tensor.to(device).requires_grad_(True)
        logit = model(input_tensor)          # cancer logit (binary)
        model.zero_grad()
        logit[:, 0].backward()
    finally:
        fwd_handle.remove()
        bwd_handle.remove()

    if hook.activations is None or hook.gradients is None:
        return np.zeros((7, 7), dtype=np.float32)

    # GAP over spatial dims → channel weights
    weights = hook.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
    cam = (weights * hook.activations).sum(dim=1).squeeze()   # (H, W)
    cam = torch.relu(cam).cpu().numpy()

    if cam.max() > 0:
        cam /= cam.max()
    return cam.astype(np.float32)


# ─── Tissue mask ─────────────────────────────────────────────────────────────

def _tissue_mask(img_rgb: np.ndarray) -> np.ndarray:
    """
    Binary mask: True where tissue is present.
    Stain background is near-white (R>200, G>200, B>200).
    """
    bg = (
        (img_rgb[:, :, 0] > 200)
        & (img_rgb[:, :, 1] > 200)
        & (img_rgb[:, :, 2] > 200)
    )
    return (~bg).astype(np.float32)


# ─── Histology anomaly map ────────────────────────────────────────────────────

def _histology_anomaly_map(img_rgb: np.ndarray) -> np.ndarray:
    """
    Pixel-level anomaly score from four histological signals.
    Returns an array normalised to [0, 1] at the same resolution as img_rgb.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # 1. Dark-dense nuclei — inverted intensity
    dark_nuclei = 1.0 - gray

    # 2. Hyperchromatic (purple/blue) regions — hematoxylin channel proxy
    #    High B and low R is typical of haematoxylin-stained nuclei
    r = img_rgb[:, :, 0].astype(np.float32) / 255.0
    g = img_rgb[:, :, 1].astype(np.float32) / 255.0
    b = img_rgb[:, :, 2].astype(np.float32) / 255.0
    hyperchromatic = np.clip(b - r + 0.5, 0, 1)

    # 3. Texture complexity — local standard deviation
    gray_u8 = (gray * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(gray_u8, (5, 5), 0).astype(np.float32)
    texture = np.abs(gray_u8.astype(np.float32) - blur)
    if texture.max() > 0:
        texture /= texture.max()

    # 4. Irregular local edges — Canny edge density
    edges = cv2.Canny(gray_u8, threshold1=50, threshold2=150).astype(np.float32)
    edges = cv2.GaussianBlur(edges, (9, 9), 0)
    if edges.max() > 0:
        edges /= edges.max()

    # Weighted blend
    anomaly = 0.30 * dark_nuclei + 0.30 * hyperchromatic + 0.20 * texture + 0.20 * edges
    if anomaly.max() > 0:
        anomaly /= anomaly.max()
    return anomaly.astype(np.float32)


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_heatmap(
    image_bytes: bytes,
    model: nn.Module | None,
    device: torch.device | None,
    output_size: int = 512,
) -> tuple[bytes, list[dict[str, Any]]]:
    """
    Generate a combined evidence heatmap for one H&E image.

    Parameters
    ----------
    image_bytes : raw image file bytes
    model       : loaded EfficientNet-B4 (or None → anomaly-only mode)
    device      : torch device matching the model
    output_size : side length of the output square image

    Returns
    -------
    heatmap_jpeg : JPEG bytes of the coloured overlay
    lesion_regions : list of dicts {x, y, w, h, mean_score}
    """
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_small = pil_img.resize((output_size, output_size), Image.LANCZOS)
    img_rgb = np.array(img_small, dtype=np.uint8)

    tissue = _tissue_mask(img_rgb)
    anomaly = _histology_anomaly_map(img_rgb)

    # ── Grad-CAM ──────────────────────────────────────────────────────────
    if model is not None and device is not None:
        input_tensor = _TRANSFORM(pil_img).unsqueeze(0)
        cam_small = _compute_gradcam(model, input_tensor, device)
        cam = cv2.resize(cam_small, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
        cam = np.clip(cam, 0, 1).astype(np.float32)
    else:
        # Anomaly-only fallback (no trained model)
        cam = np.zeros((output_size, output_size), dtype=np.float32)

    # ── Blend 58 % GradCAM + 42 % anomaly ────────────────────────────────
    evidence = 0.58 * cam + 0.42 * anomaly
    evidence = evidence * tissue          # suppress background
    if evidence.max() > 0:
        evidence /= evidence.max()

    # ── JET overlay ───────────────────────────────────────────────────────
    heat_u8 = (evidence * 255).astype(np.uint8)
    jet = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    jet_rgb = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 0.45, jet_rgb, 0.55, 0)

    # ── Lesion contours (strong regions > 60 % of max) ───────────────────
    thresh = 0.60
    strong_mask = (evidence > thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(strong_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lesion_regions: list[dict[str, Any]] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 80:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        roi_score = float(evidence[y : y + h, x : x + w].mean())
        lesion_regions.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "mean_score": round(roi_score, 4)})
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 220, 0), 2)

    # ── Encode to JPEG ────────────────────────────────────────────────────
    pil_out = Image.fromarray(overlay)
    buf = io.BytesIO()
    pil_out.save(buf, format="JPEG", quality=88)
    return buf.getvalue(), lesion_regions
