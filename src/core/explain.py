"""Conservative saliency overlays and grounded explanation text."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from src.core.dual_model import DualStreamDetector


def saliency_map(model: torch.nn.Module, image: torch.Tensor, device: torch.device) -> np.ndarray:
    """Input-gradient saliency. It describes model sensitivity, not proof of an artifact."""
    sample = image.unsqueeze(0).to(device).detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logit = model(sample)
    logit.sum().backward(retain_graph=isinstance(model, DualStreamDetector))
    gradients = sample.grad.detach().abs().amax(dim=1)[0]
    if isinstance(model, DualStreamDetector):
        # Include a frequency-specific signal for the second evidence stream.
        frequency_grad = torch.autograd.grad(
            model.components(sample)[1].sum(), sample, retain_graph=False, allow_unused=True
        )[0]
        if frequency_grad is not None:
            gradients = (gradients + frequency_grad.detach().abs().amax(dim=1)[0]) / 2
    array = gradients.cpu().numpy()
    array = (array - array.min()) / max(float(array.max() - array.min()), 1e-8)
    return array


def overlay(image: Image.Image, heatmap: np.ndarray, opacity: float = 0.42) -> Image.Image:
    heatmap_image = Image.fromarray(np.uint8(heatmap * 255)).resize(image.size, Image.Resampling.BILINEAR)
    heat = np.asarray(heatmap_image, dtype=np.float32) / 255.0
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    colored = np.stack((255 * heat, 80 * (1 - heat), 30 * (1 - heat)), axis=-1)
    return Image.fromarray(np.uint8(base * (1 - opacity) + colored * opacity))


def grounded_explanation(probability_fake: float, confidence: float) -> str:
    if probability_fake >= 0.5:
        return (
            f"Likely AI-generated ({confidence:.0%} calibrated confidence). The highlighted areas show "
            "where the detector was most sensitive to combined semantic and frequency evidence; they do not "
            "independently prove a visible artifact."
        )
    return (
        f"Likely authentic ({confidence:.0%} calibrated confidence). No single visual region is treated as "
        "conclusive; this result is a statistical assessment and may be wrong."
    )
