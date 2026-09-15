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


def clip_fft_saliency(
    model: torch.nn.Module,
    clip_model: torch.nn.Module,
    clip_image: torch.Tensor,
    fft_image: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return semantic, frequency, and combined input-gradient saliency maps."""

    model.eval()
    clip_model.eval()

    clip_input = (
        clip_image.unsqueeze(0)
        .to(device)
        .detach()
        .requires_grad_(True)
    )

    fft_input = (
        fft_image.unsqueeze(0)
        .to(device)
        .detach()
        .requires_grad_(True)
    )

    # Keep the CLIP backbone frozen while allowing gradients to its input.
    for parameter in clip_model.parameters():
        parameter.requires_grad = False

    clip_model.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)

    clip_features = clip_model.encode_image(
        clip_input
    )

    clip_features = (
        clip_features
        / clip_features.norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-8)
    ).float()

    logit = model(
        clip_features,
        fft_input,
    )

    # Separate gradients for the two evidence paths.
    semantic_gradient = torch.autograd.grad(
        logit.sum(),
        clip_input,
        retain_graph=True,
        allow_unused=False,
    )[0]

    frequency_gradient = torch.autograd.grad(
        logit.sum(),
        fft_input,
        retain_graph=False,
        allow_unused=False,
    )[0]

    semantic = semantic_gradient.detach().abs().amax(
        dim=1
    )[0]

    frequency = frequency_gradient.detach().abs().amax(
        dim=1
    )[0]

    def normalize(value: torch.Tensor) -> np.ndarray:
        value = value.detach().cpu().numpy()

        low = value.min()
        high = value.max()

        return (
            (value - low)
            / max(
                float(high - low),
                1e-8,
            )
        )

    semantic_map = normalize(
        semantic
    )

    frequency_map = normalize(
        frequency
    )

    combined_map = normalize(
        (
            torch.as_tensor(
                semantic_map,
                dtype=torch.float32,
            )
            + torch.as_tensor(
                frequency_map,
                dtype=torch.float32,
            )
        )
        / 2.0
    )

    return (
        semantic_map,
        frequency_map,
        combined_map,
    )


def occlusion_saliency(
    model: torch.nn.Module,
    clip_model: torch.nn.Module,
    image: Image.Image,
    temperature: float,
    device: torch.device,
    grid: int = 7,
    batch_size: int = 16,
) -> tuple[np.ndarray, dict]:
    """Compute smooth signed occlusion attribution for the CLIP+FFT model.

    Positive influence means masking a region lowers P(AI), so that region
    supports the AI-generated decision.

    Negative influence means masking a region raises P(AI), so that region
    opposes the AI-generated decision.
    """

    from PIL import ImageFilter
    from scripts.train_clip_fft import MultiViewTransform

    model.eval()
    clip_model.eval()

    transform = MultiViewTransform()

    with torch.inference_mode():
        clip_base, fft_base = transform(image)

        base_features = clip_model.encode_image(
            clip_base.unsqueeze(0).to(device)
        )

        base_features = (
            base_features
            / base_features.norm(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-8)
        ).float()

        base_logit = model(
            base_features,
            fft_base.unsqueeze(0).to(device),
        )

        base_probability = torch.sigmoid(
            base_logit / temperature
        ).item()

    width, height = image.size

    blurred = image.filter(
        ImageFilter.GaussianBlur(
            radius=max(
                4,
                min(width, height) / 40,
            )
        )
    )

    occluded_clip = []
    occluded_fft = []
    positions = []

    cell_width = width / grid
    cell_height = height / grid

    for row in range(grid):
        for column in range(grid):
            left = int(round(column * cell_width))
            top = int(round(row * cell_height))
            right = int(round((column + 1) * cell_width))
            bottom = int(round((row + 1) * cell_height))

            occluded = image.copy()
            occluded.paste(
                blurred.crop(
                    (
                        left,
                        top,
                        right,
                        bottom,
                    )
                ),
                (
                    left,
                    top,
                ),
            )

            clip_image, fft_image = transform(
                occluded
            )

            occluded_clip.append(
                clip_image
            )

            occluded_fft.append(
                fft_image
            )

            positions.append(
                (
                    left,
                    top,
                    right,
                    bottom,
                )
            )

    scores = []

    for start in range(
        0,
        len(positions),
        batch_size,
    ):
        end = min(
            start + batch_size,
            len(positions),
        )

        batch_clip = torch.stack(
            occluded_clip[start:end]
        ).to(
            device,
            non_blocking=torch.cuda.is_available(),
        )

        batch_fft = torch.stack(
            occluded_fft[start:end]
        ).to(
            device,
            non_blocking=torch.cuda.is_available(),
        )

        with torch.inference_mode():
            features = clip_model.encode_image(
                batch_clip
            )

            features = (
                features
                / features.norm(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-8)
            ).float()

            logits = model(
                features,
                batch_fft,
            )

            probabilities = torch.sigmoid(
                logits / temperature
            )

            scores.extend(
                probabilities.cpu().tolist()
            )

    scores_array = np.asarray(
        scores,
        dtype=np.float64,
    )

    influence = (
        base_probability
        - scores_array
    )

    influence_grid = influence.reshape(
        grid,
        grid,
    )

    max_abs = max(
        float(np.abs(influence_grid).max()),
        1e-8,
    )

    normalized_grid = (
        influence_grid
        / max_abs
    )

    grid_image = Image.fromarray(
        np.uint8(
            np.clip(
                (
                    normalized_grid
                    + 1.0
                )
                * 127.5,
                0,
                255,
            )
        ),
        mode="L",
    )

    smooth_image = grid_image.resize(
        (width, height),
        Image.Resampling.BICUBIC,
    )

    smooth_normalized = (
        np.asarray(
            smooth_image,
            dtype=np.float32,
        )
        / 127.5
        - 1.0
    )

    strength = np.abs(
        smooth_normalized
    )

    weak_cutoff = 0.20

    alpha = np.clip(
        (
            strength
            - weak_cutoff
        )
        / (
            1.0
            - weak_cutoff
        ),
        0.0,
        1.0,
    )

    alpha = (
        alpha ** 1.25
        * 185.0
    )

    heat = np.zeros(
        (
            height,
            width,
            4,
        ),
        dtype=np.uint8,
    )

    positive = (
        smooth_normalized
        >= 0
    )

    heat[..., 0] = np.where(
        positive,
        255,
        40,
    )

    heat[..., 1] = np.where(
        positive,
        45,
        90,
    )

    heat[..., 2] = np.where(
        positive,
        45,
        255,
    )

    heat[..., 3] = np.uint8(
        np.clip(
            alpha,
            0,
            255,
        )
    )

    base = np.asarray(
        image.convert("RGBA"),
        dtype=np.float32,
    )

    heat_float = heat.astype(
        np.float32
    )

    alpha_float = (
        heat_float[..., 3:4]
        / 255.0
    )

    composite = (
        base * (1.0 - alpha_float)
        + heat_float * alpha_float
    ).clip(
        0,
        255,
    ).astype(
        np.uint8
    )

    overlay_image = Image.fromarray(
        composite,
        "RGBA",
    ).convert("RGB")

    positive_values = influence[
        influence > 0
    ]

    negative_values = influence[
        influence < 0
    ]

    evidence = {
        "base_probability": float(
            base_probability
        ),
        "max_positive_influence": (
            float(positive_values.max())
            if positive_values.size
            else 0.0
        ),
        "max_negative_influence": (
            float(negative_values.min())
            if negative_values.size
            else 0.0
        ),
        "mean_absolute_influence": float(
            np.abs(influence).mean()
        ),
    }

    return (
        np.asarray(
            overlay_image,
            dtype=np.uint8,
        ),
        evidence,
    )


def overlay(image: Image.Image, heatmap: np.ndarray, opacity: float = 0.42) -> Image.Image:
    heatmap_image = Image.fromarray(np.uint8(heatmap * 255)).resize(image.size, Image.Resampling.BILINEAR)
    heat = np.asarray(heatmap_image, dtype=np.float32) / 255.0
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    colored = np.stack((255 * heat, 80 * (1 - heat), 30 * (1 - heat)), axis=-1)
    return Image.fromarray(np.uint8(base * (1 - opacity) + colored * opacity))


def grounded_explanation(
    probability_fake: float,
    confidence: float,
    threshold: float = 0.5,
) -> str:
    """Return conservative explanation language using the model threshold."""

    if probability_fake >= threshold:
        return (
            f"Likely AI-generated. "
            f"P(AI-generated): {probability_fake:.2%}. "
            "Highlighted regions show areas whose masking changed the model's "
            "AI-generation score; they do not independently prove a visual artifact."
        )

    return (
        f"Likely authentic. "
        f"P(AI-generated): {probability_fake:.2%}. "
        "Highlighted regions show areas whose masking changed the model's "
        "AI-generation score; they do not independently prove a visual artifact."
    )
