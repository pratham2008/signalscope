"""Occlusion-based explanation for the SignalScope CLIP+FFT champion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_clip_fft import MultiViewTransform
from src.core.checkpoint import load_detector


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def make_occluded_image(
    image: Image.Image,
    blurred: Image.Image,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> Image.Image:
    result = image.copy()
    result.paste(
        blurred.crop(
            (left, top, right, bottom)
        ),
        (left, top),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image",
    )

    parser.add_argument(
        "--model",
        default="model/dual/clip_fft_resolution.pt",
    )

    parser.add_argument(
        "--grid",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--output",
        default="report/occlusion_test.png",
    )

    args = parser.parse_args()

    image = Image.open(
        args.image
    ).convert("RGB")

    model, metadata = load_detector(
        args.model,
        DEVICE,
    )

    model.eval()

    config = metadata["model_config"]

    clip_model, _, _ = (
        open_clip.create_model_and_transforms(
            config["clip_model"],
            pretrained=config["clip_pretrained"],
        )
    )

    clip_model = (
        clip_model
        .eval()
        .to(DEVICE)
    )

    for parameter in clip_model.parameters():
        parameter.requires_grad = False

    transform = MultiViewTransform()

    clip_base, fft_base = transform(
        image
    )

    temperature = float(
        metadata["temperature"]
    )

    with torch.inference_mode():
        base_features = clip_model.encode_image(
            clip_base.unsqueeze(0).to(DEVICE)
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
            fft_base.unsqueeze(0).to(DEVICE),
        )

        base_probability = torch.sigmoid(
            base_logit / temperature
        ).item()

    width, height = image.size

    blurred = image.filter(
        ImageFilter.GaussianBlur(
            radius=max(4, min(width, height) / 40)
        )
    )

    occluded_clip = []
    occluded_fft = []
    positions = []

    cell_width = width / args.grid
    cell_height = height / args.grid

    for row in range(args.grid):
        for column in range(args.grid):
            left = int(round(column * cell_width))
            top = int(round(row * cell_height))
            right = int(round((column + 1) * cell_width))
            bottom = int(round((row + 1) * cell_height))

            occluded = make_occluded_image(
                image,
                blurred,
                left,
                top,
                right,
                bottom,
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

    clip_tensor = torch.stack(
        occluded_clip
    )

    fft_tensor = torch.stack(
        occluded_fft
    )

    scores = []

    for start in range(
        0,
        len(positions),
        args.batch_size,
    ):
        end = min(
            start + args.batch_size,
            len(positions),
        )

        batch_clip = clip_tensor[
            start:end
        ].to(
            DEVICE,
            non_blocking=torch.cuda.is_available(),
        )

        batch_fft = fft_tensor[
            start:end
        ].to(
            DEVICE,
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

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    # Positive influence means masking the patch lowered P(AI).
    # Negative influence means masking the patch increased P(AI).
    influence = (
        base_probability
        - scores
    )

    influence_grid = influence.reshape(
        args.grid,
        args.grid,
    )

    print(
        f"base_probability={base_probability:.6f}"
    )

    print(
        f"min_influence={influence.min():.6f}"
    )

    print(
        f"max_influence={influence.max():.6f}"
    )

    print()
    print("Influence grid")
    print("------------------------------")

    for row in influence_grid:
        print(
            " ".join(
                f"{value:+.4f}"
                for value in row
            )
        )

    # Create a smooth signed attribution overlay.
    # Positive influence = evidence toward AI.
    # Negative influence = evidence away from AI.

    influence_grid = influence.reshape(
        args.grid,
        args.grid,
    )

    max_abs = max(
        float(np.abs(influence_grid).max()),
        1e-8,
    )

    # Interpolate the coarse grid to the original image dimensions.
    normalized_grid = influence_grid / max_abs

    grid_image = Image.fromarray(
        np.uint8(
            np.clip(
                (normalized_grid + 1.0) * 127.5,
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

    # Ignore very weak attribution so the whole image does not glow.
    strength = np.abs(
        smooth_normalized
    )

    weak_cutoff = 0.20

    alpha = np.clip(
        (
            strength - weak_cutoff
        )
        / (
            1.0 - weak_cutoff
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

    positive = smooth_normalized >= 0

    # Red = positive influence toward AI.
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

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        composite,
        "RGBA",
    ).convert("RGB").save(
        output
    )

    print()
    print(
        f"overlay={output}"
    )


if __name__ == "__main__":
    main()
