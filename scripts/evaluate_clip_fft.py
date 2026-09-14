"""Evaluate the CLIP + FFT SignalScope detector on a manifest split.

The checkpoint contains:
    - frozen CLIP configuration
    - trainable FFT + fusion weights
    - validation-only temperature
    - validation-only threshold

The unseen split is never used for calibration.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.dataset import ManifestDataset
from scripts.train_clip_fft import (
    ClipFFTClassifier,
    MultiViewTransform,
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class ClipFFTInferenceDataset(Dataset):
    """Return aligned CLIP/FFT views plus metadata."""

    def __init__(
        self,
        dataset: ManifestDataset,
    ) -> None:
        self.dataset = dataset
        self.transform = MultiViewTransform()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        image, label, generator = self.dataset[index]

        clip_image, fft_image = self.transform(
            image
        )

        image_path = str(
            self.dataset.samples[index][0]
        )

        return (
            clip_image,
            fft_image,
            label,
            generator,
            image_path,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP + FFT fusion."
    )

    parser.add_argument(
        "--manifest",
        default="data/processed/generalization_manifest.csv",
    )

    parser.add_argument(
        "--split",
        default="unseen",
    )

    parser.add_argument(
        "--model",
        default="model/dual/clip_fft_fusion.pt",
    )

    parser.add_argument(
        "--output",
        default="report/clip_fft_vqdm_predictions.csv",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--clip-model",
        default="ViT-B-32",
    )

    parser.add_argument(
        "--clip-pretrained",
        default="openai",
    )

    args = parser.parse_args()

    payload = torch.load(
        args.model,
        map_location="cpu",
        weights_only=False,
    )

    model_config = payload.get(
        "model_config",
        {},
    )

    model = ClipFFTClassifier(
        clip_dim=int(
            model_config.get(
                "clip_dim",
                512,
            )
        ),
        frequency_width=int(
            model_config.get(
                "frequency_width",
                32,
            )
        ),
    )

    # The no-interaction ablation has a 256-dimensional
    # fusion input instead of the baseline's 257 dimensions.
    no_interaction = (
        payload.get("model_name")
        == "clip_fft_no_interaction"
    )

    if no_interaction:
        model.fusion = torch.nn.Sequential(
            torch.nn.Linear(256, 64),
            torch.nn.GELU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(64, 1),
        )

    model.load_state_dict(
        payload["model_state_dict"]
    )

    model = model.eval().to(DEVICE)

    temperature = float(
        payload["temperature"]
    )

    threshold = float(
        payload["threshold"]
    )

    clip_model_name = model_config.get(
        "clip_model",
        args.clip_model,
    )

    clip_pretrained = model_config.get(
        "clip_pretrained",
        args.clip_pretrained,
    )

    clip_model, _, _ = (
        open_clip.create_model_and_transforms(
            clip_model_name,
            pretrained=clip_pretrained,
        )
    )

    clip_model = (
        clip_model
        .eval()
        .to(DEVICE)
    )

    for parameter in clip_model.parameters():
        parameter.requires_grad = False

    dataset = ClipFFTInferenceDataset(
        ManifestDataset(
            args.manifest,
            split=args.split,
            transform=None,
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_labels = []
    all_probabilities = []

    print(
        f"device={DEVICE}"
    )

    print(
        f"model={args.model}"
    )

    print(
        f"split={args.split}"
    )

    print(
        f"samples={len(dataset)}"
    )

    print(
        f"temperature={temperature:.6f}"
    )

    print(
        f"threshold={threshold:.6f}"
    )

    with output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "true_label",
                "predicted_prob",
                "generator",
            ],
        )

        writer.writeheader()

        with torch.inference_mode():
            for (
                clip_images,
                fft_images,
                labels,
                generators,
                paths,
            ) in loader:

                clip_images = clip_images.to(
                    DEVICE,
                    non_blocking=torch.cuda.is_available(),
                )

                fft_images = fft_images.to(
                    DEVICE,
                    non_blocking=torch.cuda.is_available(),
                )

                clip_features = (
                    clip_model.encode_image(
                        clip_images
                    )
                )

                clip_features = (
                    clip_features
                    / clip_features.norm(
                        dim=-1,
                        keepdim=True,
                    ).clamp_min(1e-8)
                ).float()

                semantic = model.semantic_projection(
                    clip_features
                )

                frequency_features = model.frequency_encoder(
                    fft_images
                )

                frequency = model.frequency_projection(
                    frequency_features
                )

                if no_interaction:
                    fused = torch.cat(
                        (
                            semantic,
                            frequency,
                        ),
                        dim=1,
                    )
                else:
                    interaction = (
                        semantic * frequency
                    ).mean(
                        dim=1,
                        keepdim=True,
                    )

                    fused = torch.cat(
                        (
                            semantic,
                            frequency,
                            interaction,
                        ),
                        dim=1,
                    )

                logits = model.fusion(
                    fused
                ).squeeze(1)

                probabilities = torch.sigmoid(
                    logits / temperature
                ).cpu().tolist()

                for (
                    probability,
                    label,
                    generator,
                    path,
                ) in zip(
                    probabilities,
                    labels.tolist(),
                    generators,
                    paths,
                ):
                    writer.writerow(
                        {
                            "image_id": path,
                            "true_label": label,
                            "predicted_prob": probability,
                            "generator": generator,
                        }
                    )

                    all_labels.append(
                        label
                    )

                    all_probabilities.append(
                        probability
                    )

    labels = np.asarray(
        all_labels,
        dtype=np.int64,
    )

    probabilities = np.asarray(
        all_probabilities,
        dtype=np.float64,
    )

    print()
    print(
        "Prediction summary"
    )
    print(
        "------------------------------"
    )
    print(
        f"Mean P(AI): {probabilities.mean():.6f}"
    )
    print(
        f"Min P(AI):  {probabilities.min():.6f}"
    )
    print(
        f"Max P(AI):  {probabilities.max():.6f}"
    )

    predicted = probabilities >= threshold

    tn = int(
        ((labels == 0) & (predicted == 0)).sum()
    )

    fp = int(
        ((labels == 0) & (predicted == 1)).sum()
    )

    fn = int(
        ((labels == 1) & (predicted == 0)).sum()
    )

    tp = int(
        ((labels == 1) & (predicted == 1)).sum()
    )

    print()
    print(
        "Confusion matrix at frozen threshold"
    )
    print(
        "------------------------------"
    )
    print(
        f"TN={tn}"
    )
    print(
        f"FP={fp}"
    )
    print(
        f"FN={fn}"
    )
    print(
        f"TP={tp}"
    )

    print()
    print(
        f"Wrote predictions to {output}"
    )


if __name__ == "__main__":
    main()
