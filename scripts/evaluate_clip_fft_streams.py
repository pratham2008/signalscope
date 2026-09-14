"""Evaluate CLIP, FFT, and fused streams of the CLIP+FFT detector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_clip_fft import ClipFFTClassifier, MultiViewTransform
from src.core.dataset import ManifestDataset
from src.core.metrics import roc_auc


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class ClipFFTStreamDataset(Dataset):
    def __init__(self, manifest_path: str, split: str) -> None:
        self.dataset = ManifestDataset(
            manifest_path,
            split=split,
            transform=None,
        )
        self.transform = MultiViewTransform()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        image, label, generator = self.dataset[index]
        clip_image, fft_image = self.transform(image)

        return (
            clip_image,
            fft_image,
            label,
            generator,
        )


def main() -> None:
    parser = argparse.ArgumentParser()

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
        "--batch-size",
        type=int,
        default=16,
    )

    args = parser.parse_args()

    payload = torch.load(
        args.model,
        map_location="cpu",
        weights_only=False,
    )

    config = payload["model_config"]

    model = ClipFFTClassifier(
        clip_dim=int(config["clip_dim"]),
        frequency_width=int(config["frequency_width"]),
    )

    model.load_state_dict(
        payload["model_state_dict"]
    )

    model = model.eval().to(DEVICE)

    temperature = float(
        payload["temperature"]
    )

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

    dataset = ClipFFTStreamDataset(
        args.manifest,
        args.split,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    semantic_scores = []
    frequency_scores = []
    fused_scores = []
    labels_all = []
    generators_all = []

    print(f"device={DEVICE}")
    print(f"model={args.model}")
    print(f"split={args.split}")
    print(f"samples={len(dataset)}")
    print(f"temperature={temperature:.6f}")

    with torch.inference_mode():
        for (
            clip_images,
            fft_images,
            labels,
            generators,
        ) in loader:

            clip_images = clip_images.to(
                DEVICE,
                non_blocking=torch.cuda.is_available(),
            )

            fft_images = fft_images.to(
                DEVICE,
                non_blocking=torch.cuda.is_available(),
            )

            clip_features = clip_model.encode_image(
                clip_images
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

            interaction = (
                semantic * frequency
            ).mean(
                dim=1,
                keepdim=True,
            )

            fused_features = torch.cat(
                (
                    semantic,
                    frequency,
                    interaction,
                ),
                dim=1,
            )

            fused_logit = model.fusion(
                fused_features
            ).squeeze(1)

            semantic_logit = semantic.mean(dim=1)
            frequency_logit = frequency.mean(dim=1)

            semantic_prob = torch.sigmoid(
                semantic_logit / temperature
            )

            frequency_prob = torch.sigmoid(
                frequency_logit / temperature
            )

            fused_prob = torch.sigmoid(
                fused_logit / temperature
            )

            semantic_scores.extend(
                semantic_prob.cpu().tolist()
            )

            frequency_scores.extend(
                frequency_prob.cpu().tolist()
            )

            fused_scores.extend(
                fused_prob.cpu().tolist()
            )

            labels_all.extend(
                labels.tolist()
            )

            generators_all.extend(
                generators
            )

    labels = np.asarray(
        labels_all,
        dtype=np.int64,
    )

    semantic_scores = np.asarray(
        semantic_scores
    )

    frequency_scores = np.asarray(
        frequency_scores
    )

    fused_scores = np.asarray(
        fused_scores
    )

    print()
    print("Stream results")
    print("------------------------------")
    print(
        f"CLIP/semantic AUC:  "
        f"{roc_auc(labels, semantic_scores):.6f}"
    )
    print(
        f"FFT/frequency AUC: "
        f"{roc_auc(labels, frequency_scores):.6f}"
    )
    print(
        f"Fused AUC:         "
        f"{roc_auc(labels, fused_scores):.6f}"
    )

    print()
    print("Per-generator results")
    print("------------------------------")

    for generator in sorted(
        set(generators_all)
    ):
        mask = np.asarray(
            [
                g == generator
                for g in generators_all
            ]
        )

        y = labels[mask]

        print(
            f"{generator:12s} "
            f"clip={roc_auc(y, semantic_scores[mask]):.6f} "
            f"fft={roc_auc(y, frequency_scores[mask]):.6f} "
            f"fused={roc_auc(y, fused_scores[mask]):.6f}"
        )


if __name__ == "__main__":
    main()
