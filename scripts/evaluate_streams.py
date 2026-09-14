"""Evaluate SignalScope semantic, frequency, and fused streams separately."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.checkpoint import load_detector
from src.core.dataset import ManifestDataset, get_dual_val_transform
from src.core.dual_model import DualStreamDetector
from src.core.metrics import roc_auc


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class StreamDataset(Dataset):
    """Return image, label, and generator from a manifest dataset."""

    def __init__(self, dataset: ManifestDataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label, generator = self.dataset[index]
        return image, label, generator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SignalScope streams separately."
    )

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--split",
        default="unseen",
    )

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    args = parser.parse_args()

    model, metadata = load_detector(
        args.model,
        DEVICE,
    )

    if not isinstance(model, DualStreamDetector):
        raise TypeError(
            "This diagnostic requires a DualStreamDetector checkpoint."
        )

    dataset = StreamDataset(
        ManifestDataset(
            args.manifest,
            split=args.split,
            transform=get_dual_val_transform(),
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    temperature = float(
        metadata.get("temperature", 1.0)
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
        for images, labels, generators in loader:
            images = images.to(
                DEVICE,
                non_blocking=torch.cuda.is_available(),
            )

            semantic_logits, frequency_logits = model.components(images)

            fused_logits = model(images)

            semantic_prob = torch.sigmoid(
                semantic_logits / temperature
            )

            frequency_prob = torch.sigmoid(
                frequency_logits / temperature
            )

            fused_prob = torch.sigmoid(
                fused_logits / temperature
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

            labels_all.extend(labels.tolist())
            generators_all.extend(generators)

    labels = np.asarray(labels_all, dtype=np.int64)
    semantic_scores = np.asarray(semantic_scores)
    frequency_scores = np.asarray(frequency_scores)
    fused_scores = np.asarray(fused_scores)

    print()
    print("Stream results")
    print("------------------------------")
    print(
        f"Semantic stream AUC:  "
        f"{roc_auc(labels, semantic_scores):.6f}"
    )
    print(
        f"Frequency stream AUC: "
        f"{roc_auc(labels, frequency_scores):.6f}"
    )
    print(
        f"Fused stream AUC:     "
        f"{roc_auc(labels, fused_scores):.6f}"
    )

    # Also report each stream by generator.
    unique_generators = sorted(set(generators_all))

    print()
    print("Per-generator results")
    print("------------------------------")

    for generator in unique_generators:
        mask = np.asarray(
            [g == generator for g in generators_all]
        )

        y = labels[mask]

        print(
            f"{generator:12s} "
            f"semantic={roc_auc(y, semantic_scores[mask]):.6f} "
            f"frequency={roc_auc(y, frequency_scores[mask]):.6f} "
            f"fused={roc_auc(y, fused_scores[mask]):.6f}"
        )


if __name__ == "__main__":
    main()
