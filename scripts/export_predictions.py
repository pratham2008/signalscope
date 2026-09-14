"""Run a SignalScope checkpoint across a labeled split and write CSV results.

Supports:
1. CIFAKE-style REAL/FAKE directories.
2. Generator-aware CSV manifests.

In manifest mode, generator names are preserved so we can evaluate
seen and unseen generators separately.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

# Allow execution from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.checkpoint import load_detector
from src.core.dataset import (
    CIFAKEDataset,
    ManifestDataset,
    get_dual_val_transform,
    get_val_transform,
)
from src.core.dual_model import DualStreamDetector


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class ManifestInferenceDataset(Dataset):
    """Return only image and metadata needed for inference."""

    def __init__(self, dataset: ManifestDataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label, generator = self.dataset[index]
        path = self.dataset.samples[index][0]

        return image, label, generator, str(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export labeled inference results to CSV."
    )

    parser.add_argument(
        "--data-dir",
        default=None,
        help="CIFAKE-style directory containing REAL/ and FAKE/.",
    )

    parser.add_argument(
        "--manifest",
        default=None,
        help="Generator-aware CSV manifest.",
    )

    parser.add_argument(
        "--split",
        default="unseen",
        help="Manifest split to evaluate.",
    )

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("report/predictions.csv"),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional samples per class for directory mode.",
    )

    parser.add_argument(
        "--generator",
        default="",
        help="Optional generator name for directory mode.",
    )

    args = parser.parse_args()

    if args.data_dir is None and args.manifest is None:
        parser.error("Provide either --data-dir or --manifest.")

    if args.data_dir is not None and args.manifest is not None:
        parser.error("Use either --data-dir or --manifest, not both.")

    model, metadata = load_detector(
        args.model,
        DEVICE,
    )

    if isinstance(model, DualStreamDetector):
        transform = get_dual_val_transform()
    else:
        transform = get_val_transform()

    if args.manifest:
        base_dataset = ManifestDataset(
            args.manifest,
            split=args.split,
            transform=transform,
        )

        dataset = ManifestInferenceDataset(
            base_dataset
        )

        generator_mode = True

    else:
        dataset = CIFAKEDataset(
            args.data_dir,
            transform,
            args.limit,
        )

        generator_mode = False

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temperature = float(
        metadata.get("temperature", 1.0)
    )

    print(f"device={DEVICE}")
    print(f"model={args.model}")
    print(f"samples={len(dataset)}")
    print(f"temperature={temperature:.6f}")

    with args.output.open(
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
            for batch in loader:

                if generator_mode:
                    (
                        images,
                        labels,
                        generators,
                        image_paths,
                    ) = batch

                    probabilities = torch.sigmoid(
                        model(
                            images.to(
                                DEVICE,
                                non_blocking=torch.cuda.is_available(),
                            )
                        )
                        / temperature
                    ).cpu().tolist()

                    for probability, label, generator, image_path in zip(
                        probabilities,
                        labels.tolist(),
                        generators,
                        image_paths,
                    ):
                        writer.writerow(
                            {
                                "image_id": image_path,
                                "true_label": label,
                                "predicted_prob": probability,
                                "generator": generator,
                            }
                        )

                else:
                    images, labels = batch

                    probabilities = torch.sigmoid(
                        model(
                            images.to(
                                DEVICE,
                                non_blocking=torch.cuda.is_available(),
                            )
                        )
                        / temperature
                    ).cpu().tolist()

                    for probability, label in zip(
                        probabilities,
                        labels.tolist(),
                    ):
                        writer.writerow(
                            {
                                "image_id": "",
                                "true_label": label,
                                "predicted_prob": probability,
                                "generator": args.generator,
                            }
                        )

    print(
        f"Wrote {len(dataset)} predictions to {args.output}"
    )


if __name__ == "__main__":
    main()
