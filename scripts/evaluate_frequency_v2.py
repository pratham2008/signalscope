"""Evaluate SignalScope Frequency Stream v2 on a manifest split."""

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

from src.core.dataset import ManifestDataset, get_dual_val_transform
from src.core.frequency_v2 import FrequencyEncoderV2


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class FrequencyClassifierV2(torch.nn.Module):
    """Frequency Stream v2 classifier."""

    def __init__(self, width: int = 32) -> None:
        super().__init__()

        self.encoder = FrequencyEncoderV2(
            width=width
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(
                self.encoder.output_dim,
                64,
            ),
            torch.nn.GELU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(64, 1),
        )

    def forward(self, images):
        features = self.encoder(images)

        return self.classifier(
            features
        ).squeeze(1)


class InferenceDataset(Dataset):
    """Add image paths to the manifest dataset."""

    def __init__(self, dataset: ManifestDataset) -> None:
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label, generator = self.dataset[index]
        path = str(
            self.dataset.samples[index][0]
        )

        return image, label, generator, path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Frequency Stream v2."
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
        default="model/frequency/frequency_v2.pt",
    )

    parser.add_argument(
        "--output",
        default="report/frequency_v2_vqdm_predictions.csv",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    args = parser.parse_args()

    payload = torch.load(
        args.model,
        map_location="cpu",
        weights_only=False,
    )

    config = payload.get(
        "model_config",
        {},
    )

    model = FrequencyClassifierV2(
        width=int(
            config.get(
                "frequency_width",
                32,
            )
        )
    )

    model.load_state_dict(
        payload["state_dict"]
    )

    model = model.eval().to(
        DEVICE
    )

    temperature = float(
        payload["temperature"]
    )

    threshold = float(
        payload["threshold"]
    )

    dataset = InferenceDataset(
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

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels_all = []
    probabilities_all = []

    print(f"device={DEVICE}")
    print(f"model={args.model}")
    print(f"split={args.split}")
    print(f"samples={len(dataset)}")
    print(f"temperature={temperature:.6f}")
    print(f"threshold={threshold:.6f}")

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
            for images, labels, generators, paths in loader:
                images = images.to(
                    DEVICE,
                    non_blocking=torch.cuda.is_available(),
                )

                logits = model(images)

                probabilities = torch.sigmoid(
                    logits / temperature
                ).cpu().tolist()

                for probability, label, generator, path in zip(
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

                    labels_all.append(label)
                    probabilities_all.append(probability)

    labels = np.asarray(
        labels_all,
        dtype=np.int64,
    )

    probabilities = np.asarray(
        probabilities_all,
        dtype=np.float64,
    )

    predictions = (
        probabilities >= threshold
    )

    tn = int(
        ((labels == 0) & (predictions == 0)).sum()
    )

    fp = int(
        ((labels == 0) & (predictions == 1)).sum()
    )

    fn = int(
        ((labels == 1) & (predictions == 0)).sum()
    )

    tp = int(
        ((labels == 1) & (predictions == 1)).sum()
    )

    print()
    print("Confusion matrix at frozen threshold")
    print("------------------------------")
    print(f"TN={tn}")
    print(f"FP={fp}")
    print(f"FN={fn}")
    print(f"TP={tp}")

    print()
    print(f"Wrote predictions to {output}")


if __name__ == "__main__":
    main()
