"""Evaluate a frozen CLIP ViT-B/32 linear probe on a manifest split."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import open_clip
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.dataset import ManifestDataset


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class ClipProbe(torch.nn.Module):
    """The small trained classifier sitting on top of CLIP embeddings."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.classifier = torch.nn.Linear(
            embedding_dim,
            1,
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings).squeeze(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen CLIP probe on a manifest split."
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
        default="model/clip/clip_vit_b32_probe.pt",
    )

    parser.add_argument(
        "--output",
        default="report/clip_vqdm_predictions.csv",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    args = parser.parse_args()

    # Load the trained probe checkpoint.
    payload = torch.load(
        args.model,
        map_location="cpu",
        weights_only=False,
    )

    embedding_dim = int(
        payload["embedding_dim"]
    )

    temperature = float(
        payload["temperature"]
    )

    threshold = float(
        payload["threshold"]
    )

    probe = ClipProbe(
        embedding_dim
    )

    probe.load_state_dict(
        payload["probe_state_dict"]
    )

    probe = probe.eval().to(
        DEVICE
    )

    # Load the same CLIP model used during training.
    clip_model, _, preprocess = (
        open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai",
        )
    )

    clip_model = clip_model.eval().to(
        DEVICE
    )

    for parameter in clip_model.parameters():
        parameter.requires_grad = False

    dataset = ManifestDataset(
        args.manifest,
        split=args.split,
        transform=preprocess,
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
            for images, labels, generators in loader:

                images = images.to(
                    DEVICE,
                    non_blocking=torch.cuda.is_available(),
                )

                embeddings = clip_model.encode_image(
                    images
                )

                embeddings = embeddings / embeddings.norm(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-8)

                logits = probe(
                    embeddings.float()
                )

                probabilities = torch.sigmoid(
                    logits / temperature
                ).cpu().tolist()

                # ManifestDataset stores:
                # (image_path, label, generator)
                batch_size = len(labels)

                start_index = getattr(
                    main,
                    "_current_index",
                    0,
                )

                for offset in range(batch_size):

                    dataset_index = (
                        start_index + offset
                    )

                    image_path = str(
                        dataset.samples[dataset_index][0]
                    )

                    writer.writerow(
                        {
                            "image_id": image_path,
                            "true_label": labels[offset].item(),
                            "predicted_prob": probabilities[offset],
                            "generator": generators[offset],
                        }
                    )

                main._current_index = (
                    start_index + batch_size
                )

    print(
        f"Wrote predictions to {output}"
    )


if __name__ == "__main__":
    main()
