"""Train a frozen CLIP ViT-B/32 linear probe for SignalScope.

CLIP is frozen. Only the final binary linear classifier is trained.

Data protocol:
    train  -> CIFAKE + six seen generators
    val    -> CIFAKE + six seen generators
    unseen -> VQDM, never used for training/calibration
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import open_clip
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.core.calibration import fit_temperature
from src.core.dataset import ManifestDataset
from src.core.metrics import choose_threshold_at_fpr, summary


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class ClipEmbeddingDataset(Dataset):
    """Return CLIP embeddings and labels."""

    def __init__(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index):
        return self.embeddings[index], self.labels[index]


def extract_embeddings(
    clip_model: nn.Module,
    manifest_path: str,
    split: str,
    preprocess,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract frozen CLIP embeddings for one manifest split."""

    base_dataset = ManifestDataset(
        manifest_path,
        split=split,
        transform=preprocess,
    )

    loader = DataLoader(
        base_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    embeddings = []
    labels = []

    clip_model.eval()

    print(
        f"Extracting CLIP embeddings: "
        f"split={split} samples={len(base_dataset)}"
    )

    with torch.inference_mode():
        for images, batch_labels, _generators in loader:
            images = images.to(
                DEVICE,
                non_blocking=torch.cuda.is_available(),
            )

            features = clip_model.encode_image(images)

            features = features / features.norm(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-8)

            embeddings.append(
                features.float().cpu()
            )

            labels.append(
                batch_labels.float().cpu()
            )

    if not embeddings:
        raise RuntimeError(
            f"No samples found for split '{split}'."
        )

    return (
        torch.cat(embeddings),
        torch.cat(labels),
    )


class LinearProbe(nn.Module):
    """Binary linear classifier over frozen CLIP embeddings."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(
            embedding_dim,
            1,
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(
            embeddings
        ).squeeze(1)


def evaluate_probe(
    model: nn.Module,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return logits and labels from the probe."""

    dataset = ClipEmbeddingDataset(
        embeddings,
        labels,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    logits = []

    model.eval()

    with torch.inference_mode():
        for batch_embeddings, _batch_labels in loader:
            batch_embeddings = batch_embeddings.to(
                DEVICE,
                non_blocking=torch.cuda.is_available(),
            )

            logits.append(
                model(batch_embeddings).cpu()
            )

    return (
        torch.cat(logits),
        labels.cpu(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a frozen CLIP ViT-B/32 linear probe."
    )

    parser.add_argument(
        "--manifest",
        default="data/processed/generalization_manifest.csv",
    )

    parser.add_argument(
        "--output",
        default="model/clip/clip_vit_b32_probe.pt",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Embedding-batch size for the linear probe.",
    )

    parser.add_argument(
        "--image-batch-size",
        type=int,
        default=32,
        help="Image batch size during CLIP feature extraction.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--target-fpr",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if not 0 <= args.target_fpr < 1:
        raise ValueError(
            "--target-fpr must be in [0, 1)"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"device={DEVICE}")

    # Load the exact CLIP model we tested successfully.
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
    )

    clip_model = clip_model.eval().to(DEVICE)

    for parameter in clip_model.parameters():
        parameter.requires_grad = False

    embedding_dim = int(
        clip_model.visual.output_dim
    )

    print(
        f"CLIP=ViT-B-32 embedding_dim={embedding_dim}"
    )

    # Extract embeddings from seen train and validation data only.
    train_embeddings, train_labels = extract_embeddings(
        clip_model,
        args.manifest,
        "train",
        preprocess,
        args.image_batch_size,
    )

    val_embeddings, val_labels = extract_embeddings(
        clip_model,
        args.manifest,
        "val",
        preprocess,
        args.image_batch_size,
    )

    print(
        f"train_embeddings={tuple(train_embeddings.shape)}"
    )
    print(
        f"val_embeddings={tuple(val_embeddings.shape)}"
    )

    # The probe is tiny: only 512 weights + 1 bias.
    probe = LinearProbe(
        embedding_dim
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    train_dataset = ClipEmbeddingDataset(
        train_embeddings,
        train_labels,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    best_auc = float("-inf")

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        probe.train()

        for batch_embeddings, batch_labels in train_loader:
            batch_embeddings = batch_embeddings.to(
                DEVICE,
                non_blocking=torch.cuda.is_available(),
            )

            batch_labels = batch_labels.float().to(
                DEVICE,
                non_blocking=torch.cuda.is_available(),
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = probe(
                batch_embeddings
            )

            loss = criterion(
                logits,
                batch_labels,
            )

            loss.backward()
            optimizer.step()

        val_logits, val_targets = evaluate_probe(
            probe,
            val_embeddings,
            val_labels,
            args.batch_size,
        )

        probabilities = torch.sigmoid(
            val_logits
        ).numpy()

        metrics = summary(
            val_targets.numpy(),
            probabilities,
        )

        print(
            f"epoch={epoch}/{args.epochs} "
            f"val_auc={metrics['roc_auc']:.6f} "
            f"val_f1={metrics['macro_f1']:.6f}"
        )

        if metrics["roc_auc"] >= best_auc:
            best_auc = metrics["roc_auc"]

            temperature = fit_temperature(
                val_logits,
                val_targets,
            )

            calibrated = torch.sigmoid(
                val_logits / temperature
            ).numpy()

            threshold = choose_threshold_at_fpr(
                val_targets.numpy(),
                calibrated,
                args.target_fpr,
            )

            calibration_metrics = summary(
                val_targets.numpy(),
                calibrated,
                threshold,
            )

            payload = {
                "format_version": 1,
                "model_name": "clip_vit_b32_linear_probe",
                "clip_model": "ViT-B-32",
                "pretrained": "openai",
                "embedding_dim": embedding_dim,
                "probe_state_dict": probe.state_dict(),
                "temperature": float(temperature),
                "threshold": float(threshold),
                "target_fpr": float(args.target_fpr),
                "validation_metrics": calibration_metrics,
            }

            torch.save(
                payload,
                output,
            )

            output.with_suffix(
                ".metrics.json"
            ).write_text(
                json.dumps(
                    calibration_metrics,
                    indent=2,
                ),
                encoding="utf-8",
            )

            print(
                f"saved={output} "
                f"temperature={temperature:.6f} "
                f"threshold={threshold:.6f}"
            )

    print("CLIP probe training complete.")


if __name__ == "__main__":
    main()
