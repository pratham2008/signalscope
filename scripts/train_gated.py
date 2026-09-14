"""Train the feature-level gated SignalScope detector.

Initialization:
    Load the existing dual-stream baseline and reuse only its
    semantic and frequency encoders.

The new projection, gate, and classifier layers are trained from scratch.
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
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.core.calibration import fit_temperature
from src.core.dataset import (
    ManifestDataset,
    get_dual_train_transform,
    get_dual_val_transform,
)
from src.core.gated_model import GatedDualStreamDetector
from src.core.metrics import choose_threshold_at_fpr, summary


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class TwoItemDataset(Dataset):
    """Expose only image and label from ManifestDataset."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        image, label, _generator = self.dataset[index]
        return image, label


def load_baseline_encoders(
    model: GatedDualStreamDetector,
    baseline_path: str | Path,
) -> dict:
    """Load only semantic/frequency encoder weights from the old model."""

    payload = torch.load(
        baseline_path,
        map_location="cpu",
        weights_only=False,
    )

    if payload.get("model_name") != "dual_stream_detector":
        raise ValueError(
            "Baseline checkpoint is not a dual_stream_detector checkpoint."
        )

    state = payload["state_dict"]

    semantic_state = {
        key[len("semantic_encoder."):]: value
        for key, value in state.items()
        if key.startswith("semantic_encoder.")
    }

    frequency_state = {
        key[len("frequency_encoder."):]: value
        for key, value in state.items()
        if key.startswith("frequency_encoder.")
    }

    if not semantic_state:
        raise RuntimeError(
            "No semantic encoder weights found in baseline checkpoint."
        )

    if not frequency_state:
        raise RuntimeError(
            "No frequency encoder weights found in baseline checkpoint."
        )

    model.semantic_encoder.load_state_dict(
        semantic_state
    )

    model.frequency_encoder.load_state_dict(
        frequency_state
    )

    return payload


def collect(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
):
    """Collect validation logits and average loss."""

    model.eval()

    logits = []
    labels = []
    loss_sum = 0.0
    count = 0

    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(
                device,
                non_blocking=torch.cuda.is_available(),
            )

            batch_labels = batch_labels.float().to(
                device,
                non_blocking=torch.cuda.is_available(),
            )

            batch_logits = model(images)

            loss_sum += (
                criterion(
                    batch_logits,
                    batch_labels,
                ).item()
                * len(images)
            )

            logits.append(
                batch_logits.cpu()
            )

            labels.append(
                batch_labels.cpu()
            )

            count += len(images)

    if not logits:
        raise RuntimeError(
            "Validation loader produced no samples."
        )

    return (
        torch.cat(logits),
        torch.cat(labels),
        loss_sum / max(count, 1),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SignalScope gated feature-level fusion."
    )

    parser.add_argument(
        "--manifest",
        default="data/processed/generalization_manifest.csv",
    )

    parser.add_argument(
        "--baseline",
        default="model/dual/generalization_baseline.pt",
    )

    parser.add_argument(
        "--output",
        default="model/dual/gated_fusion_v1.pt",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
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

    # New gated architecture.
    model = GatedDualStreamDetector(
        pretrained_semantic=False,
        freeze_semantic=False,
    )

    # Reuse the learned encoders from our best baseline.
    baseline_payload = load_baseline_encoders(
        model,
        args.baseline,
    )

    model = model.to(DEVICE)

    train_dataset = TwoItemDataset(
        ManifestDataset(
            args.manifest,
            split="train",
            transform=get_dual_train_transform(),
        )
    )

    val_dataset = TwoItemDataset(
        ManifestDataset(
            args.manifest,
            split="val",
            transform=get_dual_val_transform(),
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(
        f"train={len(train_dataset)} "
        f"validation={len(val_dataset)}"
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=args.lr,
        weight_decay=1e-4,
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
        model.train()

        for images, labels in train_loader:
            images = images.to(
                DEVICE,
                non_blocking=torch.cuda.is_available(),
            )

            labels = labels.float().to(
                DEVICE,
                non_blocking=torch.cuda.is_available(),
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(images)

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()
            optimizer.step()

        logits, labels, val_loss = collect(
            model,
            val_loader,
            DEVICE,
            criterion,
        )

        probabilities = torch.sigmoid(
            logits
        ).numpy()

        metrics = summary(
            labels.numpy(),
            probabilities,
        )

        print(
            f"epoch={epoch}/{args.epochs} "
            f"val_loss={val_loss:.4f} "
            f"auc={metrics['roc_auc']:.6f} "
            f"f1={metrics['macro_f1']:.6f}"
        )

        if metrics["roc_auc"] >= best_auc:
            best_auc = metrics["roc_auc"]

            temperature = fit_temperature(
                logits,
                labels,
            )

            calibrated = torch.sigmoid(
                logits / temperature
            ).numpy()

            threshold = choose_threshold_at_fpr(
                labels.numpy(),
                calibrated,
                args.target_fpr,
            )

            calibration_metrics = summary(
                labels.numpy(),
                calibrated,
                threshold,
            )

            payload = {
                "format_version": 1,
                "model_name": "gated_dual_stream_detector",
                "model_config": model.model_config,
                "state_dict": model.state_dict(),
                "baseline_checkpoint": str(
                    args.baseline
                ),
                "baseline_model_config": baseline_payload.get(
                    "model_config"
                ),
                "temperature": float(
                    temperature
                ),
                "threshold": float(
                    threshold
                ),
                "target_fpr": float(
                    args.target_fpr
                ),
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


if __name__ == "__main__":
    main()
