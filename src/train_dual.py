"""Train SignalScope's dual-stream detector.

Supports:
1. Original CIFAKE directory-based training.
2. Generator-aware CSV manifest training.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.core.calibration import fit_temperature
from src.core.dataset import (
    CIFAKEDataset,
    ManifestDataset,
    get_dual_train_transform,
    get_dual_val_transform,
)
from src.core.dual_model import DualStreamDetector
from src.core.metrics import choose_threshold_at_fpr, summary


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the SignalScope dual-stream detector."
    )

    # Original CIFAKE mode
    parser.add_argument(
        "--train-dir",
        default="data/raw/cifake/train",
        help="CIFAKE-style training directory.",
    )
    parser.add_argument(
        "--val-dir",
        default="data/raw/cifake/test",
        help="CIFAKE-style validation directory.",
    )

    # Generator-aware manifest mode
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional generator-aware CSV manifest.",
    )
    parser.add_argument(
        "--train-split",
        default="train",
        help="Manifest split used for training.",
    )
    parser.add_argument(
        "--val-split",
        default="val",
        help="Manifest split used for validation/calibration.",
    )

    parser.add_argument(
        "--output",
        default="model/dual/signalscope_dual_stream.pt",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
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
        "--train-limit",
        type=int,
        default=None,
        help="Optional CIFAKE samples per class for quick runs.",
    )
    parser.add_argument(
        "--val-limit",
        type=int,
        default=None,
        help="Optional CIFAKE samples per class for quick runs.",
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
    parser.add_argument(
        "--pretrained-semantic",
        action="store_true",
        help="Use ImageNet ResNet weights.",
    )
    parser.add_argument(
        "--freeze-semantic",
        action="store_true",
        help="Freeze the semantic backbone.",
    )

    return parser.parse_args()


class TwoItemDataset(Dataset):
    """Expose only image and label from ManifestDataset."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        image, label, _generator = self.dataset[index]
        return image, label


def collect(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
):
    """Collect validation logits, labels, and average loss."""

    model.eval()

    logits = []
    labels = []
    loss_sum = 0.0
    count = 0

    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)
            batch_labels = batch_labels.float().to(device)

            batch_logits = model(images)

            if criterion is not None:
                loss_sum += (
                    criterion(batch_logits, batch_labels).item()
                    * len(images)
                )

            logits.append(batch_logits.cpu())
            labels.append(batch_labels.cpu())
            count += len(images)

    if not logits:
        raise RuntimeError("Validation loader produced no samples.")

    return (
        torch.cat(logits),
        torch.cat(labels),
        loss_sum / max(count, 1),
    )


def build_datasets(args: argparse.Namespace):
    """Build CIFAKE or manifest-backed datasets."""

    if args.manifest:
        print(f"Using manifest: {args.manifest}")
        print(
            f"Training split: {args.train_split} | "
            f"Validation split: {args.val_split}"
        )

        train_manifest = ManifestDataset(
            args.manifest,
            split=args.train_split,
            transform=get_dual_train_transform(),
        )

        val_manifest = ManifestDataset(
            args.manifest,
            split=args.val_split,
            transform=get_dual_val_transform(),
        )

        train_data = TwoItemDataset(train_manifest)
        val_data = TwoItemDataset(val_manifest)

    else:
        print("Using CIFAKE directory mode.")

        train_data = CIFAKEDataset(
            args.train_dir,
            get_dual_train_transform(),
            args.train_limit,
        )

        val_data = CIFAKEDataset(
            args.val_dir,
            get_dual_val_transform(),
            args.val_limit,
        )

    return train_data, val_data


def main() -> None:
    args = arguments()

    if not 0 <= args.target_fpr < 1:
        raise ValueError("--target-fpr must be in [0, 1)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_data, val_data = build_datasets(args)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = DualStreamDetector(
        args.pretrained_semantic,
        args.freeze_semantic,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=1e-4,
    )

    best_auc = float("-inf")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"device={device}")
    print(f"train={len(train_data)} validation={len(val_data)}")
    print(f"batch_size={args.batch_size}")
    print(f"workers={args.workers}")

    for epoch in range(1, args.epochs + 1):
        model.train()

        for images, labels in train_loader:
            images = images.to(
                device,
                non_blocking=torch.cuda.is_available(),
            )
            labels = labels.float().to(
                device,
                non_blocking=torch.cuda.is_available(),
            )

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

        logits, labels, val_loss = collect(
            model,
            val_loader,
            device,
            criterion,
        )

        probabilities = torch.sigmoid(logits).numpy()

        metrics = summary(
            labels.numpy(),
            probabilities,
        )

        print(
            f"epoch={epoch}/{args.epochs} "
            f"val_loss={val_loss:.4f} "
            f"auc={metrics['roc_auc']:.4f} "
            f"f1={metrics['macro_f1']:.4f}"
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
                "model_name": "dual_stream_detector",
                "model_config": model.model_config,
                "state_dict": model.state_dict(),
                "temperature": temperature,
                "threshold": threshold,
                "target_fpr": args.target_fpr,
                "validation_metrics": calibration_metrics,
            }

            torch.save(payload, output)

            output.with_suffix(".metrics.json").write_text(
                json.dumps(
                    calibration_metrics,
                    indent=2,
                ),
                encoding="utf-8",
            )

            print(
                f"saved={output} "
                f"temperature={temperature:.3f} "
                f"threshold={threshold:.3f}"
            )


if __name__ == "__main__":
    main()
