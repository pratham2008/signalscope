"""Run one checkpoint across a labeled split and write evaluate.py-compatible CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Allow both `python scripts/export_predictions.py` and module execution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.checkpoint import load_detector
from src.core.dataset import CIFAKEDataset, get_dual_val_transform, get_val_transform
from src.core.dual_model import DualStreamDetector


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export labeled inference results to CSV.")
    parser.add_argument("--data-dir", required=True, help="Directory containing REAL/ and FAKE/.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, default=Path("report/predictions.csv"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="Optional samples per class.")
    parser.add_argument("--generator", default="", help="Optional source/generator name written to the CSV.")
    args = parser.parse_args()
    model, metadata = load_detector(args.model, DEVICE)
    transform = get_dual_val_transform() if isinstance(model, DualStreamDetector) else get_val_transform()
    dataset = CIFAKEDataset(args.data_dir, transform, args.limit)
    loader = DataLoader(dataset, args.batch_size, shuffle=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temperature = float(metadata.get("temperature", 1.0))
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "true_label", "predicted_prob", "generator"])
        writer.writeheader(); index = 0
        with torch.inference_mode():
            for images, labels in loader:
                probabilities = torch.sigmoid(model(images.to(DEVICE)) / temperature).cpu().tolist()
                for value, label in zip(probabilities, labels.tolist()):
                    writer.writerow({"image_id": str(dataset.samples[index][0]), "true_label": label, "predicted_prob": value, "generator": args.generator})
                    index += 1
    print(f"Wrote {len(dataset)} predictions to {args.output}")


if __name__ == "__main__":
    main()
