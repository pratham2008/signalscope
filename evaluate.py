"""Evaluate SignalScope predictions from a portable CSV file.

Expected columns: image_id,true_label,predicted_prob
Optional column: generator (for per-generator metrics).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.core.metrics import choose_threshold_at_fpr, roc_auc, summary


def label(value: str) -> int:
    normalized = value.strip().lower()
    if normalized in {"1", "fake", "synthetic", "ai", "ai-generated", "generated"}:
        return 1
    if normalized in {"0", "real", "authentic"}:
        return 0
    raise ValueError(f"unrecognized true_label: {value!r}")


def load_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_id", "true_label", "predicted_prob"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} must have columns: {', '.join(sorted(required))}")
    return rows, [label(row["true_label"]) for row in rows], [float(row["predicted_prob"]) for row in rows]


def bootstrap_auc(labels: list[int], probabilities: list[float], rounds: int, seed: int) -> list[float] | None:
    if rounds <= 0 or len(set(labels)) < 2:
        return None
    rng, y, p, scores = np.random.default_rng(seed), np.asarray(labels), np.asarray(probabilities), []
    for _ in range(rounds):
        selection = rng.integers(0, len(y), len(y))
        if len(np.unique(y[selection])) == 2:
            scores.append(roc_auc(y[selection], p[selection]))
    if not scores:
        return None
    return [float(np.quantile(scores, 0.025)), float(np.quantile(scores, 0.975))]


def create_plots(labels, probabilities, threshold: float, output: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    y, p = np.asarray(labels), np.asarray(probabilities)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    thresholds = np.r_[np.inf, np.sort(np.unique(p))[::-1], -np.inf]
    fpr = [((p >= value) & (y == 0)).sum() / max((y == 0).sum(), 1) for value in thresholds]
    tpr = [((p >= value) & (y == 1)).sum() / max((y == 1).sum(), 1) for value in thresholds]
    plt.figure(); plt.plot(fpr, tpr, label=f"AUC = {roc_auc(y, p):.3f}"); plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.legend(); plt.tight_layout()
    roc_path = output / "roc_curve.png"; plt.savefig(roc_path, dpi=160); plt.close(); paths.append(str(roc_path))
    predicted = p >= threshold
    matrix = np.array([[((~predicted) & (y == 0)).sum(), (predicted & (y == 0)).sum()], [((~predicted) & (y == 1)).sum(), (predicted & (y == 1)).sum()]])
    plt.figure(); plt.imshow(matrix, cmap="Blues"); plt.xticks([0, 1], ["Real", "Synthetic"]); plt.yticks([0, 1], ["Real", "Synthetic"])
    for row in range(2):
        for column in range(2): plt.text(column, row, str(matrix[row, column]), ha="center", va="center")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    cm_path = output / "confusion_matrix.png"; plt.savefig(cm_path, dpi=160); plt.close(); paths.append(str(cm_path))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Produce SignalScope evaluation metrics and figures.")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("report/evaluation"))
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold; defaults to validation FPR policy.")
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--bootstrap-rounds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows, labels, probabilities = load_csv(args.predictions)
    threshold = args.threshold if args.threshold is not None else choose_threshold_at_fpr(labels, probabilities, args.target_fpr)
    result = summary(labels, probabilities, threshold)
    result["auc_95_ci"] = bootstrap_auc(labels, probabilities, args.bootstrap_rounds, args.seed)
    if "generator" in rows[0]:
        result["per_generator_auc"] = {}
        for generator in sorted({row.get("generator", "") for row in rows}):
            subset_labels = [label(row["true_label"]) for row in rows if row.get("generator") == generator]
            subset_probabilities = [float(row["predicted_prob"]) for row in rows if row.get("generator") == generator]
            result["per_generator_auc"][generator] = roc_auc(subset_labels, subset_probabilities) if len(set(subset_labels)) == 2 else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result["figures"] = create_plots(labels, probabilities, threshold, args.output_dir)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    print(f"\nSaved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
