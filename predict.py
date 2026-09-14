"""Single-image SignalScope inference CLI.

The public function retains the required ``predict(image_path) -> (label,
confidence)`` interface. It loads both new and original checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from src.core.checkpoint import load_detector
from src.core.dataset import get_dual_val_transform, get_val_transform
from src.core.dual_model import DualStreamDetector
from src.core.explain import grounded_explanation, overlay, saliency_map


DEFAULT_MODEL_PATH = "model/dual/signalscope_dual_stream.pt"
LEGACY_MODEL_PATH = "model/core/core_detector.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path: str | Path | None = None):
    path = Path(model_path or DEFAULT_MODEL_PATH)
    if not path.exists() and model_path is None:
        path = Path(LEGACY_MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}. Run src/train_dual.py first.")
    return load_detector(path, DEVICE)


def predict_details(image_path: str | Path, model_path: str | Path | None = None) -> dict:
    image = Image.open(image_path).convert("RGB")
    model, metadata = load_model(model_path)
    transform = get_dual_val_transform() if isinstance(model, DualStreamDetector) else get_val_transform()
    tensor = transform(image)
    with torch.inference_mode():
        raw_logit = model(tensor.unsqueeze(0).to(DEVICE)).item()
    probability_fake = float(torch.sigmoid(torch.tensor(raw_logit / float(metadata.get("temperature", 1.0)))).item())
    threshold = float(metadata.get("threshold", 0.5))
    is_fake = probability_fake >= threshold
    return {
        "label": "Likely AI-Generated" if is_fake else "Likely Authentic / Real",
        "confidence": probability_fake if is_fake else 1.0 - probability_fake,
        "probability_ai_generated": probability_fake,
        "threshold": threshold,
        "image": image,
        "tensor": tensor,
        "model": model,
    }


def predict(image_path: str | Path, model_path: str | Path | None = None) -> tuple[str, float]:
    result = predict_details(image_path, model_path)
    return result["label"], result["confidence"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one image as likely real or AI-generated.")
    parser.add_argument("image_path")
    parser.add_argument("--model", default=None)
    parser.add_argument("--explanation-output", type=Path, default=None, help="Optional saliency-overlay PNG path.")
    args = parser.parse_args()
    result = predict_details(args.image_path, args.model)
    print(f"Verdict: {result['label']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"P(AI-generated): {result['probability_ai_generated']:.4f}; threshold: {result['threshold']:.4f}")
    if args.explanation_output:
        heatmap = saliency_map(result["model"], result["tensor"], DEVICE)
        args.explanation_output.parent.mkdir(parents=True, exist_ok=True)
        overlay(result["image"], heatmap).save(args.explanation_output)
        print(grounded_explanation(result["probability_ai_generated"], result["confidence"]))
        print(f"Explanation overlay: {args.explanation_output}")


if __name__ == "__main__":
    main()
