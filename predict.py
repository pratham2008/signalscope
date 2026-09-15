"""Single-image SignalScope inference CLI.

The public function retains the required ``predict(image_path) -> (label,
confidence)`` interface. It loads both new and original checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import open_clip
from PIL import Image

from src.core.checkpoint import load_detector
from src.core.dataset import get_dual_val_transform, get_val_transform
from scripts.train_clip_fft import MultiViewTransform
from src.core.dual_model import DualStreamDetector
from src.core.explain import grounded_explanation, overlay, saliency_map, clip_fft_saliency, occlusion_saliency


DEFAULT_MODEL_PATH = "model/dual/signalscope_dual_stream.pt"
LEGACY_MODEL_PATH = "model/core/core_detector.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


_CLIP_CACHE = {}


def load_clip_model(model_name: str, pretrained: str):
    """Load and cache the frozen CLIP model used by ClipFFTClassifier."""
    key = (model_name, pretrained)

    if key not in _CLIP_CACHE:
        clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )

        clip_model = (
            clip_model
            .eval()
            .to(DEVICE)
        )

        for parameter in clip_model.parameters():
            parameter.requires_grad = False

        _CLIP_CACHE[key] = clip_model

    return _CLIP_CACHE[key]


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

    model_name = metadata.get("model_name")

    if model_name in {
        "clip_fft_fusion",
        "clip_fft_no_interaction",
        "clip_fft_hybrid",
    }:
        transform = MultiViewTransform()

        clip_image, fft_image = transform(
            image
        )

        clip_model_name = metadata.get(
            "model_config",
            {},
        ).get(
            "clip_model",
            "ViT-B-32",
        )

        clip_pretrained = metadata.get(
            "model_config",
            {},
        ).get(
            "clip_pretrained",
            "openai",
        )

        clip_model = load_clip_model(
            clip_model_name,
            clip_pretrained,
        )

        with torch.inference_mode():
            clip_features = clip_model.encode_image(
                clip_image.unsqueeze(0).to(DEVICE)
            )

            clip_features = (
                clip_features
                / clip_features.norm(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-8)
            ).float()

            fft_tensor = fft_image.unsqueeze(0).to(
                DEVICE
            )

            raw_logit = model(
                clip_features,
                fft_tensor,
            ).item()

        display_tensor = fft_image

    else:
        transform = (
            get_dual_val_transform()
            if isinstance(model, DualStreamDetector)
            else get_val_transform()
        )

        display_tensor = transform(
            image
        )

        with torch.inference_mode():
            raw_logit = model(
                display_tensor.unsqueeze(0).to(DEVICE)
            ).item()

    temperature = float(
        metadata.get(
            "temperature",
            1.0,
        )
    )

    probability_fake = float(
        torch.sigmoid(
            torch.tensor(
                raw_logit / temperature
            )
        ).item()
    )

    threshold = float(
        metadata.get(
            "threshold",
            0.5,
        )
    )

    is_fake = probability_fake >= threshold

    return {
        "label": (
            "Likely AI-Generated"
            if is_fake
            else "Likely Authentic / Real"
        ),
        "confidence": (
            probability_fake
            if is_fake
            else 1.0 - probability_fake
        ),
        "probability_ai_generated": probability_fake,
        "threshold": threshold,
        "image": image,
        "tensor": display_tensor,
        "model": model,
        "metadata": metadata,
        "model_name": model_name,
        "clip_model": (
            clip_model
            if model_name in {
                "clip_fft_fusion",
                "clip_fft_no_interaction",
                "clip_fft_hybrid",
            }
            else None
        ),
        "clip_tensor": (
            clip_image
            if model_name in {
                "clip_fft_fusion",
                "clip_fft_no_interaction",
                "clip_fft_hybrid",
            }
            else None
        ),
        "fft_tensor": (
            fft_image
            if model_name in {
                "clip_fft_fusion",
                "clip_fft_no_interaction",
                "clip_fft_hybrid",
            }
            else display_tensor
        ),
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
        args.explanation_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if result["model_name"] in {
            "clip_fft_fusion",
        }:
            heatmap, evidence = occlusion_saliency(
                result["model"],
                result["clip_model"],
                result["image"],
                float(
                    result["metadata"].get(
                        "temperature",
                        1.0,
                    )
                ),
                DEVICE,
                grid=7,
                batch_size=16,
            )

            args.explanation_output.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            Image.fromarray(
                heatmap
            ).save(
                args.explanation_output
            )

            print(
                grounded_explanation(
                    result["probability_ai_generated"],
                    result["confidence"],
                    result["threshold"],
                )
            )

            print(
                f"Strongest supporting influence: "
                f"{evidence['max_positive_influence']:.2%}"
            )

            print(
                f"Strongest opposing influence: "
                f"{evidence['max_negative_influence']:.2%}"
            )

            print(
                "Explanation method: "
                "7x7 occlusion attribution"
            )

            print(
                f"Explanation overlay: "
                f"{args.explanation_output}"
            )

        elif result["model_name"] in {
            "clip_fft_no_interaction",
            "clip_fft_hybrid",
        }:
            (
                semantic_map,
                frequency_map,
                combined_map,
            ) = clip_fft_saliency(
                result["model"],
                result["clip_model"],
                result["clip_tensor"],
                result["fft_tensor"],
                DEVICE,
            )

            overlay(
                result["image"],
                combined_map,
            ).save(
                args.explanation_output
            )

            print(
                grounded_explanation(
                    result["probability_ai_generated"],
                    result["confidence"],
                    result["threshold"],
                )
            )

            print(
                f"Combined explanation: "
                f"{args.explanation_output}"
            )

        else:
            heatmap = saliency_map(
                result["model"],
                result["tensor"],
                DEVICE,
            )

            overlay(
                result["image"],
                heatmap,
            ).save(
                args.explanation_output
            )

            print(
                grounded_explanation(
                    result["probability_ai_generated"],
                    result["confidence"],
                    result["threshold"],
                )
            )

            print(
                f"Explanation overlay: "
                f"{args.explanation_output}"
            )



if __name__ == "__main__":
    main()
