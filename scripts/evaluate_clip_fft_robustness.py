"""Evaluate CLIP+FFT SignalScope under controlled JPEG degradation."""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_clip_fft import ClipFFTClassifier, MultiViewTransform
from src.core.dataset import ManifestDataset
from src.core.metrics import summary


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def resize_roundtrip(
    image: Image.Image,
    size: int,
) -> Image.Image:
    """Downscale while preserving aspect ratio, then restore original size."""
    original_width, original_height = image.size

    scale = min(
        size / original_width,
        size / original_height,
    )

    new_width = max(
        1,
        round(original_width * scale),
    )

    new_height = max(
        1,
        round(original_height * scale),
    )

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.BICUBIC,
    )

    image = image.resize(
        (original_width, original_height),
        Image.Resampling.BICUBIC,
    )

    return image


class RobustnessDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        split: str,
        quality: int | None,
        resize_size: int | None,
    ) -> None:
        self.dataset = ManifestDataset(
            manifest_path,
            split=split,
            transform=None,
        )
        self.transform = MultiViewTransform()
        self.quality = quality
        self.resize_size = resize_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        image, label, generator = self.dataset[index]

        if self.quality is not None:
            buffer = io.BytesIO()
            image.save(
                buffer,
                format="JPEG",
                quality=self.quality,
            )
            buffer.seek(0)

            image = Image.open(buffer).convert(
                "RGB"
            ).copy()

        if self.resize_size is not None:
            image = resize_roundtrip(
                image,
                self.resize_size,
            )

        clip_image, fft_image = self.transform(
            image
        )

        image_path = str(
            self.dataset.samples[index][0]
        )

        return (
            clip_image,
            fft_image,
            label,
            generator,
            image_path,
        )


def evaluate_condition(
    *,
    clip_model,
    model,
    manifest: str,
    split: str,
    quality: int | None,
    resize_size: int | None,
    temperature: float,
    threshold: float,
    batch_size: int,
    output_path: Path,
) -> dict:

    dataset = RobustnessDataset(
        manifest,
        split,
        quality,
        resize_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    labels_all = []
    probabilities_all = []

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
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
            for (
                clip_images,
                fft_images,
                labels,
                generators,
                paths,
            ) in loader:

                clip_images = clip_images.to(
                    DEVICE,
                    non_blocking=torch.cuda.is_available(),
                )

                fft_images = fft_images.to(
                    DEVICE,
                    non_blocking=torch.cuda.is_available(),
                )

                clip_features = clip_model.encode_image(
                    clip_images
                )

                clip_features = (
                    clip_features
                    / clip_features.norm(
                        dim=-1,
                        keepdim=True,
                    ).clamp_min(1e-8)
                ).float()

                logits = model(
                    clip_features,
                    fft_images,
                )

                probabilities = torch.sigmoid(
                    logits / temperature
                ).cpu().tolist()

                for (
                    probability,
                    label,
                    generator,
                    path,
                ) in zip(
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

    labels_np = np.asarray(
        labels_all,
        dtype=np.int64,
    )

    probabilities_np = np.asarray(
        probabilities_all,
        dtype=np.float64,
    )

    return summary(
        labels_np,
        probabilities_np,
        threshold,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

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
        default="model/dual/clip_fft_jpeg.pt",
    )

    parser.add_argument(
        "--output-dir",
        default="report/robustness_jpeg",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    args = parser.parse_args()

    payload = torch.load(
        args.model,
        map_location="cpu",
        weights_only=False,
    )

    model_config = payload["model_config"]

    model = ClipFFTClassifier(
        clip_dim=int(
            model_config.get(
                "clip_dim",
                512,
            )
        ),
        frequency_width=int(
            model_config.get(
                "frequency_width",
                32,
            )
        ),
    )

    model.load_state_dict(
        payload["model_state_dict"]
    )

    model = model.eval().to(DEVICE)

    temperature = float(
        payload["temperature"]
    )

    threshold = float(
        payload["threshold"]
    )

    clip_model, _, _ = (
        open_clip.create_model_and_transforms(
            model_config["clip_model"],
            pretrained=model_config["clip_pretrained"],
        )
    )

    clip_model = (
        clip_model
        .eval()
        .to(DEVICE)
    )

    for parameter in clip_model.parameters():
        parameter.requires_grad = False

    output_dir = Path(
        args.output_dir
    )

    conditions = {
        "clean": (None, None),
        "resize_256": (None, 256),
        "resize_128": (None, 128),
        "resize_64": (None, 64),
    }

    print(f"device={DEVICE}")
    print(f"model={args.model}")
    print(f"split={args.split}")
    print(f"temperature={temperature:.6f}")
    print(f"threshold={threshold:.6f}")

    for name, (quality, resize_size) in conditions.items():
        output_path = (
            output_dir
            / f"{name}_predictions.csv"
        )

        metrics = evaluate_condition(
            clip_model=clip_model,
            model=model,
            manifest=args.manifest,
            split=args.split,
            quality=quality,
            resize_size=resize_size,
            temperature=temperature,
            threshold=threshold,
            batch_size=args.batch_size,
            output_path=output_path,
        )

        print()
        print(name)
        print("------------------------------")
        print(
            f"ROC-AUC:      {metrics['roc_auc']:.6f}"
        )
        print(
            f"Accuracy:     {metrics['accuracy']:.6f}"
        )
        print(
            f"Macro-F1:     {metrics['macro_f1']:.6f}"
        )
        print(
            f"False-positive rate: "
            f"{metrics['false_positive_rate']:.6f}"
        )
        print(
            f"TN={metrics['confusion_matrix']['tn']} "
            f"FP={metrics['confusion_matrix']['fp']} "
            f"FN={metrics['confusion_matrix']['fn']} "
            f"TP={metrics['confusion_matrix']['tp']}"
        )
        print(
            f"predictions={output_path}"
        )


if __name__ == "__main__":
    main()
