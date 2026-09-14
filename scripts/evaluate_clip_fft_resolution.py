"""Evaluate CLIP+FFT on lower-resolution inputs without upscaling first."""

from __future__ import annotations

import argparse
import csv
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


class ResolutionDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        split: str,
        max_dimension: int | None,
    ) -> None:
        self.dataset = ManifestDataset(
            manifest_path,
            split=split,
            transform=None,
        )

        self.transform = MultiViewTransform()
        self.max_dimension = max_dimension

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        image, label, generator = self.dataset[index]

        if self.max_dimension is not None:
            width, height = image.size

            scale = min(
                self.max_dimension / width,
                self.max_dimension / height,
                1.0,
            )

            if scale < 1.0:
                new_size = (
                    max(1, round(width * scale)),
                    max(1, round(height * scale)),
                )

                image = image.resize(
                    new_size,
                    Image.Resampling.BICUBIC,
                )

        clip_image, fft_image = self.transform(
            image
        )

        path = str(
            self.dataset.samples[index][0]
        )

        return (
            clip_image,
            fft_image,
            label,
            generator,
            path,
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

    config = payload["model_config"]

    model = ClipFFTClassifier(
        clip_dim=int(config["clip_dim"]),
        frequency_width=int(config["frequency_width"]),
    )

    model.load_state_dict(
        payload["model_state_dict"]
    )

    model = model.eval().to(DEVICE)

    temperature = float(
        payload["temperature"]
    )

    clip_model, _, _ = (
        open_clip.create_model_and_transforms(
            config["clip_model"],
            pretrained=config["clip_pretrained"],
        )
    )

    clip_model = (
        clip_model
        .eval()
        .to(DEVICE)
    )

    for parameter in clip_model.parameters():
        parameter.requires_grad = False

    conditions = {
        "native": None,
        "max_256": 256,
        "max_128": 128,
        "max_64": 64,
    }

    for name, max_dimension in conditions.items():
        dataset = ResolutionDataset(
            args.manifest,
            args.split,
            max_dimension,
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        labels_all = []
        probabilities_all = []

        print()
        print(name)
        print("------------------------------")

        with torch.inference_mode():
            for (
                clip_images,
                fft_images,
                labels,
                _generators,
                _paths,
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
                ).cpu().numpy()

                labels_all.extend(
                    labels.tolist()
                )

                probabilities_all.extend(
                    probabilities.tolist()
                )

        labels_np = np.asarray(
            labels_all,
            dtype=np.int64,
        )

        probabilities_np = np.asarray(
            probabilities_all,
            dtype=np.float64,
        )

        metrics = summary(
            labels_np,
            probabilities_np,
            float(payload["threshold"]),
        )

        print(
            f"ROC-AUC:  {metrics['roc_auc']:.6f}"
        )

        print(
            f"Accuracy: {metrics['accuracy']:.6f}"
        )

        print(
            f"Macro-F1: {metrics['macro_f1']:.6f}"
        )

        print(
            f"FPR:      {metrics['false_positive_rate']:.6f}"
        )


if __name__ == "__main__":
    main()
