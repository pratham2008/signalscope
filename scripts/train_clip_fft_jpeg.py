"""Train a frozen CLIP + trainable FFT fusion detector.

Controlled baseline:
    train -> CIFAKE + six seen generators
    val   -> CIFAKE + six seen generators
    unseen -> VQDM, never used for training/calibration

CLIP ViT-B/32 is frozen. Its normalized 512-D embeddings are extracted
once and kept in RAM (~78 MB for train + validation).

The FFT branch is trained directly from streamed images and uses the
same deterministic Resize(224) + CenterCrop(224) geometry as CLIP.
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
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.core.calibration import fit_temperature
from src.core.dataset import ManifestDataset
from src.core.dual_model import FrequencyEncoder
from src.core.metrics import choose_threshold_at_fpr, summary


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)


class BaseImageDataset(Dataset):
    """Read manifest images without applying a transform."""

    def __init__(
        self,
        manifest_path: str,
        split: str,
    ) -> None:
        self.dataset = ManifestDataset(
            manifest_path,
            split=split,
            transform=None,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        image, label, generator = self.dataset[index]
        return image, label, generator


class MultiViewTransform:
    """Apply the same deterministic spatial preprocessing to both streams."""

    def __init__(
        self,
        image_size: int = 224,
    ) -> None:
        self.geometry = transforms.Compose([
            transforms.Resize(
                image_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(
                image_size,
            ),
        ])

        self.clip_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

        self.fft_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ])

    def __call__(
        self,
        image: Image.Image,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.geometry(image)

        return (
            self.clip_normalize(image),
            self.fft_normalize(image),
        )


class MultiViewDataset(Dataset):
    """Dataset returning aligned CLIP and FFT views."""

    def __init__(
        self,
        manifest_path: str,
        split: str,
    ) -> None:
        self.dataset = BaseImageDataset(
            manifest_path,
            split,
        )

        self.transform = MultiViewTransform()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        image, label, generator = self.dataset[index]

        clip_image, fft_image = self.transform(
            image
        )

        return (
            clip_image,
            fft_image,
            label,
            generator,
        )


class ClipFFTClassifier(nn.Module):
    """Frozen CLIP embedding + trainable FFT encoder + fusion."""

    def __init__(
        self,
        clip_dim: int = 512,
        frequency_width: int = 32,
    ) -> None:
        super().__init__()

        self.frequency_encoder = FrequencyEncoder(
            frequency_width
        )

        frequency_dim = (
            self.frequency_encoder.output_dim
        )

        self.semantic_projection = nn.Sequential(
            nn.Linear(
                clip_dim,
                128,
            ),
            nn.GELU(),
            nn.Dropout(0.10),
        )

        self.frequency_projection = nn.Sequential(
            nn.Linear(
                frequency_dim,
                128,
            ),
            nn.GELU(),
            nn.Dropout(0.10),
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                128 * 2 + 1,
                64,
            ),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(
                64,
                1,
            ),
        )

        self.model_config = {
            "clip_model": "ViT-B-32",
            "clip_pretrained": "openai",
            "clip_dim": clip_dim,
            "frequency_width": frequency_width,
        }

    def forward(
        self,
        clip_embeddings: torch.Tensor,
        fft_images: torch.Tensor,
    ) -> torch.Tensor:
        semantic = self.semantic_projection(
            clip_embeddings
        )

        frequency_features = self.frequency_encoder(
            fft_images
        )

        frequency = self.frequency_projection(
            frequency_features
        )

        interaction = (
            semantic * frequency
        ).mean(
            dim=1,
            keepdim=True,
        )

        fused = torch.cat(
            (
                semantic,
                frequency,
                interaction,
            ),
            dim=1,
        )

        return self.fusion(
            fused
        ).squeeze(1)


def extract_clip_embeddings(
    clip_model: nn.Module,
    dataset: MultiViewDataset,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract and return frozen normalized CLIP embeddings."""

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    clip_model.eval()

    print(
        f"Extracting CLIP embeddings: "
        f"samples={len(dataset)}"
    )

    with torch.inference_mode():
        for (
            clip_images,
            _fft_images,
            batch_labels,
            _generators,
        ) in loader:
            clip_images = clip_images.to(
                DEVICE,
                non_blocking=torch.cuda.is_available(),
            )

            features = clip_model.encode_image(
                clip_images
            )

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

    return (
        torch.cat(embeddings),
        torch.cat(labels),
    )


class FusionTrainDataset(Dataset):
    """Cached CLIP embeddings + streamed FFT images."""

    def __init__(
        self,
        dataset: MultiViewDataset,
        clip_embeddings: torch.Tensor,
    ) -> None:
        if len(dataset) != len(clip_embeddings):
            raise ValueError(
                "Embedding count does not match dataset length."
            )

        self.dataset = dataset
        self.clip_embeddings = clip_embeddings

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        (
            _clip_image,
            fft_image,
            label,
            _generator,
        ) = self.dataset[index]

        return (
            self.clip_embeddings[index],
            fft_image,
            label,
        )


def evaluate(
    model: nn.Module,
    dataset: MultiViewDataset,
    clip_embeddings: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:

    eval_dataset = FusionTrainDataset(
        dataset,
        clip_embeddings,
    )

    loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    logits: list[torch.Tensor] = []

    model.eval()

    with torch.inference_mode():
        for (
            batch_clip,
            batch_fft,
            _batch_labels,
        ) in loader:

            batch_clip = batch_clip.to(
                DEVICE,
                non_blocking=torch.cuda.is_available(),
            )

            batch_fft = batch_fft.to(
                DEVICE,
                non_blocking=torch.cuda.is_available(),
            )

            logits.append(
                model(
                    batch_clip,
                    batch_fft,
                ).cpu()
            )

    return (
        torch.cat(logits),
        labels.cpu(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train frozen CLIP + trainable FFT fusion."
    )

    parser.add_argument(
        "--manifest",
        default="data/processed/generalization_manifest.csv",
    )

    parser.add_argument(
        "--output",
        default="model/dual/clip_fft_fusion.pt",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--image-batch-size",
        type=int,
        default=16,
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
        "--frequency-width",
        type=int,
        default=32,
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

    print(
        f"device={DEVICE}"
    )

    clip_model, _, _clip_preprocess = (
        open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai",
        )
    )

    clip_model = (
        clip_model
        .eval()
        .to(DEVICE)
    )

    for parameter in clip_model.parameters():
        parameter.requires_grad = False

    clip_dim = int(
        clip_model.visual.output_dim
    )

    print(
        f"CLIP=ViT-B-32 "
        f"embedding_dim={clip_dim}"
    )

    train_dataset = MultiViewDataset(
        args.manifest,
        "train",
    )

    val_dataset = MultiViewDataset(
        args.manifest,
        "val",
    )

    train_clip, train_labels = (
        extract_clip_embeddings(
            clip_model,
            train_dataset,
            args.image_batch_size,
        )
    )

    val_clip, val_labels = (
        extract_clip_embeddings(
            clip_model,
            val_dataset,
            args.image_batch_size,
        )
    )

    print(
        f"train_clip={tuple(train_clip.shape)}"
    )

    print(
        f"val_clip={tuple(val_clip.shape)}"
    )

    train_cached = FusionTrainDataset(
        train_dataset,
        train_clip,
    )

    train_loader = DataLoader(
        train_cached,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = ClipFFTClassifier(
        clip_dim=clip_dim,
        frequency_width=args.frequency_width,
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
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

        running_loss = 0.0

        for (
            batch_clip,
            batch_fft,
            batch_labels,
        ) in train_loader:

            batch_clip = batch_clip.to(
                DEVICE,
                non_blocking=torch.cuda.is_available(),
            )

            batch_fft = batch_fft.to(
                DEVICE,
                non_blocking=torch.cuda.is_available(),
            )

            batch_labels = (
                batch_labels
                .float()
                .to(
                    DEVICE,
                    non_blocking=torch.cuda.is_available(),
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                batch_clip,
                batch_fft,
            )

            loss = criterion(
                logits,
                batch_labels,
            )

            loss.backward()
            optimizer.step()

            running_loss += (
                loss.item()
                * len(batch_labels)
            )

        val_logits, val_targets = evaluate(
            model,
            val_dataset,
            val_clip,
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

        avg_loss = (
            running_loss
            / len(train_cached)
        )

        print(
            f"epoch={epoch}/{args.epochs} "
            f"loss={avg_loss:.6f} "
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
                val_logits
                / temperature
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
                "model_name": "clip_fft_fusion",
                "model_config": model.model_config,
                "temperature": float(
                    temperature
                ),
                "threshold": float(
                    threshold
                ),
                "target_fpr": float(
                    args.target_fpr
                ),
                "validation_metrics": (
                    calibration_metrics
                ),
                "model_state_dict": (
                    model.state_dict()
                ),
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

    print(
        "CLIP + FFT fusion training complete."
    )


if __name__ == "__main__":
    main()
