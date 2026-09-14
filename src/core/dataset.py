from __future__ import annotations

import io
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class RandomJPEG:
    """PIL-safe JPEG recompression used to reduce post-processing overfitting."""

    def __init__(self, probability: float = 0.5, min_quality: int = 35, max_quality: int = 95):
        self.probability = probability
        self.min_quality = min_quality
        self.max_quality = max_quality

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=random.randint(self.min_quality, self.max_quality))
        buffer.seek(0)
        return Image.open(buffer).convert("RGB").copy()


class CIFAKEDataset(Dataset):
    """
    CIFAKE dataset loader.

    Labels:
        0 = REAL
        1 = FAKE / AI-generated
    """

    def __init__(self, root_dir, transform=None, max_per_class=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.samples = []

        class_map = {
            "REAL": 0,
            "FAKE": 1,
        }

        for class_name, label in class_map.items():
            class_dir = self.root_dir / class_name

            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Dataset directory not found: {class_dir}"
                )

            files = sorted(
                p for p in class_dir.iterdir()
                if p.suffix.lower() in IMAGE_SUFFIXES
            )

            if max_per_class is not None:
                files = files[:max_per_class]

            self.samples.extend((path, label) for path in files)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

class ManifestDataset(Dataset):
    """
    Dataset loader for a generator-aware CSV manifest.

    Expected CSV columns:
        path,generator,label,split

    Labels:
        0 = REAL
        1 = FAKE / AI-generated
    """

    def __init__(
    self,
    manifest_path,
    split,
    transform=None,
    generators=None,
    max_per_class=None,
):
        import csv

        self.manifest_path = Path(manifest_path)
        self.transform = transform
        self.samples = []

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}"
            )

        allowed_generators = set(generators) if generators is not None else None

        with self.manifest_path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:
            reader = csv.DictReader(file)

            required_columns = {"path", "label", "generator", "split"}
            missing = required_columns - set(reader.fieldnames or [])

            if missing:
                raise ValueError(
                    f"Manifest is missing columns: {sorted(missing)}"
                )

            for row in reader:
                if row["split"] != split:
                    continue

                if (
                    allowed_generators is not None
                    and row["generator"] not in allowed_generators
                ):
                    continue

                image_path = Path(row["path"])

                if not image_path.exists():
                    raise FileNotFoundError(
                        f"Image referenced by manifest does not exist: {image_path}"
                    )

                self.samples.append(
                    (
                        image_path,
                        int(row["label"]),
                        row["generator"],
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label, generator = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label, generator

def get_train_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


def get_dual_train_transform(image_size: int = 224):
    """Augmentations that model common social-media and screenshot degradation."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.72, 1.0)),
        transforms.RandomHorizontalFlip(),
        RandomJPEG(),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.2))], p=0.25),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def get_dual_val_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
