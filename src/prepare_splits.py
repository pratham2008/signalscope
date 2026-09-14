from __future__ import annotations

import argparse
import csv
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# These are the Tiny GenImage generator folders you actually downloaded.
GENERATOR_DIRS = {
    "biggan": "imagenet_ai_0419_biggan",
    "vqdm": "imagenet_ai_0419_vqdm",
    "sdv5": "imagenet_ai_0424_sdv5",
    "wukong": "imagenet_ai_0424_wukong",
    "adm": "imagenet_ai_0508_adm",
    "glide": "imagenet_glide",
    "midjourney": "imagenet_midjourney",
}


def image_files(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def add_rows(
    rows: list[dict],
    directory: Path,
    label: int,
    generator: str,
    split: str,
    limit_per_class: int | None = None,
) -> None:
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = image_files(directory)

    if limit_per_class is not None:
        files = files[:limit_per_class]

    for path in files:
        rows.append(
            {
                "path": str(path.resolve()),
                "label": label,
                "generator": generator,
                "split": split,
            }
        )


def build_manifest(
    cifake_root: Path,
    genimage_root: Path,
    output_path: Path,
    seen_generators: list[str],
    cifake_train_limit: int | None,
    genimage_train_limit: int | None,
    genimage_val_limit: int | None,
) -> None:
    rows: list[dict] = []

    # -------------------------
    # CIFAKE
    # -------------------------
    add_rows(
        rows,
        cifake_root / "train" / "REAL",
        label=0,
        generator="cifake",
        split="train",
        limit_per_class=cifake_train_limit,
    )

    add_rows(
        rows,
        cifake_root / "train" / "FAKE",
        label=1,
        generator="cifake",
        split="train",
        limit_per_class=cifake_train_limit,
    )

    # Use CIFAKE test as a normal in-domain validation set.
    add_rows(
        rows,
        cifake_root / "test" / "REAL",
        label=0,
        generator="cifake",
        split="val",
        limit_per_class=cifake_train_limit,
    )

    add_rows(
        rows,
        cifake_root / "test" / "FAKE",
        label=1,
        generator="cifake",
        split="val",
        limit_per_class=cifake_train_limit,
    )

    # -------------------------
    # Tiny GenImage
    # -------------------------
    for generator in seen_generators:
        folder = GENERATOR_DIRS[generator]
        root = genimage_root / folder

        add_rows(
            rows,
            root / "train" / "nature",
            label=0,
            generator=generator,
            split="train",
            limit_per_class=genimage_train_limit,
        )

        add_rows(
            rows,
            root / "train" / "ai",
            label=1,
            generator=generator,
            split="train",
            limit_per_class=genimage_train_limit,
        )

        add_rows(
            rows,
            root / "val" / "nature",
            label=0,
            generator=generator,
            split="val",
            limit_per_class=genimage_val_limit,
        )

        add_rows(
            rows,
            root / "val" / "ai",
            label=1,
            generator=generator,
            split="val",
            limit_per_class=genimage_val_limit,
        )

    # -------------------------
    # VQDM = completely unseen
    # -------------------------
    vqdm_root = genimage_root / GENERATOR_DIRS["vqdm"]

    add_rows(
        rows,
        vqdm_root / "val" / "nature",
        label=0,
        generator="vqdm",
        split="unseen",
    )

    add_rows(
        rows,
        vqdm_root / "val" / "ai",
        label=1,
        generator="vqdm",
        split="unseen",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["path", "label", "generator", "split"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Manifest written to: {output_path}")
    print(f"Total rows: {len(rows)}")

    for split in ("train", "val", "unseen"):
        count = sum(row["split"] == split for row in rows)
        print(f"{split}: {count}")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create generator-aware SignalScope dataset manifests."
    )

    parser.add_argument(
        "--cifake-root",
        default="data/raw/cifake",
    )

    parser.add_argument(
        "--genimage-root",
        default="data/raw/genimage",
    )

    parser.add_argument(
        "--output",
        default="data/processed/generalization_manifest.csv",
    )

    parser.add_argument(
        "--cifake-train-limit",
        type=int,
        default=5000,
        help="CIFAKE samples per class for train and validation.",
    )

    parser.add_argument(
        "--genimage-train-limit",
        type=int,
        default=1000,
        help="Tiny GenImage samples per class per seen generator.",
    )

    parser.add_argument(
        "--genimage-val-limit",
        type=int,
        default=500,
        help="Tiny GenImage validation samples per class per seen generator.",
    )

    parser.add_argument(
        "--seen-generators",
        nargs="+",
        default=["biggan", "sdv5", "wukong", "adm", "glide", "midjourney"],
        choices=list(GENERATOR_DIRS),
    )

    return parser.parse_args()


def main() -> None:
    args = arguments()

    build_manifest(
        cifake_root=Path(args.cifake_root),
        genimage_root=Path(args.genimage_root),
        output_path=Path(args.output),
        seen_generators=args.seen_generators,
        cifake_train_limit=args.cifake_train_limit,
        genimage_train_limit=args.genimage_train_limit,
        genimage_val_limit=args.genimage_val_limit,
    )


if __name__ == "__main__":
    main()
