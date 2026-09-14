import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.core.dataset import (
    CIFAKEDataset,
    get_train_transform,
    get_val_transform,
)
from src.core.model import CoreDetector


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the SignalScope Core Detector."
    )

    parser.add_argument(
        "--train-limit",
        type=int,
        default=5000,
        help="Maximum REAL and FAKE training images each.",
    )

    parser.add_argument(
        "--val-limit",
        type=int,
        default=1000,
        help="Maximum REAL and FAKE validation images each.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )

    return parser.parse_args()


def evaluate(model, loader, criterion, device, dataset_size):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)

            predictions = (torch.sigmoid(logits) >= 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / dataset_size
    accuracy = correct / total

    return avg_loss, accuracy


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Train limit per class:", args.train_limit)
    print("Validation limit per class:", args.val_limit)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Learning rate:", args.lr)

    train_dataset = CIFAKEDataset(
        "data/raw/cifake/train",
        transform=get_train_transform(),
        max_per_class=args.train_limit,
    )

    val_dataset = CIFAKEDataset(
        "data/raw/cifake/test",
        transform=get_val_transform(),
        max_per_class=args.val_limit,
    )

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = CoreDetector().to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    best_val_loss = float("inf")

    Path("model/core").mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()

        total_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * images.size(0)

            predictions = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

        train_loss = total_train_loss / len(train_dataset)
        train_accuracy = train_correct / train_total

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device,
            len(val_dataset),
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                "model/core/core_detector.pt",
            )

            print(
                f"  ✓ Best model saved "
                f"(Val Loss: {val_loss:.4f})"
            )

    print("\nTraining complete.")
    print("Best checkpoint: model/core/core_detector.pt")


if __name__ == "__main__":
    main()
