"""Gated dual-stream detector for SignalScope.

The model keeps the semantic and frequency encoders separate, then learns
a per-image gate over their feature representations before producing the
final binary classification score.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from src.core.dual_model import FrequencyEncoder


class GatedDualStreamDetector(nn.Module):
    """ResNet semantic stream + FFT stream + learned feature-level gating."""

    def __init__(
        self,
        pretrained_semantic: bool = False,
        freeze_semantic: bool = False,
        frequency_width: int = 32,
    ) -> None:
        super().__init__()

        # -------------------------
        # Semantic stream
        # -------------------------
        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained_semantic
            else None
        )

        backbone = resnet18(weights=weights)

        self.semantic_encoder = nn.Sequential(
            *list(backbone.children())[:-1],
            nn.Flatten(),
        )

        self.semantic_dim = 512

        if freeze_semantic:
            for parameter in self.semantic_encoder.parameters():
                parameter.requires_grad = False

        # -------------------------
        # Frequency stream
        # -------------------------
        self.frequency_encoder = FrequencyEncoder(
            frequency_width
        )

        self.frequency_dim = (
            self.frequency_encoder.output_dim
        )

        # -------------------------
        # Per-stream projections
        # -------------------------
        self.semantic_projection = nn.Sequential(
            nn.Linear(self.semantic_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )

        self.frequency_projection = nn.Sequential(
            nn.Linear(self.frequency_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )

        # -------------------------
        # Learned gate
        # -------------------------
        # Produces one value in [0, 1] per image.
        #
        # gate ≈ 1 -> trust semantic stream more
        # gate ≈ 0 -> trust frequency stream more
        self.gate = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # -------------------------
        # Final classifier
        # -------------------------
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )

        self.model_config = {
            "pretrained_semantic": pretrained_semantic,
            "freeze_semantic": freeze_semantic,
            "frequency_width": frequency_width,
        }

    def stream_features(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw semantic and frequency feature vectors."""

        semantic_features = self.semantic_encoder(
            images
        )

        frequency_features = self.frequency_encoder(
            images
        )

        return semantic_features, frequency_features

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        semantic_features, frequency_features = (
            self.stream_features(images)
        )

        semantic_features = self.semantic_projection(
            semantic_features
        )

        frequency_features = self.frequency_projection(
            frequency_features
        )

        combined = torch.cat(
            [
                semantic_features,
                frequency_features,
            ],
            dim=1,
        )

        gate = self.gate(
            combined
        )

        fused = (
            gate * semantic_features
            + (1.0 - gate) * frequency_features
        )

        return self.classifier(
            fused
        ).squeeze(1)

    def gate_values(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Return the learned semantic-trust gate for each image."""

        with torch.no_grad():
            semantic_features, frequency_features = (
                self.stream_features(images)
            )

            semantic_features = self.semantic_projection(
                semantic_features
            )

            frequency_features = self.frequency_projection(
                frequency_features
            )

            combined = torch.cat(
                [
                    semantic_features,
                    frequency_features,
                ],
                dim=1,
            )

            return self.gate(
                combined
            ).squeeze(1)
