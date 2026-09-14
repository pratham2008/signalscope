"""Generator-invariant, dual-stream image detector.

The semantic stream and FFT stream intentionally observe different evidence.
They are fused from logits, not probabilities, so calibration can be applied once
to the combined decision score.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def frequency_representation(images: torch.Tensor) -> torch.Tensor:
    """Return per-channel, normalized log FFT magnitude maps for RGB batches."""
    centered = images - images.mean(dim=(-2, -1), keepdim=True)
    spectrum = torch.fft.fftshift(torch.fft.fft2(centered, norm="ortho"), dim=(-2, -1))
    magnitude = torch.log1p(spectrum.abs())
    low = magnitude.amin(dim=(-2, -1), keepdim=True)
    high = magnitude.amax(dim=(-2, -1), keepdim=True)
    return (magnitude - low) / (high - low).clamp_min(1e-6)


class FrequencyEncoder(nn.Module):
    def __init__(self, width: int = 32) -> None:
        super().__init__()
        channels = (width, width * 2, width * 4, width * 4)
        blocks: list[nn.Module] = []
        current = 3
        for output in channels:
            blocks.extend([
                nn.Conv2d(current, output, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(output),
                nn.GELU(),
                nn.MaxPool2d(2),
            ])
            current = output
        blocks.extend([nn.AdaptiveAvgPool2d(1), nn.Flatten()])
        self.features = nn.Sequential(*blocks)
        self.output_dim = channels[-1]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.features(frequency_representation(images))


class DualStreamDetector(nn.Module):
    """ResNet semantic probe + FFT CNN with a learned interaction gate."""

    def __init__(self, pretrained_semantic: bool = False, freeze_semantic: bool = False, frequency_width: int = 32) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained_semantic else None
        backbone = resnet18(weights=weights)
        self.semantic_encoder = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
        self.semantic_head = nn.Linear(512, 1)
        self.frequency_encoder = FrequencyEncoder(frequency_width)
        self.frequency_head = nn.Linear(self.frequency_encoder.output_dim, 1)
        self.fusion = nn.Sequential(
            nn.Linear(3, 16), nn.GELU(), nn.Dropout(0.15), nn.Linear(16, 1)
        )
        self.model_config = {
            "pretrained_semantic": pretrained_semantic,
            "freeze_semantic": freeze_semantic,
            "frequency_width": frequency_width,
        }
        if freeze_semantic:
            for parameter in self.semantic_encoder.parameters():
                parameter.requires_grad = False

    def components(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        semantic_logit = self.semantic_head(self.semantic_encoder(images)).squeeze(1)
        frequency_logit = self.frequency_head(self.frequency_encoder(images)).squeeze(1)
        return semantic_logit, frequency_logit

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        semantic_logit, frequency_logit = self.components(images)
        features = torch.stack((semantic_logit, frequency_logit, semantic_logit * frequency_logit), dim=1)
        return self.fusion(features).squeeze(1)
