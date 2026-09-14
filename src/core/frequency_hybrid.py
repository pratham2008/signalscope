"""Hybrid frequency representation for SignalScope.

Combines the proven Frequency v1 representation with a spatial
high-pass residual.

Channels:
    3 x normalized log FFT magnitude
    3 x high-pass residual
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def frequency_representation_v1(
    images: torch.Tensor,
) -> torch.Tensor:
    """Return the proven v1 normalized log FFT representation."""

    centered = images - images.mean(
        dim=(-2, -1),
        keepdim=True,
    )

    spectrum = torch.fft.fftshift(
        torch.fft.fft2(
            centered,
            norm="ortho",
        ),
        dim=(-2, -1),
    )

    magnitude = torch.log1p(
        spectrum.abs()
    )

    low = magnitude.amin(
        dim=(-2, -1),
        keepdim=True,
    )

    high = magnitude.amax(
        dim=(-2, -1),
        keepdim=True,
    )

    return (
        magnitude - low
    ) / (
        high - low
    ).clamp_min(1e-6)


def high_pass_residual(
    images: torch.Tensor,
) -> torch.Tensor:
    """Return a simple spatial high-pass residual."""

    blurred = F.avg_pool2d(
        images,
        kernel_size=5,
        stride=1,
        padding=2,
    )

    return images - blurred


def hybrid_frequency_representation(
    images: torch.Tensor,
) -> torch.Tensor:
    """Concatenate v1 FFT evidence and high-pass residual evidence."""

    fft = frequency_representation_v1(
        images
    )

    residual = high_pass_residual(
        images
    )

    residual_mean = residual.mean(
        dim=(-2, -1),
        keepdim=True,
    )

    residual_std = residual.std(
        dim=(-2, -1),
        keepdim=True,
    ).clamp_min(1e-6)

    residual = (
        residual - residual_mean
    ) / residual_std

    return torch.cat(
        (
            fft,
            residual,
        ),
        dim=1,
    )


class HybridFrequencyEncoder(nn.Module):
    """Lightweight CNN over v1 FFT + high-pass residual."""

    def __init__(
        self,
        width: int = 32,
    ) -> None:
        super().__init__()

        channels = (
            width,
            width * 2,
            width * 4,
            width * 4,
        )

        blocks: list[nn.Module] = []

        current = 6

        for output in channels:
            blocks.extend(
                [
                    nn.Conv2d(
                        current,
                        output,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(
                        output
                    ),
                    nn.GELU(),
                    nn.MaxPool2d(2),
                ]
            )

            current = output

        blocks.extend(
            [
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            ]
        )

        self.features = nn.Sequential(
            *blocks
        )

        self.output_dim = channels[-1]

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        return self.features(
            hybrid_frequency_representation(
                images
            )
        )
