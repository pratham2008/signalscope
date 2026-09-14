"""Frequency Stream v2 for SignalScope.

Uses two complementary forms of low-level evidence:

1. Log FFT magnitude maps.
2. High-pass residual maps.

The two representations are concatenated and processed by a lightweight CNN.
"""

from __future__ import annotations

import torch
from torch import nn


def fft_magnitude(images: torch.Tensor) -> torch.Tensor:
    """Return log-scaled FFT magnitude maps without per-image min-max scaling."""

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

    return magnitude


def high_pass_residual(images: torch.Tensor) -> torch.Tensor:
    """Extract a simple high-pass residual from the image."""

    blurred = torch.nn.functional.avg_pool2d(
        images,
        kernel_size=5,
        stride=1,
        padding=2,
    )

    residual = images - blurred

    return residual


def frequency_representation_v2(
    images: torch.Tensor,
) -> torch.Tensor:
    """Combine FFT magnitude and high-pass residual evidence."""

    fft = fft_magnitude(images)

    residual = high_pass_residual(images)

    # Normalize each representation independently over spatial dimensions.
    fft_mean = fft.mean(
        dim=(-2, -1),
        keepdim=True,
    )
    fft_std = fft.std(
        dim=(-2, -1),
        keepdim=True,
    ).clamp_min(1e-6)

    residual_mean = residual.mean(
        dim=(-2, -1),
        keepdim=True,
    )
    residual_std = residual.std(
        dim=(-2, -1),
        keepdim=True,
    ).clamp_min(1e-6)

    fft = (fft - fft_mean) / fft_std
    residual = (residual - residual_mean) / residual_std

    return torch.cat(
        [
            fft,
            residual,
        ],
        dim=1,
    )


class FrequencyEncoderV2(nn.Module):
    """Lightweight CNN operating on FFT + high-pass representations."""

    def __init__(self, width: int = 32) -> None:
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
                    nn.BatchNorm2d(output),
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
        representation = frequency_representation_v2(
            images
        )

        return self.features(
            representation
        )
