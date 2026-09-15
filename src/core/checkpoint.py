"""Portable checkpoint loading for the current and legacy SignalScope models."""

from __future__ import annotations

from pathlib import Path

import torch

from src.core.dual_model import DualStreamDetector
from src.core.model import CoreDetector
from scripts.train_clip_fft import ClipFFTClassifier


def load_detector(path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    payload = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(payload, dict):
        # Original project checkpoints were plain CoreDetector state_dicts.
        model = CoreDetector().to(device)
        model.load_state_dict(payload)
        model.eval()
        return model, {
            "model_name": "legacy_core_detector",
            "temperature": 1.0,
            "threshold": 0.5,
        }

    model_name = payload.get("model_name")

    if model_name == "dual_stream_detector":
        model = DualStreamDetector(
            **payload.get("model_config", {})
        ).to(device)

        model.load_state_dict(
            payload["state_dict"]
        )

        model.eval()
        return model, payload

    if model_name in {
        "clip_fft_fusion",
        "clip_fft_no_interaction",
        "clip_fft_hybrid",
    }:
        config = payload.get(
            "model_config",
            {},
        )

        model = ClipFFTClassifier(
            clip_dim=int(
                config.get(
                    "clip_dim",
                    512,
                )
            ),
            frequency_width=int(
                config.get(
                    "frequency_width",
                    32,
                )
            ),
        )

        if model_name == "clip_fft_hybrid":
            from src.core.frequency_hybrid import HybridFrequencyEncoder

            model.frequency_encoder = HybridFrequencyEncoder(
                int(
                    config.get(
                        "frequency_width",
                        32,
                    )
                )
            )

        if model_name == "clip_fft_no_interaction":
            model.fusion = torch.nn.Sequential(
                torch.nn.Linear(256, 64),
                torch.nn.GELU(),
                torch.nn.Dropout(0.15),
                torch.nn.Linear(64, 1),
            )

        model.load_state_dict(
            payload["model_state_dict"]
        )

        model.eval().to(device)
        return model, payload

    # Unknown dictionary checkpoint: preserve the previous legacy behavior.
    model = CoreDetector().to(device)
    model.load_state_dict(payload)
    model.eval()

    return model, {
        "model_name": "legacy_core_detector",
        "temperature": 1.0,
        "threshold": 0.5,
    }
