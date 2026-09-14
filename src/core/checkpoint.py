"""Portable checkpoint loading for the current and legacy SignalScope models."""

from __future__ import annotations

from pathlib import Path

import torch

from src.core.dual_model import DualStreamDetector
from src.core.model import CoreDetector


def load_detector(path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and payload.get("model_name") == "dual_stream_detector":
        model = DualStreamDetector(**payload.get("model_config", {})).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model, payload
    # Original project checkpoints were a plain CoreDetector state_dict.
    model = CoreDetector().to(device)
    model.load_state_dict(payload)
    model.eval()
    return model, {"model_name": "legacy_core_detector", "temperature": 1.0, "threshold": 0.5}
