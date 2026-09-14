"""Temperature scaling fitted only on validation logits."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_steps: int = 50) -> float:
    logits, labels = logits.detach().float().cpu(), labels.detach().float().cpu()
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_steps, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits / log_temperature.exp().clamp(0.05, 20), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().detach().clamp(0.05, 20).item())
