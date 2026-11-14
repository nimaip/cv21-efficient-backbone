from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ForwardResult:
    """Logits plus optional regularizers from a backbone forward pass."""

    logits: Tensor
    regularizers: dict[str, Tensor] = field(default_factory=dict)
    loss_logits: Tensor | None = None
    debug: dict = field(default_factory=dict)

    def loss(self, targets: Tensor, label_smoothing: float = 0.0) -> Tensor:
        predictions = self.logits if self.loss_logits is None else self.loss_logits
        total = F.cross_entropy(
            predictions, targets, label_smoothing=label_smoothing
        )
        for value in self.regularizers.values():
            total = total + value
        return total
