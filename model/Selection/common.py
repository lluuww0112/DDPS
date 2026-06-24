from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class PatchSelectionResult:
    selected_indices: torch.Tensor | None = None
    selected_features: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
