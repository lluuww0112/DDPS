from __future__ import annotations

from typing import Any

__all__ = [
    "BaseVLM",
    "VLMInterface",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .vlm import BaseVLM, VLMInterface

        exports = {
            "BaseVLM": BaseVLM,
            "VLMInterface": VLMInterface,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
