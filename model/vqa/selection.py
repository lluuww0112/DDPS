from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(slots=True)
class ImageSelectionResult:
    image: Image.Image
    metadata: dict[str, Any] = field(default_factory=dict)


def _resize_image(image: Image.Image, max_side: int | None) -> Image.Image:
    if max_side is None:
        return image
    if max_side <= 0:
        raise ValueError(f"`max_side` must be positive, got {max_side}.")

    width, height = image.size
    scale = min(float(max_side) / max(width, height), 1.0)
    if scale >= 1.0:
        return image

    resized_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return image.resize(resized_size, Image.Resampling.BICUBIC)


def load_image(
    image_path: str,
    *,
    max_side: int | None = None,
) -> ImageSelectionResult:
    path = Path(image_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    image = Image.open(path).convert("RGB")
    original_size = image.size
    image = _resize_image(image, max_side=max_side)
    return ImageSelectionResult(
        image=image,
        metadata={
            "image_path": str(path),
            "original_size": list(original_size),
            "image_size": list(image.size),
            "max_side": max_side,
        },
    )


__all__ = [
    "ImageSelectionResult",
    "load_image",
]
