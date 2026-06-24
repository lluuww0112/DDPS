from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from PIL import Image

from ..common import PatchSelectionResult


def _resolve_budget(
    token_count: int,
    *,
    budget: int | None = None,
    keep_ratio: float | None = None,
) -> int:
    if token_count <= 0:
        return 0
    if budget is not None:
        return max(1, min(int(budget), token_count))
    if keep_ratio is None:
        return token_count
    if keep_ratio <= 0.0 or keep_ratio > 1.0:
        raise ValueError(f"`keep_ratio` must be in (0, 1], got {keep_ratio}.")
    return max(1, min(int(math.ceil(token_count * float(keep_ratio))), token_count))


def _resolve_sample_key(
    *,
    sample_key: str | None,
    visual_metadata: Mapping[str, Any] | None,
    image: Image.Image | None,
) -> str | None:
    if sample_key is not None and str(sample_key).strip():
        return str(sample_key).strip()

    if visual_metadata is not None:
        for key in ("image_path", "path", "image_id", "sample_id", "id"):
            value = visual_metadata.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()

    if image is not None:
        return f"image:{image.width}x{image.height}"
    return None


def _resolve_generator_seed(
    *,
    seed: int | None,
    per_sample_seed: bool,
    sample_key: str | None,
) -> int | None:
    if seed is None:
        return None

    resolved_seed = int(seed)
    if not per_sample_seed or sample_key is None:
        return resolved_seed

    digest = hashlib.blake2b(
        f"{resolved_seed}:{sample_key}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _sample_patch_indices(
    token_count: int,
    *,
    keep_count: int,
    device: torch.device,
    seed: int | None,
) -> torch.Tensor:
    if keep_count >= token_count:
        return torch.arange(token_count, device=device, dtype=torch.long)

    if seed is None:
        indices = torch.randperm(token_count)[:keep_count].sort().values
    else:
        generator = torch.Generator()
        generator.manual_seed(seed)
        indices = torch.randperm(token_count, generator=generator)[:keep_count].sort().values
    return indices.to(device=device, dtype=torch.long)


def uniform_random_patch_selection(
    image_features: torch.Tensor,
    *,
    budget: int | None = None,
    keep_ratio: float | None = None,
    seed: int | None = 0,
    per_sample_seed: bool = True,
    sample_key: str | None = None,
    visual_metadata: Mapping[str, Any] | None = None,
    image: Image.Image | None = None,
    **_: Any,
) -> PatchSelectionResult:
    token_count = int(image_features.shape[0])
    keep_count = _resolve_budget(
        token_count,
        budget=budget,
        keep_ratio=keep_ratio,
    )
    resolved_sample_key = _resolve_sample_key(
        sample_key=sample_key,
        visual_metadata=visual_metadata,
        image=image,
    )
    resolved_seed = _resolve_generator_seed(
        seed=seed,
        per_sample_seed=per_sample_seed,
        sample_key=resolved_sample_key,
    )
    selected_indices = _sample_patch_indices(
        token_count,
        keep_count=keep_count,
        device=image_features.device,
        seed=resolved_seed,
    )
    return PatchSelectionResult(
        selected_indices=selected_indices,
        metadata={
            "selector_type": "uniform_random_patch_selection",
            "selection_mode": "uniform_without_replacement",
            "budget": None if budget is None else int(budget),
            "keep_ratio": None if keep_ratio is None else float(keep_ratio),
            "keep_count": int(keep_count),
            "seed": None if seed is None else int(seed),
            "resolved_seed": None if resolved_seed is None else int(resolved_seed),
            "per_sample_seed": bool(per_sample_seed),
            "sample_key": resolved_sample_key,
            "selected_token_count": int(selected_indices.numel()),
            "image_token_count": token_count,
        },
    )


def uniform_random_patch_selection_batch(
    image_features: torch.Tensor,
    *,
    budget: int | None = None,
    keep_ratio: float | None = None,
    seed: int | None = 0,
    per_sample_seed: bool = True,
    sample_keys: Sequence[str | None] | None = None,
    visual_metadata: Sequence[Mapping[str, Any] | None] | None = None,
    images: Sequence[Image.Image | None] | None = None,
    **_: Any,
) -> list[PatchSelectionResult]:
    if image_features.ndim != 3:
        raise ValueError(
            "Expected batched image features with shape (B, N, D), "
            f"but got {tuple(image_features.shape)}."
        )

    batch_size = int(image_features.shape[0])
    if sample_keys is not None and len(sample_keys) != batch_size:
        raise ValueError(
            "`sample_keys` batch length must match image features: "
            f"sample_keys={len(sample_keys)}, batch_size={batch_size}."
        )

    metadata_batch = [None] * batch_size if visual_metadata is None else list(visual_metadata)
    image_batch = [None] * batch_size if images is None else list(images)
    if len(metadata_batch) != batch_size:
        raise ValueError(
            "`visual_metadata` batch length must match image features: "
            f"visual_metadata={len(metadata_batch)}, batch_size={batch_size}."
        )
    if len(image_batch) != batch_size:
        raise ValueError(
            "`images` batch length must match image features: "
            f"images={len(image_batch)}, batch_size={batch_size}."
        )

    results: list[PatchSelectionResult] = []
    for item_index in range(batch_size):
        item_result = uniform_random_patch_selection(
            image_features[item_index],
            budget=budget,
            keep_ratio=keep_ratio,
            seed=seed,
            per_sample_seed=per_sample_seed,
            sample_key=(sample_keys[item_index] if sample_keys is not None else None),
            visual_metadata=metadata_batch[item_index],
            image=image_batch[item_index],
        )
        results.append(
            PatchSelectionResult(
                selected_indices=item_result.selected_indices,
                selected_features=item_result.selected_features,
                metadata={
                    **item_result.metadata,
                    "batch_index": item_index,
                },
            )
        )
    return results


random_patch_selection = uniform_random_patch_selection
random_patch_selection_batch = uniform_random_patch_selection_batch
uniform_random_patch_selection.batch = uniform_random_patch_selection_batch
random_patch_selection.batch = random_patch_selection_batch
