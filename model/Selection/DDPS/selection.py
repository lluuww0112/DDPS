from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open
from transformers import AutoTokenizer, CLIPImageProcessor
from transformers.utils.hub import cached_file

from ..common import PatchSelectionResult
from .clip_model import CLIPTextModel, CLIPVisionModelV2


CLIP_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}



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


def identity_patch_selection(
    image_features: torch.Tensor,
    **_: Any,
) -> PatchSelectionResult:
    selected_indices = torch.arange(
        image_features.shape[0],
        device=image_features.device,
        dtype=torch.long,
    )
    return PatchSelectionResult(
        selected_indices=selected_indices,
        metadata={
            "strategy": "identity",
            "selected_token_count": int(selected_indices.numel()),
            "image_token_count": int(image_features.shape[0]),
        },
    )


def topk_norm_patch_selection(
    image_features: torch.Tensor,
    *,
    budget: int | None = None,
    keep_ratio: float | None = None,
    **_: Any,
) -> PatchSelectionResult:
    token_count = int(image_features.shape[0])
    resolved_budget = _resolve_budget(
        token_count,
        budget=budget,
        keep_ratio=keep_ratio,
    )
    if resolved_budget >= token_count:
        selected_indices = torch.arange(
            token_count,
            device=image_features.device,
            dtype=torch.long,
        )
    else:
        scores = image_features.float().norm(dim=-1)
        selected_indices = torch.topk(scores, k=resolved_budget).indices.sort().values
    return PatchSelectionResult(
        selected_indices=selected_indices,
        metadata={
            "strategy": "topk_norm",
            "budget": resolved_budget,
            "selected_token_count": int(selected_indices.numel()),
            "image_token_count": token_count,
        },
    )


def _configure_clip_image_processor(
    image_processor: CLIPImageProcessor,
    *,
    clip_do_center_crop: bool | None,
) -> CLIPImageProcessor:
    if clip_do_center_crop is not None:
        image_processor.do_center_crop = bool(clip_do_center_crop)

    if not bool(getattr(image_processor, "do_center_crop", True)):
        crop_size = getattr(image_processor, "crop_size", {}) or {}
        resize_size = getattr(image_processor, "size", {}) or {}
        target_height = crop_size.get("height") or resize_size.get("height")
        target_width = crop_size.get("width") or resize_size.get("width")
        if target_height is None:
            target_height = resize_size.get("shortest_edge")
        if target_width is None:
            target_width = resize_size.get("shortest_edge")
        if target_height is not None and target_width is not None:
            image_processor.do_resize = True
            image_processor.size = {
                "height": int(target_height),
                "width": int(target_width),
            }
    return image_processor


def _resolve_model_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    try:
        first_parameter = next(model.parameters())
    except (AttributeError, StopIteration, TypeError):
        return torch.device("cpu")
    return first_parameter.device


def _resolve_device(
    device: str | torch.device | None,
    reference_tensor: torch.Tensor,
) -> torch.device:
    if device is None:
        return reference_tensor.device
    return torch.device(device)


def _resolve_device_key(device: torch.device) -> str:
    if device.index is None:
        return device.type
    return str(device)


def _resolve_clip_dtype(
    clip_dtype: str | torch.dtype | None,
) -> tuple[torch.dtype | None, str]:
    if clip_dtype is None:
        return None, "default"
    if isinstance(clip_dtype, torch.dtype):
        return clip_dtype, str(clip_dtype).replace("torch.", "")

    normalized = str(clip_dtype).strip().lower()
    if normalized == "default":
        return None, "default"
    resolved = CLIP_DTYPE_MAP.get(normalized)
    if resolved is None:
        available = ", ".join(sorted(CLIP_DTYPE_MAP))
        raise ValueError(f"Unsupported `clip_dtype`: {clip_dtype}. Available: {available}.")
    return resolved, normalized


@lru_cache(maxsize=16)
def _load_query_cached(
    resolved_path: str,
    modified_time_ns: int,
) -> str:
    del modified_time_ns
    path = Path(resolved_path)
    query_lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not query_lines:
        raise ValueError(f"Query file does not contain a non-empty line: {path}")
    if len(query_lines) > 1:
        raise ValueError(f"Query file must contain exactly one non-empty line: {path}")
    return query_lines[0]


def _load_query(query_file: str | Path) -> str:
    path = Path(query_file).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Query file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Query file path must point to a file: {path}")
    stat = path.stat()
    return _load_query_cached(str(path.resolve()), stat.st_mtime_ns)


@lru_cache(maxsize=4)
def _load_maskclip_components(
    clip_model_name: str,
    device_key: str,
    clip_dtype_key: str,
    clip_do_center_crop: bool | None,
) -> tuple[CLIPImageProcessor, Any, CLIPVisionModelV2, CLIPTextModel]:
    clip_dtype, _ = _resolve_clip_dtype(clip_dtype_key)
    image_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
    image_processor = _configure_clip_image_processor(
        image_processor,
        clip_do_center_crop=clip_do_center_crop,
    )
    tokenizer = AutoTokenizer.from_pretrained(clip_model_name)
    vision_model = CLIPVisionModelV2(clip_model_name)
    text_model = CLIPTextModel(clip_model_name)
    if clip_dtype is None:
        vision_model = vision_model.to(device_key)
        text_model = text_model.to(device_key)
    else:
        vision_model = vision_model.to(device=device_key, dtype=clip_dtype)
        text_model = text_model.to(device=device_key, dtype=clip_dtype)
    vision_model.eval()
    text_model.eval()
    return image_processor, tokenizer, vision_model, text_model


@lru_cache(maxsize=4)
def _load_internal_maskclip_components(
    clip_model_name: str,
    device_key: str,
    clip_dtype_key: str,
) -> tuple[Any, CLIPTextModel, torch.Tensor]:
    clip_dtype, _ = _resolve_clip_dtype(clip_dtype_key)
    tokenizer = AutoTokenizer.from_pretrained(clip_model_name)
    text_model = CLIPTextModel(clip_model_name)
    checkpoint_path = cached_file(
        clip_model_name,
        "model.safetensors",
    )
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"Could not resolve model.safetensors for `{clip_model_name}`."
        )
    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        projection_weight = checkpoint.get_tensor("visual_projection.weight")

    if clip_dtype is None:
        text_model = text_model.to(device_key)
        projection_weight = projection_weight.to(device=device_key)
    else:
        text_model = text_model.to(device=device_key, dtype=clip_dtype)
        projection_weight = projection_weight.to(device=device_key, dtype=clip_dtype)
    text_model.eval()
    return tokenizer, text_model, projection_weight


def _resolve_internal_vision_tower(model: Any) -> Any:
    for candidate in (getattr(model, "model", None), model):
        vision_tower = getattr(candidate, "vision_tower", None)
        if vision_tower is not None:
            return vision_tower
    raise RuntimeError("Could not resolve LLaVA vision tower for internal MaskCLIP.")


def _compute_internal_maskclip_scores(
    *,
    model: Any,
    extraction_metadata: dict[str, Any],
    text_embeddings: torch.Tensor,
    projection_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    hidden_states = extraction_metadata.get("_maskclip_internal_hidden_states")
    pooled_output = extraction_metadata.get("_maskclip_internal_pooled_output")
    if not isinstance(hidden_states, torch.Tensor) or not isinstance(
        pooled_output, torch.Tensor
    ):
        raise ValueError(
            "Internal MaskCLIP features were not provided by the VLM extraction path."
        )

    vision_tower = _resolve_internal_vision_tower(model)
    last_layer = vision_tower.vision_model.encoder.layers[-1]
    dense_features = last_layer.layer_norm1(hidden_states)
    dense_features = last_layer.self_attn.v_proj(dense_features)
    dense_features = last_layer.self_attn.out_proj(dense_features)
    dense_features = F.linear(dense_features, projection_weight)[:, 1:, :]
    dense_features = F.normalize(dense_features, dim=-1)
    image_features = F.normalize(
        F.linear(pooled_output, projection_weight),
        dim=-1,
    )

    if text_embeddings.ndim == 1:
        patch_scores = dense_features @ text_embeddings
        image_scores = image_features @ text_embeddings
    elif text_embeddings.ndim == 2:
        patch_scores = torch.einsum("bnd,bd->bn", dense_features, text_embeddings)
        image_scores = torch.einsum("bd,bd->b", image_features, text_embeddings)
    else:
        raise ValueError(
            "Text embedding must have shape (D,) or (B, D), "
            f"got {tuple(text_embeddings.shape)}."
        )

    token_count = int(patch_scores.shape[1])
    grid_size = int(math.isqrt(token_count))
    if grid_size * grid_size != token_count:
        raise ValueError(
            "Internal MaskCLIP currently requires a square vision patch grid, "
            f"got {token_count} tokens."
        )
    return (
        patch_scores.view(patch_scores.shape[0], grid_size, grid_size),
        image_scores,
        (grid_size, grid_size),
    )


def preload_maskclip_patch_selection(
    *,
    clip_model_name: str = "openai/clip-vit-base-patch16",
    clip_dtype: str | torch.dtype | None = None,
    clip_do_center_crop: bool | None = None,
    use_internal_rep: bool = False,
    device: str | torch.device | None = None,
    model: Any,
    **_: Any,
) -> None:
    preload_device = torch.device(device) if device is not None else _resolve_model_device(model)
    device_key = _resolve_device_key(preload_device)
    _, clip_dtype_key = _resolve_clip_dtype(clip_dtype)
    if use_internal_rep:
        _load_internal_maskclip_components(
            clip_model_name,
            device_key,
            clip_dtype_key,
        )
    else:
        _load_maskclip_components(
            clip_model_name,
            device_key,
            clip_dtype_key,
            clip_do_center_crop,
        )


def _encode_text_queries(
    queries: list[str],
    *,
    tokenizer: Any,
    text_model: CLIPTextModel,
    device: torch.device,
) -> torch.Tensor:
    with torch.inference_mode():
        text_inputs = tokenizer(
            queries,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
        text_embedding = text_model(**text_inputs)
        return F.normalize(text_embedding, dim=-1)


def _encode_text_query(
    query: str,
    *,
    tokenizer: Any,
    text_model: CLIPTextModel,
    device: torch.device,
) -> torch.Tensor:
    return _encode_text_queries(
        [query],
        tokenizer=tokenizer,
        text_model=text_model,
        device=device,
    )[0]


@lru_cache(maxsize=16)
def _load_text_embedding(
    clip_model_name: str,
    device_key: str,
    clip_dtype_key: str,
    clip_do_center_crop: bool | None,
    query: str,
) -> torch.Tensor:
    _, tokenizer, _, text_model = _load_maskclip_components(
        clip_model_name,
        device_key,
        clip_dtype_key,
        clip_do_center_crop,
    )
    return _encode_text_query(
        query,
        tokenizer=tokenizer,
        text_model=text_model,
        device=torch.device(device_key),
    )


def _prepare_image_batch(images: list[Image.Image] | list[Any]) -> list[Any]:
    prepared = []
    for image in images:
        if isinstance(image, Image.Image):
            prepared.append(image.convert("RGB"))
        else:
            prepared.append(image)
    return prepared


def _compute_dense_patch_score_maps_and_image_scores(
    images: list[Image.Image] | list[Any],
    *,
    image_processor: CLIPImageProcessor,
    vision_model: CLIPVisionModelV2,
    text_embedding: torch.Tensor,
    batch_size: int,
    device: torch.device,
    clip_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    if batch_size <= 0:
        raise ValueError(f"`batch_size` must be positive, got {batch_size}.")

    score_chunks: list[torch.Tensor] = []
    image_score_chunks: list[torch.Tensor] = []
    clip_grid: tuple[int, int] | None = None
    patch_size = int(vision_model.config.patch_size)
    prepared_images = _prepare_image_batch(images)

    with torch.inference_mode():
        for start in range(0, len(prepared_images), batch_size):
            batch_images = prepared_images[start : start + batch_size]
            pixel_values = image_processor(
                images=batch_images,
                return_tensors="pt",
            )["pixel_values"]
            if clip_dtype is None:
                pixel_values = pixel_values.to(device=device, non_blocking=True)
            else:
                pixel_values = pixel_values.to(
                    device=device,
                    dtype=clip_dtype,
                    non_blocking=True,
                )

            patch_embeddings, image_latents = vision_model(pixel_values)
            patch_embeddings = F.normalize(patch_embeddings, dim=-1)
            image_latents = F.normalize(image_latents, dim=-1)

            if text_embedding.ndim == 1:
                patch_scores = patch_embeddings @ text_embedding
                image_scores = image_latents @ text_embedding
            elif text_embedding.ndim == 2:
                batch_offset = start
                batch_text_embedding = text_embedding[
                    batch_offset : batch_offset + patch_embeddings.shape[0]
                ]
                if int(batch_text_embedding.shape[0]) != int(patch_embeddings.shape[0]):
                    raise ValueError(
                        "Text embedding batch length must match image batch length."
                    )
                patch_scores = torch.einsum(
                    "bnd,bd->bn",
                    patch_embeddings,
                    batch_text_embedding,
                )
                image_scores = torch.einsum(
                    "bd,bd->b",
                    image_latents,
                    batch_text_embedding,
                )
            else:
                raise ValueError(
                    "Text embedding must have shape (D,) or (B, D), "
                    f"got {tuple(text_embedding.shape)}."
                )

            clip_h = int(pixel_values.shape[-2] // patch_size)
            clip_w = int(pixel_values.shape[-1] // patch_size)
            if clip_h * clip_w != int(patch_scores.shape[1]):
                raise ValueError(
                    "Failed to infer CLIP patch grid from processor output: "
                    f"expected {clip_h}x{clip_w}={clip_h * clip_w} patches, "
                    f"but got {int(patch_scores.shape[1])}."
                )

            if clip_grid is None:
                clip_grid = (clip_h, clip_w)
            elif clip_grid != (clip_h, clip_w):
                raise ValueError(
                    "CLIP processor returned inconsistent patch grids across batches: "
                    f"{clip_grid} vs {(clip_h, clip_w)}."
                )

            score_chunks.append(patch_scores.view(patch_scores.shape[0], clip_h, clip_w))
            image_score_chunks.append(image_scores)

    if clip_grid is None:
        raise ValueError("No images were provided for dense patch scoring.")
    return (
        torch.cat(score_chunks, dim=0),
        torch.cat(image_score_chunks, dim=0),
        clip_grid,
    )

def _resize_score_maps(
    score_maps: torch.Tensor,
    *,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    if score_maps.shape[-2:] == (target_height, target_width):
        return score_maps
    return F.interpolate(
        score_maps.unsqueeze(1),
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


def _factor_grid_for_aspect(
    token_count: int,
    *,
    aspect_ratio: float,
) -> tuple[int, int]:
    best_h = 1
    best_w = token_count
    best_error = float("inf")
    for candidate_h in range(1, int(math.sqrt(token_count)) + 1):
        if token_count % candidate_h != 0:
            continue
        candidate_w = token_count // candidate_h
        for h, w in ((candidate_h, candidate_w), (candidate_w, candidate_h)):
            error = abs((w / h) - aspect_ratio)
            if error < best_error:
                best_h = h
                best_w = w
                best_error = error
    return best_h, best_w


def infer_feature_grid(
    token_count: int,
    *,
    image: Image.Image | None = None,
    grid_hw: tuple[int, int] | None = None,
) -> tuple[int, int]:
    if token_count <= 0:
        raise ValueError(f"`token_count` must be positive, got {token_count}.")
    if grid_hw is not None:
        grid_h, grid_w = int(grid_hw[0]), int(grid_hw[1])
        if grid_h > 0 and grid_w > 0 and grid_h * grid_w == token_count:
            return grid_h, grid_w
        raise ValueError(
            "Explicit feature grid does not match token count: "
            f"grid=({grid_h}, {grid_w}), tokens={token_count}."
        )

    square = int(math.isqrt(token_count))
    if square * square == token_count:
        return square, square

    aspect_ratio = 1.0
    if image is not None and image.height > 0:
        aspect_ratio = float(image.width) / float(image.height)
    return _factor_grid_for_aspect(token_count, aspect_ratio=aspect_ratio)


def maskclip_patch_selection(
    image_features: torch.Tensor,
    *,
    image: Image.Image,
    query_file: str | None = None,
    query: str | None = None,
    keep_ratio: float = 0.5,
    clip_model_name: str = "openai/clip-vit-base-patch16",
    clip_dtype: str | torch.dtype | None = None,
    clip_do_center_crop: bool | None = None,
    use_internal_rep: bool = False,
    batch_size: int = 8,
    image_grid_hw: tuple[int, int] | None = None,
    device: str | torch.device | None = None,
    extraction_metadata: dict[str, Any] | None = None,
    model: Any = None,
    **_: Any,
) -> PatchSelectionResult:
    token_count = int(image_features.shape[0])
    keep_count = _resolve_budget(token_count, keep_ratio=keep_ratio)
    grid_h, grid_w = infer_feature_grid(
        token_count,
        image=image,
        grid_hw=image_grid_hw,
    )
    selector_device = _resolve_device(device, image_features)
    selector_device_key = _resolve_device_key(selector_device)
    resolved_clip_dtype, clip_dtype_key = _resolve_clip_dtype(clip_dtype)
    resolved_query = str(query).strip() if query is not None else ""
    if not resolved_query:
        if query_file is None:
            raise ValueError("`query` or `query_file` must be provided for patch selection.")
        resolved_query = _load_query(query_file)
    if use_internal_rep:
        tokenizer, text_model, projection_weight = _load_internal_maskclip_components(
            clip_model_name,
            selector_device_key,
            clip_dtype_key,
        )
        text_embedding = _encode_text_query(
            resolved_query,
            tokenizer=tokenizer,
            text_model=text_model,
            device=selector_device,
        )
        dense_score_maps, image_scores, clip_grid = _compute_internal_maskclip_scores(
            model=model,
            extraction_metadata=dict(extraction_metadata or {}),
            text_embeddings=text_embedding,
            projection_weight=projection_weight,
        )
    else:
        image_processor, _, vision_model, _ = _load_maskclip_components(
            clip_model_name,
            selector_device_key,
            clip_dtype_key,
            clip_do_center_crop,
        )
        text_embedding = _load_text_embedding(
            clip_model_name,
            selector_device_key,
            clip_dtype_key,
            clip_do_center_crop,
            resolved_query,
        )
        dense_score_maps, image_scores, clip_grid = (
            _compute_dense_patch_score_maps_and_image_scores(
                [image],
                image_processor=image_processor,
                vision_model=vision_model,
                text_embedding=text_embedding,
                batch_size=batch_size,
                device=selector_device,
                clip_dtype=resolved_clip_dtype,
            )
        )
    score_map = _resize_score_maps(
        dense_score_maps,
        target_height=grid_h,
        target_width=grid_w,
    )[0]
    flat_scores = score_map.flatten()
    if keep_count >= token_count:
        selected_indices = torch.arange(
            token_count,
            device=flat_scores.device,
            dtype=torch.long,
        )
    else:
        selected_indices = torch.topk(
            flat_scores,
            k=keep_count,
            largest=True,
            sorted=False,
        ).indices.sort().values
    selected_indices = selected_indices.to(
        device=image_features.device,
        dtype=torch.long,
    )

    metadata = {
        "selector_type": "maskclip_patch_selection_vqa",
        "selection_mode": "single_image_topk",
        "clip_model_name": clip_model_name,
        "use_internal_rep": bool(use_internal_rep),
        "clip_dtype": clip_dtype_key,
        "clip_do_center_crop": (
            None if clip_do_center_crop is None else bool(clip_do_center_crop)
        ),
        "query_file": str(Path(query_file).expanduser()) if query_file is not None else None,
        "query": resolved_query,
        "keep_ratio": float(keep_ratio),
        "keep_count": int(keep_count),
        "clip_grid_hw": [int(clip_grid[0]), int(clip_grid[1])],
        "image_grid_hw": [grid_h, grid_w],
        "score_min": float(flat_scores.min().item()),
        "score_max": float(flat_scores.max().item()),
        "image_importance_score": float(image_scores[0].item()),
        "selected_token_count": int(selected_indices.numel()),
        "image_token_count": token_count,
    }
    return PatchSelectionResult(
        selected_indices=selected_indices,
        metadata=metadata,
    )


def _query_from_prompt_value(prompt: Any) -> str | None:
    if isinstance(prompt, dict):
        query = prompt.get("query")
        if query is not None and str(query).strip():
            return str(query).strip()
        user_prompt = prompt.get("user")
        if user_prompt is not None and str(user_prompt).strip():
            return str(user_prompt).strip()
    elif prompt is not None and str(prompt).strip():
        return str(prompt).strip()
    return None


def _resolve_batch_queries(
    *,
    batch_size: int,
    queries: list[str] | tuple[str, ...] | None = None,
    prompts: list[Any] | tuple[Any, ...] | None = None,
    query_file: str | None = None,
) -> list[str]:
    resolved_queries: list[str] = []
    if queries is not None:
        resolved_queries = [str(query).strip() for query in queries]
    elif prompts is not None:
        resolved_queries = [
            _query_from_prompt_value(prompt) or ""
            for prompt in prompts
        ]
    elif query_file is not None:
        query = _load_query(query_file)
        resolved_queries = [query] * batch_size

    if len(resolved_queries) != batch_size or any(not query for query in resolved_queries):
        raise ValueError(
            "Batch patch selection requires one non-empty query per image."
        )
    return resolved_queries


def maskclip_patch_selection_batch(
    image_features: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    *,
    images: list[Image.Image] | tuple[Image.Image, ...],
    query_file: str | None = None,
    queries: list[str] | tuple[str, ...] | None = None,
    prompts: list[Any] | tuple[Any, ...] | None = None,
    keep_ratio: float = 0.5,
    clip_model_name: str = "openai/clip-vit-base-patch16",
    clip_dtype: str | torch.dtype | None = None,
    clip_do_center_crop: bool | None = None,
    use_internal_rep: bool = False,
    batch_size: int = 8,
    image_grid_hw: tuple[int, int] | None = None,
    device: str | torch.device | None = None,
    extraction_metadata: dict[str, Any] | None = None,
    model: Any = None,
    **_: Any,
) -> list[PatchSelectionResult]:
    image_batch = list(images)
    if torch.is_tensor(image_features):
        if image_features.ndim == 2:
            feature_batch = [image_features]
        elif image_features.ndim == 3:
            feature_batch = [image_features[index] for index in range(image_features.shape[0])]
        else:
            raise ValueError(
                "Batched image features must have shape (B, N, D), "
                f"got {tuple(image_features.shape)}."
            )
    else:
        feature_batch = list(image_features)

    if len(feature_batch) != len(image_batch):
        raise ValueError(
            "Image feature batch length must match images: "
            f"features={len(feature_batch)}, images={len(image_batch)}."
        )
    if not image_batch:
        return []

    resolved_queries = _resolve_batch_queries(
        batch_size=len(image_batch),
        queries=queries,
        prompts=prompts,
        query_file=query_file,
    )
    selector_device = _resolve_device(device, feature_batch[0])
    selector_device_key = _resolve_device_key(selector_device)
    resolved_clip_dtype, clip_dtype_key = _resolve_clip_dtype(clip_dtype)
    if use_internal_rep:
        tokenizer, text_model, projection_weight = _load_internal_maskclip_components(
            clip_model_name,
            selector_device_key,
            clip_dtype_key,
        )
        text_embeddings = _encode_text_queries(
            resolved_queries,
            tokenizer=tokenizer,
            text_model=text_model,
            device=selector_device,
        )
        dense_score_maps, image_scores, clip_grid = _compute_internal_maskclip_scores(
            model=model,
            extraction_metadata=dict(extraction_metadata or {}),
            text_embeddings=text_embeddings,
            projection_weight=projection_weight,
        )
    else:
        image_processor, tokenizer, vision_model, text_model = _load_maskclip_components(
            clip_model_name,
            selector_device_key,
            clip_dtype_key,
            clip_do_center_crop,
        )
        text_embeddings = _encode_text_queries(
            resolved_queries,
            tokenizer=tokenizer,
            text_model=text_model,
            device=selector_device,
        )
        dense_score_maps, image_scores, clip_grid = (
            _compute_dense_patch_score_maps_and_image_scores(
                image_batch,
                image_processor=image_processor,
                vision_model=vision_model,
                text_embedding=text_embeddings,
                batch_size=batch_size,
                device=selector_device,
                clip_dtype=resolved_clip_dtype,
            )
        )

    results: list[PatchSelectionResult] = []

    if torch.is_tensor(image_features) and image_features.ndim == 3:
        token_count = int(image_features.shape[1])
        keep_count = _resolve_budget(token_count, keep_ratio=keep_ratio)
        grids = [
            infer_feature_grid(
                token_count,
                image=image,
                grid_hw=image_grid_hw,
            )
            for image in image_batch
        ]
        if len(set(grids)) == 1:
            grid_h, grid_w = grids[0]
            resized_scores = _resize_score_maps(
                dense_score_maps,
                target_height=grid_h,
                target_width=grid_w,
            )
            flat_scores = resized_scores.flatten(start_dim=1)
            if keep_count >= token_count:
                selected_indices_batch = torch.arange(
                    token_count,
                    device=flat_scores.device,
                    dtype=torch.long,
                ).expand(len(image_batch), -1)
            else:
                selected_indices_batch = torch.topk(
                    flat_scores,
                    k=keep_count,
                    dim=1,
                    largest=True,
                    sorted=False,
                ).indices.sort(dim=1).values

            selected_indices_batch = selected_indices_batch.to(
                device=image_features.device,
                dtype=torch.long,
            )
            selected_features_batch = torch.gather(
                image_features,
                dim=1,
                index=selected_indices_batch.unsqueeze(-1).expand(
                    -1,
                    -1,
                    image_features.shape[-1],
                ),
            )
            score_stats = torch.stack(
                (
                    flat_scores.amin(dim=1),
                    flat_scores.amax(dim=1),
                    image_scores,
                ),
                dim=1,
            ).float().cpu().tolist()

            for item_index, query in enumerate(resolved_queries):
                selected_indices = selected_indices_batch[item_index]
                metadata = {
                    "selector_type": "maskclip_patch_selection_vqa",
                    "selection_mode": "batch_image_topk",
                    "clip_model_name": clip_model_name,
                    "use_internal_rep": bool(use_internal_rep),
                    "clip_dtype": clip_dtype_key,
                    "clip_do_center_crop": (
                        None if clip_do_center_crop is None else bool(clip_do_center_crop)
                    ),
                    "query_file": (
                        str(Path(query_file).expanduser())
                        if query_file is not None
                        else None
                    ),
                    "query": query,
                    "keep_ratio": float(keep_ratio),
                    "keep_count": int(keep_count),
                    "clip_grid_hw": [int(clip_grid[0]), int(clip_grid[1])],
                    "image_grid_hw": [grid_h, grid_w],
                    "score_min": float(score_stats[item_index][0]),
                    "score_max": float(score_stats[item_index][1]),
                    "image_importance_score": float(score_stats[item_index][2]),
                    "selected_token_count": int(keep_count),
                    "image_token_count": token_count,
                }
                results.append(
                    PatchSelectionResult(
                        selected_indices=selected_indices,
                        selected_features=selected_features_batch[item_index],
                        metadata=metadata,
                    )
                )
            return results

    for item_index, (item_features, image, query) in enumerate(
        zip(feature_batch, image_batch, resolved_queries)
    ):
        token_count = int(item_features.shape[0])
        keep_count = _resolve_budget(token_count, keep_ratio=keep_ratio)
        grid_h, grid_w = infer_feature_grid(
            token_count,
            image=image,
            grid_hw=image_grid_hw,
        )
        score_map = _resize_score_maps(
            dense_score_maps[item_index : item_index + 1],
            target_height=grid_h,
            target_width=grid_w,
        )[0]
        flat_scores = score_map.flatten()
        if keep_count >= token_count:
            selected_indices = torch.arange(
                token_count,
                device=flat_scores.device,
                dtype=torch.long,
            )
        else:
            selected_indices = torch.topk(
                flat_scores,
                k=keep_count,
                largest=True,
                sorted=False,
            ).indices.sort().values
        selected_indices = selected_indices.to(
            device=item_features.device,
            dtype=torch.long,
        )
        metadata = {
            "selector_type": "maskclip_patch_selection_vqa",
            "selection_mode": "batch_image_topk",
            "clip_model_name": clip_model_name,
            "use_internal_rep": bool(use_internal_rep),
            "clip_dtype": clip_dtype_key,
            "clip_do_center_crop": (
                None if clip_do_center_crop is None else bool(clip_do_center_crop)
            ),
            "query_file": str(Path(query_file).expanduser()) if query_file is not None else None,
            "query": query,
            "keep_ratio": float(keep_ratio),
            "keep_count": int(keep_count),
            "clip_grid_hw": [int(clip_grid[0]), int(clip_grid[1])],
            "image_grid_hw": [grid_h, grid_w],
            "score_min": float(flat_scores.min().item()),
            "score_max": float(flat_scores.max().item()),
            "image_importance_score": float(image_scores[item_index].item()),
            "selected_token_count": int(selected_indices.numel()),
            "image_token_count": token_count,
        }
        results.append(
            PatchSelectionResult(
                selected_indices=selected_indices,
                metadata=metadata,
            )
        )
    return results


maskclip_patch_selection.preload = preload_maskclip_patch_selection
maskclip_patch_selection.batch = maskclip_patch_selection_batch
