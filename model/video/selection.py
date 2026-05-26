from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from ..DDPS.selection import (
    PatchSelectionResult,
    _compute_dense_patch_score_maps_and_image_scores,
    _load_maskclip_components,
    _load_query,
    _load_text_embedding,
    _resolve_clip_dtype,
    _resolve_device,
    _resolve_device_key,
    infer_feature_grid,
    preload_maskclip_patch_selection,
)


@dataclass(slots=True)
class FrameSelectionResult:
    frames: torch.Tensor | None
    metadata: dict[str, Any] = field(default_factory=dict)


def _inspect_video_capture(video_path: str) -> tuple[cv2.VideoCapture, int, float]:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) if cap.isOpened() else 0.0
    return cap, total_frames, fps


def _transcode_video_for_opencv(video_path: str) -> Path | None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        return None

    temp_dir = Path(tempfile.mkdtemp(prefix="ddps_opencv_decode_"))
    output_path = temp_dir / f"{Path(video_path).stem}_opencv_h264.mp4"
    command = [
        ffmpeg_path,
        "-y",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError):
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None
    return output_path if output_path.exists() else None


def _open_video_for_sampling(
    video_path: str,
) -> tuple[cv2.VideoCapture, int, float, Path | None]:
    cap, total_frames, fps = _inspect_video_capture(video_path)
    if cap.isOpened() and total_frames > 0:
        return cap, total_frames, fps, None

    cap.release()
    transcoded_path = _transcode_video_for_opencv(video_path)
    if transcoded_path is None:
        raise RuntimeError(
            "Failed to decode video with OpenCV. Install ffmpeg or transcode the "
            "source video to a broadly supported H.264 mp4 file."
        )

    cap, total_frames, fps = _inspect_video_capture(str(transcoded_path))
    if cap.isOpened() and total_frames > 0:
        return cap, total_frames, fps, transcoded_path

    cap.release()
    shutil.rmtree(transcoded_path.parent, ignore_errors=True)
    raise RuntimeError(f"Failed to decode video after transcoding: {video_path}")


def _resize_frame(frame_rgb: np.ndarray, *, max_side: int | None) -> np.ndarray:
    if max_side is None:
        return frame_rgb
    if max_side <= 0:
        raise ValueError(f"`max_side` must be positive, got {max_side}.")

    height, width = frame_rgb.shape[:2]
    scale = min(float(max_side) / max(height, width), 1.0)
    if scale >= 1.0:
        return frame_rgb

    resized_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return cv2.resize(frame_rgb, resized_size, interpolation=cv2.INTER_AREA)


def _decode_target_frames(
    cap: cv2.VideoCapture,
    indices: list[int],
    *,
    max_side: int | None,
) -> tuple[list[np.ndarray], list[int]]:
    frames: list[np.ndarray] = []
    sampled_indices: list[int] = []

    for frame_idx in indices:
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx)):
            continue
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(_resize_frame(frame_rgb, max_side=max_side))
        sampled_indices.append(int(frame_idx))

    return frames, sampled_indices


def _decode_target_frames_sequentially(
    cap: cv2.VideoCapture,
    indices: list[int],
    *,
    num_frames: int,
    max_side: int | None,
) -> tuple[list[np.ndarray], list[int], int, bool, bool]:
    target_set = set(indices)
    frames: list[np.ndarray] = []
    prefix_frames: list[np.ndarray] = []
    sampled_indices: list[int] = []
    frame_idx = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        should_keep_prefix = frame_idx < num_frames
        should_sample = frame_idx in target_set
        if should_keep_prefix or should_sample:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = _resize_frame(frame_rgb, max_side=max_side)
            if should_keep_prefix:
                prefix_frames.append(frame_rgb)
            if should_sample:
                frames.append(frame_rgb)
                sampled_indices.append(frame_idx)
        frame_idx += 1

    actual_total_frames = frame_idx
    used_short_video_fallback = False
    used_index_mismatch_fallback = False
    if len(frames) != len(indices):
        recoverable_frame_count = min(actual_total_frames, num_frames)
        if recoverable_frame_count <= 0 or len(prefix_frames) != recoverable_frame_count:
            raise RuntimeError(
                f"Expected {len(indices)} sampled frames, but got {len(frames)}."
            )
        frames = prefix_frames
        sampled_indices = list(range(recoverable_frame_count))
        used_short_video_fallback = actual_total_frames < num_frames
        used_index_mismatch_fallback = not used_short_video_fallback

    return (
        frames,
        sampled_indices,
        actual_total_frames,
        used_short_video_fallback,
        used_index_mismatch_fallback,
    )


def uniform_sampling(
    video_path: str,
    num_frames: int = 8,
    max_side: int | None = 720,
) -> FrameSelectionResult:
    if num_frames <= 0:
        raise ValueError(f"`num_frames` must be positive, got {num_frames}.")

    cap, total_frames, fps, transcoded_path = _open_video_for_sampling(video_path)
    if total_frames <= num_frames:
        indices = list(range(total_frames))
    elif num_frames == 1:
        indices = [total_frames // 2]
    else:
        indices = np.linspace(0, total_frames - 1, num_frames).round().astype(int).tolist()

    try:
        frames, sampled_indices = _decode_target_frames(
            cap,
            indices,
            max_side=max_side,
        )
        actual_total_frames = total_frames
        used_short_video_fallback = False
        used_index_mismatch_fallback = False
        sampling_strategy = "indexed_seek"
        if len(frames) != len(indices):
            cap.release()
            fallback_video_path = str(transcoded_path) if transcoded_path is not None else video_path
            cap, total_frames, fps = _inspect_video_capture(fallback_video_path)
            if not cap.isOpened() or total_frames <= 0:
                raise RuntimeError(f"Failed to reopen video for fallback: {video_path}")
            (
                frames,
                sampled_indices,
                actual_total_frames,
                used_short_video_fallback,
                used_index_mismatch_fallback,
            ) = _decode_target_frames_sequentially(
                cap,
                indices,
                num_frames=num_frames,
                max_side=max_side,
            )
            sampling_strategy = "sequential_fallback"
    finally:
        cap.release()
        if transcoded_path is not None:
            shutil.rmtree(transcoded_path.parent, ignore_errors=True)

    if not frames:
        raise RuntimeError(f"No frames were decoded from video: {video_path}")

    base_height, base_width = frames[0].shape[:2]
    normalized_frames = []
    for frame in frames:
        if frame.shape[:2] != (base_height, base_width):
            frame = cv2.resize(
                frame,
                (base_width, base_height),
                interpolation=cv2.INTER_AREA,
            )
        normalized_frames.append(frame)

    video_np = np.stack(normalized_frames, axis=0)
    return FrameSelectionResult(
        frames=torch.from_numpy(video_np),
        metadata={
            "video_path": video_path,
            "decoded_video_path": str(transcoded_path) if transcoded_path is not None else video_path,
            "sampled_indices": sampled_indices,
            "num_frames": len(normalized_frames),
            "total_frames": total_frames,
            "decoded_total_frames": actual_total_frames,
            "fps": fps if fps > 0 else None,
            "frame_shape": list(video_np.shape[1:]),
            "sampling_strategy": sampling_strategy,
            "used_short_video_fallback": used_short_video_fallback,
            "used_index_mismatch_fallback": used_index_mismatch_fallback,
        },
    )


def frames_to_contact_sheet(
    frame_selection: FrameSelectionResult,
    *,
    columns: int | None = None,
    annotate: bool = True,
) -> Image.Image:
    frames = frame_selection.frames
    if frames is None or not torch.is_tensor(frames) or frames.ndim != 4:
        raise ValueError("Expected frames with shape (T, H, W, C).")

    frame_count = int(frames.shape[0])
    if frame_count <= 0:
        raise ValueError("Cannot build a contact sheet from an empty frame set.")

    if columns is None:
        columns = int(np.ceil(np.sqrt(frame_count)))
    columns = max(1, int(columns))
    rows = int(np.ceil(frame_count / columns))
    frame_images = [
        Image.fromarray(frames[index].detach().cpu().numpy().astype(np.uint8), mode="RGB")
        for index in range(frame_count)
    ]
    width, height = frame_images[0].size
    label_height = 18 if annotate else 0
    sheet = Image.new("RGB", (columns * width, rows * (height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)

    sampled_indices = frame_selection.metadata.get("sampled_indices") or list(range(frame_count))
    for index, frame in enumerate(frame_images):
        row, col = divmod(index, columns)
        x = col * width
        y = row * (height + label_height)
        sheet.paste(frame, (x, y + label_height))
        if annotate:
            label = f"frame {sampled_indices[index]}"
            draw.text((x + 4, y + 2), label, fill=(0, 0, 0))

    return sheet



def _resolve_total_budget(
    *,
    token_count: int,
    keep_ratio: float,
    total_budget: int | None,
) -> int:
    if token_count <= 0:
        return 0
    if total_budget is not None:
        resolved_total_budget = int(total_budget)
        if resolved_total_budget < 0:
            raise ValueError(f"`total_budget` must be non-negative, got {total_budget}.")
    else:
        if keep_ratio <= 0.0 or keep_ratio > 1.0:
            raise ValueError(f"`keep_ratio` must be in (0, 1], got {keep_ratio}.")
        resolved_total_budget = max(1, int(np.ceil(token_count * float(keep_ratio))))
    return min(resolved_total_budget, token_count)


def _allocate_budget_with_softmax_capacities(
    frame_scores: torch.Tensor,
    *,
    total_budget: int,
    capacities: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if temperature <= 0.0:
        raise ValueError(f"`temperature` must be positive, got {temperature}.")
    if total_budget < 0:
        raise ValueError(f"`total_budget` must be non-negative, got {total_budget}.")

    scores = frame_scores.to(dtype=torch.float32).flatten()
    scaled_scores = scores / temperature
    resolved_capacities = capacities.to(device=scores.device, dtype=torch.long).flatten()
    if resolved_capacities.shape != scores.shape:
        raise ValueError(
            "Capacity shape must match score shape: "
            f"scores={tuple(scores.shape)}, capacities={tuple(resolved_capacities.shape)}."
        )
    if torch.any(resolved_capacities < 0):
        raise ValueError("Capacities must be non-negative.")

    total_capacity = int(resolved_capacities.sum().item())
    allocatable_budget = min(int(total_budget), total_capacity)
    quotas = torch.zeros_like(scores, dtype=torch.float32)
    weights = torch.zeros_like(scores, dtype=torch.float32)
    active_capacity_mask = resolved_capacities > 0
    active_indices = torch.nonzero(active_capacity_mask, as_tuple=False).flatten()
    if active_indices.numel() > 0:
        weights[active_indices] = torch.softmax(scaled_scores[active_indices], dim=0)

    if allocatable_budget == 0:
        return torch.zeros_like(resolved_capacities), quotas, weights

    remaining_budget = float(allocatable_budget)
    active_mask = active_capacity_mask.clone()
    while remaining_budget > 1e-6:
        current_indices = torch.nonzero(active_mask, as_tuple=False).flatten()
        if current_indices.numel() == 0:
            break

        current_weights = torch.softmax(scaled_scores[current_indices], dim=0)
        current_capacities = resolved_capacities[current_indices].to(dtype=torch.float32)
        tentative = current_weights * remaining_budget
        saturated = tentative >= current_capacities
        if torch.any(saturated):
            saturated_indices = current_indices[saturated]
            quotas[saturated_indices] = resolved_capacities[saturated_indices].to(dtype=torch.float32)
            remaining_budget -= float(resolved_capacities[saturated_indices].sum().item())
            active_mask[saturated_indices] = False
            continue

        quotas[current_indices] = tentative
        remaining_budget = 0.0

    allocated = torch.floor(quotas).to(dtype=torch.long)
    remaining_integer_budget = allocatable_budget - int(allocated.sum().item())
    if remaining_integer_budget > 0:
        fractional = quotas - allocated.to(dtype=quotas.dtype)
        available_mask = allocated < resolved_capacities
        candidate_indices = torch.nonzero(available_mask, as_tuple=False).flatten()
        if candidate_indices.numel() > 0:
            candidate_fractional = fractional[candidate_indices]
            candidate_scores = scores[candidate_indices]
            ranking = torch.argsort(
                candidate_fractional + candidate_scores * 1e-6,
                descending=True,
            )
            allocated[candidate_indices[ranking][:remaining_integer_budget]] += 1

    if int(allocated.sum().item()) != allocatable_budget:
        raise RuntimeError(
            "Allocated patch budget does not match the allocatable budget: "
            f"allocated={int(allocated.sum().item())}, allocatable={allocatable_budget}."
        )
    return allocated, quotas, weights


def _coerce_video_frames(frame_selection: FrameSelectionResult) -> torch.Tensor:
    frames = frame_selection.frames
    if frames is None:
        raise ValueError("Frame selection did not provide video frames.")
    if not torch.is_tensor(frames):
        raise TypeError("Frame selection frames must be a torch.Tensor.")
    if frames.ndim != 4:
        raise ValueError(
            "Expected sampled video frames with shape (T, H, W, C), "
            f"but got {tuple(frames.shape)}."
        )
    return frames


def _prepare_frame_arrays(frames: torch.Tensor) -> list[np.ndarray]:
    frames_cpu = frames.detach()
    if frames_cpu.device.type != "cpu":
        frames_cpu = frames_cpu.cpu()
    if not frames_cpu.is_contiguous():
        frames_cpu = frames_cpu.contiguous()
    return list(frames_cpu.numpy())


def _resolve_contact_sheet_layout(
    frame_selection: FrameSelectionResult,
    visual_metadata: dict[str, Any],
    image: Image.Image,
) -> dict[str, int]:
    frames = _coerce_video_frames(frame_selection)
    frame_count = int(frames.shape[0])
    if frame_count <= 0:
        raise ValueError("Cannot run video patch selection on an empty frame set.")

    frame_h = int(frames.shape[1])
    frame_w = int(frames.shape[2])
    columns_value = visual_metadata.get("contact_sheet_columns")
    if columns_value is None:
        columns = int(np.ceil(np.sqrt(frame_count)))
    else:
        columns = int(columns_value)
    columns = max(1, columns)
    rows = int(np.ceil(frame_count / columns))
    label_height = int(visual_metadata.get("contact_sheet_label_height", 18))
    sheet_size = visual_metadata.get("contact_sheet_size")
    if isinstance(sheet_size, (list, tuple)) and len(sheet_size) == 2:
        sheet_w, sheet_h = int(sheet_size[0]), int(sheet_size[1])
    else:
        sheet_w, sheet_h = image.size

    return {
        "frame_count": frame_count,
        "frame_h": frame_h,
        "frame_w": frame_w,
        "columns": columns,
        "rows": rows,
        "label_height": label_height,
        "sheet_w": sheet_w,
        "sheet_h": sheet_h,
    }


def _project_frame_scores_to_feature_grid(
    dense_score_maps: torch.Tensor,
    *,
    layout: dict[str, int],
    grid_h: int,
    grid_w: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, int]]]:
    fill_value = torch.finfo(dense_score_maps.dtype).min
    score_grid = dense_score_maps.new_full((grid_h, grid_w), fill_value)
    region_masks = torch.zeros(
        (layout["frame_count"], grid_h, grid_w),
        device=dense_score_maps.device,
        dtype=torch.bool,
    )
    regions: list[dict[str, int]] = []

    frame_h = layout["frame_h"]
    frame_w = layout["frame_w"]
    columns = layout["columns"]
    label_height = layout["label_height"]
    sheet_h = max(layout["sheet_h"], 1)
    sheet_w = max(layout["sheet_w"], 1)

    for frame_idx in range(layout["frame_count"]):
        row, col = divmod(frame_idx, columns)
        x0 = col * frame_w
        x1 = x0 + frame_w
        y0 = row * (frame_h + label_height) + label_height
        y1 = y0 + frame_h

        gx0 = max(0, min(grid_w - 1, int(np.floor(x0 * grid_w / sheet_w))))
        gx1 = max(gx0 + 1, min(grid_w, int(np.ceil(x1 * grid_w / sheet_w))))
        gy0 = max(0, min(grid_h - 1, int(np.floor(y0 * grid_h / sheet_h))))
        gy1 = max(gy0 + 1, min(grid_h, int(np.ceil(y1 * grid_h / sheet_h))))

        resized_scores = F.interpolate(
            dense_score_maps[frame_idx].view(1, 1, *dense_score_maps.shape[-2:]),
            size=(gy1 - gy0, gx1 - gx0),
            mode="bilinear",
            align_corners=False,
        ).view(gy1 - gy0, gx1 - gx0)
        score_grid[gy0:gy1, gx0:gx1] = resized_scores
        region_masks[frame_idx, gy0:gy1, gx0:gx1] = True
        regions.append(
            {
                "frame_index": frame_idx,
                "grid_y0": gy0,
                "grid_y1": gy1,
                "grid_x0": gx0,
                "grid_x1": gx1,
                "grid_token_count": int((gy1 - gy0) * (gx1 - gx0)),
            }
        )

    return score_grid, region_masks, regions


def _select_topk_from_video_regions(
    score_grid: torch.Tensor,
    *,
    eligible_mask: torch.Tensor,
    region_masks: torch.Tensor,
    frame_scores: torch.Tensor,
    frame_weights: torch.Tensor,
    raw_budget: torch.Tensor,
    allocated_budget: torch.Tensor,
    eligible_counts: torch.Tensor,
    regions: list[dict[str, int]],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    flat_scores = score_grid.flatten()
    selected_chunks: list[torch.Tensor] = []
    frame_metadata: list[dict[str, Any]] = []

    for frame_idx, region in enumerate(regions):
        frame_region_mask = region_masks[frame_idx] & eligible_mask
        keep_count = int(allocated_budget[frame_idx].item())
        eligible_count = int(eligible_counts[frame_idx].item())
        if keep_count > eligible_count:
            raise ValueError(
                "Allocated frame budget exceeds eligible patch count: "
                f"frame_index={frame_idx}, keep_count={keep_count}, "
                f"eligible_count={eligible_count}."
            )

        if keep_count > 0:
            flat_region_mask = frame_region_mask.flatten()
            masked_scores = flat_scores.masked_fill(
                ~flat_region_mask,
                torch.finfo(flat_scores.dtype).min,
            )
            selected_chunks.append(
                torch.topk(
                    masked_scores,
                    k=keep_count,
                    largest=True,
                    sorted=False,
                ).indices.sort().values
            )

        raw_region_scores = score_grid[region_masks[frame_idx]]
        frame_metadata.append(
            {
                **region,
                "eligible_count": eligible_count,
                "initial_allocated_budget": keep_count,
                "initial_keep_count": keep_count,
                "final_keep_count": keep_count,
                "reallocated_count": 0,
                "score_min": float(raw_region_scores.min().item()),
                "score_max": float(raw_region_scores.max().item()),
                "frame_importance_score": float(frame_scores[frame_idx].item()),
                "softmax_weight": float(frame_weights[frame_idx].item()),
                "raw_budget": float(raw_budget[frame_idx].item()),
            }
        )

    if not selected_chunks:
        selected_indices = torch.empty(0, device=score_grid.device, dtype=torch.long)
    else:
        selected_indices = torch.cat(selected_chunks, dim=0).unique(sorted=True)
    return selected_indices, frame_metadata


def maskclip_patch_selection(
    image_features: torch.Tensor,
    *,
    image: Image.Image,
    visual_metadata: dict[str, Any] | None = None,
    frame_selection: FrameSelectionResult | None = None,
    query_file: str,
    clip_model_name: str = "openai/clip-vit-base-patch16",
    clip_dtype: str | torch.dtype | None = None,
    clip_do_center_crop: bool | None = None,
    keep_ratio: float = 0.5,
    total_budget: int | None = None,
    temperature: float = 1.0,
    patch_score_threshold: float = 0.0,
    batch_size: int = 8,
    image_grid_hw: tuple[int, int] | None = None,
    device: str | torch.device | None = None,
    **kwargs: Any,
) -> PatchSelectionResult:
    visual_metadata = dict(visual_metadata or {})
    if frame_selection is None:
        candidate = visual_metadata.get("_frame_selection")
        if isinstance(candidate, FrameSelectionResult):
            frame_selection = candidate

    if frame_selection is None:
        from ..DDPS.selection import maskclip_patch_selection as ddps_maskclip_patch_selection

        return ddps_maskclip_patch_selection(
            image_features=image_features,
            image=image,
            query_file=query_file,
            clip_model_name=clip_model_name,
            clip_dtype=clip_dtype,
            clip_do_center_crop=clip_do_center_crop,
            keep_ratio=keep_ratio,
            batch_size=batch_size,
            image_grid_hw=image_grid_hw,
            device=device,
            **kwargs,
        )
    if patch_score_threshold is None:
        raise ValueError("`patch_score_threshold` must be a non-null float.")

    frames = _coerce_video_frames(frame_selection)
    token_count = int(image_features.shape[0])
    grid_h, grid_w = infer_feature_grid(
        token_count,
        image=image,
        grid_hw=image_grid_hw,
    )
    layout = _resolve_contact_sheet_layout(frame_selection, visual_metadata, image)
    selector_device = _resolve_device(device, image_features)
    selector_device_key = _resolve_device_key(selector_device)
    resolved_clip_dtype, clip_dtype_key = _resolve_clip_dtype(clip_dtype)
    query = _load_query(query_file)
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
        query,
    )
    dense_score_maps, frame_scores, clip_grid = (
        _compute_dense_patch_score_maps_and_image_scores(
            _prepare_frame_arrays(frames),
            image_processor=image_processor,
            vision_model=vision_model,
            text_embedding=text_embedding,
            batch_size=batch_size,
            device=selector_device,
            clip_dtype=resolved_clip_dtype,
        )
    )
    score_grid, region_masks, regions = _project_frame_scores_to_feature_grid(
        dense_score_maps,
        layout=layout,
        grid_h=grid_h,
        grid_w=grid_w,
    )
    valid_region_mask = region_masks.any(dim=0)
    total_video_tokens = int(valid_region_mask.sum().item())
    total_budget_value = _resolve_total_budget(
        token_count=total_video_tokens,
        keep_ratio=keep_ratio,
        total_budget=total_budget,
    )
    eligible_mask = (score_grid >= float(patch_score_threshold)) & valid_region_mask
    eligible_counts = (region_masks & eligible_mask.unsqueeze(0)).reshape(
        layout["frame_count"],
        -1,
    ).sum(dim=1).to(dtype=torch.long)
    allocated_budget, raw_budget, frame_weights = _allocate_budget_with_softmax_capacities(
        frame_scores,
        total_budget=total_budget_value,
        capacities=eligible_counts,
        temperature=temperature,
    )
    selected_indices, frame_metadata = _select_topk_from_video_regions(
        score_grid,
        eligible_mask=eligible_mask,
        region_masks=region_masks,
        frame_scores=frame_scores,
        frame_weights=frame_weights,
        raw_budget=raw_budget,
        allocated_budget=allocated_budget,
        eligible_counts=eligible_counts,
        regions=regions,
    )
    selected_indices = selected_indices.to(
        device=image_features.device,
        dtype=torch.long,
    )

    selected_token_count = int(selected_indices.numel())
    eligible_token_count = int(eligible_counts.sum().item())
    allocatable_budget = min(total_budget_value, eligible_token_count)
    metadata = {
        "selector_type": "maskclip_patch_selection_v4_video",
        "selection_mode": "contact_sheet_frame_regions",
        "allocation_strategy": "one_pass_capacity_aware",
        "clip_model_name": clip_model_name,
        "clip_dtype": clip_dtype_key,
        "clip_do_center_crop": (
            None if clip_do_center_crop is None else bool(clip_do_center_crop)
        ),
        "query_file": str(Path(query_file).expanduser()),
        "query": query,
        "keep_ratio": float(keep_ratio),
        "total_budget": int(total_budget_value),
        "allocatable_budget": int(allocatable_budget),
        "temperature": float(temperature),
        "patch_score_threshold": float(patch_score_threshold),
        "frame_count": layout["frame_count"],
        "contact_sheet_columns": layout["columns"],
        "contact_sheet_rows": layout["rows"],
        "contact_sheet_label_height": layout["label_height"],
        "clip_grid_hw": [int(clip_grid[0]), int(clip_grid[1])],
        "image_grid_hw": [grid_h, grid_w],
        "eligible_token_count": eligible_token_count,
        "selected_token_count": selected_token_count,
        "video_token_count": total_video_tokens,
        "image_token_count": token_count,
        "original_video_tokens": token_count,
        "selected_video_tokens": selected_token_count,
        "underfilled_token_count": int(total_budget_value - selected_token_count),
        "reallocated_token_count": 0,
        "frame_importance_scores": [float(score) for score in frame_scores.tolist()],
        "frame_softmax_weights": [float(weight) for weight in frame_weights.tolist()],
        "frame_raw_budgets": [float(budget) for budget in raw_budget.tolist()],
        "initial_frame_allocated_budgets": [
            int(budget) for budget in allocated_budget.tolist()
        ],
        "final_frame_allocated_budgets": [
            int(budget) for budget in allocated_budget.tolist()
        ],
        "frame_allocated_budgets": [int(budget) for budget in allocated_budget.tolist()],
        "per_frame": frame_metadata,
    }
    return PatchSelectionResult(
        selected_indices=selected_indices,
        metadata=metadata,
    )


maskclip_patch_selection.preload = preload_maskclip_patch_selection
