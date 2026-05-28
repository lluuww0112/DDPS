from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _coerce_average_metric(
    value: float | int | None,
) -> float | None:
    if value is None:
        return None
    return float(value)


def _coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_runtime_metrics(vlm: Any) -> dict[str, int | bool | None]:
    timing_info = getattr(vlm, "last_timing_info", {}) or {}
    patch_info = getattr(vlm, "last_patch_selection_info", {}) or {}
    selector_metadata = patch_info.get("selector_metadata")
    if not isinstance(selector_metadata, Mapping):
        selector_metadata = {}

    input_sequence_length = _coerce_int(
        timing_info.get("input_sequence_length")
    )
    if input_sequence_length is None:
        input_sequence_length = _coerce_int(
            patch_info.get("input_length_after")
        )

    input_sequence_length_before_patch = _coerce_int(
        patch_info.get("input_length_before")
    )
    original_llm_input_sequence_length = input_sequence_length_before_patch
    if original_llm_input_sequence_length is None:
        original_llm_input_sequence_length = input_sequence_length

    selected_video_tokens = _coerce_int(
        patch_info.get("selected_video_tokens")
    )
    original_video_tokens = _coerce_int(
        patch_info.get("original_video_tokens")
    )
    reallocated_patch_count = _coerce_int(
        patch_info.get("reallocated_token_count")
    )
    if reallocated_patch_count is None:
        reallocated_patch_count = _coerce_int(
            selector_metadata.get("reallocated_token_count")
        )

    return {
        "patch_selection_applied": bool(patch_info.get("applied", False)),
        "llm_input_sequence_length": input_sequence_length,
        "original_llm_input_sequence_length": original_llm_input_sequence_length,
        "llm_input_sequence_length_before_patch_selection": input_sequence_length_before_patch,
        "visual_token_count": selected_video_tokens,
        "original_visual_token_count": original_video_tokens,
        "original_video_token_count": original_video_tokens,
        "selected_video_token_count": selected_video_tokens,
        "reallocated_patch_count": reallocated_patch_count,
    }


def init_runtime_metric_totals() -> dict[str, float | int]:
    return {
        "visual_token_sum": 0.0,
        "visual_token_count": 0,
        "input_sequence_length_sum": 0.0,
        "input_sequence_length_count": 0,
        "original_visual_token_sum": 0.0,
        "original_visual_token_count": 0,
        "original_input_sequence_length_sum": 0.0,
        "original_input_sequence_length_count": 0,
        "reallocated_patch_sum": 0.0,
        "reallocated_patch_count": 0,
    }


def update_runtime_metric_totals(
    totals: dict[str, float | int],
    metrics: Mapping[str, int | bool | None],
) -> None:
    visual_token_count = metrics.get("visual_token_count")
    if isinstance(visual_token_count, int):
        totals["visual_token_sum"] += float(visual_token_count)
        totals["visual_token_count"] += 1

    input_sequence_length = metrics.get("llm_input_sequence_length")
    if isinstance(input_sequence_length, int):
        totals["input_sequence_length_sum"] += float(input_sequence_length)
        totals["input_sequence_length_count"] += 1

    original_visual_token_count = metrics.get("original_visual_token_count")
    if isinstance(original_visual_token_count, int):
        totals["original_visual_token_sum"] += float(original_visual_token_count)
        totals["original_visual_token_count"] += 1

    original_input_sequence_length = metrics.get("original_llm_input_sequence_length")
    if isinstance(original_input_sequence_length, int):
        totals["original_input_sequence_length_sum"] += float(original_input_sequence_length)
        totals["original_input_sequence_length_count"] += 1

    reallocated_patch_count = metrics.get("reallocated_patch_count")
    if isinstance(reallocated_patch_count, int):
        totals["reallocated_patch_sum"] += float(reallocated_patch_count)
        totals["reallocated_patch_count"] += 1


def summarize_runtime_metric_totals(
    totals: Mapping[str, float | int],
) -> dict[str, float | int | None]:
    visual_token_count = int(totals.get("visual_token_count", 0))
    input_count = int(totals.get("input_sequence_length_count", 0))
    original_visual_token_count = int(totals.get("original_visual_token_count", 0))
    original_input_count = int(totals.get("original_input_sequence_length_count", 0))
    reallocation_count = int(totals.get("reallocated_patch_count", 0))
    visual_token_sum = float(totals.get("visual_token_sum", 0.0))
    input_sum = float(totals.get("input_sequence_length_sum", 0.0))
    original_visual_token_sum = float(totals.get("original_visual_token_sum", 0.0))
    original_input_sum = float(totals.get("original_input_sequence_length_sum", 0.0))
    reallocation_sum = float(totals.get("reallocated_patch_sum", 0.0))

    return {
        "avg_visual_token_count": (
            visual_token_sum / visual_token_count if visual_token_count > 0 else None
        ),
        "visual_token_samples": visual_token_count,
        "avg_llm_input_sequence_length": (
            input_sum / input_count if input_count > 0 else None
        ),
        "llm_input_sequence_length_samples": input_count,
        "avg_original_visual_token_count": (
            original_visual_token_sum / original_visual_token_count
            if original_visual_token_count > 0 else None
        ),
        "original_visual_token_samples": original_visual_token_count,
        "avg_original_llm_input_sequence_length": (
            original_input_sum / original_input_count
            if original_input_count > 0 else None
        ),
        "original_llm_input_sequence_length_samples": original_input_count,
        "avg_reallocated_patch_count": (
            reallocation_sum / reallocation_count if reallocation_count > 0 else None
        ),
        "reallocated_patch_samples": reallocation_count,
    }


def _format_average_metric(value: float | int | None, *, samples: int) -> str:
    resolved_value = _coerce_average_metric(value)
    if resolved_value is None:
        return "N/A"
    return f"{resolved_value:.2f} (n={samples})"


def format_runtime_summary_lines(
    summary: Mapping[str, float | int | None],
) -> list[tuple[str, str]]:
    return [
        (
            "Avg Visual",
            _format_average_metric(
                summary.get("avg_visual_token_count"),
                samples=int(summary.get("visual_token_samples", 0) or 0),
            ),
        ),
        (
            "Avg LLM Seq",
            _format_average_metric(
                summary.get("avg_llm_input_sequence_length"),
                samples=int(summary.get("llm_input_sequence_length_samples", 0) or 0),
            ),
        ),
        (
            "Avg Orig Vis",
            _format_average_metric(
                summary.get("avg_original_visual_token_count"),
                samples=int(summary.get("original_visual_token_samples", 0) or 0),
            ),
        ),
        (
            "Avg Orig Seq",
            _format_average_metric(
                summary.get("avg_original_llm_input_sequence_length"),
                samples=int(summary.get("original_llm_input_sequence_length_samples", 0) or 0),
            ),
        ),
        (
            "Avg Realloc",
            _format_average_metric(
                summary.get("avg_reallocated_patch_count"),
                samples=int(summary.get("reallocated_patch_samples", 0) or 0),
            ),
        ),
    ]
