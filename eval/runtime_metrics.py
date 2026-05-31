from __future__ import annotations

from collections.abc import Mapping
from typing import Any

MetricValue = int | float | bool | None


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


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _config_get(config: Any, key: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(key)
    return getattr(config, key, None)


def _iter_model_configs(vlm: Any) -> list[Any]:
    if vlm is None:
        return []

    model = getattr(vlm, "model", vlm)
    configs: list[Any] = []
    for attr in ("language_model", "text_model"):
        module = getattr(model, attr, None)
        config = getattr(module, "config", None)
        if config is not None:
            configs.append(config)

    model_config = getattr(model, "config", None)
    if model_config is not None:
        text_config = _config_get(model_config, "text_config")
        if text_config is not None:
            configs.append(text_config)
        configs.append(model_config)
    return configs


def _first_config_int(vlm: Any, key: str) -> int | None:
    for config in _iter_model_configs(vlm):
        value = _coerce_int(_config_get(config, key))
        if value is not None:
            return value
    return None


def _estimate_image_token_count(vlm: Any) -> int | None:
    if vlm is None:
        return None

    model = getattr(vlm, "model", vlm)
    config = getattr(model, "config", None)
    vision_config = _config_get(config, "vision_config")
    image_size = _coerce_int(_config_get(vision_config, "image_size"))
    patch_size = _coerce_int(_config_get(vision_config, "patch_size"))
    if image_size is None or patch_size is None or patch_size <= 0:
        return None

    grid = image_size // patch_size
    if grid <= 0:
        return None

    token_count = grid * grid
    strategy = str(_config_get(config, "vision_feature_select_strategy") or "default")
    if strategy != "default":
        token_count += 1
    return token_count


def estimate_prefill_tflops(vlm: Any, sequence_length: int | None) -> float | None:
    """Estimate decoder-only LLM prefill TFLOPs for one sample.

    This model-shape estimate covers the language-model prefill pass. It
    excludes the vision encoder, projector, embeddings, and lm_head.
    """
    if sequence_length is None or sequence_length <= 0:
        return None

    hidden_size = _first_config_int(vlm, "hidden_size")
    intermediate_size = _first_config_int(vlm, "intermediate_size")
    num_layers = _first_config_int(vlm, "num_hidden_layers")
    num_heads = _first_config_int(vlm, "num_attention_heads")
    num_kv_heads = _first_config_int(vlm, "num_key_value_heads") or num_heads
    if (
        hidden_size is None
        or intermediate_size is None
        or num_layers is None
        or num_heads is None
        or num_heads <= 0
    ):
        return None

    length = float(sequence_length)
    hidden = float(hidden_size)
    intermediate = float(intermediate_size)
    layers = float(num_layers)
    kv_ratio = float(num_kv_heads or num_heads) / float(num_heads)

    attention_linear_flops = (4.0 + 4.0 * kv_ratio) * length * hidden * hidden
    mlp_flops = 6.0 * length * hidden * intermediate
    attention_context_flops = 4.0 * length * length * hidden
    return layers * (
        attention_linear_flops + mlp_flops + attention_context_flops
    ) / 1_000_000_000_000.0


def _with_prefill_metrics(
    metrics: dict[str, MetricValue],
    vlm: Any,
) -> dict[str, MetricValue]:
    prefill_tokens = _coerce_int(metrics.get("llm_input_sequence_length"))
    original_prefill_tokens = _coerce_int(
        metrics.get("original_llm_input_sequence_length")
    )
    if original_prefill_tokens is None:
        original_prefill_tokens = prefill_tokens

    metrics["prefill_token_count"] = prefill_tokens
    metrics["original_prefill_token_count"] = original_prefill_tokens
    metrics["prefill_tflops"] = estimate_prefill_tflops(vlm, prefill_tokens)
    metrics["original_prefill_tflops"] = estimate_prefill_tflops(
        vlm,
        original_prefill_tokens,
    )
    return metrics


def extract_runtime_metrics(
    vlm: Any,
    *,
    timing_info: Mapping[str, Any] | None = None,
    patch_info: Mapping[str, Any] | None = None,
    prefer_patch_sequence_lengths: bool = False,
) -> dict[str, MetricValue]:
    timing_info = _as_mapping(
        timing_info if timing_info is not None else getattr(vlm, "last_timing_info", {})
    )
    patch_info = _as_mapping(
        patch_info if patch_info is not None else getattr(vlm, "last_patch_selection_info", {})
    )
    selector_metadata = patch_info.get("selector_metadata")
    if not isinstance(selector_metadata, Mapping):
        selector_metadata = {}

    timing_input_sequence_length = _coerce_int(timing_info.get("input_sequence_length"))
    if timing_input_sequence_length is None:
        timing_input_sequence_length = _coerce_int(
            timing_info.get("llm_input_sequence_length")
        )
    patch_input_sequence_length = _coerce_int(patch_info.get("input_length_after"))
    if prefer_patch_sequence_lengths:
        input_sequence_length = patch_input_sequence_length or timing_input_sequence_length
    else:
        input_sequence_length = timing_input_sequence_length or patch_input_sequence_length

    input_sequence_length_before_patch = _coerce_int(
        patch_info.get("input_length_before")
    )
    original_llm_input_sequence_length = input_sequence_length_before_patch
    if original_llm_input_sequence_length is None:
        original_llm_input_sequence_length = input_sequence_length

    selected_video_tokens = _coerce_int(patch_info.get("selected_video_tokens"))
    if selected_video_tokens is None:
        selected_video_tokens = _coerce_int(patch_info.get("selected_image_tokens"))
    if selected_video_tokens is None:
        selected_video_tokens = _coerce_int(selector_metadata.get("selected_token_count"))

    original_video_tokens = _coerce_int(patch_info.get("original_video_tokens"))
    if original_video_tokens is None:
        original_video_tokens = _coerce_int(patch_info.get("original_image_tokens"))
    if original_video_tokens is None:
        original_video_tokens = _coerce_int(patch_info.get("image_token_count"))
    if original_video_tokens is None:
        original_video_tokens = _coerce_int(selector_metadata.get("image_token_count"))

    if selected_video_tokens is None and not bool(patch_info.get("applied", False)):
        selected_video_tokens = _estimate_image_token_count(vlm)
    if original_video_tokens is None:
        original_video_tokens = selected_video_tokens

    reallocated_patch_count = _coerce_int(patch_info.get("reallocated_token_count"))
    if reallocated_patch_count is None:
        reallocated_patch_count = _coerce_int(
            selector_metadata.get("reallocated_token_count")
        )

    return _with_prefill_metrics({
        "patch_selection_applied": bool(patch_info.get("applied", False)),
        "llm_input_sequence_length": input_sequence_length,
        "original_llm_input_sequence_length": original_llm_input_sequence_length,
        "llm_input_sequence_length_before_patch_selection": input_sequence_length_before_patch,
        "visual_token_count": selected_video_tokens,
        "original_visual_token_count": original_video_tokens,
        "original_video_token_count": original_video_tokens,
        "selected_video_token_count": selected_video_tokens,
        "reallocated_patch_count": reallocated_patch_count,
    }, vlm)


def extract_runtime_metrics_from_result(
    row: Mapping[str, Any],
    vlm: Any = None,
) -> dict[str, MetricValue]:
    runtime = _as_mapping(row.get("runtime"))
    patch_info = _as_mapping(row.get("patch_selection"))
    return extract_runtime_metrics(
        vlm,
        timing_info=runtime,
        patch_info=patch_info,
        prefer_patch_sequence_lengths=True,
    )


def init_runtime_metric_totals() -> dict[str, float | int]:
    return {
        "visual_token_sum": 0.0,
        "visual_token_count": 0,
        "prefill_token_sum": 0.0,
        "prefill_token_count": 0,
        "prefill_tflops_sum": 0.0,
        "prefill_tflops_count": 0,
        "input_sequence_length_sum": 0.0,
        "input_sequence_length_count": 0,
        "original_visual_token_sum": 0.0,
        "original_visual_token_count": 0,
        "original_prefill_token_sum": 0.0,
        "original_prefill_token_count": 0,
        "original_prefill_tflops_sum": 0.0,
        "original_prefill_tflops_count": 0,
        "original_input_sequence_length_sum": 0.0,
        "original_input_sequence_length_count": 0,
        "reallocated_patch_sum": 0.0,
        "reallocated_patch_count": 0,
    }


def _add_average_value(
    totals: dict[str, float | int],
    *,
    sum_key: str,
    count_key: str,
    value: Any,
) -> None:
    if _is_number(value):
        totals[sum_key] += float(value)
        totals[count_key] += 1


def update_runtime_metric_totals(
    totals: dict[str, float | int],
    metrics: Mapping[str, MetricValue],
) -> None:
    _add_average_value(
        totals,
        sum_key="visual_token_sum",
        count_key="visual_token_count",
        value=metrics.get("visual_token_count"),
    )
    _add_average_value(
        totals,
        sum_key="prefill_token_sum",
        count_key="prefill_token_count",
        value=metrics.get("prefill_token_count"),
    )
    _add_average_value(
        totals,
        sum_key="prefill_tflops_sum",
        count_key="prefill_tflops_count",
        value=metrics.get("prefill_tflops"),
    )
    _add_average_value(
        totals,
        sum_key="input_sequence_length_sum",
        count_key="input_sequence_length_count",
        value=metrics.get("llm_input_sequence_length"),
    )
    _add_average_value(
        totals,
        sum_key="original_visual_token_sum",
        count_key="original_visual_token_count",
        value=metrics.get("original_visual_token_count"),
    )
    _add_average_value(
        totals,
        sum_key="original_prefill_token_sum",
        count_key="original_prefill_token_count",
        value=metrics.get("original_prefill_token_count"),
    )
    _add_average_value(
        totals,
        sum_key="original_prefill_tflops_sum",
        count_key="original_prefill_tflops_count",
        value=metrics.get("original_prefill_tflops"),
    )
    _add_average_value(
        totals,
        sum_key="original_input_sequence_length_sum",
        count_key="original_input_sequence_length_count",
        value=metrics.get("original_llm_input_sequence_length"),
    )
    _add_average_value(
        totals,
        sum_key="reallocated_patch_sum",
        count_key="reallocated_patch_count",
        value=metrics.get("reallocated_patch_count"),
    )


def summarize_runtime_metric_totals(
    totals: Mapping[str, float | int],
) -> dict[str, float | int | None]:
    visual_token_count = int(totals.get("visual_token_count", 0))
    prefill_token_count = int(totals.get("prefill_token_count", 0))
    prefill_tflops_count = int(totals.get("prefill_tflops_count", 0))
    input_count = int(totals.get("input_sequence_length_count", 0))
    original_visual_token_count = int(totals.get("original_visual_token_count", 0))
    original_prefill_token_count = int(totals.get("original_prefill_token_count", 0))
    original_prefill_tflops_count = int(totals.get("original_prefill_tflops_count", 0))
    original_input_count = int(totals.get("original_input_sequence_length_count", 0))
    reallocation_count = int(totals.get("reallocated_patch_count", 0))
    visual_token_sum = float(totals.get("visual_token_sum", 0.0))
    prefill_token_sum = float(totals.get("prefill_token_sum", 0.0))
    prefill_tflops_sum = float(totals.get("prefill_tflops_sum", 0.0))
    input_sum = float(totals.get("input_sequence_length_sum", 0.0))
    original_visual_token_sum = float(totals.get("original_visual_token_sum", 0.0))
    original_prefill_token_sum = float(totals.get("original_prefill_token_sum", 0.0))
    original_prefill_tflops_sum = float(totals.get("original_prefill_tflops_sum", 0.0))
    original_input_sum = float(totals.get("original_input_sequence_length_sum", 0.0))
    reallocation_sum = float(totals.get("reallocated_patch_sum", 0.0))

    return {
        "avg_visual_token_count": (
            visual_token_sum / visual_token_count if visual_token_count > 0 else None
        ),
        "visual_token_samples": visual_token_count,
        "avg_prefill_token_count": (
            prefill_token_sum / prefill_token_count if prefill_token_count > 0 else None
        ),
        "prefill_token_samples": prefill_token_count,
        "avg_prefill_tflops": (
            prefill_tflops_sum / prefill_tflops_count if prefill_tflops_count > 0 else None
        ),
        "prefill_tflops_samples": prefill_tflops_count,
        "avg_llm_input_sequence_length": (
            input_sum / input_count if input_count > 0 else None
        ),
        "llm_input_sequence_length_samples": input_count,
        "avg_original_visual_token_count": (
            original_visual_token_sum / original_visual_token_count
            if original_visual_token_count > 0 else None
        ),
        "original_visual_token_samples": original_visual_token_count,
        "avg_original_prefill_token_count": (
            original_prefill_token_sum / original_prefill_token_count
            if original_prefill_token_count > 0 else None
        ),
        "original_prefill_token_samples": original_prefill_token_count,
        "avg_original_prefill_tflops": (
            original_prefill_tflops_sum / original_prefill_tflops_count
            if original_prefill_tflops_count > 0 else None
        ),
        "original_prefill_tflops_samples": original_prefill_tflops_count,
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
            "Avg Vis Tok",
            _format_average_metric(
                summary.get("avg_visual_token_count"),
                samples=int(summary.get("visual_token_samples", 0) or 0),
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
            "Avg Prefill",
            _format_average_metric(
                summary.get("avg_prefill_token_count"),
                samples=int(summary.get("prefill_token_samples", 0) or 0),
            ),
        ),
        (
            "Avg Orig Pre",
            _format_average_metric(
                summary.get("avg_original_prefill_token_count"),
                samples=int(summary.get("original_prefill_token_samples", 0) or 0),
            ),
        ),
        (
            "Avg TFLOPs",
            _format_average_metric(
                summary.get("avg_prefill_tflops"),
                samples=int(summary.get("prefill_tflops_samples", 0) or 0),
            ),
        ),
        (
            "Avg Orig TFLOPs",
            _format_average_metric(
                summary.get("avg_original_prefill_tflops"),
                samples=int(summary.get("original_prefill_tflops_samples", 0) or 0),
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
            "Avg Realloc",
            _format_average_metric(
                summary.get("avg_reallocated_patch_count"),
                samples=int(summary.get("reallocated_patch_samples", 0) or 0),
            ),
        ),
    ]
