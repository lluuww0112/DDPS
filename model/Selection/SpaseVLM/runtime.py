from __future__ import annotations

"""SparseVLM runtime patch for DDPS.

This module ports the core SparseVLM inference idea into DDPS without pulling
in the full upstream LLaVA fork. Like the existing FastV integration, it wraps
Hugging Face LLaVA's LLaMA decoder at generation time.
"""

import math
import types
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import create_causal_mask


_SPARSE_TOKEN_SCHEDULES = {
    False: {
        192: (300, 200, 110),
        128: (303, 110, 36),
        96: (238, 48, 26),
        64: (66, 30, 17),
    },
    True: {
        192: (300, 200, 118),
        128: (238, 108, 60),
        96: (246, 54, 28),
        64: (66, 34, 20),
    },
}


@dataclass(slots=True)
class SparseVLMState:
    enabled: bool = True
    retain_tokens: int = 192
    use_v2: bool = False
    pruning_layers: tuple[int, ...] = (2, 6, 15)
    keep_schedule: tuple[int, ...] | None = None
    enable_token_recycling: bool = False
    recycle_topk_ratio: float = 0.3
    recycle_cluster_divisor: int = 10
    image_token_ranges: list[tuple[int, int]] = field(default_factory=list)
    last_info: dict[str, Any] = field(default_factory=dict)
    generation_infos: list[dict[str, Any]] = field(default_factory=list)
    pending_generation_kwargs: dict[str, torch.Tensor] = field(default_factory=dict)


def _resolve_language_model(model: Any) -> Any:
    candidates = [
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "language_model", None),
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate
    raise TypeError(
        "SparseVLM currently expects a LLaVA model exposing a LLaMA language_model."
    )


def _resolve_keep_schedule(state: SparseVLMState) -> tuple[int, ...]:
    if state.keep_schedule is not None:
        schedule = tuple(int(value) for value in state.keep_schedule)
    else:
        version_schedules = _SPARSE_TOKEN_SCHEDULES[state.use_v2]
        if state.retain_tokens not in version_schedules:
            raise ValueError(
                "Unsupported `retain_tokens` for SparseVLM. "
                "Use one of {64, 96, 128, 192} or pass `keep_schedule` explicitly."
            )
        schedule = version_schedules[state.retain_tokens]

    if len(schedule) != len(state.pruning_layers):
        raise ValueError(
            "SparseVLM `keep_schedule` length must match `pruning_layers`: "
            f"schedule={len(schedule)}, pruning_layers={len(state.pruning_layers)}."
        )
    return schedule


def _run_decoder_layer_with_attention(
    decoder_layer: Any,
    hidden_states: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
    position_ids: torch.LongTensor | None,
    past_key_values: Any,
    use_cache: bool | None,
    cache_position: torch.LongTensor | None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    kwargs: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    residual = hidden_states
    hidden_states = decoder_layer.input_layernorm(hidden_states)
    attention_output, attention_weights = decoder_layer.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + attention_output

    residual = hidden_states
    hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
    hidden_states = decoder_layer.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states, attention_weights


def _gather_sequence(values: torch.Tensor, keep_indices: torch.Tensor) -> torch.Tensor:
    batch_size = int(values.shape[0])
    batch_indices = torch.arange(batch_size, device=values.device).unsqueeze(1)
    return values[batch_indices, keep_indices.to(values.device)]


def _gather_cache_sequence(values: torch.Tensor, keep_indices: torch.Tensor) -> torch.Tensor:
    batch_size = int(values.shape[0])
    batch_indices = torch.arange(batch_size, device=values.device).view(-1, 1, 1)
    head_indices = torch.arange(values.shape[1], device=values.device).view(1, -1, 1)
    sequence_indices = keep_indices.to(values.device).unsqueeze(1)
    return values[batch_indices, head_indices, sequence_indices, :]


def _cluster_and_merge_cache(x: torch.Tensor, cluster_num: int) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(
            f"Expected cache merge candidates with shape (B, H, N, D), got {tuple(x.shape)}."
        )
    batch_size, num_heads, token_count, head_dim = x.shape
    flattened = x.permute(0, 2, 1, 3).reshape(batch_size, token_count, num_heads * head_dim)
    merged = _cluster_and_merge(flattened, cluster_num)
    return merged.reshape(batch_size, cluster_num, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()


def _insert_recycled_cache_tokens(
    selected_cache: torch.Tensor,
    *,
    merged_cache: torch.Tensor,
    image_token_ranges: list[tuple[int, int]],
    kept_visual_tokens: int,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for batch_index, (start, _) in enumerate(image_token_ranges):
        insert_at = int(start) + int(kept_visual_tokens)
        row = torch.cat(
            (
                selected_cache[batch_index, :, :insert_at, :],
                merged_cache[batch_index],
                selected_cache[batch_index, :, insert_at:, :],
            ),
            dim=1,
        )
        rows.append(row)
    return torch.stack(rows, dim=0)


def _prune_dynamic_cache(
    past_key_values: Any,
    keep_indices: torch.Tensor,
    *,
    pruned_indices: torch.Tensor | None = None,
    top_candidate_indices: torch.Tensor | None = None,
    merged_count: int = 0,
    image_token_ranges: list[tuple[int, int]] | None = None,
    kept_visual_tokens: int = 0,
) -> None:
    layers = getattr(past_key_values, "layers", None)
    if not isinstance(layers, list):
        raise TypeError("SparseVLM KV-cache pruning requires a Transformers DynamicCache.")

    for layer in layers:
        if not bool(getattr(layer, "is_initialized", False)):
            continue
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
            raise TypeError("SparseVLM encountered an unsupported cache layer.")
        if int(keys.shape[-2]) <= int(keep_indices.max().item()):
            continue

        selected_keys = _gather_cache_sequence(keys, keep_indices)
        selected_values = _gather_cache_sequence(values, keep_indices)
        if (
            merged_count <= 0
            or pruned_indices is None
            or top_candidate_indices is None
            or image_token_ranges is None
        ):
            layer.keys = selected_keys
            layer.values = selected_values
            continue

        pruned_keys = _gather_cache_sequence(keys, pruned_indices)
        pruned_values = _gather_cache_sequence(values, pruned_indices)
        merge_candidate_keys = _gather_cache_sequence(pruned_keys, top_candidate_indices)
        merge_candidate_values = _gather_cache_sequence(pruned_values, top_candidate_indices)
        merged_keys = _cluster_and_merge_cache(merge_candidate_keys, merged_count)
        merged_values = _cluster_and_merge_cache(merge_candidate_values, merged_count)
        layer.keys = _insert_recycled_cache_tokens(
            selected_keys,
            merged_cache=merged_keys,
            image_token_ranges=image_token_ranges,
            kept_visual_tokens=kept_visual_tokens,
        )
        layer.values = _insert_recycled_cache_tokens(
            selected_values,
            merged_cache=merged_values,
            image_token_ranges=image_token_ranges,
            kept_visual_tokens=kept_visual_tokens,
        )


def _compute_text_rater_indices(
    hidden_states: torch.Tensor,
    *,
    image_token_ranges: list[tuple[int, int]],
) -> list[torch.Tensor]:
    raters: list[torch.Tensor] = []
    for batch_index, (start, length) in enumerate(image_token_ranges):
        text_start = int(start) + int(length)
        visual_tokens = hidden_states[batch_index, int(start) : text_start, :]
        text_tokens = hidden_states[batch_index, text_start:, :]
        if visual_tokens.numel() == 0 or text_tokens.numel() == 0:
            raters.append(torch.zeros(0, dtype=torch.long, device=hidden_states.device))
            continue

        text_scores = (visual_tokens @ text_tokens.transpose(0, 1)).softmax(dim=1).mean(dim=0)
        selected = torch.nonzero(text_scores > text_scores.mean(), as_tuple=False).flatten()
        if int(selected.numel()) == 0:
            selected = torch.topk(text_scores, k=1, largest=True, sorted=True).indices
        raters.append(selected.to(dtype=torch.long))
    return raters


def _cluster_and_merge(x: torch.Tensor, cluster_num: int) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected merge candidates with shape (B, N, C), got {tuple(x.shape)}.")
    batch_size, token_count, hidden_dim = x.shape
    if token_count == 0:
        return x[:, :0, :]

    cluster_num = max(1, min(int(cluster_num), int(token_count)))
    if cluster_num >= token_count:
        return x

    # CUDA `cdist` does not support bf16 inputs, so compute pairwise distances in fp32.
    distance = torch.cdist(x.float(), x.float(), p=2) / math.sqrt(float(hidden_dim))
    dist_nearest, _ = torch.topk(distance, k=cluster_num, dim=-1, largest=False)
    density = (-(dist_nearest**2).mean(dim=-1)).exp()
    density = density + torch.rand_like(density) * 1e-6

    mask = (density[:, None, :] > density[:, :, None]).to(x.dtype)
    dist_max = distance.flatten(1).max(dim=-1).values[:, None, None]
    dist, _ = (distance * mask + dist_max * (1.0 - mask)).min(dim=-1)
    score = dist * density
    _, centers = torch.topk(score, k=cluster_num, dim=-1)

    batch_indices = torch.arange(batch_size, device=x.device)[:, None].expand(batch_size, cluster_num)
    dist_to_centers = distance[batch_indices, centers]
    cluster_index = dist_to_centers.argmin(dim=1)

    cluster_slots = torch.arange(cluster_num, device=x.device)[None, :].expand(batch_size, cluster_num)
    cluster_index[batch_indices.reshape(-1), centers.reshape(-1)] = cluster_slots.reshape(-1)

    flat_cluster_index = cluster_index + torch.arange(batch_size, device=x.device)[:, None] * cluster_num
    token_weight = x.new_ones(batch_size, token_count, 1)
    all_weight = x.new_zeros(batch_size * cluster_num, 1)
    all_weight.index_add_(0, flat_cluster_index.reshape(batch_size * token_count), token_weight.reshape(batch_size * token_count, 1))
    all_weight = all_weight + 1e-6
    norm_weight = token_weight / all_weight[flat_cluster_index]

    merged = x.new_zeros(batch_size * cluster_num, hidden_dim)
    merged.index_add_(0, flat_cluster_index.reshape(batch_size * token_count), (x * norm_weight).reshape(batch_size * token_count, hidden_dim))
    return merged.reshape(batch_size, cluster_num, hidden_dim)


def _build_sparse_keep_indices(
    attention_weights: torch.Tensor,
    *,
    image_token_ranges: list[tuple[int, int]],
    text_rater_indices: list[torch.Tensor],
    keep_count: int,
    use_v2: bool,
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    if attention_weights.ndim != 4:
        raise ValueError(
            "SparseVLM expected attention weights with shape (B, H, Q, K), "
            f"got {tuple(attention_weights.shape)}."
        )

    batch_size = int(attention_weights.shape[0])
    keep_rows: list[torch.Tensor] = []
    pruned_rows: list[torch.Tensor] = []
    relation_rows: list[torch.Tensor] = []
    item_infos: list[dict[str, Any]] = []
    num_heads = int(attention_weights.shape[1])
    top_head_count = max(1, num_heads // 2)

    for batch_index, (start, length) in enumerate(image_token_ranges):
        start = int(start)
        length = int(length)
        text_start = start + length
        if length <= 0:
            raise ValueError("SparseVLM image token length must be positive.")
        if start < 0 or text_start > sequence_length:
            raise ValueError(
                "SparseVLM image token range is outside the current sequence: "
                f"start={start}, length={length}, sequence_length={sequence_length}."
            )

        relative_text = text_rater_indices[batch_index]
        if int(relative_text.numel()) == 0:
            text_indices = torch.arange(text_start, sequence_length, device=attention_weights.device)
        else:
            text_indices = relative_text.to(attention_weights.device) + text_start
            text_indices = text_indices[text_indices < sequence_length]
            if int(text_indices.numel()) == 0:
                text_indices = torch.arange(text_start, sequence_length, device=attention_weights.device)

        head_slice = attention_weights[batch_index]
        if use_v2 and int(text_indices.numel()) > 0 and num_heads > 1:
            head_scores = head_slice[:, text_indices, start:text_start].sum(dim=(1, 2))
            selected_heads = torch.topk(head_scores, k=top_head_count, largest=True, sorted=False).indices
            head_slice = head_slice[selected_heads]

        if int(text_indices.numel()) == 0:
            relation = head_slice.mean(dim=0)[-1, start:text_start]
        else:
            relation = head_slice.mean(dim=0)[text_indices, start:text_start].mean(dim=0)

        current_keep_count = max(1, min(int(keep_count), length))
        keep_relative = torch.topk(
            relation,
            k=current_keep_count,
            largest=True,
            sorted=False,
        ).indices
        keep_mask = torch.zeros(length, dtype=torch.bool, device=attention_weights.device)
        keep_mask[keep_relative] = True
        keep_visual_indices = (keep_mask.nonzero(as_tuple=False).flatten() + start).sort().values
        pruned_visual_indices = ((~keep_mask).nonzero(as_tuple=False).flatten() + start).sort().values
        keep_indices = torch.cat(
            (
                torch.arange(start, device=attention_weights.device),
                keep_visual_indices,
                torch.arange(text_start, sequence_length, device=attention_weights.device),
            )
        )

        keep_rows.append(keep_indices)
        pruned_rows.append(pruned_visual_indices)
        relation_rows.append(relation)
        item_infos.append(
            {
                "image_token_start_index": start,
                "original_image_tokens": length,
                "selected_image_tokens": int(current_keep_count),
                "selected_text_tokens": int(text_indices.numel()),
                "pruned_image_tokens": int(length - current_keep_count),
            }
        )

    return (
        torch.stack(keep_rows, dim=0),
        torch.stack(pruned_rows, dim=0),
        torch.stack(relation_rows, dim=0),
        item_infos,
    )


def _insert_recycled_tokens(
    selected_hidden_states: torch.Tensor,
    *,
    selected_attention_mask: torch.Tensor | None,
    selected_position_ids: torch.Tensor,
    merged_tokens: torch.Tensor,
    image_token_ranges: list[tuple[int, int]],
    kept_visual_tokens: int,
    use_v2: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    merged_count = int(merged_tokens.shape[1])
    rows_hidden: list[torch.Tensor] = []
    rows_mask: list[torch.Tensor] = []
    rows_position_ids: list[torch.Tensor] = []

    for batch_index, (start, _) in enumerate(image_token_ranges):
        insert_at = int(start) + int(kept_visual_tokens)
        row_hidden = torch.cat(
            (
                selected_hidden_states[batch_index, :insert_at, :],
                merged_tokens[batch_index],
                selected_hidden_states[batch_index, insert_at:, :],
            ),
            dim=0,
        )
        rows_hidden.append(row_hidden)

        if selected_attention_mask is not None:
            row_mask = torch.cat(
                (
                    selected_attention_mask[batch_index, :insert_at],
                    torch.ones(merged_count, dtype=selected_attention_mask.dtype, device=selected_attention_mask.device),
                    selected_attention_mask[batch_index, insert_at:],
                ),
                dim=0,
            )
            rows_mask.append(row_mask)

        if use_v2 and merged_count == 0:
            row_position_ids = selected_position_ids[batch_index]
        else:
            row_position_ids = torch.arange(
                row_hidden.shape[0],
                dtype=selected_position_ids.dtype,
                device=selected_position_ids.device,
            )
        rows_position_ids.append(row_position_ids)

    hidden_states = torch.stack(rows_hidden, dim=0)
    attention_mask = torch.stack(rows_mask, dim=0) if rows_mask else None
    position_ids = torch.stack(rows_position_ids, dim=0)
    return hidden_states, attention_mask, position_ids


def install_sparsevlm(
    model: Any,
    sparsevlm_config: dict[str, Any] | None = None,
) -> SparseVLMState:
    language_model = _resolve_language_model(model)
    existing_state = getattr(language_model, "_ddps_sparsevlm_state", None)
    config = dict(sparsevlm_config or {})

    if isinstance(existing_state, SparseVLMState):
        existing_state.enabled = bool(config.get("use_sparsevlm", existing_state.enabled))
        existing_state.retain_tokens = int(config.get("retain_tokens", existing_state.retain_tokens))
        existing_state.use_v2 = bool(config.get("use_v2", existing_state.use_v2))
        existing_state.pruning_layers = tuple(config.get("pruning_layers", existing_state.pruning_layers))
        keep_schedule = config.get("keep_schedule")
        existing_state.keep_schedule = tuple(keep_schedule) if keep_schedule is not None else existing_state.keep_schedule
        existing_state.enable_token_recycling = bool(
            config.get("enable_token_recycling", existing_state.enable_token_recycling)
        )
        existing_state.recycle_topk_ratio = float(
            config.get("recycle_topk_ratio", existing_state.recycle_topk_ratio)
        )
        existing_state.recycle_cluster_divisor = int(
            config.get("recycle_cluster_divisor", existing_state.recycle_cluster_divisor)
        )
        return existing_state

    state = SparseVLMState(
        enabled=bool(config.get("use_sparsevlm", True)),
        retain_tokens=int(config.get("retain_tokens", 192)),
        use_v2=bool(config.get("use_v2", False)),
        pruning_layers=tuple(config.get("pruning_layers", (2, 6, 15))),
        keep_schedule=(tuple(config["keep_schedule"]) if config.get("keep_schedule") is not None else None),
        enable_token_recycling=bool(config.get("enable_token_recycling", False)),
        recycle_topk_ratio=float(config.get("recycle_topk_ratio", 0.3)),
        recycle_cluster_divisor=int(config.get("recycle_cluster_divisor", 10)),
    )
    original_forward = language_model.forward
    original_update_model_kwargs_for_generation = getattr(
        model,
        "_ddps_sparsevlm_original_update_model_kwargs_for_generation",
        model._update_model_kwargs_for_generation,
    )

    def sparsevlm_forward(
        self: Any,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast:
        if not state.enabled or not state.image_token_ranges:
            return original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                use_cache=use_cache,
                **kwargs,
            )

        is_prefill = past_key_values is None or past_key_values.get_seq_length() == 0
        if not is_prefill:
            return original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                use_cache=use_cache,
                **kwargs,
            )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if not use_cache:
            raise ValueError(
                "SparseVLM prefill pruning requires `use_cache=True` so subsequent "
                "decode steps can reuse the pruned KV cache."
            )
        if past_key_values is None:
            from transformers.cache_utils import DynamicCache

            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.shape[0] == 1 and inputs_embeds.shape[0] > 1:
            position_ids = position_ids.expand(int(inputs_embeds.shape[0]), -1)

        keep_schedule = _resolve_keep_schedule(state)
        batch_size = int(inputs_embeds.shape[0])
        sequence_length = int(inputs_embeds.shape[1])
        current_ranges = [(int(start), int(length)) for start, length in state.image_token_ranges]
        original_ranges = list(current_ranges)
        original_sequence_length = sequence_length
        current_visual_length = current_ranges[0][1]
        state.pending_generation_kwargs = {}
        text_rater_indices = _compute_text_rater_indices(inputs_embeds, image_token_ranges=current_ranges)
        original_attn_implementation = self.config._attn_implementation

        self.config._attn_implementation = "eager"
        try:
            causal_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
        finally:
            self.config._attn_implementation = original_attn_implementation

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        item_records = [
            {
                "applied": False,
                "selector_type": "sparsevlm",
                "retain_tokens": int(state.retain_tokens),
                "use_v2": bool(state.use_v2),
                "stages": [],
            }
            for _ in range(batch_size)
        ]

        for layer_index, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if layer_index in state.pruning_layers and sequence_length > 1 and current_visual_length > 0:
                stage_index = tuple(state.pruning_layers).index(int(layer_index))
                if stage_index is None or stage_index >= len(keep_schedule):
                    raise ValueError(
                        f"SparseVLM pruning layer {layer_index} does not have a configured keep schedule."
                    )

                self.config._attn_implementation = "eager"
                try:
                    hidden_states, captured_attention = _run_decoder_layer_with_attention(
                        decoder_layer,
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        kwargs=kwargs,
                    )
                finally:
                    self.config._attn_implementation = original_attn_implementation
                if captured_attention is None:
                    raise RuntimeError("SparseVLM could not capture eager attention weights.")

                keep_indices, pruned_indices, relation_scores, stage_infos = _build_sparse_keep_indices(
                    captured_attention,
                    image_token_ranges=current_ranges,
                    text_rater_indices=text_rater_indices,
                    keep_count=int(keep_schedule[stage_index]),
                    use_v2=state.use_v2,
                    sequence_length=sequence_length,
                )

                kept_visual_tokens = int(stage_infos[0]["selected_image_tokens"])
                selected_hidden_states = _gather_sequence(hidden_states, keep_indices)
                selected_attention_mask = (
                    _gather_sequence(attention_mask, keep_indices) if attention_mask is not None else None
                )
                selected_position_ids = _gather_sequence(position_ids, keep_indices)

                merged_count = 0
                top_candidate_indices: torch.Tensor | None = None
                if (
                    state.enable_token_recycling
                    and int(pruned_indices.shape[1]) > 0
                    and float(state.recycle_topk_ratio) > 0.0
                ):
                    pruned_hidden_states = _gather_sequence(hidden_states, pruned_indices)
                    pruned_count = int(pruned_hidden_states.shape[1])
                    top_candidate_count = max(
                        1,
                        min(pruned_count, int(pruned_count * float(state.recycle_topk_ratio)) + 1),
                    )
                    relation_pruned = relation_scores[
                        torch.arange(batch_size, device=relation_scores.device).unsqueeze(1),
                        (pruned_indices - torch.tensor([start for start, _ in current_ranges], device=relation_scores.device).unsqueeze(1)),
                    ]
                    top_candidate_indices = torch.topk(
                        relation_pruned,
                        k=top_candidate_count,
                        largest=True,
                        sorted=False,
                    ).indices
                    batch_indices = torch.arange(batch_size, device=pruned_hidden_states.device).unsqueeze(1)
                    merge_candidates = pruned_hidden_states[batch_indices, top_candidate_indices]
                    merged_count = max(
                        1,
                        min(
                            int(merge_candidates.shape[1]),
                            int(int(merge_candidates.shape[1]) / max(1, state.recycle_cluster_divisor)) + 1,
                        ),
                    )
                    merged_tokens = _cluster_and_merge(merge_candidates, merged_count)
                    hidden_states, attention_mask, position_ids = _insert_recycled_tokens(
                        selected_hidden_states,
                        selected_attention_mask=selected_attention_mask,
                        selected_position_ids=selected_position_ids,
                        merged_tokens=merged_tokens,
                        image_token_ranges=current_ranges,
                        kept_visual_tokens=kept_visual_tokens,
                        use_v2=state.use_v2,
                    )
                else:
                    hidden_states = selected_hidden_states
                    attention_mask = selected_attention_mask
                    position_ids = selected_position_ids if state.use_v2 else torch.arange(
                        selected_hidden_states.shape[1],
                        dtype=selected_position_ids.dtype,
                        device=selected_position_ids.device,
                    ).unsqueeze(0).expand(batch_size, -1)

                current_visual_length = kept_visual_tokens + merged_count
                current_ranges = [(start, current_visual_length) for start, _ in current_ranges]
                sequence_length = int(hidden_states.shape[1])
                cache_position = torch.arange(sequence_length, device=hidden_states.device)
                _prune_dynamic_cache(
                    past_key_values,
                    keep_indices,
                    pruned_indices=pruned_indices,
                    top_candidate_indices=top_candidate_indices,
                    merged_count=merged_count,
                    image_token_ranges=current_ranges,
                    kept_visual_tokens=kept_visual_tokens,
                )
                self.config._attn_implementation = "eager"
                try:
                    causal_mask = create_causal_mask(
                        config=self.config,
                        inputs_embeds=hidden_states,
                        attention_mask=attention_mask,
                        cache_position=cache_position,
                        past_key_values=None,
                        position_ids=position_ids,
                    )
                finally:
                    self.config._attn_implementation = original_attn_implementation
                position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

                for batch_index, stage_info in enumerate(stage_infos):
                    applied = int(stage_info["pruned_image_tokens"]) > 0 or merged_count > 0
                    item_records[batch_index]["applied"] = item_records[batch_index]["applied"] or applied
                    item_records[batch_index]["stages"].append(
                        {
                            "layer": int(layer_index),
                            "applied": applied,
                            "keep_visual_tokens": int(stage_info["selected_image_tokens"]),
                            "merged_visual_tokens": int(merged_count),
                            "output_visual_tokens": int(current_visual_length),
                            "selected_text_tokens": int(stage_info["selected_text_tokens"]),
                        }
                    )
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs,
                )

        hidden_states = self.norm(hidden_states)
        selected_counts = [int(length) for _, length in current_ranges]
        original_counts = [int(length) for _, length in original_ranges]
        for batch_index, record in enumerate(item_records):
            record["original_image_tokens"] = original_counts[batch_index]
            record["selected_image_tokens"] = selected_counts[batch_index]
            record["input_length_before"] = int(original_sequence_length)
            record["input_length_after"] = int(sequence_length)
            record["selected_text_tokens"] = int(text_rater_indices[batch_index].numel())
            record["backend"] = "llava"

        pending_generation_kwargs: dict[str, torch.Tensor] = {
            "cache_position": cache_position,
            "position_ids": position_ids,
        }
        if attention_mask is not None:
            pending_generation_kwargs["attention_mask"] = attention_mask
        state.pending_generation_kwargs = pending_generation_kwargs

        applied = any(bool(item.get("applied", False)) for item in item_records)
        state.last_info = {
            "applied": applied,
            "backend": "llava",
            "selector_type": "sparsevlm",
            "retain_tokens": int(state.retain_tokens),
            "use_v2": bool(state.use_v2),
            "pruning_layers": list(state.pruning_layers),
            "keep_schedule": list(keep_schedule),
            "batch_size": batch_size,
            "original_image_tokens": original_counts[0] if len(set(original_counts)) == 1 else None,
            "selected_image_tokens": selected_counts[0] if len(set(selected_counts)) == 1 else None,
            "input_length_before": int(original_sequence_length),
            "input_length_after": int(sequence_length),
            "items": item_records,
            "selector_metadata": {
                "selector_type": "sparsevlm",
                "retain_tokens": int(state.retain_tokens),
                "use_v2": bool(state.use_v2),
                "pruning_layers": list(state.pruning_layers),
                "keep_schedule": list(keep_schedule),
                "image_token_count": original_counts[0] if len(set(original_counts)) == 1 else None,
                "selected_token_count": selected_counts[0] if len(set(selected_counts)) == 1 else None,
            },
        }
        state.generation_infos.append(dict(state.last_info))
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def sparsevlm_update_model_kwargs_for_generation(
        self: Any,
        outputs: Any,
        model_kwargs: dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> dict[str, Any]:
        if state.pending_generation_kwargs:
            model_kwargs.update(state.pending_generation_kwargs)
            state.pending_generation_kwargs = {}
        return original_update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )

    language_model.forward = types.MethodType(sparsevlm_forward, language_model)
    language_model._ddps_sparsevlm_state = state
    language_model._ddps_sparsevlm_original_forward = original_forward
    model._update_model_kwargs_for_generation = types.MethodType(
        sparsevlm_update_model_kwargs_for_generation,
        model,
    )
    model._ddps_sparsevlm_original_update_model_kwargs_for_generation = (
        original_update_model_kwargs_for_generation
    )
    return state
