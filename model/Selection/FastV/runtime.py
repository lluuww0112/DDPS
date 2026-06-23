from __future__ import annotations

"""Standalone DDPS FastV port.

This module intentionally does not import from the cloned official FastV repo.
The official implementation prunes visual tokens in the LLaMA decoder after
ranking them with attention from layer K - 1; this file carries that behavior
as a local runtime patch for Hugging Face LLaVA models.
"""

import types
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers.models.llama.modeling_llama import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast


@dataclass(slots=True)
class FastVState:
    enabled: bool = True
    fastv_k: int = 3
    fastv_r: float = 0.5
    image_token_ranges: list[tuple[int, int]] = field(default_factory=list)
    last_info: dict[str, Any] = field(default_factory=dict)
    generation_infos: list[dict[str, Any]] = field(default_factory=list)


def _resolve_language_model(model: Any) -> Any:
    candidates = [
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "language_model", None),
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate
    raise TypeError("FastV currently expects a LLaVA model exposing a LLaMA language_model.")


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


def _build_keep_indices(
    attention_weights: torch.Tensor,
    *,
    image_token_ranges: list[tuple[int, int]],
    reduction_ratio: float,
    sequence_length: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    if attention_weights.ndim != 4:
        raise ValueError(
            "FastV expected attention weights with shape (B, H, Q, K), "
            f"got {tuple(attention_weights.shape)}."
        )

    batch_size = int(attention_weights.shape[0])
    if len(image_token_ranges) != batch_size:
        raise ValueError(
            "FastV image token range count must match batch size: "
            f"ranges={len(image_token_ranges)}, batch={batch_size}."
        )

    keep_rows: list[torch.Tensor] = []
    item_infos: list[dict[str, Any]] = []
    device = attention_weights.device
    reduction_ratio = max(0.0, min(float(reduction_ratio), 1.0))

    for batch_index, (start, length) in enumerate(image_token_ranges):
        start = int(start)
        length = int(length)
        if length <= 0:
            raise ValueError("FastV image token length must be positive.")
        if start < 0 or start + length > sequence_length:
            raise ValueError(
                "FastV image token range is outside the current sequence: "
                f"start={start}, length={length}, sequence_length={sequence_length}."
            )

        keep_image_count = int(round(length * (1.0 - reduction_ratio)))
        keep_image_count = max(1, min(keep_image_count, length))
        image_attention = attention_weights[batch_index].mean(dim=0)[-1, start : start + length]
        top_image_indices = torch.topk(
            image_attention,
            k=keep_image_count,
            largest=True,
            sorted=False,
        ).indices + start
        keep_indices = torch.cat(
            (
                torch.arange(start, device=device),
                top_image_indices,
                torch.arange(start + length, sequence_length, device=device),
            )
        ).sort().values
        keep_rows.append(keep_indices)
        item_infos.append(
            {
                "applied": True,
                "selector_type": "fastv",
                "fastv_k": None,
                "fastv_r": reduction_ratio,
                "image_token_start_index": start,
                "original_image_tokens": length,
                "selected_image_tokens": keep_image_count,
                "input_length_before": sequence_length,
                "input_length_after": int(keep_indices.numel()),
            }
        )

    return torch.stack(keep_rows, dim=0), item_infos


def _gather_sequence(values: torch.Tensor, keep_indices: torch.Tensor) -> torch.Tensor:
    batch_size = int(values.shape[0])
    batch_indices = torch.arange(batch_size, device=values.device).unsqueeze(1)
    return values[batch_indices, keep_indices.to(values.device)]


def _gather_cache_sequence(
    values: torch.Tensor,
    keep_indices: torch.Tensor,
) -> torch.Tensor:
    batch_size = int(values.shape[0])
    batch_indices = torch.arange(batch_size, device=values.device).view(-1, 1, 1)
    head_indices = torch.arange(values.shape[1], device=values.device).view(1, -1, 1)
    sequence_indices = keep_indices.to(values.device).unsqueeze(1)
    return values[batch_indices, head_indices, sequence_indices, :]


def _prune_dynamic_cache(
    past_key_values: Any,
    keep_indices: torch.Tensor,
) -> None:
    layers = getattr(past_key_values, "layers", None)
    if not isinstance(layers, list):
        raise TypeError("FastV KV-cache pruning requires a Transformers DynamicCache.")

    for layer in layers:
        if not bool(getattr(layer, "is_initialized", False)):
            continue
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
            raise TypeError("FastV encountered an unsupported cache layer.")
        if int(keys.shape[-2]) <= int(keep_indices.max().item()):
            continue
        layer.keys = _gather_cache_sequence(keys, keep_indices)
        layer.values = _gather_cache_sequence(values, keep_indices)


def install_fastv(model: Any, fastv_config: dict[str, Any] | None = None) -> FastVState:
    language_model = _resolve_language_model(model)
    existing_state = getattr(language_model, "_ddps_fastv_state", None)
    config = dict(fastv_config or {})

    if isinstance(existing_state, FastVState):
        existing_state.enabled = bool(config.get("use_fastv", existing_state.enabled))
        existing_state.fastv_k = int(config.get("fastv_k", existing_state.fastv_k))
        existing_state.fastv_r = float(config.get("fastv_r", existing_state.fastv_r))
        return existing_state

    state = FastVState(
        enabled=bool(config.get("use_fastv", True)),
        fastv_k=int(config.get("fastv_k", 3)),
        fastv_r=float(config.get("fastv_r", 0.5)),
    )
    original_forward = language_model.forward

    def fastv_forward(
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
                "FastV prefill pruning requires `use_cache=True` so subsequent "
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

        sequence_length = int(inputs_embeds.shape[1])
        batch_size = int(inputs_embeds.shape[0])
        fastv_k = int(state.fastv_k)
        if fastv_k <= 0 or fastv_k >= int(self.config.num_hidden_layers):
            raise ValueError(
                "`fastv_k` must be in [1, num_hidden_layers - 1], "
                f"got {fastv_k} for {self.config.num_hidden_layers} layers."
            )

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
        captured_attention: torch.Tensor | None = None
        pruned = False
        item_infos: list[dict[str, Any]] = []

        for layer_index, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if layer_index == fastv_k and not pruned and sequence_length > 1:
                if captured_attention is None:
                    raise RuntimeError("FastV could not capture attention from layer fastv_k - 1.")
                keep_indices, item_infos = _build_keep_indices(
                    captured_attention,
                    image_token_ranges=state.image_token_ranges,
                    reduction_ratio=state.fastv_r,
                    sequence_length=sequence_length,
                )
                hidden_states = _gather_sequence(hidden_states, keep_indices)
                if attention_mask is not None:
                    attention_mask = _gather_sequence(attention_mask, keep_indices)
                position_ids = _gather_sequence(position_ids.expand(batch_size, -1), keep_indices)
                sequence_length = int(hidden_states.shape[1])
                cache_position = torch.arange(sequence_length, device=hidden_states.device)
                _prune_dynamic_cache(past_key_values, keep_indices)
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
                pruned = True

            if layer_index == fastv_k - 1:
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
                    raise RuntimeError("FastV could not capture eager attention weights.")
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
        if item_infos:
            for item in item_infos:
                item["fastv_k"] = fastv_k
            selected = [int(item["selected_image_tokens"]) for item in item_infos]
            original = [int(item["original_image_tokens"]) for item in item_infos]
            state.last_info = {
                "applied": True,
                "backend": "llava",
                "selector_type": "fastv",
                "fastv_k": fastv_k,
                "fastv_r": float(state.fastv_r),
                "batch_size": batch_size,
                "original_image_tokens": original[0] if len(set(original)) == 1 else None,
                "selected_image_tokens": selected[0] if len(set(selected)) == 1 else None,
                "input_length_before": item_infos[0]["input_length_before"],
                "input_length_after": item_infos[0]["input_length_after"],
                "items": item_infos,
                "selector_metadata": {
                    "selector_type": "fastv",
                    "fastv_k": fastv_k,
                    "fastv_r": float(state.fastv_r),
                    "image_token_count": original[0] if len(set(original)) == 1 else None,
                    "selected_token_count": selected[0] if len(set(selected)) == 1 else None,
                },
            }
            state.generation_infos.append(dict(state.last_info))
        else:
            state.last_info = {
                "applied": False,
                "backend": "llava",
                "reason": "sequence_too_short",
            }
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    language_model.forward = types.MethodType(fastv_forward, language_model)
    language_model._ddps_fastv_state = state
    language_model._ddps_fastv_original_forward = original_forward
    return state
