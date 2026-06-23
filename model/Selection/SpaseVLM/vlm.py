from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ...vlm import BaseVLM
from .runtime import install_sparsevlm


class SparseVLMVLM(BaseVLM):
    def __init__(
        self,
        *args: Any,
        sparsevlm_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        self.sparsevlm_config = dict(sparsevlm_config or {})
        self._sparsevlm_state: Any | None = None
        super().__init__(*args, **kwargs)
        self._install_sparsevlm()

    def _install_sparsevlm(self) -> None:
        self._sparsevlm_state = install_sparsevlm(self.model, self.sparsevlm_config)

    def _resolve_image_token_ranges(
        self,
        model_inputs: Mapping[str, Any],
    ) -> list[tuple[int, int]]:
        input_ids = model_inputs.get("input_ids")
        if input_ids is None:
            raise ValueError("SparseVLM requires `input_ids` to infer image token positions.")

        image_token_id = getattr(self.model.config, "image_token_index", None)
        if image_token_id is None:
            image_token_id = getattr(self.model.config, "image_token_id", None)
        if image_token_id is None:
            raise ValueError("Could not find the image placeholder token id for SparseVLM.")

        ranges: list[tuple[int, int]] = []
        for item_index in range(int(input_ids.shape[0])):
            image_positions = torch.nonzero(
                input_ids[item_index] == int(image_token_id),
                as_tuple=False,
            ).flatten()
            if int(image_positions.numel()) == 0:
                raise ValueError(f"SparseVLM could not find image tokens in batch item {item_index}.")
            start = int(image_positions.min().item())
            end = int(image_positions.max().item()) + 1
            if end - start != int(image_positions.numel()):
                raise ValueError("SparseVLM expects image placeholder tokens to be contiguous.")
            ranges.append((start, end - start))
        return ranges

    def _prepare_sparse_generation(self, model_inputs: Mapping[str, Any]) -> None:
        if self._sparsevlm_state is None:
            self._install_sparsevlm()
        self._sparsevlm_state.image_token_ranges = self._resolve_image_token_ranges(model_inputs)
        self._sparsevlm_state.last_info = {}

    def _finish_sparse_generation(self) -> None:
        sparse_info = {}
        if self._sparsevlm_state is not None:
            sparse_info = dict(self._sparsevlm_state.last_info or {})
        self.last_patch_selection_info = sparse_info or {
            "applied": False,
            "backend": self.backend,
            "reason": "sparsevlm_not_run",
        }
        if self._sparsevlm_state is not None:
            self._sparsevlm_state.image_token_ranges = []

    def _run_standard_generation_batch(
        self,
        model_inputs: dict[str, Any],
        *,
        timing_path: str = "sparsevlm_generation_batch",
    ) -> list[str]:
        self._prepare_sparse_generation(model_inputs)
        try:
            outputs = super()._run_standard_generation_batch(
                model_inputs,
                timing_path=timing_path,
            )
        finally:
            self._finish_sparse_generation()

        timing_info = dict(self.last_timing_info or {})
        if self.last_patch_selection_info.get("applied"):
            timing_info["path"] = timing_path
            timing_info["input_sequence_length"] = int(
                self.last_patch_selection_info.get(
                    "input_length_after",
                    timing_info.get("input_sequence_length", 0),
                )
            )
            timing_info["original_input_sequence_length"] = int(
                self.last_patch_selection_info.get(
                    "input_length_before",
                    timing_info["input_sequence_length"],
                )
            )
            items = self.last_patch_selection_info.get("items")
            if isinstance(items, list) and items:
                timing_info["items"] = [
                    {
                        **timing_info,
                        "input_sequence_length": int(
                            item.get(
                                "input_length_after",
                                timing_info["input_sequence_length"],
                            )
                        ),
                        "original_input_sequence_length": int(
                            item.get(
                                "input_length_before",
                                timing_info["original_input_sequence_length"],
                            )
                        ),
                    }
                    for item in items
                    if isinstance(item, Mapping)
                ]
            self.last_timing_info = timing_info
        return outputs

    def _answer_batch_with_visual(self, *args: Any, **kwargs: Any) -> list[str]:
        if self._sparsevlm_state is None:
            self._install_sparsevlm()
        self._sparsevlm_state.generation_infos = []
        outputs = super()._answer_batch_with_visual(*args, **kwargs)

        generation_infos = list(self._sparsevlm_state.generation_infos)
        if generation_infos:
            items = [
                item
                for info in generation_infos
                for item in info.get("items", [])
                if isinstance(item, Mapping)
            ]
            last_info = dict(generation_infos[-1])
            last_info["batch_size"] = len(outputs)
            if items:
                last_info["items"] = items
            self.last_patch_selection_info = last_info
        return outputs
