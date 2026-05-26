from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection


def _freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


class CLIPVisionModelV2(nn.Module):
    def __init__(self, pretrained_name: str):
        super().__init__()
        original_model = CLIPVisionModelWithProjection.from_pretrained(pretrained_name)
        self.model = original_model
        self.vision_model = original_model.vision_model
        self.visual_projection = original_model.visual_projection
        self.config = original_model.config
        _freeze_module(self.model)

    def _forward_to_final_block_input(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.vision_model.embeddings(pixel_values=pixel_values)
        hidden_states = self.vision_model.pre_layrnorm(hidden_states)

        encoder_layers = self.vision_model.encoder.layers
        for encoder_layer in encoder_layers[:-1]:
            layer_outputs = encoder_layer(
                hidden_states=hidden_states,
                attention_mask=None,
                causal_attention_mask=None,
                output_attentions=False,
            )
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
        return hidden_states

    def _forward_maskclip_dense_from_final_block_input(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        last_layer = self.vision_model.encoder.layers[-1]
        hidden_states = last_layer.layer_norm1(hidden_states)
        hidden_states = last_layer.self_attn.v_proj(hidden_states)
        hidden_states = last_layer.self_attn.out_proj(hidden_states)
        projected_tokens = self.visual_projection(hidden_states)
        return projected_tokens[:, 1:, :]

    def _forward_global_latent_from_final_block_input(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        last_layer = self.vision_model.encoder.layers[-1]
        layer_outputs = last_layer(
            hidden_states=hidden_states,
            attention_mask=None,
            causal_attention_mask=None,
            output_attentions=False,
        )
        last_hidden_state = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
        pooled_output = self.vision_model.post_layernorm(last_hidden_state[:, 0, :])
        return self.visual_projection(pooled_output)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self._forward_to_final_block_input(pixel_values)
        patch_embeddings = self._forward_maskclip_dense_from_final_block_input(hidden_states)
        image_latent = self._forward_global_latent_from_final_block_input(hidden_states)
        return patch_embeddings, image_latent


class CLIPTextModel(nn.Module):
    def __init__(self, pretrained_name: str):
        super().__init__()
        original_model = CLIPTextModelWithProjection.from_pretrained(pretrained_name)
        self.text_model = original_model.text_model
        self.text_projection = original_model.text_projection
        _freeze_module(original_model)

    def forward(self, **kwargs: Any) -> torch.Tensor:
        output = self.text_model(**kwargs)
        last_hidden_state = output.last_hidden_state
        input_ids = kwargs["input_ids"]
        eos_token_indices = input_ids.argmax(dim=-1)
        batch_indices = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
        eos_hidden_states = last_hidden_state[batch_indices, eos_token_indices]
        return self.text_projection(eos_hidden_states)
