"""
Qwen2 VTP Wrapper for LLaVA-OneVision.

Wraps Qwen2Model to capture attention weights at a selected layer
and perform visual token pruning via ElasticCacheOV.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast


class Qwen2ModelVTP(nn.Module):
    """
    Qwen2Model wrapper that intercepts forward pass at selected_layer
    to capture attention weights for visual token pruning.

    This replaces the standard Qwen2Model inside Qwen2ForCausalLM.
    """

    def __init__(self, qwen2_model, cache):
        """
        Args:
            qwen2_model: original Qwen2Model instance
            cache: ElasticCacheOV instance for pruning
        """
        super().__init__()
        # Copy all attributes from the original model
        self.config = qwen2_model.config
        self.embed_tokens = qwen2_model.embed_tokens
        self.layers = qwen2_model.layers
        self.norm = qwen2_model.norm
        self.gradient_checkpointing = qwen2_model.gradient_checkpointing

        # Keep reference to original model for _update_causal_mask
        self._original_model = qwen2_model

        self.cache = cache
        self.selected_layer = cache.selected_layer

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, list]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Store input_ids for image token detection during pruning (only during prefill)
        # input_ids is stored on the cache by LlavaOVForConditionalGeneration.forward()
        # because Qwen2ForCausalLM.forward() converts input_ids -> inputs_embeds

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = ()  # Always init as tuple so we can append
        next_decoder_cache = None

        is_prefill = hidden_states.shape[1] > 1

        layer_idx = 0
        for decoder_layer in self.layers:
            # During decoding, extend causal mask if needed
            if hidden_states.shape[1] == 1 and causal_mask.shape[-1] != (past_key_values.key_cache[layer_idx].shape[2] + 1):
                causal_mask = causal_mask[:, :, :, -1:].repeat(1, 1, 1, past_key_values.key_cache[layer_idx].shape[2] + 1)

            # Only collect attention at the selected layer during prefill
            # Must force True at selected_layer even if parent set it False
            layer_output_attentions = is_prefill and layer_idx == self.selected_layer

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    layer_output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=layer_output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if layer_output_attentions else 1]

            if layer_output_attentions:
                all_self_attns += (layer_outputs[1],)

            # === VTP PRUNING at selected layer ===
            input_ids = getattr(self.cache, '_input_ids', None)
            if is_prefill and layer_idx == self.selected_layer and input_ids is not None and len(layer_outputs) >= 3 and layer_outputs[1] is not None:
                attentions = layer_outputs[1]  # [batch, heads, seq_len, seq_len]

                topk_indices = self.cache.process_attention(
                    attentions, hidden_states, input_ids
                )

                if topk_indices is not None:
                    img_start = self.cache.img_start
                    img_end = self.cache.img_end

                    # Build index list: before image + kept image tokens + after image
                    index_list_pre = torch.arange(0, img_start, device=hidden_states.device)
                    index_list_post = torch.arange(img_end + 1, hidden_states.shape[1], device=hidden_states.device)
                    index_list = torch.cat([index_list_pre, topk_indices.sort(descending=False)[0], index_list_post])

                    # Prune KV cache for all layers up to selected layer
                    for l in range(self.selected_layer + 1):
                        past_key_values.key_cache[l] = past_key_values.key_cache[l][:, :, index_list, :].contiguous()
                        past_key_values.value_cache[l] = past_key_values.value_cache[l][:, :, index_list, :].contiguous()

                    # Prune hidden states
                    hidden_states = hidden_states[:, index_list, :]

                    # Prune causal mask
                    if causal_mask is not None:
                        causal_mask = causal_mask[:, :, index_list, :][:, :, :, index_list]

                    # Prune position IDs
                    position_ids = position_ids[:, index_list]

                    # Update cache position
                    cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)

            layer_idx += 1

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions):
        """Delegate to original Qwen2Model's _update_causal_mask."""
        return self._original_model._update_causal_mask(
            attention_mask, input_tensor, cache_position, past_key_values, output_attentions
        )
