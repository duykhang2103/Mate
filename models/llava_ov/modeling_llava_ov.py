"""
LLaVA-OneVision with Visual Token Pruning.

Wraps HuggingFace's LlavaOnevisionForConditionalGeneration with VTP support:
- Replaces Qwen2Model with Qwen2ModelVTP for attention-based pruning
- Supports both attention-based and cluster-based pruning
"""

import torch
import torch.nn as nn
from transformers import LlavaOnevisionForConditionalGeneration

from models.llava_ov.qwen2_vtp import Qwen2ModelVTP
from models.llava_ov.elastic_cache_ov import ElasticCacheOV


class LlavaOVForConditionalGeneration(LlavaOnevisionForConditionalGeneration):
    """LLaVA-OneVision with Visual Token Pruning."""

    def __init__(self, config):
        super().__init__(config)

        self.selected_layer = getattr(config, 'selected_layer', 10)
        self.alpha = getattr(config, 'alpha', 0.4)
        self.use_cluster_pruning = getattr(config, 'use_cluster_pruning', False)
        self.cluster_pruning_topk = getattr(config, 'cluster_pruning_topk', 0.4)

        # Create the pruning cache
        image_token_id = getattr(config, 'image_token_index', 151646)
        self.cache = ElasticCacheOV(
            alpha=self.alpha,
            total_num_layers=config.text_config.num_hidden_layers,
            selected_layer=self.selected_layer,
            image_token_id=image_token_id,
            use_cluster_pruning=self.use_cluster_pruning,
            cluster_pruning_topk=self.cluster_pruning_topk,
        )

        # Replace Qwen2Model with Qwen2ModelVTP
        # self.language_model is Qwen2ForCausalLM
        # self.language_model.model is Qwen2Model (the transformer backbone)
        original_model = self.language_model.model
        self.language_model.model = Qwen2ModelVTP(original_model, self.cache)

        # Token count tracking
        self._last_pruned_token_count = None

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None,
                image_sizes=None, pixel_values_videos=None, image_sizes_videos=None,
                labels=None, use_cache=None, output_attentions=None,
                output_hidden_states=True, return_dict=None, **kwargs):
        """Forward pass with VTP pruning at the selected layer."""

        # Always request attention weights so VTP can use them at the selected layer
        if output_attentions is None:
            output_attentions = True

        # Pass input_ids to cache before parent converts them to inputs_embeds
        # Parent forward() converts input_ids -> inputs_embeds, then passes inputs_embeds
        # to language_model, so input_ids never reaches Qwen2ModelVTP
        if input_ids is not None:
            self.cache._input_ids = input_ids

        # Track input shape for token counting
        if input_ids is not None:
            input_len = input_ids.shape[1]
        else:
            input_len = None

        # Call parent forward — VTP pruning happens inside Qwen2ModelVTP.forward()
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            pixel_values_videos=pixel_values_videos,
            image_sizes_videos=image_sizes_videos,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

        # Track token count after pruning
        if input_len is not None and self.cache.num_tokens_after_prune is not None:
            self._last_pruned_token_count = self.cache.num_tokens_after_prune

        return outputs
