import torch
import torch.nn as nn
from transformers import LlavaOnevisionForConditionalGeneration


class LlavaOVForConditionalGeneration(LlavaOnevisionForConditionalGeneration):
    """LLaVA-OneVision with Visual Token Pruning."""

    def __init__(self, config):
        super().__init__(config)
        self.selected_layer = getattr(config, 'selected_layer', 10)
        self.alpha = getattr(config, 'alpha', 0.4)
        self.tau = getattr(config, 'tau', 0.8)
        self.use_cluster_pruning = getattr(config, 'use_cluster_pruning', False)
        self.cluster_pruning_topk = getattr(config, 'cluster_pruning_topk', 0.4)

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None,
                image_sizes=None, pixel_values_videos=None, image_sizes_videos=None,
                labels=None, use_cache=None, output_attentions=None,
                output_hidden_states=True, return_dict=None, **kwargs):
        """Forward pass with optional VTP pruning."""

        # Enable output_attentions to get attention weights
        if output_attentions is None:
            output_attentions = True

        return super().forward(
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
