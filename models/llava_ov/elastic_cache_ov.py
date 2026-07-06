"""
Elastic Cache for LLaVA-OneVision Visual Token Pruning.

Adapted from PLLaVA's elastic_cache.py to work with LLaVA-OV's architecture:
- Image tokens identified by image_token_id (151646), not pad_token_id
- No static/dynamic window structure (LLaVA-OV uses temporal position embeddings)
- Simpler pruning: just topk on attention scores across all image tokens
"""

import torch
import torch.nn.functional as F
import numpy as np


def slice1d(x, start, end):
    return x[:, start:end, ...]


def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}


def cluster_dpc_knn(x, cluster_num, k=5, token_mask=None):
    """Cluster tokens with DPC-KNN algorithm.
    Return:
        idx_cluster (Tensor[B, N]): cluster index of each token.
        cluster_num (int): actual cluster number.
    Args:
        x: input token feature, [B, N, C]
        cluster_num (int): cluster number
        k (int): number of the nearest neighbor used for local density.
        token_mask (Tensor[B, N]): mask indicate whether the token is padded.
    """
    with torch.no_grad():
        B, N, C = x.shape

        dist_matrix = torch.cdist(x.float(), x.float()) / (C ** 0.5)

        if token_mask is not None:
            token_mask = token_mask > 0
            dist_matrix = dist_matrix * token_mask[:, None, :] + \
                          (dist_matrix.max() + 1) * (~token_mask[:, None, :])

        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
        density = density + torch.rand(
            density.shape, device=density.device, dtype=density.dtype) * 1e-6

        if token_mask is not None:
            density = density * token_mask

        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
        dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

        score = dist * density
        _, index_down = torch.topk(score, k=cluster_num, dim=-1)

        dist_matrix_t = torch.cdist(x.float(), x[index_down].float()) / (C ** 0.5)
        idx_cluster = dist_matrix_t.argmin(dim=1)

        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)
    return idx_cluster, cluster_num


class ElasticCacheOV:
    """
    Vision token pruning cache for LLaVA-OneVision.

    Simpler than PLLaVA's VTPWindowCache because LLaVA-OV doesn't have
    the static/dynamic window structure. All image tokens are treated equally
    and pruned based on attention scores or clustering.
    """

    def __init__(
        self,
        alpha=0.4,
        total_num_layers=32,
        selected_layer=10,
        image_token_id=151646,
        video_token_id=151647,
        use_cluster_pruning=False,
        cluster_pruning_topk=0.4,
    ):
        self.alpha = alpha
        self.total_num_layers = total_num_layers
        self.selected_layer = selected_layer
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.use_cluster_pruning = use_cluster_pruning
        self.cluster_pruning_topk = cluster_pruning_topk

        self.img_start = None
        self.img_end = None
        self.num_tokens_after_prune = None

    def process_attention(self, attentions, hidden_states, input_ids):
        """
        Process attention weights at the selected layer to determine which tokens to keep.

        Args:
            attentions: [batch, heads, seq_len, seq_len] attention weights from the selected layer
            hidden_states: [batch, seq_len, hidden_dim] hidden states from the selected layer
            input_ids: [batch, seq_len] input token IDs

        Returns:
            topk_indices: indices of tokens to KEEP (absolute positions in the sequence)
        """
        assert attentions.shape[0] == 1, "Batch size must be 1"

        # Find image/video token positions (LLaVA-OV uses 151646 for images, 151647 for videos)
        image_mask = (input_ids == self.image_token_id) | (input_ids == self.video_token_id)
        img_positions = image_mask.nonzero(as_tuple=True)
        if len(img_positions[1]) == 0:
            return None

        img_start = img_positions[1][0].item()
        img_end = img_positions[1][-1].item()
        num_img_tokens = img_end - img_start + 1

        self.img_start = img_start
        self.img_end = img_end

        # Cluster pruning path
        if self.use_cluster_pruning and hidden_states is not None:
            return self._cluster_prune(hidden_states, img_start, img_end)

        # Attention-based pruning path
        # attentions: [batch, heads, seq_len, seq_len]
        # Use text-to-image attention: text tokens attending to image tokens
        # Shape: [heads, num_text_tokens, num_img_tokens]
        text_to_image_attn = attentions[0, :, img_end + 1:, img_start:img_end + 1]

        if text_to_image_attn.shape[1] == 0 or text_to_image_attn.shape[2] == 0:
            # No text tokens or no image tokens — fall back to uniform selection
            num_keep = max(int(num_img_tokens * self.alpha), 1)
            step = max(num_img_tokens // num_keep, 1)
            topk_indices = torch.arange(img_start, img_end + 1, step=step, device=input_ids.device)[:num_keep]
            return topk_indices

        # Aggregate across heads: mean attention per image token
        # [num_img_tokens]
        attn_scores = text_to_image_attn.mean(dim=0).max(dim=0)[0]

        # Select top-k image tokens
        num_keep = max(int(num_img_tokens * self.alpha), 1)
        _, topk_local = torch.topk(attn_scores, k=num_keep, dim=-1)
        topk_indices = topk_local + img_start

        self.num_tokens_after_prune = (
            img_start  # tokens before image
            + num_keep  # kept image tokens
            + (input_ids.shape[1] - img_end - 1)  # tokens after image
        )

        return topk_indices

    def _cluster_prune(self, hidden_states, img_start, img_end):
        """
        Content-adaptive pruning using DPC-KNN clustering on LLM hidden states.
        """
        num_img_tokens = img_end - img_start + 1
        img_hidden = hidden_states[:, img_start:img_end + 1, :]

        num_keep = max(int(num_img_tokens * self.alpha), 1)
        num_clusters = min(num_keep, num_img_tokens // 2) if num_img_tokens > 2 else num_img_tokens
        num_clusters = max(num_clusters, 1)

        idx_cluster, actual_clusters = cluster_dpc_knn(img_hidden, cluster_num=num_clusters, k=5)

        cluster_sizes = torch.zeros(actual_clusters, device=img_hidden.device, dtype=torch.long)
        for c in range(actual_clusters):
            cluster_sizes[c] = (idx_cluster[0] == c).sum()

        _, sorted_cluster_indices = torch.sort(cluster_sizes, descending=True)

        selected_mask = torch.zeros(num_img_tokens, device=img_hidden.device, dtype=torch.bool)
        tokens_selected = 0

        for cluster_idx in sorted_cluster_indices:
            if tokens_selected >= num_keep:
                break
            cluster_mask = (idx_cluster[0] == cluster_idx)
            cluster_count = cluster_mask.sum().item()

            if tokens_selected + cluster_count <= num_keep:
                selected_mask |= cluster_mask
                tokens_selected += cluster_count
            else:
                cluster_token_indices = torch.where(cluster_mask)[0]
                cluster_hidden = img_hidden[0, cluster_token_indices, :]
                importance = cluster_hidden.norm(dim=-1)
                num_to_take = num_keep - tokens_selected
                if num_to_take > 0 and len(cluster_token_indices) > 0:
                    num_to_take = min(num_to_take, len(cluster_token_indices))
                    _, topk_within = torch.topk(importance, k=num_to_take)
                    selected_mask[cluster_token_indices[topk_within]] = True
                    tokens_selected += num_to_take

        topk_indices = torch.where(selected_mask)[0] + img_start

        self.num_tokens_after_prune = (
            img_start
            + num_keep
            + (hidden_states.shape[1] - img_end - 1)
        )

        return topk_indices

    def prune_kv_cache(self, past_key_values, topk_indices, img_start, img_end):
        """
        Prune KV cache to keep only the selected tokens.

        Args:
            past_key_values: DynamicCache object
            topk_indices: indices of tokens to KEEP (absolute positions)
            img_start: start of image tokens in the sequence
            img_end: end of image tokens in the sequence

        Returns:
            pruned past_key_values
        """
        # Build full index list: tokens before image + kept image tokens + tokens after image
        index_list_pre = torch.arange(0, img_start, device=topk_indices.device)
        index_list_post = torch.arange(img_end + 1, past_key_values.key_cache[0].shape[2], device=topk_indices.device)
        index_list = torch.cat([index_list_pre, topk_indices.sort(descending=False)[0], index_list_post])

        # Prune all layers up to and including the selected layer
        for layer_idx in range(self.selected_layer + 1):
            past_key_values.key_cache[layer_idx] = past_key_values.key_cache[layer_idx][:, :, index_list, :].contiguous()
            past_key_values.value_cache[layer_idx] = past_key_values.value_cache[layer_idx][:, :, index_list, :].contiguous()

        return past_key_values, index_list
