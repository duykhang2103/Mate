"""Attention-free distortion routing for SafePruneVid.

The router treats each spatial location across a temporal window as a
trajectory.  Collapsing a trajectory to its temporal centroid incurs cosine
distortion; retaining it as per-frame tokens removes that distortion at a cost
of ``window_size - 1`` additional pre-clustering tokens.  A single threshold is
applied across every window in a sample, so scarce tokens are allocated to the
highest-value trajectories instead of independently reserving the same ratio
inside every window.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

import torch
import torch.nn.functional as F


def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    score_min = scores.amin(dim=-1, keepdim=True)
    score_max = scores.amax(dim=-1, keepdim=True)
    return (scores - score_min) / (score_max - score_min).clamp_min(1e-6)


def compute_distortion_controlled_static_masks(
    windows: Sequence[torch.Tensor],
    epsilon: float,
    query_embedding: torch.Tensor | None = None,
    query_weight: float = 0.7,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """Select globally thresholded temporal trajectories.

    Args:
        windows: Temporal windows shaped ``[B, W_i, L, C]``.  All windows
            share batch, spatial-token, and channel dimensions.
        epsilon: Minimum distortion reduction per additional token required to
            preserve a trajectory dynamically.
        query_embedding: Optional projected query shaped ``[B, C]``.
        query_weight: Non-negative multiplicative relevance strength.  Query
            relevance can only increase a trajectory's routing score.

    Returns:
        ``static_masks`` contains one boolean ``[B, L]`` mask per window where
        ``True`` means temporally merge the trajectory.  ``diagnostics``
        contains per-sample routing statistics.
    """
    if not windows:
        raise ValueError("windows must contain at least one temporal window")
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    if query_weight < 0:
        raise ValueError(
            f"query_weight must be non-negative, got {query_weight}"
        )

    batch_size, _, spatial_tokens, channels = windows[0].shape
    if spatial_tokens < 2:
        raise ValueError(
            "distortion routing requires at least two spatial trajectories"
        )

    normalized_query = None
    if query_embedding is not None:
        if query_embedding.shape != (batch_size, channels):
            raise ValueError(
                "Query and visual feature shapes are incompatible: "
                f"query={query_embedding.shape}, window={windows[0].shape}"
            )
        normalized_query = F.normalize(query_embedding.float(), p=2, dim=-1)

    scores = []
    distortions = []
    costs = []
    query_reliabilities = []
    for window in windows:
        if window.ndim != 4:
            raise ValueError(
                f"each window must have shape [B, W, L, C], got {window.shape}"
            )
        if (
            window.shape[0] != batch_size
            or window.shape[2] != spatial_tokens
            or window.shape[3] != channels
        ):
            raise ValueError(
                "all windows must share batch, spatial, and channel dimensions"
            )

        scoring_window = F.normalize(window.float(), p=2, dim=-1)
        temporal_centroid = F.normalize(
            window.float().mean(dim=1, keepdim=True), p=2, dim=-1
        )
        distortion = (
            1.0 - (scoring_window * temporal_centroid).sum(dim=-1)
        ).amax(dim=1).clamp_min(0.0)
        extra_token_cost = max(int(window.shape[1]) - 1, 1)
        value_per_token = distortion / float(extra_token_cost)

        if normalized_query is not None:
            query_scores = torch.einsum(
                "bwlc,bc->bwl", scoring_window, normalized_query
            ).amax(dim=1)
            query_spread = query_scores.std(
                dim=-1, keepdim=True, unbiased=False
            )
            distortion_spread = distortion.std(
                dim=-1, keepdim=True, unbiased=False
            )
            reliability = query_spread / (
                query_spread + distortion_spread + 1e-6
            )
            value_per_token = value_per_token * (
                1.0
                + float(query_weight)
                * reliability
                * _normalize_scores(query_scores)
            )
            query_reliabilities.append(reliability)

        scores.append(value_per_token)
        distortions.append(distortion)
        costs.append(
            torch.full_like(value_per_token, float(extra_token_cost))
        )

    flat_scores = torch.cat(scores, dim=-1)
    flat_distortions = torch.cat(distortions, dim=-1)
    flat_costs = torch.cat(costs, dim=-1)
    flat_dynamic = flat_scores > float(epsilon)
    window_size_values = torch.tensor(
        [window.shape[1] for window in windows],
        device=flat_scores.device,
        dtype=flat_scores.dtype,
    )

    static_masks = []
    start = 0
    for window_scores in scores:
        end = start + spatial_tokens
        dynamic_mask = flat_dynamic[:, start:end].clone()
        for batch_index in range(batch_size):
            dynamic_count = int(dynamic_mask[batch_index].sum().item())
            if dynamic_count == 0:
                highest = window_scores[batch_index].argmax()
                dynamic_mask[batch_index, highest] = True
            elif dynamic_count == spatial_tokens:
                lowest = window_scores[batch_index].argmin()
                dynamic_mask[batch_index, lowest] = False
        static_masks.append(~dynamic_mask)
        flat_dynamic[:, start:end] = dynamic_mask
        start = end

    quantiles = torch.tensor(
        [0.25, 0.5, 0.75, 0.95],
        device=flat_scores.device,
        dtype=flat_scores.dtype,
    )
    score_quantiles = torch.quantile(flat_scores, quantiles, dim=-1).T
    distortion_quantiles = torch.quantile(
        flat_distortions, quantiles, dim=-1
    ).T
    diagnostics: dict[str, Any] = {
        "epsilon": torch.full(
            (batch_size,),
            float(epsilon),
            device=flat_scores.device,
            dtype=flat_scores.dtype,
        ),
        "num_windows": torch.full(
            (batch_size,),
            len(windows),
            device=flat_scores.device,
            dtype=torch.long,
        ),
        "window_size_mean": window_size_values.mean().expand(batch_size),
        "window_size_std": window_size_values.std(
            unbiased=False
        ).expand(batch_size),
        "window_size_min": window_size_values.amin().expand(batch_size),
        "window_size_max": window_size_values.amax().expand(batch_size),
        "total_tracks": torch.full(
            (batch_size,),
            flat_scores.shape[-1],
            device=flat_scores.device,
            dtype=torch.long,
        ),
        "dynamic_tracks": flat_dynamic.sum(dim=-1),
        "dynamic_fraction": flat_dynamic.float().mean(dim=-1),
        "extra_token_cost": (flat_dynamic.float() * flat_costs).sum(dim=-1),
        "score_mean": flat_scores.mean(dim=-1),
        "score_std": flat_scores.std(dim=-1, unbiased=False),
        "score_q25": score_quantiles[:, 0],
        "score_q50": score_quantiles[:, 1],
        "score_q75": score_quantiles[:, 2],
        "score_q95": score_quantiles[:, 3],
        "distortion_mean": flat_distortions.mean(dim=-1),
        "distortion_std": flat_distortions.std(dim=-1, unbiased=False),
        "distortion_q25": distortion_quantiles[:, 0],
        "distortion_q50": distortion_quantiles[:, 1],
        "distortion_q75": distortion_quantiles[:, 2],
        "distortion_q95": distortion_quantiles[:, 3],
    }
    if query_reliabilities:
        diagnostics["query_reliability"] = torch.cat(
            query_reliabilities, dim=-1
        ).mean(dim=-1)

    return static_masks, diagnostics


def estimate_post_cluster_tokens(
    static_masks: Sequence[torch.Tensor],
    windows: Sequence[torch.Tensor],
    cluster_ratio: float,
) -> torch.Tensor:
    """Predict merged token count using PruneVid's exact count contract."""
    if len(static_masks) != len(windows) or not windows:
        raise ValueError("static masks and windows must be non-empty and aligned")
    if not 0 < cluster_ratio <= 1:
        raise ValueError("cluster_ratio must be in (0, 1]")

    totals = torch.zeros(
        static_masks[0].shape[0],
        device=static_masks[0].device,
        dtype=torch.long,
    )
    for static_mask, window in zip(static_masks, windows):
        static_count = static_mask.sum(dim=-1)
        dynamic_count = static_mask.shape[-1] - static_count
        static_after_cluster = torch.where(
            static_count > 14,
            (static_count.float() * cluster_ratio).to(torch.long),
            static_count,
        )
        dynamic_after_cluster = torch.where(
            dynamic_count > 14,
            (dynamic_count.float() * cluster_ratio).to(torch.long),
            dynamic_count,
        )
        totals += static_after_cluster + int(window.shape[1]) * dynamic_after_cluster
    return totals


def apply_learned_safety_router(
    adaptive_static_masks: Sequence[torch.Tensor],
    baseline_static_masks: Sequence[torch.Tensor],
    diagnostics: Mapping[str, torch.Tensor],
    predicted_merged_tokens: torch.Tensor,
    router: Mapping[str, Any],
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Choose baseline masks for samples predicted to be unsafe."""
    if len(adaptive_static_masks) != len(baseline_static_masks):
        raise ValueError("adaptive and baseline masks must be aligned")
    fields = list(router.get("feature_fields") or [])
    weights = list(router.get("weights") or [])
    means = list(router.get("means") or [])
    scales = list(router.get("scales") or [])
    if not fields or not (
        len(fields) == len(weights) == len(means) == len(scales)
    ):
        raise ValueError("invalid safety-router feature metadata")

    feature_columns = []
    for field in fields:
        if field == "merged_vision_tokens":
            value = predicted_merged_tokens
        elif field == "query_merge_reliability":
            value = diagnostics.get("query_reliability")
        elif field.startswith("routing_"):
            value = diagnostics.get(field.removeprefix("routing_"))
        else:
            value = None
        if value is None:
            raise ValueError(f"safety-router feature is unavailable: {field}")
        feature_columns.append(value.float())
    features = torch.stack(feature_columns, dim=-1)
    weights_tensor = torch.as_tensor(
        weights, device=features.device, dtype=features.dtype
    )
    means_tensor = torch.as_tensor(
        means, device=features.device, dtype=features.dtype
    )
    scales_tensor = torch.as_tensor(
        scales, device=features.device, dtype=features.dtype
    ).clamp_min(1e-8)
    bias = float(router.get("bias", 0.0))
    threshold = float(router["threshold"])
    risks = torch.sigmoid(
        ((features - means_tensor) / scales_tensor) @ weights_tensor + bias
    )
    use_baseline = risks >= threshold
    selected_masks = [
        torch.where(use_baseline.unsqueeze(-1), baseline, adaptive)
        for adaptive, baseline in zip(
            adaptive_static_masks, baseline_static_masks
        )
    ]
    return selected_masks, risks, use_baseline
