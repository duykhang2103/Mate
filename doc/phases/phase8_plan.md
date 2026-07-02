# Phase 8 Plan — Fix Accuracy Degradation with Real Motion Signal

> **Created**: 2026-07-02
> **Status**: PLANNING
> **Goal**: Improve accuracy on motion-heavy and temporally complex videos (currently ALL methods underperform baseline)
> **Core problem**: Every modification to PruneVid so far has degraded accuracy

---

## Failure Analysis — Why Every Method Failed

### Current Results (5 tasks, ~1000 samples)

| Method | Avg | Action Seq | Action Pred | Unexpected | Obj Interact | Moving Dir |
|--------|-----|-----------|-------------|-----------|-------------|-----------|
| Baseline (fixed L10) | **50.71** | 55.85 | 51.50 | 62.50 | 63.50 | 20.50 |
| Entropy+Borderline | 50.20 | 53.72 | 51.50 | 62.00 | 63.50 | 20.50 |
| Delta | **-0.51** | -2.13 | 0.00 | -0.50 | 0.00 | 0.00 |

### Root Cause 1: Motion Signal Is Zero (THE FUNDAMENTAL FLAW)

The two-tier retention system computes motion scores from **attention temporal variance** (`elastic_cache.py:197-226`). But PLLaVA's attention is uniform (entropy=0.9999), so:

```
Motion scores: [0.000000, 0.000000, ...]
Max motion: 0.000001
All windows: motion_ratio=0.0000 → alpha=0.4000 (unchanged)
```

**The motion-adaptive mode does literally nothing.** It's not that it helps a little — it's that the signal it relies on doesn't exist in PLLaVA.

### Root Cause 2: Static/Dynamic Split Uses Wrong Signal

`merge_frames_dynamic()` (`modeling_pllava.py:777-840`) splits tokens into static/dynamic using **feature cosine similarity** between frames. This is a weak proxy:

- High similarity → classified as "static" → fewer tokens retained in LLM pruning
- Low similarity → classified as "dynamic" → more tokens retained

But feature similarity ≠ motion. A slow pan across a static scene has high feature change but low motion. A fast-moving object in a complex scene might have moderate feature change.

### Root Cause 3: Borderline Preservation Adds Noise

Tokens near the pruning cutoff are, by definition, low-attention. Keeping them adds noise to the model's representation. The -2.13% drop on Action Sequence comes from this.

### Root Cause 4: Moving Direction Is Random (20.5%)

This task is essentially at chance level for both baseline and modified. Something is fundamentally wrong with how this task is evaluated, or the model simply cannot solve it with 16 frames.

---

## The Plan: 4 Changes, Ordered by Impact

### Change 1: Optical-Flow Static/Dynamic Split (FOUNDATION)

**What**: Replace feature-similarity threshold with dense optical flow magnitude for the static/dynamic token split in `merge_frames_dynamic()`.

**Why**: This is the foundation everything else builds on. Without a real motion signal, no motion-aware mechanism can work.

**Where**: `models/pllava/modeling_pllava.py`, `merge_frames_dynamic()` method (lines 777-840)

**Current code** (lines 797-803):
```python
frames_normed = F.normalize(current_frames, p=2, dim=-1)
frames_sim = einsum('b w l c, b t l c -> b w t l', frames_normed, frames_normed)
frames_sim = (frames_sim.sum(dim=-2) - 1).sum(dim=-2) / (window_size*(window_size-1))
mask = frames_sim > threshold
```

**New code**:
```python
# Compute optical flow magnitude between consecutive frames
# flow_mag: [B, window_size-1, H, W] — magnitude per frame pair
flow_mag = self._compute_flow_magnitude(current_frames)  # NEW method

# Pool flow magnitude to match token grid (12x12 per frame)
# flow_mag_pooled: [B, window_size, 144] — per-token motion score
flow_mag_pooled = self._pool_flow_to_tokens(flow_mag, H=12, W=12)

# A token is "dynamic" if its average flow magnitude across the window exceeds threshold
avg_flow = flow_mag_pooled.mean(dim=1)  # [B, 144]
threshold = torch.quantile(avg_flow, 1.0 - self.config.cluster_ratio)  # top cluster_ratio% are dynamic
mask = avg_flow > threshold  # [B, 144] — True = static, False = dynamic
```

**New methods to add to `PllavaForConditionalGeneration`**:

```python
def _compute_flow_magnitude(self, frames):
    """
    Compute optical flow magnitude between consecutive frames.
    Args:
        frames: [B, W, L, C] — window of frames (already in feature space)
    Returns:
        flow_mag: [B, W-1, H, W] — flow magnitude per frame pair
    """
    # Convert frames back to pixel space for flow computation
    # Or: use a lightweight flow estimator on the feature maps
    # Option A: Farneback on grayscale (cheap, accurate)
    # Option B: RAFT (expensive, more accurate)
    # Recommendation: Farneback for speed, since this runs once per window
    pass

def _pool_flow_to_tokens(self, flow_mag, H=12, W=12):
    """
    Pool dense flow magnitude to match the token grid.
    Args:
        flow_mag: [B, T, h, w] — dense flow magnitude
        H, W: target grid size (12x12 for PLLaVA)
    Returns:
        pooled: [B, T, H*W] — per-token motion score
    """
    # Adaptive average pooling to target grid size
    pass
```

**Implementation choice — Farneback vs RAFT**:

| Aspect | Farneback | RAFT |
|--------|-----------|------|
| Speed | ~5ms per frame pair | ~50ms per frame pair |
| Accuracy | Good for large motion | Better for small/subtle motion |
| Dependencies | OpenCV (already installed) | Requires custom model weights |
| Memory | Minimal | ~200MB GPU |
| Recommendation | **Use this** | Overkill for this task |

Farneback is sufficient because:
1. We only need coarse motion magnitude per token, not precise flow vectors
2. This runs once per window during vision merge, not per LLM layer
3. The existing `flow_correlation_sanity_check.py` already uses Farneback successfully

**Parameters to tune**:
- `cluster_ratio`: What fraction of tokens are "dynamic" (default 0.5 → top 50% by flow magnitude)
- Flow threshold: Use quantile-based (top X% are dynamic) instead of fixed threshold

**Expected impact**: Dynamic tokens (high motion) get more retention in LLM pruning. This directly helps Moving Direction, Action Sequence, and other motion-heavy tasks.

---

### Change 2: Fix Two-Tier Retention to Use Flow-Based Motion (CONNECT TO CHANGE 1)

**What**: Make the two-tier retention in `elastic_cache.py` actually use the optical flow signal from Change 1.

**Where**: `models/pllava/elastic_cache.py`, `process_attention()` and `_compute_motion_scores()`

**Current problem**: `_compute_motion_scores()` computes attention temporal variance → always ~0.

**New approach**: Pass flow-based motion scores from the vision merge stage to the LLM pruning stage.

**Data flow**:
```
merge_frames_dynamic() → stores flow_mag_pooled on self
    ↓
VTPWindowCache.process_attention() → reads flow_mag_pooled instead of computing attention variance
```

**Changes needed**:

1. In `modeling_pllava.py`, after `merge_frames_dynamic()`, store the flow scores:
```python
merged, static_sizes, dynamic_sizes, window_sizes = self.merge_frames_dynamic(image_features)
self._last_flow_motion_scores = flow_mag_pooled  # [B, window_size, 144]
```

2. In `elastic_cache.py`, `_compute_motion_scores()` now reads stored flow scores:
```python
def _compute_motion_scores(self, flow_motion_scores, static_sizes, dynamic_sizes, window_sizes):
    """
    Compute per-window motion score from optical flow magnitude.
    Args:
        flow_motion_scores: [B, max_window, 144] — per-token flow magnitude
    """
    motion_scores = []
    token_idx = 0
    for static_size, dynamic_size, window_size in zip(static_sizes, dynamic_sizes, window_sizes):
        # Average flow magnitude across all tokens in this window
        window_flow = flow_motion_scores[0, :window_size, :].mean().item()
        motion_scores.append(window_flow)
    return motion_scores
```

3. In `prompt_prefill()`, pass flow scores to `process_attention()`:
```python
flow_scores = getattr(self, '_last_flow_motion_scores', None)
topk_indices = self.process_attention(
    text_to_image_attentions, static_sizes, dynamic_sizes, window_sizes,
    flow_motion_scores=flow_scores  # NEW parameter
)
```

**Expected impact**: Two-tier retention now actually differentiates motion-heavy vs static windows. Motion-heavy windows retain more tokens.

---

### Change 3: Kill Borderline Preservation (SIMPLE FIX)

**What**: Disable borderline preservation entirely.

**Why**: The -2.13% drop on Action Sequence is directly caused by keeping low-importance tokens near the pruning cutoff. These tokens add noise, not signal.

**Where**: `models/pllava/elastic_cache.py`, `process_attention()` lines 164-171

**Change**: Simply skip the borderline block:
```python
# REMOVED: borderline preservation
# The old code kept tokens near the cutoff, adding noise
# Just use top-k directly without borderline expansion
```

Or, more conservatively, set `use_borderline_preservation=False` as default everywhere.

**Expected impact**: Eliminates the noise injection that caused Action Sequence to drop 2.13%.

---

### Change 4: Temporal Pivot Anchoring in Merge (TARGETED FIX)

**What**: When merging pruned tokens into kept tokens, bias pivot selection toward temporally adjacent kept tokens instead of purely feature-similar ones.

**Why**: The current `TextPivotMerge_LayerWise` (`llama.py:1626-1733`) selects pivots by cosine similarity. A pruned token from frame *t* could get merged into a kept token from frame *t+3*, losing directional/sequential information.

**Where**: `models/pllava/llama.py`, `TextPivotMerge_LayerWise.__call__()` lines 1692-1704

**Current code** (lines 1694-1695):
```python
similarity = (k_hh_pruned / torch.norm(...)) @ (k_hh_recent / torch.norm(...)).transpose(-1, -2)
max_values, max_indices = similarity.max(dim=-1)
```

**New code**:
```python
# Pure cosine similarity
cos_sim = (k_hh_pruned / torch.norm(...)) @ (k_hh_recent / torch.norm(...)).transpose(-1, -2)

# Temporal proximity bonus: tokens closer in time get a bonus
# Compute temporal distance between pruned and kept tokens
# pruned_positions: [num_pruned] — temporal positions of pruned tokens
# kept_positions: [num_kept] — temporal positions of kept tokens
temporal_dist = torch.cdist(
    pruned_positions.float().unsqueeze(0),
    kept_positions.float().unsqueeze(0)
)  # [1, num_pruned, num_kept]

# Bonus: exponential decay with temporal distance
temporal_bonus = torch.exp(-temporal_dist / self.temporal_sigma)  # sigma=2.0 default

# Combined score: weighted sum of similarity and temporal proximity
combined = (1 - self.temporal_weight) * cos_sim + self.temporal_weight * temporal_bonus
max_values, max_indices = combined.max(dim=-1)
```

**New parameters**:
- `temporal_weight`: How much to weight temporal proximity (0.0 = ignore time, 1.0 = only time). Default: 0.3
- `temporal_sigma`: Decay rate for temporal distance (higher = more tolerant of distant frames). Default: 2.0

**Challenge**: We need to know the temporal position of each token. This information is available from `window_sizes` and `static_sizes`/`dynamic_sizes` — each token's position can be inferred from the window structure.

**Expected impact**: Motion trajectories are preserved better during merge. Helps Action Sequence and Moving Direction.

---

## Execution Plan

### Step 1: Implement Change 1 + 2 (Optical Flow + Two-Tier Fix) — ~4-6 hrs

1. Add `_compute_flow_magnitude()` and `_pool_flow_to_tokens()` to `modeling_pllava.py`
2. Modify `merge_frames_dynamic()` to use flow instead of feature similarity
3. Store flow scores on the model instance
4. Update `elastic_cache.py` to use stored flow scores
5. Smoke test: verify different motion_scale values produce different retention ratios

### Step 2: Implement Change 3 (Kill Borderline) — ~15 min

1. Set `use_borderline_preservation=False` as default
2. Or remove the borderline code entirely from `process_attention()`

### Step 3: Run 5-Task Eval — ~4-6 hrs

Run baseline vs flow-based two-tier on 5 tasks, all samples:

```bash
# Baseline (already done): 50.71%
# Flow-based two-tier:
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --use_lora --lora_alpha 14 \
    --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --cluster_ratio 0.5 --temporal_segment_ratio 0.25 \
    --use_motion_adaptive --motion_scale 1.0 \
    --conv_mode plain --num_frames 16 \
    --tasks "Action Sequence,Action Prediction,Moving Direction,Object Interaction,Unexpected Action"
```

### Step 4: Decision Gate

| Result | Action |
|--------|--------|
| Flow-based > baseline by ≥1% | Proceed to Change 4 + full benchmarks |
| Flow-based > baseline by 0.5-1% | Proceed, frame as marginal |
| Flow-based ≈ baseline (±0.5%) | Try different motion_scale, then proceed |
| Flow-based < baseline | Debug flow computation, check if flow magnitude actually varies |

### Step 5: Implement Change 4 (Temporal Pivot) — ~2-3 hrs

Only if Change 1+2 shows improvement or neutrality.

### Step 6: Full Benchmarks — ~24-48 hrs

- Full 20-task MVBench (baseline vs best config)
- MVBench motion-heavy subset (14 tasks)
- MVBench temporal-complex subset (5 tasks)
- VideoMME long videos (if data available)

---

## Parameters to Sweep

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `motion_scale` | 0.5, 1.0, 2.0 | 1.0 | How much flow magnitude affects retention |
| `cluster_ratio` | 0.3, 0.5, 0.7 | 0.5 | Fraction of tokens classified as dynamic |
| `alpha` | 0.3, 0.4, 0.5 | 0.4 | Base retention ratio |
| `temporal_weight` | 0.1, 0.3, 0.5 | 0.3 | Weight for temporal pivot anchoring |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Flow computation too slow | Low | Medium | Farneback is ~5ms/frame, runs once per window |
| Flow magnitude doesn't vary across MVBench videos | Medium | High | Run sanity check first on 20 videos |
| Two-tier retention still doesn't help | Medium | High | The signal is real now; if it doesn't help, the problem is elsewhere |
| Temporal pivot anchoring breaks things | Low | Medium | Small surgical change, easy to ablate |

---

## Key Insight

**The fundamental problem was never about the pruning algorithm — it was about the motion signal.**

- Entropy signal: uniform (0.9999) → useless
- Attention variance: uniform → useless (motion scores ~0)
- Feature similarity: weak proxy → unreliable

Optical flow magnitude is the only signal that actually measures motion. Everything else is a proxy that fails on PLLaVA because of its uniform attention architecture.

---

## Files to Modify

| File | Changes |
|------|---------|
| `models/pllava/modeling_pllava.py` | Add `_compute_flow_magnitude()`, `_pool_flow_to_tokens()`, modify `merge_frames_dynamic()` |
| `models/pllava/elastic_cache.py` | Update `_compute_motion_scores()` to use flow scores, update `process_attention()` signature |
| `models/pllava/configuration_pllava.py` | Add `temporal_weight`, `temporal_sigma` params |
| `models/pllava/llama.py` | Add temporal proximity to `TextPivotMerge_LayerWise` (Change 4 only) |
| `tasks/eval/mvbench/pllava_eval_mvbench.py` | Add `--temporal_weight` CLI arg |
| `tasks/eval/model_utils.py` | Pass new params through |

---

## What We're NOT Doing (And Why)

1. **NOT using RAFT** — Overkill for coarse motion detection. Farneback is sufficient and faster.
2. **NOT changing the pruning layer** — Layer 10 works for the baseline. The issue is WHAT gets pruned, not WHEN.
3. **NOT changing alpha** — 0.4 is fine. The issue is that alpha doesn't vary by motion.
4. **NOT adding more training** — This is inference-time only. No fine-tuning needed.

---

## Expected Outcome

If optical flow provides a real motion signal:
- Motion-heavy tasks (Moving Direction, Action Sequence) should improve
- Static tasks (Object Interaction, Action Prediction) should stay the same
- Overall accuracy should improve by 1-3% on motion-heavy subsets
- FLOPs reduction should remain similar (~0.16x)
