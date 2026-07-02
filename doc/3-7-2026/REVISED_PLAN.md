# Revised Plan: PruneVid+ (Updated 3-7-2026)

## Critical Finding: Flow Vision Merge is Harmful

The full 988-sample ablation proved that **flow-based vision merge is worse than feature-similarity merge**:

| Config | Avg Accuracy | Delta vs Baseline |
|--------|-------------|-------------------|
| Baseline (feature-similarity merge, no LLM prune) | **50.71%** | — |
| Flow merge only (selected_layer=999) | **46.05%** | **-4.66%** |
| Flow merge + LLM prune (selected_layer=10) | **45.24%** | **-5.47%** |

**Conclusion**: The regression is primarily from the vision merge change, NOT from LLM pruning. The flow merge produces uniform token counts (CV=4-5%) and destroys the content-adaptive allocation that makes baseline work (CV=27-42%).

**Implication**: Method #1 must work at the **LLM pruning level**, not the vision merge level. The vision merge should stay as feature-similarity (baseline behavior).

---

## Core Constraint (non-negotiable)

- **PLLaVA attention is uniform**: entropy ≈ 0.9999 at all 32 layers.
- All attention-based signals are useless.
- Only pixel-level signals (optical flow, frame difference) work.
- No training allowed (inference-time only).

---

## 4 Viable Methods

### Method 1: Motion-Adaptive LLM Pruning (Optical Flow Scores)

**What**: Use optical flow scores to decide which tokens to keep or drop at the LLM pruning layer (layer 10).

**How it works**:
1. Compute optical flow between consecutive frames (already implemented in `_compute_flow_from_pixels()`)
2. Pool flow magnitude to token grid (already implemented in `_pool_flow_to_tokens()`)
3. Store flow scores on the model (already implemented: `self.language_model._flow_motion_scores`)
4. In `process_attention()`, use flow scores instead of attention variance for two-tier retention:
   - High-flow tokens (dynamic) → higher retention ratio
   - Low-flow tokens (static) → lower retention ratio
   - Average retention stays the same as baseline (alpha=0.4)

**What stays the same**:
- Vision merge: feature-similarity (baseline behavior, CV=27-42%)
- LLM pruning layer: 10 (baseline)
- Base alpha: 0.4 (baseline)

**What changes**:
- `process_attention()` uses `flow_motion_scores` for per-window alpha scaling
- Dynamic windows get higher alpha, static windows get lower alpha
- Total token budget stays roughly the same as baseline

**Existing code**:
- `_compute_flow_from_pixels()` — `modeling_pllava.py:~830`
- `_pool_flow_to_tokens()` — `modeling_pllava.py:~840`
- `_compute_flow_motion_scores()` — `elastic_cache.py:232`
- `process_attention()` with `flow_motion_scores` param — `elastic_cache.py:126`

**What needs to change**:
- Remove flow-based vision merge (revert `merge_frames_dynamic()` to feature-similarity only)
- Keep flow computation in `forward()` for LLM-level scoring
- Ensure flow scores are passed to `process_attention()` correctly

**Expected contribution**: Improve temporal task accuracy by keeping more tokens in dynamic regions while pruning more aggressively in static regions, at the same total token budget.

---

### Method 2: Variance-Weighted Merging

**What**: Replace simple average merge with variance-weighted merge for static tokens.

**How it works**:
1. For each static token, compute feature variance across frames
2. Use variance as weight when merging: high-variance tokens get larger weight
3. This preserves important temporal information that simple averaging dilutes

**Status**: Not implemented. Needs implementation in `merge_frames_dynamic()`.

**Expected contribution**: Better preservation of important temporal information without hurting accuracy.

---

### Method 3: Task-Specific Tuning

**What**: Optimize pruning parameters per benchmark instead of using one fixed setting.

**Parameters to tune**:
- `alpha` (base retention ratio): [0.3, 0.4, 0.5]
- `tau` (similarity threshold): [0.5, 0.8, 1.0]
- `cluster_ratio` (spatial merge ratio): [0.3, 0.5, 0.7]
- `temporal_segment_ratio`: [0.15, 0.25, 0.35]
- `motion_scale` (for flow-based alpha scaling): [0.3, 0.5, 0.7, 1.0]

**Process**:
1. Run cheap ablations on 5-task subset
2. Pick best config per benchmark
3. Run full evaluation with tuned configs

**Expected contribution**: Show that tuned method outperforms fixed PruneVid config at same token budget.

---

### Method 4: Multi-Objective Evaluation

**What**: Extend evaluation to include efficiency metrics alongside accuracy.

**Metrics to report**:
- Token retention ratio (%)
- FLOPs (relative to baseline)
- TTFT (time-to-first-token)
- Prefill time
- Peak GPU memory

**Plots**:
- Accuracy vs token budget curves (10%, 15%, 20%, 25%)
- Efficiency vs accuracy

**Expected contribution**: Show same accuracy with better efficiency, or better robustness across pruning ratios.

---

## Experiment Plan

### Step 1: Revert Flow Vision Merge

- Revert `merge_frames_dynamic()` to use feature-similarity only (baseline behavior)
- Keep optical flow computation in `forward()` for LLM-level scoring
- Verify accuracy returns to baseline (~50.71% on 5-task)

### Step 2: Implement Motion-Adaptive LLM Pruning

- Use flow scores in `process_attention()` for per-window alpha scaling
- Keep total token budget similar to baseline
- Run 5-task eval to check impact

### Step 3: Implement Variance-Weighted Merging

- Replace simple average merge with variance-weighted merge
- Run 5-task eval to check impact

### Step 4: Task-Specific Tuning

- Grid search on 5-task subset
- Pick best config per task family

### Step 5: Full Benchmark Evaluation

- 20-task MVBench
- VideoMME
- EgoSchema
- Multiple token budgets (10%, 15%, 20%, 25%)
- Efficiency metrics

### Step 6: Robustness Curves and Plots

- Accuracy vs token budget
- Efficiency vs accuracy
- Compare against PruneVid published numbers

---

## Target Claim

> "PruneVid+ improves temporal task accuracy at the same token budget through optical flow-guided adaptive LLM pruning and variance-weighted merging, while maintaining comparable performance on other video benchmarks and offering better robustness across pruning ratios."

---

## What Must NOT Be Tried

- Anything attention-based (entropy, attention variance, cross-attention selection, sink tokens)
- Anything requiring training (MLP re-scoring, temporal attention, uptraining)
- Anything requiring extra models (object detection, segmentation)
- Flow-based vision merge (proven harmful: -4.66%)

---

## Competitive Landscape

| Method | Retained | FLOPs | MVBench | Where to beat |
|--------|----------|-------|---------|---------------|
| PruneVid | 16.2% | 0.23x | 47.6 | EgoSchema Fullset (42.6), VCGBench (2.98) |
| FastV | 30.0% | 0.33x | 46.1 | Everywhere (weaker) |
| Prumerge | 55.7% | 0.53x | 45.6 | Everywhere (weaker) |
| Look-M | 20.0% | 1.00x | 46.6 | No FLOPs savings |

**Target**: Match PruneVid's accuracy at same or lower token budget, with better temporal task performance.
