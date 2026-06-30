# Phase 7 Debugging Guide

> **Created**: 2026-06-30
> **Status**: In Progress — Phase A partially implemented
> **Purpose**: Document all bugs found and fixes needed so you can debug and complete the work

---

## Executive Summary

We identified **3 fundamental issues** with PruneVid's pruning mechanisms on PLLaVA, plus several implementation bugs. Two mechanisms (entropy and attention-based motion) are fundamentally broken because PLLaVA's attention is uniform (entropy=0.9999), making any attention-based scoring useless.

**What works**: The pruning infrastructure (token counting, Stage 1 merge, fallback pruning) is solid.
**What's broken**: All attention-based scoring mechanisms (entropy, motion, query) produce uniform scores.

---

## Issue 1: Entropy-Adaptive Pruning (DEAD END)

### Root Cause
All 32 layers have entropy = 0.9999 (uniform attention). The threshold `tau=0.8` never fires.

### Evidence
```
Smoke test results (cooking.mp4):
Layer 0: entropy=0.9999
Layer 1: entropy=0.9999
...
Layer 31: entropy=0.9999
```

### Why
PLLaVA uses a vision-language model where attention is distributed evenly across all image tokens. Unlike text models where attention is sparse, vision tokens receive uniform attention.

### Recommendation
**Abandon entropy-based pruning for PLLaVA.** This is a negative result worth reporting.

---

## Issue 2: Motion-Adaptive Pruning (BROKEN — needs fix)

### Root Cause
Motion scores are computed from attention temporal variance, but attention is uniform → variance is near zero → all motion scores are identical.

### Evidence
```
Motion scores: ['0.000000', '0.000000']
Max motion: 0.000001
Window 0: motion_ratio=0.0000 -> alpha=0.4000 (base=0.4)
Window 1: motion_ratio=0.0000 -> alpha=0.4000 (base=0.4)
```

### Current Code State
I started implementing a fix (two-tier retention) but left it partially done. Here's what needs to happen:

### Fix: Two-Tier Retention Budget

**Concept**: Instead of using attention-based motion scores, use the Stage 1 static/dynamic split. Dynamic tokens (which Stage 1 identified as changing across frames) get higher retention.

**File**: `models/pllava/elastic_cache.py`, `process_attention()` method

**What I changed** (may need cleanup):
- Removed `alpha = self.alpha` before the loop
- Added `dynamic_alpha = min(self.alpha * (1.0 + self.motion_scale), 1.0)` for dynamic tokens
- Static tokens still use `self.alpha`
- Dynamic tokens use `dynamic_alpha`

**What still needs work**:
1. **Lines 131-134 in `elastic_cache.py`**: `_compute_motion_scores()` is called but result is unused. Remove these lines:
   ```python
   # DELETE these 3 lines (currently at lines 131-134):
   motion_scores = self._compute_motion_scores(text_to_image_attentions, static_sizes, dynamic_sizes, window_sizes)
   max_motion = max(motion_scores) if motion_scores else 1.0
   max_motion = max(max_motion, 1e-6)  # avoid div by zero
   ```
2. The `_compute_motion_scores()` method (lines 189-210) is now dead code — can be removed entirely
3. The smoke test script (`scripts/smoke_test_motion.py`) has a monkey-patch that's no longer needed — can be simplified

**Parameters to tune**:
- `motion_scale`: How much more dynamic tokens retain. Try [0.5, 1.0, 2.0]
  - `motion_scale=0.5`: dynamic tokens retain 1.5x more than static
  - `motion_scale=1.0`: dynamic tokens retain 2x more than static
  - `motion_scale=2.0`: dynamic tokens retain 3x more than static

**Smoke test command**:
```bash
conda run -n pllava --no-capture-output python -m scripts.smoke_test_motion
```

**Expected behavior**: With `motion_scale=1.0`, dynamic tokens should have ~2x retention ratio compared to static tokens.

---

## Issue 3: Query-Conditioned Pruning (NOT IMPLEMENTED)

### Current State
- `text_indices` parameter is threaded through 4 files but always `None`
- No code computes which tokens correspond to the query vs image

### What Needs to Be Done

**Step 1**: Compute `text_indices` from `input_ids` in `modeling_pllava.py`

**File**: `models/pllava/modeling_pllava.py`, in the forward pass before calling `merge_frames_dynamic()`

**Logic**:
```python
# Find where image tokens start in input_ids
# PLLaVA uses <image> placeholder tokens
# text_indices = indices of non-image tokens (query tokens)
image_token_id = self.config.image_token_index  # or however PLLaVA marks image tokens
text_indices = (input_ids != image_token_id).nonzero(as_tuple=True)[0]
```

**Step 2**: Forward `text_indices` to cache in `llama.py`

**File**: `models/pllava/llama.py`, in `LlamaModelVTP.forward()`

The `text_indices` parameter already exists in the function signature — just need to pass it through.

**Step 3**: Use `text_indices` in scoring in `elastic_cache.py`

**File**: `models/pllava/elastic_cache.py`, `process_attention()`

**Current scoring** (line 141):
```python
window_attentions = text_to_image_attentions[0].max(dim=0)[0].max(dim=0)[0]
```

**New scoring**:
```python
max_attn = text_to_image_attentions[0].max(dim=0)[0].max(dim=0)[0]
if text_indices is not None and len(text_indices) > 0:
    # Attention from query tokens only
    query_attn = text_to_image_attentions[0, :, text_indices].max(dim=0)[0].max(dim=0)[0]
    # Combine: weighted average of max attention and query attention
    window_attentions = (1 - self.query_weight) * max_attn + self.query_weight * query_attn
else:
    window_attentions = max_attn
```

**Step 4**: Add `query_weight` config parameter

**File**: `models/pllava/configuration_pllava.py`

Add to `PllavaConfig`:
```python
query_weight: float = 0.5  # Weight for query-conditioned scoring
```

**Step 5**: Pass `query_weight` through the stack

**Files to modify**:
- `tasks/eval/model_utils.py`: Pass `query_weight` to config
- `tasks/eval/mvbench/pllava_eval_mvbench.py`: Add `--query_weight` CLI arg

---

## Issue 4: `pruned_tokens: None` Bug (FIXED)

### Root Cause
`PllavaForConditionalGeneration` wraps `LlamaModelVTP` with a `.model` attribute. After the forward pass, `lm_model.last_pruned_token_count` accesses the wrapper, which doesn't have the attribute.

### Fix Applied
**File**: `tasks/eval/model_utils.py`, line 379-381
```python
if not hasattr(lm_model, 'last_pruned_token_count'):
    lm_model = lm_model.model  # Unwrap PeftModel
```

### Status
**Fixed in commit `b2db5b5`**. No action needed.

---

## Issue 5: VTP-Entropy Log Capture (FIXED)

### Root Cause
`logging.basicConfig()` in eval scripts defaulted to WARNING level, so INFO messages from the pruning module were not captured.

### Fix Applied
Changed `logging.basicConfig()` to `logging.basicConfig(level=logging.INFO)` in:
- `tasks/eval/mvbench/pllava_eval_mvbench.py`
- `tasks/eval/egoschema/pllava_eval_egoschema.py`
- `tasks/eval/videomme/pllava_eval_videomme.py`

### Status
**Fixed in commit `b2db5b5`**. No action needed.

---

## Issue 6: `last_dynamic_selected_layer` Reset Bug (FIXED)

### Root Cause
In `llama.py:1584-1590`, the `else` branch was resetting entropy state during decoding (when `current_length_len > 1`), which wiped out the pruning decision made during prefill.

### Fix Applied
Changed to only reset entropy state on prefill:
```python
if current_length_len <= 1:
    # Prefill: reset entropy state
    layer.self_attn.vtp_cache.last_dynamic_selected_layer = None
    layer.self_attn.vtp_cache.last_pruned_token_count = None
else:
    # Decoding: preserve pruning decision
    pass
```

### Status
**Fixed in commit `b2db5b5`**. No action needed.

---

## Issue 7: `FRIEND_ADVICE.md` — Optical Flow (NOT YET APPLIED)

### What It Says
The friend recommended using optical flow instead of attention-based motion scoring. They also suggested:
1. Multi-scale temporal windows
2. Two-tier retention budget (dynamic tokens keep more)
3. Query-conditioned pruning
4. Test on motion-heavy categories first

### What's Applicable
- **Two-tier retention**: Implemented in Issue 2 fix above
- **Query-conditioned**: To be implemented (Issue 3)
- **Optical flow**: The model exists in `modeling_pllava_flow.py` but is NOT used by the active model. Using it would require significant refactoring. **Recommendation**: Skip optical flow for now, use the simpler two-tier approach.
- **Motion-heavy categories**: Good advice for evaluation — test on "Moving Direction", "Action Sequence" first

### Status
Partially applied. Two-tier retention is implemented. Query-conditioned is TODO.

---

## Current Code State

### What's Working
1. Token counting (raw, merged, pruned) — correct
2. Stage 1 vision merge (DPC-KNN) — working
3. Fallback pruning — working
4. Logging — fixed
5. LoRA unwrapping — fixed

### What's Partially Done
1. **Two-tier retention** in `elastic_cache.py` — code changed but needs cleanup:
   - `_compute_motion_scores()` is now dead code
   - `motion_scores` parameter in `process_attention()` is unused
   - Need to test with different `motion_scale` values

### What's Not Done
1. Query-conditioned pruning (Issue 3)
2. Ablation study (Phase C)

---

## How to Debug

### Step 1: Verify Two-Tier Retention Works
```bash
# Run smoke test
conda run -n pllava --no-capture-output python -m scripts.smoke_test_motion

# Expected: dynamic tokens should have higher retention
# Check output for token counts
```

### Step 2: Test Different motion_scale Values
```bash
# In infer_single_video.py or smoke test, try:
# motion_scale=0.5: dynamic retain 1.5x more
# motion_scale=1.0: dynamic retain 2x more
# motion_scale=2.0: dynamic retain 3x more
```

### Step 3: Run 5-Task Eval
```bash
# Baseline (already done): 50.71%
# Run with motion-scale=1.0
conda run -n pllava python -m tasks.eval.mvbench.pllava_eval_mvbench \
  --use_lora --lora_alpha 14 \
  --selected_layer 10 --alpha 0.4 --tau 0.8 \
  --cluster_ratio 0.5 --temporal_segment_ratio 0.25 \
  --use_motion_adaptive --motion_scale 1.0 \
  --conv_mode plain --num_frames 16 \
  --max_new_tokens 10 \
  --output_dir doc/eval/mvbench_dev_motion_scale_1.0

# Compare results
python -c "
import json
b = json.load(open('doc/eval/mvbench_dev_baseline_all/all_results.json'))['results']
m = json.load(open('doc/eval/mvbench_dev_motion_scale_1.0/all_results.json'))['results']
for t in ['action_sequence','action_prediction','moving_direction','object_interaction','unexpected_action']:
    print(f'{t}: baseline={b[t][\"acc\"]:.2f} motion={m[t][\"acc\"]:.2f} diff={m[t][\"acc\"]-b[t][\"acc\"]:+.2f}')
print(f'AVG: baseline={sum(v[\"acc\"] for v in b.values())/len(b):.2f} motion={sum(v[\"acc\"] for v in m.values())/len(m):.2f}')
"
```

### Step 4: If Two-Tier Helps, Implement Query-Conditioned (Issue 3)
Follow the steps in Issue 3 above.

### Step 5: Ablation Study (Phase C)
Run 4 configs and compare:
1. Baseline (done: 50.71%)
2. Motion-only (Step 3 above)
3. Query-only (after Step 4)
4. Motion+Query (after Step 4)

---

## Decision Gate

After ablation study:

| Result | Action |
|--------|--------|
| Any config beats baseline by >=1% | Full 20-task MVBench + EgoSchema + VideoMME |
| Any config beats baseline by 0.5-1% | Full MVBench, frame as "marginal improvement" |
| All configs approx baseline (+-0.5%) | Still run full MVBench — larger samples may reveal effect |
| All configs < baseline | Pivot to FLOPs reduction narrative |

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `models/pllava/elastic_cache.py` | VTPWindowCache — pruning logic, two-tier retention |
| `models/pllava/llama.py` | LlamaModelVTP — forward pass, entropy state management |
| `models/pllava/modeling_pllava.py` | PLLaVA model — merge_frames_dynamic, text_indices computation |
| `models/pllava/configuration_pllava.py` | Config params (motion_scale, query_weight) |
| `tasks/eval/model_utils.py` | load_pllava, pllava_answer — token info extraction |
| `tasks/eval/mvbench/pllava_eval_mvbench.py` | MVBench eval script |
| `doc/eval/mvbench_dev_baseline_all/` | Baseline results (50.71%) |
| `scripts/smoke_test_motion.py` | Smoke test for motion-adaptive pruning |

---

## Quick Reference: Current Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `selected_layer` | 10 | Layer to apply pruning (fixed) |
| `alpha` | 0.4 | Base retention ratio (40% tokens kept) |
| `tau` | 0.8 | Entropy threshold (unused — always 0.9999) |
| `cluster_ratio` | 0.5 | DPC-KNN clustering ratio for Stage 1 |
| `temporal_segment_ratio` | 0.25 | Window size as fraction of frames |
| `use_motion_adaptive` | False | Enable two-tier retention |
| `motion_scale` | 1.0 | How much more dynamic tokens retain |
| `use_borderline_preservation` | False | Keep borderline tokens near threshold |
| `borderline_margin` | 0.1 | Margin for borderline preservation |
| `query_weight` | N/A | Weight for query-conditioned scoring (not yet implemented) |
