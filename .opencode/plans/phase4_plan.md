# Phase 4 Plan — Borderline Token Preservation

> **Created**: 2026-06-25
> **Status**: PLANNING
> **Goal**: Preserve borderline tokens (near the pruning cutoff) to reduce cascading errors

---

## Problem

Single-shot pruning at one layer permanently removes tokens that might become important in deeper layers. Tokens with attention scores just below the cutoff are "borderline" — they could flip to important as representations evolve. Dropping them early creates cascading information loss.

---

## Design: Borderline Margin Preservation

**Core idea**: After selecting top-k tokens by attention score, also keep any tokens whose score is within a configurable margin (e.g., 10%) of the cutoff score. This adds a safety buffer without changing the pruning architecture.

**Before** (current):
```
top-k selection → keep exactly alpha% of tokens
```

**After**:
```
top-k selection → compute cutoff score → keep top-k + tokens within margin of cutoff
```

**Effect**: Slightly more tokens retained per window (~5-15% increase), but the retained set is more robust to representation shift in deeper layers.

---

## Implementation Plan

### Step 1: Add Config Params

**File**: `models/pllava/configuration_pllava.py`

Add to `PllavaConfig.__init__()`:
```python
use_borderline_preservation=False,   # enable borderline margin
borderline_margin=0.1,               # margin as fraction of cutoff score (0.1 = 10%)
```

### Step 2: Modify `process_attention()` in VTPWindowCache

**File**: `models/pllava/elastic_cache.py`

Update `__init__` to accept new params:
```python
def __init__(self, ..., use_borderline_preservation=False, borderline_margin=0.1):
    self.use_borderline_preservation = use_borderline_preservation
    self.borderline_margin = borderline_margin
```

Update static token selection (around line 143-146):
```python
num_retain_static_tokens = int(static_size * alpha)
_, static_topk_indices = torch.topk(static_attentions, k=num_retain_static_tokens, dim=-1)

if self.use_borderline_preservation and num_retain_static_tokens > 0:
    cutoff_score = static_attentions[static_topk_indices[-1]]  # lowest score in top-k
    margin_threshold = cutoff_score * (1.0 - self.borderline_margin)
    borderline_mask = static_attentions >= margin_threshold
    # Union of top-k and borderline
    combined_mask = torch.zeros_like(static_attentions, dtype=torch.bool)
    combined_mask[static_topk_indices] = True
    combined_mask = combined_mask | borderline_mask
    static_topk_indices = torch.where(combined_mask)[0]
```

Same pattern for dynamic tokens (around line 149-156):
```python
num_retain_dynamic_tokens = int(dynamic_attentions.shape[-1] * alpha)
_, dynamic_topk_indices = torch.topk(dynamic_attentions, k=num_retain_dynamic_tokens, dim=-1)

if self.use_borderline_preservation and num_retain_dynamic_tokens > 0:
    # For dynamic: compute per-frame cutoff, then apply borderline across all frames
    all_dynamic_scores = dynamic_attentions.reshape(-1)
    all_topk_flat = dynamic_topk_indices + torch.arange(window_size, device=dynamic_attentions.device).unsqueeze(1) * dynamic_attentions.shape[-1]
    all_topk_flat = all_topk_flat.reshape(-1)
    cutoff_score = all_dynamic_scores[all_topk_flat[-1]]
    margin_threshold = cutoff_score * (1.0 - self.borderline_margin)
    borderline_mask = all_dynamic_scores >= margin_threshold
    combined_mask = torch.zeros_like(all_dynamic_scores, dtype=torch.bool)
    combined_mask[all_topk_flat] = True
    combined_mask = combined_mask | borderline_mask
    dynamic_topk_indices = torch.where(combined_mask)[0]
    # Reshape back to (num_frames, tokens_per_frame) if needed
```

### Step 3: Propagate Config Params Through Call Chain

**Files to modify**:
1. `models/pllava/configuration_pllava.py` — add params (Step 1)
2. `models/pllava/modeling_pllava.py` — propagate to LLM config
3. `models/pllava/llama.py` — pass to VTPWindowCache constructor
4. `tasks/eval/model_utils.py` — accept in `load_pllava()`
5. `scripts/infer_single_video.py` — add CLI args
6. `tasks/eval/mvbench/pllava_eval_mvbench.py` — add CLI args

### Step 4: Smoke Test

Run on `example/cooking.mp4` with borderline preservation:
```bash
python scripts/infer_single_video.py \
    --video example/cooking.mp4 \
    --model_dir MODELS/pllava-7b \
    --weight_dir MODELS/pllava-7b \
    --use_lora --lora_alpha 14 \
    --use_entropy_adaptive --tau_entropy 0.8 \
    --use_borderline_preservation --borderline_margin 0.1 \
    --alpha 0.4
```

Verify: token count should be slightly higher than without borderline (more tokens retained).

### Step 5: MVBench Dev Eval

Compare: entropy-only vs entropy+borderline.

```bash
# Baseline: entropy-only (already done in Phase 2/3)

# New: entropy + borderline
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_entropy_borderline_pruned_5_tasks_75_samples \
    --num_frames 16 \
    --use_lora --lora_alpha 14 \
    --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 \
    --selected_layer 10 \
    --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 \
    --cluster_ratio 0.5 \
    --use_entropy_adaptive \
    --tau_entropy 0.8 \
    --use_borderline_preservation --borderline_margin 0.1 \
    --tasks "Action Sequence,Action Prediction,Moving Direction,Object Interaction,Unexpected Action" \
    --max_samples 75 > log_mvbench_entropy_borderline_pruned_5_tasks_75_samples.log 2>&1
```

### Step 6: Compare Results

| Result | Action |
|--------|--------|
| Borderline > entropy-only | Keep borderline preservation |
| Borderline ≈ entropy-only | Drop borderline, focus on entropy + motion |
| Borderline < entropy-only by >1% | Debug or drop |

---

## Files Modified

| File | Change |
|------|--------|
| `models/pllava/configuration_pllava.py` | Add `use_borderline_preservation`, `borderline_margin` |
| `models/pllava/elastic_cache.py` | Add borderline logic in `process_attention()` |
| `models/pllava/llama.py` | Pass borderline params to VTPWindowCache |
| `models/pllava/modeling_pllava.py` | Propagate borderline params |
| `tasks/eval/model_utils.py` | Accept borderline params in `load_pllava()` |
| `scripts/infer_single_video.py` | Add borderline CLI args |
| `tasks/eval/mvbench/pllava_eval_mvbench.py` | Add borderline CLI args |

---

## Estimated Time: ~45 min

- Config + propagation: 20 min
- Borderline logic in `process_attention()`: 10 min
- Smoke test: 10 min
- MVBench dev eval: 5 min (user runs)
