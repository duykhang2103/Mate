# Phase 4 Plan — Borderline Token Preservation

> **Created**: 2026-06-25
> **Status**: IMPLEMENTED — Awaiting eval results
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

**Effect**: Slightly more tokens retained per window (~13% increase in unit test: 35.6% → 48.6%), but the retained set is more robust to representation shift in deeper layers.

**Design decision**: Borderline is applied only to **static tokens**, not dynamic tokens. Dynamic tokens use per-frame top-k which requires uniform per-frame counts — borderline would produce non-uniform counts and break the `.view(window_size, -1)` reshape. Static tokens are the majority of retained tokens and benefit most from borderline preservation.

---

## Implementation Details

### Config params
```python
use_borderline_preservation=False,   # enable borderline margin
borderline_margin=0.1,               # margin as fraction of cutoff score (0.1 = 10%)
```

### Core logic in `process_attention()` (static tokens only)
```python
num_retain_static_tokens = int(static_size * alpha)
_, static_topk_indices = torch.topk(static_attentions, k=num_retain_static_tokens, dim=-1)

if self.use_borderline_preservation and num_retain_static_tokens > 0:
    cutoff_score = static_attentions[static_topk_indices[-1]]
    margin_threshold = cutoff_score * (1.0 - self.borderline_margin)
    borderline_mask = static_attentions >= margin_threshold
    combined_mask = torch.zeros_like(static_attentions, dtype=torch.bool)
    combined_mask[static_topk_indices] = True
    combined_mask = combined_mask | borderline_mask
    static_topk_indices = torch.where(combined_mask)[0]
```

---

## Files Modified

| File | Change |
|------|--------|
| `models/pllava/configuration_pllava.py` | Add `use_borderline_preservation`, `borderline_margin` params |
| `models/pllava/elastic_cache.py` | Add borderline logic in `process_attention()` (static tokens) |
| `models/pllava/llama.py` | Read params from config, pass to VTPWindowCache |
| `models/pllava/modeling_pllava.py` | Propagate params to `text_config` |
| `tasks/eval/model_utils.py` | Accept params in `load_pllava()` |
| `scripts/infer_single_video.py` | Add `--use_borderline_preservation`, `--borderline_margin` CLI args |
| `tasks/eval/mvbench/pllava_eval_mvbench.py` | Add CLI args, propagate through `load_model_and_dataset()` |

---

## Smoke Test Results

Unit test with mock data:
- **Borderline ON**: 286/584 tokens retained (49.0%)
- **Borderline OFF**: 208/584 tokens retained (35.6%)
- **Delta**: +78 tokens (+13.4%)

---

## MVBench Dev Eval Command

```bash
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

## Decision Gate

| Result | Action |
|--------|--------|
| Borderline > entropy-only (50.93) | Keep borderline preservation |
| Borderline ≈ entropy-only (±0.5%) | Keep borderline if tokens retained is higher (more robust) |
| Borderline < entropy-only by >1% | Debug or drop |

---

## Estimated Time: ~45 min (actual: ~30 min)
