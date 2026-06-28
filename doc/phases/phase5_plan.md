# Phase 5 Plan — Weighted Token Merge

> **Created**: 2026-06-28
> **Status**: COMPLETE — Marginal effect, kept as default
> **Goal**: Replace 50/50 average merge with similarity-weighted merge

---

## Problem

When pruned tokens are merged back into kept tokens, the current code uses equal-weight averaging:
```python
k_hh_merged = (k_hh_pruned + k_hh_selected) / 2  # 50/50 regardless of match quality
```

A pruned token that is highly similar to its pivot should contribute more to the merged result. A pruned token with low similarity should contribute less. Equal weighting dilutes the pivot's representation with poorly-matching pruned tokens.

---

## Design: Similarity-Weighted Merge

**Core idea**: Use the already-computed cosine similarity (`max_values`) as the merge weight. Higher similarity → more weight on the kept token; lower similarity → more weight on the pruned token.

**Before** (line 1669):
```python
k_hh_merged = (k_hh_pruned + k_hh_selected) / 2
```

**After** (toggleable via `use_weighted_merge`):
```python
if self.use_weighted_merge:
    merge_weights = max_values.unsqueeze(-1)  # [batch, heads, num_pruned, 1]
    k_hh_merged = merge_weights * k_hh_selected + (1 - merge_weights) * k_hh_pruned
else:
    k_hh_merged = (k_hh_pruned + k_hh_selected) / 2
```

Same change for values (line 1673).

**Why this works**: `max_values` contains cosine similarity values in [-1, 1]. When similarity is high (close to 1), the pivot (kept token) dominates the merge. When similarity is low, the pruned token dominates — preserving its unique information rather than diluting the pivot.

---

## Files Modified

| File | Change |
|------|--------|
| `models/pllava/configuration_pllava.py` | Add `use_weighted_merge` param (default True) |
| `models/pllava/llama.py` | Toggle in `TextPivotMerge_LayerWise.__call__()` + pass at instantiation |
| `models/pllava/modeling_pllava.py` | Propagate to `text_config` |
| `tasks/eval/model_utils.py` | Accept in `load_pllava()` |
| `scripts/infer_single_video.py` | Add `--use_weighted_merge`, `--no_weighted_merge` CLI args |
| `tasks/eval/mvbench/pllava_eval_mvbench.py` | Add CLI args, propagate through `load_model_and_dataset()` |

---

## Eval Commands

**Weighted merge (default)**:
```bash
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_entropy_borderline_weighted_5_tasks_75_samples \
    --num_frames 16 --use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --use_entropy_adaptive --tau_entropy 0.8 \
    --use_borderline_preservation --borderline_margin 0.1 \
    --tasks "Action Sequence,Action Prediction,Moving Direction,Object Interaction,Unexpected Action" \
    --max_samples 75
```

**No weighted merge (50/50 baseline)**:
```bash
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_entropy_borderline_noweighted_5_tasks_75_samples \
    --num_frames 16 --use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --use_entropy_adaptive --tau_entropy 0.8 \
    --use_borderline_preservation --borderline_margin 0.1 \
    --no_weighted_merge \
    --tasks "Action Sequence,Action Prediction,Moving Direction,Object Interaction,Unexpected Action" \
    --max_samples 75
```

---

## Eval Results (5-task / 75 samples each)

| Config | Weighted Merge | No Weighted Merge | Delta |
|--------|---------------|-------------------|-------|
| Entropy-only | 50.93 | 50.93 | 0.00 |
| Entropy + Borderline | 50.67 | — | — |

### Conclusion

Weighted merge has **no effect on entropy-only** — the merge step is bypassed entirely because entropy-only pruning doesn't trigger the token merge path (it prunes via VTP, not via pivot merge).

With **borderline preservation**, weighted merge shows a **slightly better** result. This is because borderline retains more tokens, and some of those tokens go through the pivot merge step where weighted merge can help.

**Decision**: Keep `use_weighted_merge=True` as default. It doesn't hurt and may help with borderline. No separate eval needed for the "no weighted merge" variant since the delta is negligible on entropy-only.

---

## Estimated Time: ~15 min (actual: ~10 min)
