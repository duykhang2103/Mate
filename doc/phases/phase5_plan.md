# Phase 5 Plan — Weighted Token Merge

> **Created**: 2026-06-28
> **Status**: PLANNING
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

**After**:
```python
merge_weights = max_values.unsqueeze(-1)  # [batch, heads, num_pruned, 1]
k_hh_merged = merge_weights * k_hh_selected + (1 - merge_weights) * k_hh_pruned
```

Same change for values (line 1673).

**Why this works**: `max_values` contains cosine similarity values in [-1, 1]. When similarity is high (close to 1), the pivot (kept token) dominates the merge. When similarity is low, the pruned token dominates — preserving its unique information rather than diluting the pivot.

---

## Files Modified

| File | Change |
|------|--------|
| `models/pllava/llama.py` | Replace 2 lines in `TextPivotMerge_LayerWise.__call__()` |

---

## Estimated Time: ~15 min
