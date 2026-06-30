# Phase 7 Plan — Motion-Aware + Query-Conditioned Pruning

> **Created**: 2026-06-30
> **Status**: IN PROGRESS — Phase A
> **Goal**: Improve accuracy on motion-heavy and temporally complex videos while maintaining FLOPs reduction
> **Fallback**: If no accuracy improvement, pivot to FLOPs reduction narrative

---

## Context

### What failed
- **Entropy-adaptive pruning**: All 32 layers have entropy = 0.9999 (uniform attention). The threshold `tau=0.8` never fires. Entropy cannot distinguish important from unimportant tokens in PLLaVA.
- **Motion-adaptive pruning**: Dead code — motion scores are computed in `elastic_cache.py:181-210` but never used to modulate `alpha` (line 140 always uses `self.alpha`).
- **Query-conditioned pruning**: Does not exist — `text_indices` parameter is threaded through but always `None` and never used.

### What works
- **PruneVid baseline (fixed layer 10)**: 50.71% on 5-task MVBench (988 samples)
- **FLOPs tracking**: Token counts correctly captured (raw=2304, merged=~341-987, pruned=~245)
- **Stage 1 (vision merge)**: Working — DPC-KNN temporal clustering + static/dynamic split

### Current accuracy (5-task, 988 samples)
| Method | Avg | Action Seq | Action Pred | Unexpected | Obj Interact | Moving Dir |
|--------|-----|-----------|-------------|-----------|-------------|-----------|
| Baseline (fixed L10) | 50.71 | 55.85 | 51.50 | 62.50 | 63.50 | 20.50 |
| Entropy+Borderline | 50.20 | 53.72 | 51.50 | 62.00 | 63.50 | 20.50 |

---

## Plan: 3 Phases

### Phase A: Fix Motion-Aware Pruning (connect dead code)

**What**: The motion scores are already computed but never used. Connect them to per-window alpha modulation.

**Where to change**: `models/pllava/elastic_cache.py`, `process_attention()` (lines 125-179)

**Current code** (line 138-140):
```python
alpha = self.alpha
for i, (static_size, dynamic_size, window_size) in enumerate(...):
    end_idx = start_idx + static_size + dynamic_size
```

**New code**:
```python
for i, (static_size, dynamic_size, window_size) in enumerate(...):
    end_idx = start_idx + static_size + dynamic_size
    
    # Motion-adaptive alpha: high motion -> keep more tokens
    if self.use_motion_adaptive and motion_scores:
        motion_ratio = motion_scores[i] / max_motion  # normalized to [0, 1]
        alpha = self.alpha * (1.0 + self.motion_scale * motion_ratio)
        alpha = min(alpha, 1.0)  # cap at 1.0
    else:
        alpha = self.alpha
```

**Parameters to tune**:
- `motion_scale`: How much motion affects retention. Try [0.5, 1.0, 2.0]
- `alpha`: Base retention ratio. Current=0.4, try [0.3, 0.4, 0.5]

**Expected impact**: Motion-heavy windows retain more tokens, static windows retain fewer. Should help on Moving Direction, Action Sequence.

**Time**: 30 min implementation + 1 hr smoke test

---

### Phase B: Add Query-Conditioned Pruning

**What**: Use the question's attention pattern to weight which image tokens are important. Different questions need different visual information.

**Where to change**: `models/pllava/elastic_cache.py`, `process_attention()` and `obtain_language_attention()`

**Approach**: 
1. In `obtain_language_attention()`, also extract the query-specific attention (attention from the question tokens, not all text tokens)
2. In `process_attention()`, combine query attention with max attention as the importance score

**Current scoring** (line 141):
```python
window_attentions = text_to_image_attentions[0].max(dim=0)[0].max(dim=0)[0]
```

**New scoring**:
```python
max_attn = text_to_image_attentions[0].max(dim=0)[0].max(dim=0)[0]
if text_indices is not None and len(text_indices) > 0:
    query_attn = text_to_image_attentions[0, :, text_indices].max(dim=0)[0].max(dim=0)[0]
    window_attentions = (1 - self.query_weight) * max_attn + self.query_weight * query_attn
else:
    window_attentions = max_attn
```

**Challenge**: `text_indices` is always `None`. Need to compute it from `input_ids` in `modeling_pllava.py`.

**Time**: 1-2 hr implementation + 1 hr smoke test

---

### Phase C: Combined Ablation Study

**What**: Run full ablation with all combinations on 5 tasks / all samples.

| # | Config | Motion | Query | Description |
|---|--------|--------|-------|-------------|
| 0 | Baseline | No | No | Fixed layer 10, alpha=0.4 (DONE) |
| 1 | Motion-only | Yes | No | motion_scale=1.0 |
| 2 | Query-only | No | Yes | query_weight=0.5 |
| 3 | Motion+Query | Yes | Yes | motion_scale=1.0, query_weight=0.5 |

**Time**: ~4-6 hrs total (1-1.5 hr per config)

---

### Decision Gate (after Phase C)

| Result | Action |
|--------|--------|
| Any config beats baseline by >=1% | Full 20-task MVBench + EgoSchema + VideoMME |
| Any config beats baseline by 0.5-1% | Full MVBench, frame as "marginal improvement" |
| All configs approx baseline (+-0.5%) | Still run full MVBench — larger samples may reveal effect |
| All configs < baseline | Pivot to FLOPs reduction narrative |

---

## Files to Modify

| File | Change |
|------|--------|
| `models/pllava/elastic_cache.py` | Connect motion scores to alpha; add query attention scoring |
| `models/pllava/modeling_pllava.py` | Compute `text_indices` from input_ids |
| `models/pllava/llama.py` | Forward `text_indices` to cache |
| `models/pllava/configuration_pllava.py` | Add `query_weight` config param |
| `tasks/eval/model_utils.py` | Pass `query_weight` to config |
| `tasks/eval/mvbench/pllava_eval_mvbench.py` | Add `--use_query_conditioned`, `--query_weight` CLI args |

---

## Timeline

| Day | Task | Time |
|-----|------|------|
| Day 1 | Phase A: Fix motion-adaptive + smoke test | 2 hr |
| Day 1 | Phase B: Add query-conditioned + smoke test | 3 hr |
| Day 2 | Phase C: Run ablation (4 configs x 5 tasks) | 6 hr |
| Day 2 | Decision gate + next steps | 1 hr |
| **Total** | | **~12 hr** |
