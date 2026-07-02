# Current Situation

> **Updated**: 2026-07-02
> **Status**: All methods underperform baseline. Root cause identified.

---

## Benchmark Results (5 tasks, ~1000 samples)

| Method | Avg | Action Seq | Action Pred | Unexpected | Obj Interact | Moving Dir |
|--------|-----|-----------|-------------|-----------|-------------|-----------|
| Baseline (fixed L10, alpha=0.4) | **50.71** | 55.85 | 51.50 | 62.50 | 63.50 | 20.50 |
| Entropy + Borderline | 50.20 | 53.72 | 51.50 | 62.00 | 63.50 | 20.50 |

**Moving Direction is at 20.5% — essentially random (1/5 = 20%).**

---

## What Failed and Why

### 1. Entropy-Adaptive Pruning — DEAD END
- All 32 layers have entropy = 0.9999 (uniform attention)
- Threshold τ=0.8 never fires
- Falls back to layer 20 (worse than fixed layer 10)
- **Verdict**: Abandon. PLLaVA attention is architecturally uniform.

### 2. Motion-Adaptive Pruning — FUNDAMENTALLY BROKEN
- Motion scores computed from attention temporal variance
- Because attention is uniform, variance ≈ 0 for all windows
- Motion scores: [0.000000, 0.000000, ...]
- The `--use_motion_adaptive` flag does literally nothing
- **Verdict**: Need optical flow instead of attention-based motion.

### 3. Borderline Preservation — HARMFUL
- Keeps tokens near pruning cutoff (low-importance tokens)
- Added noise → Action Sequence dropped 2.13%
- **Verdict**: Disable entirely.

### 4. Query-Conditioned Pruning — NOT IMPLEMENTED
- `text_indices` is hardcoded to `None` and ignored
- **Verdict**: Implement after fixing motion signal.

### 5. Temporal Pivot Anchoring — NOT IMPLEMENTED
- Merge selects pivots by cosine similarity only
- No temporal awareness → motion trajectories get smeared
- **Verdict**: Implement after fixing motion signal.

---

## Root Cause: The Motion Signal Doesn't Exist

Every modification so far has relied on **attention-based signals** (entropy, attention variance, text-to-image attention). But PLLaVA's attention is **architecturally uniform** — all tokens receive equal attention regardless of content.

This means:
- Entropy = 0.9999 (always high, never triggers)
- Attention variance ≈ 0 (always uniform)
- Motion scores ≈ 0 (always the same)

**The only way to get a real motion signal is to use optical flow** — actual pixel-level motion between frames.

---

## The Fix: Optical Flow (Phase 8)

See `doc/phases/phase8_plan.md` for the full plan.

**Summary of 4 changes**:
1. **Optical-flow static/dynamic split** — Replace feature similarity with Farneback flow magnitude
2. **Fix two-tier retention** — Use flow-based motion scores instead of attention variance
3. **Kill borderline preservation** — Remove noise injection
4. **Temporal pivot anchoring** — Bias merge toward temporally adjacent tokens

---

## Available Infrastructure

| Component | Status | Location |
|-----------|--------|----------|
| Farneback flow | Working | `scripts/flow_correlation_sanity_check.py` |
| RAFT flow model | Exists (unused) | `models/pllava/modeling_pllava_flow.py` |
| Static/dynamic split | Working (wrong signal) | `models/pllava/modeling_pllava.py:777-840` |
| Two-tier retention | Working (wrong signal) | `models/pllava/elastic_cache.py:126-195` |
| Token counting | Working | `models/pllava/elastic_cache.py:272-273` |
| FLOPs tracking | Working | `tasks/eval/model_utils.py` |
| MVBench eval | Working | `tasks/eval/mvbench/pllava_eval_mvbench.py` |
| VideoMME eval | Working | `tasks/eval/videomme/pllava_eval_videomme.py` |
| VideoMME videos | 183/760 available | `DATAS/Video-MME/data/` |
| VideoMME JSON | Unknown | `DATAS/Video-MME/json/` |

---

## Next Step

Implement optical flow in `merge_frames_dynamic()` per `doc/phases/phase8_plan.md`.
