# Method Survey (Updated 3-7-2026)

## All Methods Tested

| # | Method | Accuracy | Delta | Status | Root Cause |
|---|--------|----------|-------|--------|------------|
| 0 | Baseline (fixed L10, alpha=0.4) | **50.71%** | — | WORKING | — |
| 1 | Entropy-adaptive pruning | 50.20% | -0.51 | DEAD END | PLLaVA attention uniform (entropy=0.9999) |
| 2 | Motion-adaptive (attention variance) | 49.87% | -0.84 | DEAD END | Attention variance ≈ 0 for all tokens |
| 3 | Borderline preservation | 50.67% | -0.26 | HARMFUL | Low-attention tokens are noise, not signal |
| 4 | Weighted token merge | 50.93% | +0.22 | NEUTRAL | Marginal, doesn't hurt |
| 5 | Flow vision merge only | 46.05% | -4.66 | HARMFUL | Destroys content-adaptive token allocation |
| 6 | Flow vision merge + LLM prune | 45.24% | -5.47 | HARMFUL | Compounds merge harm + pruning harm |

## Root Cause: Why Everything Fails

PLLaVA's attention is **architecturally uniform** (entropy=0.9999 across all 32 layers). This means:
- Attention-based signals (entropy, variance, cross-attention scores) are useless
- Only pixel-level signals (optical flow, frame difference) work
- The feature-similarity merge in baseline is actually doing something useful — it allocates tokens content-adaptively (CV=27-42%)

## What's Left to Try

### Viable (no training, no attention)

1. **Motion-adaptive LLM pruning** — use flow scores for per-window alpha at layer 10
2. **Variance-weighted merging** — feature variance weighting instead of simple average
3. **Task-specific tuning** — per-benchmark parameter optimization
4. **Multi-objective evaluation** — accuracy + FLOPs + memory + latency

### Not Viable

| Method | Why Not |
|--------|---------|
| Anything attention-based | PLLaVA attention is uniform |
| Anything requiring training | Inference-time only constraint |
| Anything requiring extra models | No object detector/segmentation available |
| Flow vision merge | Proven harmful (-4.66%) |

## Competitive Position

PruneVid (16.2% retained, 0.23x FLOPs) is the only method that improves accuracy while reducing FLOPs. Target weaknesses:
- EgoSchema Fullset: 42.6 (0.0 improvement)
- VCGBench: 2.98 (-0.01 degradation)
- Moving Direction: ~20% (random)
- Fixed layer 10, no adaptivity

## Revised Plan

See `REVISED_PLAN.md` for the updated experiment plan using only viable methods.
