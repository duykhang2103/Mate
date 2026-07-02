# Feedback on FRIEND_ADVICE_1.md

## Summary

The 14 advice items are **good generic guidance for video token pruning** but ~40% don't apply to PLLaVA specifically. The core issue: PLLaVA's attention is **architecturally uniform** (entropy = 0.9999 across all 32 layers). This makes all attention-based signals useless. The plan needs to be rewritten around this constraint.

---

## The Fundamental Constraint

PLLaVA's attention mechanism produces nearly identical attention weights for all tokens at every layer:

```
Layer 0:  entropy = 0.9999
Layer 1:  entropy = 0.9999
...
Layer 31: entropy = 0.9999
```

This means:
- **Entropy** cannot distinguish important from unimportant tokens
- **Attention variance** is ~0 for all tokens (no motion signal from attention)
- **Cross-attention scores** at any layer are uniform (no selection signal)
- **Sink tokens** don't exist (all tokens receive equal attention)

Every method that relies on attention-based scoring is fundamentally broken for this model.

---

## Advice-by-Advice Feedback

### 1. Adaptive per-frame pruning ratios — PARTIALLY APPLICABLE

**What works**: The motion-based component. Using optical flow (pixel-level motion) to set per-frame pruning ratios is viable. We implemented this and the signal is real — optical flow magnitude varies across frames and tokens.

**What doesn't work**: The entropy-based component. Since PLLaVA attention is uniform, entropy scores are identical for all tokens and cannot guide pruning.

**What we tried**: We implemented Farneback optical flow for static/dynamic token classification. The flow signal is valid, but the LLM-level pruning we added alongside it caused a -5.47% regression. The flow merge alone (no LLM pruning) appears roughly neutral based on a 250-sample ablation (49.6% vs 50.71% baseline).

**Reference**: FastVID's dynamic density pruning is promising but requires training, which we're avoiding.

---

### 2. Information-aware / entropy-aware merging — PARTIALLY APPLICABLE

**What works**: Variance-based weighting on token features. Computing feature variance across frames and using it to weight merging is viable because it operates on raw features, not attention.

**What doesn't work**: "Entropy-aware" merging. Same uniform attention problem.

**Status**: Not implemented yet. This is a viable next step.

---

### 3. Importance re-scoring (MLP) — NOT APPLICABLE

**Problem**: Requires training an MLP to produce importance scores. Our constraint is inference-time only (no training/fine-tuning).

**If training were allowed**: This would be a strong approach. SharpV's results show re-scoring can improve accuracy at low token budgets.

---

### 4. Sink-token suppression — NOT APPLICABLE

**Problem**: Sink tokens are defined as "high attention but low semantic content." In PLLaVA, ALL tokens have equal attention (entropy = 0.9999). There are no distinct sink tokens to suppress.

**Evidence**: We tested borderline token preservation (keeping tokens near the pruning threshold). It degraded accuracy by -0.26%, confirming that low-attention tokens in PLLaVA are noise, not signal.

---

### 5. Temporal attention / weighted temporal fusion — NOT APPLICABLE

**Problem**: Requires training a temporal attention module. Violates the no-training constraint.

**If training were allowed**: This would help. Attention-weighted fusion would preserve critical temporal information better than simple averaging.

---

### 6. Motion-guided temporal pruning — APPLICABLE (BEST ADVICE)

**Why it works**: Uses pixel-level motion (optical flow or frame difference) which is independent of attention. This is the only motion signal that actually works for PLLaVA.

**Status**: Partially implemented. We have optical flow computation in the vision merge path. Need to complete the ablation and tune parameters.

**This should be the primary method in any revised plan.**

---

### 7. Multi-scale / region-aware spatial merging — NOT APPLICABLE

**Problem**: Requires object detection or segmentation to identify regions. This adds an extra model, increases complexity, and may require training.

**The multi-scale concept is sound** but implementation needs to be simple and model-free.

---

### 8. Multi-layer cross-attention — NOT APPLICABLE

**Problem**: Uses cross-attention from multiple layers (M-1, M, M+1). But PLLaVA has uniform attention across ALL 32 layers. Multi-layer = same as single-layer = useless.

**We verified this**: Entropy is 0.9999 at every layer. Adding more layers doesn't help.

---

### 9. Weighted KV merging — ALREADY IMPLEMENTED

**Status**: We have `TextPivotMerge_LayerWise` in `llama.py` which uses similarity-weighted merge instead of simple drop. It's the default behavior.

**Result**: Marginal improvement (+0.22% when tested). Not a game-changer but doesn't hurt.

---

### 10. Uptraining / robustness training — NOT APPLICABLE

**Problem**: Requires training. Our constraint is inference-time only.

**If training were allowed**: This would be the strongest approach. EVS and SharpV both show training with random pruning rates dramatically improves robustness.

---

### 11. Multi-objective optimization — APPLICABLE

**What it means**: Tune pruning thresholds to balance accuracy, FLOPs, memory, and latency simultaneously.

**Status**: Not implemented. We've been optimizing for accuracy only. Adding FLOPs/memory metrics would strengthen the paper.

**This is a good addition to the evaluation strategy.**

---

### 12. Task-specific or benchmark-specific tuning — APPLICABLE

**What it means**: Optimize pruning parameters (alpha, tau, cluster_ratio) per benchmark instead of using one fixed setting.

**Status**: Not implemented. We've been using the same parameters everywhere.

**This is practical and likely to help.**

---

### 13. Hybrid methods — DEPENDS ON INDIVIDUAL COMPONENTS

**Problem**: Only useful if the individual methods work within our constraints. Of the 14 advice items, only 4-5 are applicable. A hybrid of 4-5 methods is still a valid contribution.

---

### 14. Evaluation strategy — CORRECT AND IMPORTANT

**What's right**: 
- Accuracy vs token budget curves
- Robustness analysis across pruning ratios
- TTFT, memory, and efficiency metrics
- Fine-grained / long-video subsets

**Status**: We haven't done this yet. Our evaluation has been limited to 5-task MVBench. This needs to be a priority.

---

## Revised Plan: What Actually Works for PLLaVA

### Methods (4 viable out of 14)

1. **Motion-adaptive pruning ratios** (optical flow) — already implemented, needs ablation
2. **Variance-weighted merging** (token feature variance) — not implemented, viable
3. **Task-specific tuning** (per-benchmark optimization) — not implemented, practical
4. **Multi-objective evaluation** (accuracy + FLOPs + memory) — not implemented, strengthens paper

### What We Cannot Do

- Anything requiring training (MLP re-scoring, temporal attention, uptraining)
- Anything attention-based (entropy, sink tokens, multi-layer cross-attention)
- Anything requiring extra models (object detection, segmentation)

### Evaluation Strategy (from advice #14, excellent)

1. Run full 20-task MVBench (not just 5-task dev eval)
2. Run VideoMME + EgoSchema
3. Plot accuracy vs token budget curves (10%, 15%, 20%, 25%)
4. Measure TTFT, prefill time, memory
5. Compare against PruneVid's published numbers

### Target Claims (from the plan's Scenario 1-4)

Best realistic claim: **"PruneVid+ improves temporal task accuracy at same token budget through optical flow-guided adaptive pruning."**

This is honest, paper-backed, and achievable within our constraints.

---

## Bottom Line

The advice document is well-researched and cites good papers. But it's written for a generic video token pruning scenario, not specifically for PLLaVA. The 40% that doesn't apply is mostly due to PLLaVA's uniform attention architecture. The 60% that does apply — motion-adaptive ratios, variance-weighted merging, task-specific tuning, and the evaluation strategy — form a solid foundation for a revised plan.
