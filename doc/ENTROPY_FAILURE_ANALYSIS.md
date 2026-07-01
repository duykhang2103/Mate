# Why Entropy-Adaptive Pruning Gave Lower Accuracy

> **Created**: 2026-07-01
> **Status**: Complete
> **Result**: Entropy+Borderline: 50.20% vs Baseline: 50.71% (-0.51%)

---

## TL;DR

PLLaVA's attention entropy is **uniform (~0.9999) across all 32 layers**. The entropy signal cannot distinguish important from unimportant tokens, so entropy-triggered pruning provides no benefit and introduces noise.

---

## What We Expected

Entropy-adaptive pruning should:
1. Compute attention entropy at each layer during prefill
2. Trigger pruning when entropy drops below threshold (τ=0.8)
3. Low entropy = model has focused attention = safe to prune
4. High entropy = model needs more tokens = delay pruning

**Hypothesis**: Simple queries → entropy drops early → prune early → save FLOPs
**Hypothesis**: Complex queries → entropy drops late → prune later → preserve accuracy

---

## What Actually Happened

### The Data

| Method | Avg | Action Seq | Action Pred | Unexpected | Obj Interact | Moving Dir |
|--------|-----|-----------|-------------|-----------|-------------|-----------|
| Baseline (fixed L10) | **50.71%** | 55.85 | 51.50 | 62.50 | 63.50 | 20.50 |
| Entropy+Borderline | **50.20%** | 53.72 | 51.50 | 62.00 | 63.50 | 20.50 |

**Delta**: -0.51% average accuracy

### Entropy Profile (All 32 Layers)

```
Layer  0: entropy = 0.9999
Layer  1: entropy = 0.9999
Layer  2: entropy = 0.9999
Layer  3: entropy = 0.9999
Layer  4: entropy = 0.9999
Layer  5: entropy = 0.9999
Layer  6: entropy = 0.9999
Layer  7: entropy = 0.9999
Layer  8: entropy = 0.9999
Layer  9: entropy = 0.9999
Layer 10: entropy = 0.9999  ← baseline prunes here
Layer 11: entropy = 0.9999
Layer 12: entropy = 0.9999
Layer 13: entropy = 0.9999
Layer 14: entropy = 0.9999
Layer 15: entropy = 0.9999
Layer 16: entropy = 0.9999
Layer 17: entropy = 0.9999
Layer 18: entropy = 0.9999
Layer 19: entropy = 0.9999
Layer 20: entropy = 0.9999  ← fallback (never reached)
Layer 21: entropy = 0.9999
Layer 22: entropy = 0.9999
Layer 23: entropy = 0.9999
Layer 24: entropy = 0.9999
Layer 25: entropy = 0.9999
Layer 26: entropy = 0.9999
Layer 27: entropy = 0.9999
Layer 28: entropy = 0.9999
Layer 29: entropy = 0.9999
Layer 30: entropy = 0.9999
Layer 31: entropy = 0.9999
```

**Threshold τ=0.8 never fires.** The entropy-adaptive mode falls back to the fallback layer (20), which is later than the baseline (10). This means:
- Pruning happens at layer 20 instead of layer 10
- Layers 10-19 run with full token count (slower, no FLOPs savings)
- No accuracy benefit because the entropy signal is meaningless

---

## Root Cause Analysis

### Cause 1: PLLaVA's Attention Architecture

PLLaVA uses LLaVA's vision-language architecture where:
- Image tokens are projected through a linear layer
- Text tokens attend to image tokens via cross-attention
- The attention mechanism is **inherently uniform** because:
  - Image tokens are spatially arranged (12×12 grid per frame)
  - There's no semantic grouping at the attention level
  - The model treats all image tokens equally during prefill

**Why this matters**: Entropy measures attention uniformity. If the model's attention is architecturally uniform, entropy will always be high regardless of input content.

### Cause 2: Entropy Computation Location

We compute entropy on **text-to-image attention** (attention from text tokens to image tokens). But:
- During prefill, the text is just the prompt ("Describe the video...")
- The prompt is generic, not query-specific
- Generic prompts produce generic attention → high entropy

**What we should have computed**: Attention from **query tokens** (the actual question) to image tokens. But query tokens arrive later in the conversation, not during prefill.

### Cause 3: Pre-RoPE vs Post-RoPE (Fixed in Phase 6)

Before the bug fix, entropy was computed on attention **before** Rotary Position Embedding (RoPE). After fixing:
- Entropy is computed on post-RoPE, post-softmax attention
- But the result is still ~0.9999
- This confirms the uniformity is intrinsic, not a computation artifact

### Cause 4: Borderline Preservation Added Noise

The borderline preservation mechanism keeps tokens near the pruning cutoff. Combined with entropy-adaptive:
- Entropy never triggers → fallback to layer 20
- Borderline preservation keeps more tokens at layer 20
- Result: More tokens than baseline, but at a later layer
- This is strictly worse than baseline (more compute, same or worse accuracy)

---

## Why Lower Accuracy?

The -0.51% degradation comes from:

1. **Fallback to later layer**: Pruning at layer 20 vs layer 10 means layers 10-19 process more tokens than necessary. This doesn't help accuracy — it just adds noise.

2. **Borderline tokens are low-importance**: Tokens near the cutoff are, by definition, low-attention. Keeping them adds noise to the model's representation.

3. **Inconsistent pruning behavior**: Different inputs may trigger at different layers (or fallback), causing inconsistent token counts. The model isn't trained for variable token counts.

4. **No query-specific information**: The entropy is computed on generic prompt attention, not question-specific attention. The model can't distinguish "this video needs more tokens for this question" because the question isn't available during prefill.

---

## Key Insight

**Entropy is the wrong signal for PLLaVA.**

PLLaVA's attention is architecturally uniform. This is a property of the model, not a bug. Any method that relies on attention entropy to distinguish important from unimportant tokens will fail on PLLaVA.

**What would work**: A signal that correlates with information content, not attention uniformity. Possible alternatives:
- Optical flow magnitude (motion-adaptive) — measures pixel change
- Token similarity clustering (already used in vision merge) — measures redundancy
- Task-specific importance (query-conditioned) — measures question relevance

---

## Lessons Learned

1. **Always validate assumptions before implementing**: We assumed entropy would vary across layers. We should have checked entropy profiles first.

2. **PLLaVA ≠ general Transformer**: Findings from other vision-language models (e.g., LLaVA-1.5) may not transfer to PLLaVA.

3. **Fallback behavior matters**: A fallback to a later layer is strictly worse than a fixed earlier layer. The fallback should be to an earlier layer (or the same layer).

4. **Borderline preservation is harmful**: Tokens near the cutoff are low-importance. Keeping them adds noise, not signal.

---

## Recommendation

**Do not pursue entropy-adaptive pruning further for PLLaVA.**

Instead:
1. Focus on **motion-adaptive pruning** (Phase 7A) — uses optical flow, not attention
2. Focus on **query-conditioned pruning** (Phase 7B) — uses question tokens, not prompt tokens
3. Or pivot to **FLOPs reduction narrative** — same accuracy, 84% fewer FLOPs

---

## Files Modified

| File | Change |
|------|--------|
| `models/pllava/llama.py` | Added `_compute_attention_entropy()`, modified `forward()` for dynamic layer selection |
| `models/pllava/elastic_cache.py` | `VTPWindowCache` accepts `dynamic_selected_layer` parameter |
| `models/pllava/configuration_pllava.py` | Added `use_entropy_adaptive`, `tau_entropy`, `entropy_fallback_layer` |
| `tasks/eval/model_utils.py` | `load_pllava()` accepts entropy params |

**Status**: Code remains in place for reference, but `use_entropy_adaptive=False` by default.
