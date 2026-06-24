# Phase 2: Multi-Head Entropy Fix

> **Goal**: Fix the broken entropy computation (single-head collapse) and validate entropy-adaptive pruning
> **Duration**: Weeks 3–4
> **Status**: IN PROGRESS

---

## Background

Entropy-adaptive infrastructure is **already implemented** (Phase 2 is about fixing, not building):

- [x] Config params: `use_entropy_adaptive`, `tau_entropy`, `entropy_fallback_layer`
- [x] `VTPWindowCache` accepts `dynamic_selected_layer`
- [x] `LlamaModelVTP.forward()` computes entropy per layer, triggers pruning at first layer where H < τ
- [x] `scripts/infer_single_video.py` supports `--use_entropy_adaptive` flag
- [x] Entropy profile stored on model for post-inference visualization

**The problem**: The entropy computation collapses all attention heads into a single scalar, losing inter-head diversity.

---

## The Bug: Single-Head Collapse

**File**: `models/pllava/llama.py` → `_compute_attention_entropy()`

Current code:
```python
# text_to_image_attn shape: [batch, heads, num_text, num_img]
attn_scores = text_to_image_attn[0].max(dim=0)[0].max(dim=0)[0]  # [num_img_tokens]
#            ^^^^^^^^
#            This collapses ALL heads into one via max
attn_probs = F.softmax(attn_scores, dim=-1)
entropy = -torch.sum(attn_probs * torch.log(attn_probs))
```

**Why this is wrong**: Different attention heads attend to different visual regions. Head 1 might focus on motion, Head 2 on objects, Head 3 on background. Collapsing via max discards this diversity — the "uncertain" signal from Head 3 is overwritten by the "confident" signal from Head 1.

---

## The Fix: Per-Head Entropy

```python
def _compute_attention_entropy(self, attentions, input_ids_new):
    """Compute per-head Shannon entropy of text-to-image attention."""
    pad_token = self.pad_token_id
    batch_img_indices, seq_img_indices = torch.where(input_ids_new == pad_token)

    if len(seq_img_indices) == 0:
        return float('inf')

    img_start, img_end = seq_img_indices[0].item(), seq_img_indices[-1].item()
    text_to_image_attn = attentions[:, :, img_end+1:, img_start: img_end+1]

    if text_to_image_attn.shape[2] == 0 or text_to_image_attn.shape[3] == 0:
        return float('inf')

    num_img_tokens = text_to_image_attn.shape[3]
    if num_img_tokens <= 1:
        return 0.0

    # Per-head entropy
    head_entropies = []
    for h in range(text_to_image_attn.shape[1]):
        # Per-head: max across text tokens → [num_img_tokens]
        head_attn = text_to_image_attn[0, h].max(dim=0)[0]
        head_probs = F.softmax(head_attn, dim=-1)
        head_entropy = -torch.sum(head_probs * torch.log(head_probs + 1e-10))
        head_entropies.append(head_entropy)

    # Aggregate: mean across heads (robust) or max (conservative)
    entropy = torch.stack(head_entropies).mean()
    normalized_entropy = entropy / torch.log(torch.tensor(num_img_tokens, dtype=torch.float, device=entropy.device))

    return normalized_entropy.item()
```

**Key difference**: Entropy is computed per-head first, then aggregated. This preserves the signal that some heads are uncertain while others are confident.

---

## Validation Steps

### Step 1: Quick Smoke Test (~5 min)

Run inference on 1 video to verify no crashes:

```bash
python scripts/infer_single_video.py \
    --video example/cooking.mp4 \
    --model_dir MODELS/pllava-7b \
    --weight_dir MODELS/pllava-7b \
    --use_lora --lora_alpha 14 \
    --use_entropy_adaptive \
    --tau_entropy 0.8 \
    --skip_generation
```

**Expected**: Entropy profile printed, no errors, trigger layer shown.

### Step 2: Entropy Profile Sanity Check (~10 min)

Run on 3 different videos to check entropy trends:

```bash
for video in example/cooking.mp4 example/yoga.mp4 example/1917.mp4; do
    python scripts/infer_single_video.py \
        --video "$video" \
        --model_dir MODELS/pllava-7b \
        --weight_dir MODELS/pllava-7b \
        --use_lora --lora_alpha 14 \
        --use_entropy_adaptive \
        --tau_entropy 0.8 \
        --alpha 0.4
done
```

**Check**:
- Entropy decreases across layers (monotonic or near-monotonic)
- Simple queries trigger earlier than complex queries
- Trigger layer varies per video (not always the same)

### Step 3: Compare Fixed vs Entropy-Adaptive (~30 min)

Run both modes on same videos, compare answers and timing:

```bash
# Fixed layer 10
python scripts/infer_single_video.py \
    --video example/cooking.mp4 \
    --model_dir MODELS/pllava-7b \
    --weight_dir MODELS/pllava-7b \
    --use_lora --lora_alpha 14 \
    --selected_layer 10 --alpha 0.4

# Entropy-adaptive
python scripts/infer_single_video.py \
    --video example/cooking.mp4 \
    --model_dir MODELS/pllava-7b \
    --weight_dir MODELS/pllava-7b \
    --use_lora --lora_alpha 14 \
    --use_entropy_adaptive \
    --tau_entropy 0.8 --alpha 0.4
```

**Check**:
- Entropy-adaptive triggers at a different layer than 10
- Answers are reasonable (not garbage)
- Wall-clock time is comparable or better

### Step 4: MVBench Quick Eval (~1 hour)

Run entropy-adaptive on 1 MVBench task:

```bash
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_entropy \
    --num_frames 16 \
    --use_lora --lora_alpha 14 \
    --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 \
    --selected_layer 10 \
    --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 \
    --cluster_ratio 0.5 \
    --tasks "Action Sequence" \
    --max_samples 50
```

**Note**: This requires adding `--use_entropy_adaptive` and `--tau_entropy` args to the eval script. Do this as part of Phase 1 eval setup.

**Check**:
- Accuracy is comparable to fixed-layer baseline (±1%)
- No crashes or errors

---

## Decision Gate

After Step 4, decide:

| Result | Action |
|--------|--------|
| Entropy-adaptive ≥ fixed-layer | Proceed to Phase 3 (motion-adaptive) |
| Entropy-adaptive < fixed-layer by >2% | Debug entropy computation, check τ threshold |
| Entropy-adaptive crashes | Fix bugs, re-run |

---

## Files to Modify

| File | Change |
|------|--------|
| `models/pllava/llama.py` | Fix `_compute_attention_entropy()` to use per-head computation |
| `doc/ENTROPY_PROGRESS.md` | Update with validation results |

---

## Expected Outcome

- Entropy-adaptive pruning works correctly with per-head entropy
- Trigger layer varies per query (simple → early, complex → late)
- Accuracy matches or exceeds fixed-layer baseline
- Ready for Phase 3 (motion-aware adaptive retention)
