# Plan: Break Through the Dead End

> **Date**: 2026-07-06
> **Status**: Active
> **Goal**: Find a path to beat PruneVid baseline on at least one benchmark

---

## Situation Assessment

### What We Know

| Fact | Implication |
|------|-------------|
| PLLaVA attention is uniform (entropy=0.9999) | All attention-based token selection fails |
| Every method on PLLaVA degraded accuracy (-0.26% to -5.57%) | PLLaVA is architecturally incompatible with pruning |
| LLaVA-OV baseline is 52.0% on 20 samples/task | Need full eval to know real standing |
| PruneVid paper: 58.0 MVBench, 58.2 VideoMME on LLaVA-OV | 6% gap to close (or baseline is wrong) |
| VTP on LLaVA-OV is NOT implemented yet | We're declaring defeat before fighting |
| Cluster pruning was abandoned after one PLLaVA test | Only attention-agnostic method, untested on LLaVA-OV |

### The Three Paths

| Path | Description | Risk | Reward |
|------|-------------|------|--------|
| **A: VTP on LLaVA-OV** | Implement the obvious next step | Medium | High (matches PruneVid) |
| **B: Cluster pruning on LLaVA-OV** | Content-based, no attention needed | Low | Medium (model-agnostic contribution) |
| **C: Negative result paper** | Publish why pruning fails on uniform models | Low | Medium (honest analysis) |

---

## Path A: VTP on LLaVA-OV (Primary)

### Why This First

The entire PruneVid paper's contribution is that token selection works on models with non-uniform attention. LLaVA-OV has non-uniform attention. We haven't tested this. This is the obvious move.

### Step 1: Full Baseline Evaluation

Run all 20 MVBench tasks on LLaVA-OV baseline (no pruning).

```bash
conda run -n pllava python -m tasks.eval.mvbench.ov_eval_mvbench \
    --pretrained_model_name_or_path MODELS/llava-onevision-7b \
    --save_path test_results/mvbench_ov_baseline_full_20tasks \
    --num_frames 16 \
    --tasks "Action Sequence,Action Prediction,Action Antonym,Fine-grained Action,Unexpected Action,Object Existence,Object Interaction,Object Shuffle,Moving Direction,Action Localization,Scene Transition,Action Count,Moving Count,Moving Attribute,State Change,Fine-grained Pose,Character Order,Egocentric Navigation,Episodic Reasoning,Counterfactual Inference" \
    > log_mvb_ov_baseline_full_20tasks.log 2>&1
```

**20 MVBench tasks** (from `tasks/eval/mvbench/__init__.py`):
Action Sequence, Action Prediction, Action Antonym, Fine-grained Action, Unexpected Action, Object Existence, Object Interaction, Object Shuffle, Moving Direction, Action Localization, Scene Transition, Action Count, Moving Count, Moving Attribute, State Change, Fine-grained Pose, Character Order, Egocentric Navigation, Episodic Reasoning, Counterfactual Inference.

**Decision gate**:
- If result is ~58%: baseline is correct, proceed with VTP
- If result is ~52%: something is off with eval setup (preprocessing, samples, etc.) — debug first
- If result is <50%: fundamental issue, stop and diagnose

### Step 2: Implement Qwen2 VTP Wrapper

**File**: `models/llava_ov/qwen2_vtp.py` (new)

What it does:
- Wraps Qwen2Model to capture attention weights at `selected_layer`
- Hooks into the forward pass at a specific layer
- Stores attention weights for the pruning step

Key difference from PLLaVA's `llama.py`:
- Qwen2 uses `Qwen2SdpaAttention` or `Qwen2FlashAttention2`
- Need to use `attn_implementation="eager"` to get attention weights
- Image token ID is fixed at 151646 (not dynamic like PLLaVA)

### Step 3: Implement Token Merge + Pruning

**File**: `models/llava_ov/elastic_cache_ov.py` (new)

What it does:
- Merges visual tokens across frames (feature-similarity, same as PLLaVA baseline)
- Prunes tokens at selected layer using attention scores
- Supports cluster pruning as alternative to attention-based pruning

Key difference from PLLaVA's `elastic_cache.py`:
- Token layout: `[batch, num_frames * spatial_pool, hidden_size]` (LLaVA-OV)
- Image tokens are identified by `input_ids == 151646` (not `pad_token_id`)
- Need to handle variable token counts from dynamic resolution

### Step 4: Integrate into Forward Pass

**File**: `models/llava_ov/modeling_llava_ov.py` (modify)

Changes:
- After vision encoder processes frames, merge tokens
- After embedding layer, hook into Qwen2 at selected layer
- At selected layer: collect attention, call pruning, continue with reduced tokens
- Track token counts and FLOPs

### Step 5: Baseline vs VTP Comparison

Run both on same 5-task subset first, then full 20-task:

```bash
# Baseline (already done)
# VTP with alpha=0.4
conda run -n pllava python -m tasks.eval.mvbench.ov_eval_mvbench \
    --pretrained_model_name_or_path MODELS/llava-onevision-7b \
    --save_path test_results/mvbench_ov_vtp_alpha04 \
    --num_frames 16 \
    --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --tasks "Action Sequence,Action Prediction,Moving Direction,Object Interaction,Unexpected Action" \
    --max_samples 200 \
    > log_mvb_ov_vtp_alpha04.log 2>&1
```

---

## Path B: Cluster Pruning on LLaVA-OV (Fallback)

### Why This Matters

Cluster pruning is the **only method that doesn't depend on attention being non-uniform**. It clusters on LLM hidden states (content features), which SHOULD be discriminative on LLaVA-OV even if attention isn't.

### What's Already Implemented

- `cluster_dpc_knn()` function in `elastic_cache.py`
- `_cluster_prune()` method in `VTPWindowCache`
- CLI flags: `--use_cluster_pruning`, `--cluster_pruning_topk`

### What Needs to Change for LLaVA-OV

1. Port `_cluster_prune()` to `elastic_cache_ov.py`
2. Adapt token layout handling (LLaVA-OV has different grid structure)
3. Hook into LLaVA-OV forward pass

### Evaluation

Run same comparison: baseline vs cluster pruning on 5-task subset.

```bash
# Cluster pruning with topk=0.4
conda run -n pllava python -m tasks.eval.mvbench.ov_eval_mvbench \
    --pretrained_model_name_or_path MODELS/llava-onevision-7b \
    --save_path test_results/mvbench_ov_cluster_topk04 \
    --num_frames 16 \
    --use_cluster_pruning --cluster_pruning_topk 0.4 \
    --tasks "Action Sequence,Action Prediction,Moving Direction,Object Interaction,Unexpected Action" \
    --max_samples 200 \
    > log_mvb_ov_cluster_topk04.log 2>&1
```

---

## Path C: Negative Result Paper (Last Resort)

### When to Choose This

If:
- LLaVA-OV baseline is ~52% (not ~58%)
- VTP on LLaVA-OV doesn't improve over baseline
- Cluster pruning on LLaVA-OV doesn't help either

### Paper Structure

**Title**: "When Visual Token Pruning Fails: Architectural Determinants of Token Selection Effectiveness"

**Claim**: "Token selection effectiveness is determined by attention uniformity, not the pruning algorithm. Models with uniform attention (PLLaVA) cannot benefit from any token selection method, while models with non-uniform attention (LLaVA-OneVision) show consistent improvements."

**Key findings to report**:
1. PLLaVA: all methods fail (entropy=0.9999 everywhere)
2. LLaVA-OV: methods work (non-uniform attention)
3. Ablation: attention entropy correlates with pruning effectiveness
4. Recommendation: use non-uniform attention models for token-efficient video understanding

**Target venue**: Workshop or short paper at main conference

---

## Sanity Check: Does Flow Actually Separate Motion Tasks?

Before committing to any path, run the 15-minute check your friend suggested.

**File**: `scripts/flow_sanity_check.py` (new)

```python
"""
Check if optical flow magnitude actually separates motion-heavy from static MVBench tasks.
Run on 20-30 videos across different task types.
"""
import json
import cv2
import numpy as np
from decord import VideoReader, cpu

def compute_flow_magnitude(video_path, num_frames=16):
    """Compute average flow magnitude between consecutive frames."""
    vr = VideoReader(video_path, ctx=cpu(0))
    indices = np.linspace(0, len(vr)-1, num_frames, dtype=int)
    frames = [vr[i].asnumpy() for i in indices]

    magnitudes = []
    for i in range(len(frames)-1):
        gray1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frames[i+1], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        magnitudes.append(mag.mean())
    return np.mean(magnitudes)

# Load MVBench, sample 3 videos per task type, compute flow
# Print: task_type -> avg_flow_magnitude
# If motion-heavy tasks have significantly higher flow → direction is viable
```

**Expected outcome**:
- If motion-heavy tasks (Moving Direction, Action Sequence) have 2x+ higher flow than static tasks → flow-based methods should work
- If flow is similar across all tasks → the entire motion-adaptive direction is dead

---

## Known Blockers

### Transformers Version

**Current**: 4.39.3 (does NOT have `LlavaOnevisionForConditionalGeneration`)
**Required**: 4.46.3 (or at least 4.45.0+)

**Fix**: `pip install transformers==4.46.3`

**Risk**: Upgrading may break PLLaVA if it depends on 4.39.3 internals. Test PLLaVA eval after upgrade.

---

## Execution Order

| Step | Action | Time | Dependencies |
|------|--------|------|-------------|
| 0 | Upgrade transformers to 4.46.3 | ~5 min | None |
| 1 | Run full 20-task MVBench baseline on LLaVA-OV | ~3-6 hrs | Step 0 |
| 2 | Run flow sanity check on 20-30 MVBench videos | ~15 min | None |
| 3 | Implement Qwen2 VTP wrapper | ~2-3 hrs | Step 1 baseline |
| 4 | Implement elastic_cache_ov.py | ~2-3 hrs | Step 3 |
| 5 | Integrate into forward pass | ~2-3 hrs | Step 4 |
| 6 | Run VTP comparison (5 tasks) | ~1-2 hrs | Step 5 |
| 7 | Run VTP comparison (20 tasks) | ~3-6 hrs | Step 6 |
| 8 | If VTP fails: implement cluster pruning on LLaVA-OV | ~2-3 hrs | Step 7 |
| 9 | If everything fails: write negative result paper | ~1-2 weeks | Step 8 |

**Total estimated time**: 15-25 hours for Paths A+B, plus 1-2 weeks for Path C if needed.

---

## Decision Tree

```
Start
  |
  v
[Step 1] Full 20-task baseline on LLaVA-OV
  |
  +--> Result ~58%?
  |      |
  |      +--> Yes: Baseline is correct, proceed to Step 3 (VTP)
  |      |
  |      +--> No (~52%): Debug eval setup
  |             |
  |             +--> Check: wrong samples? wrong preprocessing?
  |             +--> Check: processor config? num_frames?
  |             +--> Fix and re-run
  |
  v
[Step 2] Flow sanity check
  |
  +--> Flow separates motion tasks?
  |      |
  |      +--> Yes: Motion-adaptive direction viable
  |      |
  |      +--> No: Skip motion-adaptive, focus on content-clustering
  |
  v
[Step 3-7] Implement + evaluate VTP on LLaVA-OV
  |
  +--> VTP > baseline?
  |      |
  |      +--> Yes: "Non-Uniform Attention Exploitation" paper
  |      |
  |      +--> No: Try cluster pruning (Step 8)
  |
  v
[Step 8] Cluster pruning on LLaVA-OV
  |
  +--> Cluster > baseline?
  |      |
  |      +--> Yes: "Model-Agnostic Content Clustering" paper
  |      |
  |      +--> No: Negative result paper (Step 9)
  |
  v
[Step 9] Write paper on why pruning fails on uniform models
```

---

## What NOT to Do

| Don't | Why |
|-------|-----|
| Try more methods on PLLaVA | Dead end. Stop. |
| Use attention-based signals on PLLaVA | Entropy=0.9999 everywhere. Useless. |
| Implement flow-based vision merge | Proven harmful (-4.66%). Reverted. |
| Skip full baseline evaluation | You need reliable numbers before comparing. |
| Declare defeat before VTP on LLaVA-OV | It hasn't been tried yet. |

---

## Key Insight

**The dead end is PLLaVA, not the research direction.**

LLaVA-OV with non-uniform attention is the correct target. The PruneVid paper already shows results on it. Your job is to:
1. Verify your baseline matches theirs (~58%)
2. Implement VTP on LLaVA-OV
3. Show improvement or publish honest analysis

The fact that PLLaVA fails is actually a **finding**, not a failure. It tells you something real about model architecture and token selection.
