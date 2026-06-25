# Phase 2 Execution Plan — Multi-Head Entropy Fix + Dev Eval

> **Created**: 2026-06-25
> **Status**: IN PROGRESS (Steps 1-3 done, Steps 4-6 blocked on MVBench dataset)
> **Goal**: Fix broken entropy computation (single-head collapse) and validate entropy-adaptive pruning on MVBench dev subset

---

## Progress

| Step | Status | Notes |
|------|--------|-------|
| 1. Fix entropy bug | DONE | Per-head entropy in `llama.py:1337-1400` |
| 2. Add CLI args to eval | DONE | `--use_entropy_adaptive`, `--tau_entropy`, `--entropy_fallback_layer` |
| 3. Smoke test | DONE | 2304→627→245 tokens, 89.4% reduction, no crashes |
| 4. Fixed-layer baseline eval | BLOCKED | Needs `DATAS/MVBench/json/` annotation files |
| 5. Entropy-adaptive eval | BLOCKED | Same blocker |
| 6. Compare results | BLOCKED | Depends on 4-5 |

**Blocker**: `DATAS/MVBench/json/` directory with 20 annotation JSONs is missing. Video zips exist but annotations need to be downloaded from HuggingFace.

---

## Dev Eval Spec

- **Dataset**: MVBench only
- **Tasks** (5): Action Sequence, Action Prediction, Moving Direction, Object Interaction, Unexpected Action
- **Samples per task**: 75 (375 total)
- **Purpose**: Fast iteration for Phase 2 entropy fix validation

---

## Step 1: Fix Entropy Bug ✅

**File**: `models/pllava/llama.py:1337-1400`

**Bug**: `_compute_attention_entropy()` collapsed all attention heads via `max(dim=0).max(dim=0)`, losing inter-head diversity.

**Fix**: Compute entropy per head, then aggregate via mean. Preserves signal that some heads are uncertain while others are confident.

---

## Step 2: Add Entropy CLI Args to MVBench Eval ✅

**File**: `tasks/eval/mvbench/pllava_eval_mvbench.py`

Added to `parse_args()`:
- `--use_entropy_adaptive` (store_true)
- `--tau_entropy` (float, default=0.8)
- `--entropy_fallback_layer` (int, default=20)

Passed through `load_model_and_dataset()` → `load_pllava()` → `PllavaConfig`.

---

## Step 3: Single-Video Smoke Test ✅

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

**Result**: Passed. 2304 raw → 627 after vision merge → 245 after LLM pruning (89.4% reduction). Entropy-adaptive mode enabled without errors.

---

## Step 4: MVBench Fixed-Layer Baseline ⏳ BLOCKED

```bash
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_dev_fixed \
    --num_frames 16 \
    --use_lora --lora_alpha 14 \
    --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 \
    --selected_layer 10 \
    --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 \
    --cluster_ratio 0.5 \
    --tasks "Action Sequence,Action Prediction,Moving Direction,Object Interaction,Unexpected Action" \
    --max_samples 75
```

---

## Step 5: MVBench Entropy-Adaptive ⏳ BLOCKED

```bash
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_entropy_pruned_5_tasks_75_samples \
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
    --tasks "Action Sequence,Action Prediction,Moving Direction,Object Interaction,Unexpected Action" \
    --max_samples 75 > log_mvbench_entropy_pruned_5_tasks_75_samples.log 2>&1
```

---

## Step 6: Compare Results

- `test_results/mvbench_dev_fixed/upload_leaderboard.json`
- `test_results/mvbench_dev_entropy/upload_leaderboard.json`

| Result | Action |
|--------|--------|
| Entropy >= fixed | Proceed to Phase 3 |
| Entropy < fixed by >2% | Debug entropy computation |
| Entropy crashes | Fix bugs, re-run |

---

## Estimated Time

| Step | Time |
|------|------|
| Code fix (Step 1-2) | ~10 min |
| Smoke test (Step 3) | ~5-10 min |
| Fixed-layer baseline (Step 4) | ~25-35 min |
| Entropy-adaptive eval (Step 5) | ~25-35 min |
| Comparison (Step 6) | ~5 min |
| **Total** | **~70-95 min** |
