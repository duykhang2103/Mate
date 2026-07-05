# Plan: Fix Regression + VideoMME Evaluation

> **Date**: 2026-07-05
> **Status**: Ready to execute
> **Goal**: Improve accuracy on motion-heavy and temporally complex videos

---

## Situation

| Experiment                               | MVBench 5-task (1000 samples) | Delta      |
| ---------------------------------------- | ----------------------------- | ---------- |
| Baseline (fixed L10, alpha=0.4)          | **50.71%**                    | —          |
| Modified (flow vision merge + LLM prune) | **45.24%**                    | **-5.47%** |

**Root cause**: Flow-based vision merge is active in `merge_frames_dynamic()` (lines 856-874 of `modeling_pllava.py`). Your own `FLOW_REGRESSION_UPDATED.md` proved **85% of the regression (-4.66%) comes from the vision merge**, not LLM pruning. Step 1 of `REVISED_PLAN.md` was never executed.

**The code is in an inconsistent state**: both flow vision merge and flow LLM pruning coexist, but flow vision merge is harmful and must be removed first.

---

## Plan Overview

| Step | What                                           | Why                                    | Time    |
| ---- | ---------------------------------------------- | -------------------------------------- | ------- |
| 1    | Revert flow vision merge                       | Fix the -5.47% regression              | 5 min   |
| 2    | Verify baseline on MVBench 5-task              | Confirm accuracy returns to ~50.71%    | ~1 hr   |
| 3    | Baseline on VideoMME long (100 samples)        | Establish baseline on target benchmark | ~15 min |
| 4    | Implement motion-adaptive LLM pruning only     | The actual method contribution         | 15 min  |
| 5    | Evaluate method on VideoMME long (100 samples) | Compare against baseline               | ~15 min |

---

## Step 1: Revert Flow Vision Merge

**What to change**: In `models/pllava/modeling_pllava.py`, `merge_frames_dynamic()` method (line 834), remove the flow-based static/dynamic classification branch. Always use feature-similarity fallback.

**File**: `models/pllava/modeling_pllava.py`

**Lines to modify**: 856-881

Current code flow:

```python
if flow_mag is not None and window_size > 1:
    # Flow-based static/dynamic classification  <-- REMOVE THIS BRANCH
    ...
else:
    # Fallback: feature similarity (original behavior)  <-- ALWAYS USE THIS
    ...
```

**Change**: Comment out or remove lines 856-874 (the `if flow_mag is not None` branch). The `else` branch (lines 876-881) becomes the only path.

After the change, `merge_frames_dynamic()` will always use feature-similarity classification regardless of whether `flow_mag` is available. The `flow_mag` variable will still be computed in `forward()` (line 1038-1045) and stored on the language model (line 1124-1128) for use by `process_attention()` in `elastic_cache.py`.

**Verification**: The flow scores are still passed to the LLM pruning layer — only the vision merge classification is affected.

---

## Step 2: Verify Baseline on MVBench

Run the 5-task MVBench eval with `use_flow_pruning=True` but WITHOUT `use_motion_adaptive` (to verify that disabling flow vision merge restores baseline accuracy).

```bash
conda run -n pllava python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_verify_baseline_after_revert \
    --num_frames 16 --use_lora --lora_alpha 14 \
    --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 \
    --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --tasks "Action Sequence,Action Prediction,Moving Direction,Object Interaction,Unexpected Action" \
    --use_flow_pruning --flow_dynamic_ratio 0.5 \
    --max_samples 200 \
    > log_mvb_verify_baseline_after_revert.log 2>&1
```

**Expected**: Accuracy should return to ~50.71% (baseline). If it does, the regression is confirmed to come from flow vision merge.

**If accuracy does NOT return to ~50%**: There may be another issue. Check the log for errors.

---

## Step 3: Baseline on VideoMME Long Videos

Run baseline (no motion adaptive, no flow pruning) on VideoMME long split, 100 samples.

### Pre-check: verify data availability

```bash
conda run -n pllava python -c "
import json, os
data = json.load(open('DATAS/Video-MME/json/long.json'))
available = sum(1 for d in data if os.path.exists(f'DATAS/Video-MME/data/{d[\"video\"]}'))
print(f'Long videos available: {available}/{len(data)}')
"
```

### Run baseline

```bash
conda run -n pllava python -m tasks.eval.videomme.pllava_eval_videomme \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/videomme_long_baseline_100 \
    --num_frames 16 --use_lora --lora_alpha 14 \
    --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 \
    --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --tasks "Long Video" --max_samples 100 \
    > log_videomme_long_baseline_100.log 2>&1
```

### Check results

```bash
python -c "
import json
r = json.load(open('test_results/videomme_long_baseline_100/upload_leaderboard.json'))
print('Baseline VideoMME Long:', json.dumps(r, indent=2))
"
```

---

## Step 4: Implement Motion-Adaptive LLM Pruning Only

This is already implemented in the code. The key is using the correct flags:

- `--use_motion_adaptive` — enables per-window alpha scaling based on flow scores
- `--motion_scale 1.0` — how much motion affects retention ratio
- `--use_flow_pruning` — needed to trigger flow computation (but vision merge is now disabled per Step 1)

**No code changes needed for this step.** The `process_attention()` in `elastic_cache.py:126-199` already:

1. Reads `flow_motion_scores` from the language model
2. Calls `_compute_flow_motion_scores()` to get per-window scores
3. Scales alpha: high motion -> higher alpha -> keep more tokens

---

## Step 5: Evaluate Method on VideoMME Long Videos

Run the same eval with motion-adaptive pruning enabled.

```bash
conda run -n pllava python -m tasks.eval.videomme.pllava_eval_videomme \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/videomme_long_motion_100 \
    --num_frames 16 --use_lora --lora_alpha 14 \
    --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 \
    --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --use_flow_pruning --flow_dynamic_ratio 0.5 \
    --use_motion_adaptive --motion_scale 1.0 \
    --tasks "Long Video" --max_samples 100 \
    > log_videomme_long_motion_100.log 2>&1
```

### Compare results

```bash
python -c "
import json, os
for name, path in [
    ('Baseline', 'test_results/videomme_long_baseline_100'),
    ('Motion 1.0', 'test_results/videomme_long_motion_100'),
]:
    fp = os.path.join(path, 'upload_leaderboard.json')
    if os.path.exists(fp):
        data = json.load(open(fp))
        print(f'{name}: {json.dumps(data, indent=2)}')
    else:
        print(f'{name}: not found')
"
```

---

## Quick Reference: All Commands

```bash
# Step 2: Verify baseline after revert
conda run -n pllava python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_verify_baseline_after_revert \
    --num_frames 16 --use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --tasks "Action Sequence,Action Prediction,Moving Direction,Object Interaction,Unexpected Action" \
    --use_flow_pruning --flow_dynamic_ratio 0.5 --max_samples 200 \
    > log_mvb_verify_baseline_after_revert.log 2>&1

# Step 3: VideoMME baseline
conda run -n pllava python -m tasks.eval.videomme.pllava_eval_videomme \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/videomme_long_baseline_100 \
    --num_frames 16 --use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --tasks "Long Video" --max_samples 100 \
    > log_videomme_long_baseline_100.log 2>&1

# Step 5: VideoMME with motion-adaptive pruning
conda run -n pllava python -m tasks.eval.videomme.pllava_eval_videomme \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/videomme_long_motion_100 \
    --num_frames 16 --use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --use_flow_pruning --flow_dynamic_ratio 0.5 \
    --use_motion_adaptive --motion_scale 1.0 \
    --tasks "Long Video" --max_samples 100 \
    > log_videomme_long_motion_100.log 2>&1
```

---

## What to Expect

| Scenario        | VideoMME Long Result | Interpretation                                      |
| --------------- | -------------------- | --------------------------------------------------- |
| **Best case**   | Method > Baseline    | Motion-adaptive pruning helps on long videos        |
| **Likely case** | Method ≈ Baseline    | Motion signal is neutral; method doesn't hurt       |
| **Worst case**  | Method < Baseline    | Motion signal is noisy on long videos; needs tuning |

**If result is best/likely case**: This is a valid finding — your method maintains accuracy on long videos while reducing tokens. Combined with MVBench temporal task results, this supports the paper claim.

**If result is worst case**: Try `--motion_scale 0.5` (less aggressive) or analyze per-sample which videos are hurt.

---

## Next Steps After This Plan

1. **If VideoMME shows improvement or neutral**: Run full VideoMME (all 3 splits, all samples) + EgoSchema
2. **Implement variance-weighted merging** (Step 3 of REVISED_PLAN)
3. **Task-specific tuning** (grid search on alpha, tau, motion_scale)
4. **Efficiency metrics** (TTFT, memory, FLOPs)

---

RESULT:
Both baseline and motion adaptive give same accuracy:
{
"Long Video": 45.0,
"Avg": 45.0
}

but motion adaptive approach prune less token than baseline -> LOST
