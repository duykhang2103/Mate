# Phase 6 Plan — Full Benchmark Evaluation

> **Created**: 2026-06-28
> **Status**: PLANNING
> **Goal**: Validate improvements across MVBench, EgoSchema, VideoMME

---

## Problem

Current results (5-task, 75 samples) show small deltas (~0.5%) that could be noise.
Need larger sample counts and more benchmarks to confirm improvements are real.

---

## Priorities

### Priority 1: Validate MVBench (confidence building)

**Run 1 — Baseline (original PruneVid code), 300 samples, 5 tasks:**
```bash
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_baseline_300samples \
    --num_frames 16 --use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --tasks "Action Sequence,Action Prediction,Moving Direction,Object Interaction,Unexpected Action" \
    --max_samples 300
```

**Run 2 — Best config (entropy + borderline + weighted), 300 samples, 5 tasks:**
```bash
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_entropy_borderline_300samples \
    --num_frames 16 --use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --use_entropy_adaptive --tau_entropy 0.8 \
    --use_borderline_preservation --borderline_margin 0.1 \
    --tasks "Action Sequence,Action Prediction,Moving Direction,Object Interaction,Unexpected Action" \
    --max_samples 300
```

**Decision gate:**
- Gap < 0.5% → improvements are noise, skip to Priority 2 with current numbers
- Gap ≥ 0.5% → run full 20-task MVBench

### Priority 2: Full MVBench (all 20 tasks)

Run baseline + best config on all 20 MVBench tasks, 300 samples each.
Publishable numbers for paper.

### Priority 3: EgoSchema

Egocentric video understanding — tests temporal reasoning.
Less motion-heavy than MVBench but good for showing generalization.

### Priority 4: VideoMME

Broader benchmark — shows method doesn't hurt on non-motion tasks.

### Stretch: TemporalBench

If results are optimistic, add TemporalBench — specifically tests temporal understanding.

---

## Execution Timeline

| Priority | Runs | Est. Time | GPU-hours |
|----------|------|-----------|-----------|
| 1: Validate MVBench | 2 | ~2 hrs each | ~4 hrs |
| 2: Full MVBench | 2 | ~6 hrs each | ~12 hrs |
| 3: EgoSchema | 2 | ~4 hrs each | ~8 hrs |
| 4: VideoMME | 2 | ~6 hrs each | ~12 hrs |
| **Total** | **8** | | **~36 hrs** |

---

## Decision Gates

| After Priority 1 | Action |
|-------------------|--------|
| Gap ≥ 0.5% | Continue to Priority 2 |
| Gap < 0.5% | Still run Priority 2 (for ablation table), but frame as "marginal improvement" |

| After Priority 2 | Action |
|-------------------|--------|
| Full MVBench improvement ≥ 1% | Strong result, proceed to paper |
| Full MVBench improvement 0.5-1% | Moderate result, include with caveats |
| Full MVBench improvement < 0.5% | Focus paper on efficiency (tokens saved) rather than accuracy |

---

## Ablation Table (to fill in)

| Config | 5-task Avg | Full MVBench | EgoSchema | VideoMME |
|--------|-----------|--------------|-----------|----------|
| Baseline (PruneVid) | 50.40 | ? | ? | ? |
| + Entropy | 50.93 | ? | ? | ? |
| + Borderline | 50.67 | ? | ? | ? |
| + Weighted Merge | ~50.67 | ? | ? | ? |
| Full method (all) | ? | ? | ? | ? |
