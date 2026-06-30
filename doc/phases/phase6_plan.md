# Phase 6 Plan — Full Benchmark Evaluation

> **Created**: 2026-06-28
> **Updated**: 2026-06-30 (after bug fixes)
> **Status**: READY TO EXECUTE
> **Goal**: Validate improvements across MVBench, EgoSchema, VideoMME with FLOPs comparison

---

## Bug Fixes Applied (2026-06-30)

Three critical bugs were found and fixed in `models/pllava/llama.py`:

### Bug 1 (CRITICAL): Fallback pruning unreachable
- **Was**: When no entropy trigger fired, fallback set `dynamic_selected_layer` AFTER the loop exited, so `self.cache()` was never called.
- **Fix**: Store `fallback_layer_outputs` during the loop, execute pruning after loop when fallback triggers.

### Bug 2 (SIGNIFICENT): Entropy on pre-RoPE attention
- **Was**: Entropy computed on `attn_weights_wo_pos` (before Rotary Position Embedding).
- **Fix**: Compute entropy on post-RoPE, post-softmax `attn_weights` (the actual attention the model uses).

### Bug 3 (MODERATE): Aggressive max-over-heads aggregation
- **Was**: `max(dim=0).max(dim=0)` collapsed all heads and text tokens into one scalar per image token.
- **Fix**: Per-head entropy computation, then mean aggregation across heads.

**Impact**: These fixes may significantly change results. The fallback bug meant entropy-adaptive mode was silently doing NOTHING for many inputs.

---

## FLOPs Estimation (for Paper)

The PruneVid paper reports FLOPs as a **ratio** (e.g., "0.23x"), not absolute TFLOPS. This is GPU-agnostic.

### Formula

```
Baseline FLOPs  = original_tokens × total_layers
Our FLOPs       = vision_merged_tokens × pruning_layer + llm_pruned_tokens × (total_layers - pruning_layer)
FLOPs Ratio     = Our FLOPs / Baseline FLOPs
```

### PLLaVA-7B Parameters
- `original_tokens = 2304` (16 frames × 144 pooled)
- `total_layers = 32`
- `hidden_dim = 4096`

### Example Calculation
- Vision merge: 2304 → 627 tokens
- Pruning at layer 10: 627 → 245 tokens
- Baseline: 2304 × 32 = 73,728
- Ours: 627 × 10 + 245 × 22 = 6,270 + 5,390 = 11,660
- **FLOPs Ratio: 11,660 / 73,728 = 0.158x** (~84% reduction)

### Where to Compute

The FLOPs ratio is computed from token counts collected during eval:
- `raw_visual_tokens`: from `measure_vision_pruning()`
- `after_vision_merge_tokens`: after Stage 1
- `llm_pruned_tokens`: after Stage 2 (estimated from `alpha` and token counts)
- `pruning_layer`: from `selected_layer` or `dynamic_selected_layer`

---

## Execution Plan

### Step 1: Smoke Test (5 min)

Verify bug fixes work without crashes:

```bash
python scripts/infer_single_video.py \
    --video example/cooking.mp4 \
    --model_dir MODELS/pllava-7b \
    --weight_dir MODELS/pllava-7b \
    --use_lora --lora_alpha 14 \
    --use_entropy_adaptive --tau_entropy 0.8 \
    --use_borderline_preservation --borderline_margin 0.1 \
    --alpha 0.4 --skip_generation
```

**Check**:
- No crashes
- Entropy profile printed with trigger layer shown
- Token counts: 2304 → ~600 → ~250

### Step 2: Dev Eval — 3 configs, 5 tasks, 75 samples (~2.5 hrs)

#### Config A: Baseline (fixed layer 10)
```bash
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_dev_baseline \
    --num_frames 16 --use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --tasks "Action Sequence,Action Prediction,Moving Direction,Object Interaction,Unexpected Action" 
```

#### Config B: Entropy + Borderline (best method)
```bash
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_dev_entropy_borderline \
    --num_frames 16 --use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --use_entropy_adaptive --tau_entropy 0.8 \
    --use_borderline_preservation --borderline_margin 0.1 \
    --tasks "Action Sequence,Action Prediction,Moving Direction,Object Interaction,Unexpected Action" 
```

#### Config C: Entropy-only (ablation)
```bash
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_dev_entropy_only \
    --num_frames 16 --use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --use_entropy_adaptive --tau_entropy 0.8 \
    --tasks "Action Sequence,Action Prediction,Moving Direction,Object Interaction,Unexpected Action" 
```

### Step 3: Decision Gate

Check results in `test_results/mvbench_dev_*/upload_leaderboard.json`:

| Result | Action |
|--------|--------|
| Entropy+borderline > baseline by ≥1% | Full benchmarks with confidence |
| Entropy+borderline > baseline by 0.5-1% | Full benchmarks, frame as "marginal" |
| Entropy+borderline ≈ baseline (±0.5%) | Still proceed — larger samples may reveal effect |
| Entropy+borderline < baseline | Debug further, check if Bug 2 fix changed entropy behavior |

### Step 4: Full MVBench — 20 tasks, 300 samples/task (~12 hrs × 2)

#### Run 4a: Baseline
```bash
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_full_baseline \
    --num_frames 16 --use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --max_samples 300
```

#### Run 4b: Best config (entropy + borderline + weighted)
```bash
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_full_entropy_borderline \
    --num_frames 16 --use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --use_entropy_adaptive --tau_entropy 0.8 \
    --use_borderline_preservation --borderline_margin 0.1 \
    --max_samples 300
```

### Step 5: EgoSchema (~8 hrs × 2)

#### Run 5a: Baseline
```bash
python -m tasks.eval.egoshcema.pllava_eval_egoschema \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/egoschema_baseline \
    --num_frames 16 --use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5
```

#### Run 5b: Best config
```bash
python -m tasks.eval.egoshcema.pllava_eval_egoschema \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/egoschema_entropy_borderline \
    --num_frames 16 --use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --use_entropy_adaptive --tau_entropy 0.8 \
    --use_borderline_preservation --borderline_margin 0.1
```

### Step 6: VideoMME (~12 hrs × 2)

#### Run 6a: Baseline
```bash
python -m tasks.eval.videomme.pllava_eval_videomme \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/videomme_baseline \
    --num_frames 16 --use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5
```

#### Run 6b: Best config
```bash
python -m tasks.eval.videomme.pllava_eval_videomme \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/videomme_entropy_borderline \
    --num_frames 16 --use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --use_entropy_adaptive --tau_entropy 0.8 \
    --use_borderline_preservation --borderline_margin 0.1
```

### Step 7: MVBench Motion-Heavy Subset Analysis

Use `--tasks` flag with task lists from `TEMPORAL_LONG_DATASET.md`:

#### Motion-heavy tasks (14 tasks)
```bash
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_motion_heavy_baseline \
    --num_frames 16 --use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --tasks "Action Antonym,Action Count,Action Localization,Action Prediction,Action Sequence,Fine-grained Action,Fine-grained Pose,Moving Attribute,Moving Count,Moving Direction,Object Interaction,Object Shuffle,State Change,Unexpected Action" \
    --max_samples 200
```

```bash
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_motion_heavy_entropy_borderline \
    --num_frames 16 --use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --use_entropy_adaptive --tau_entropy 0.8 \
    --use_borderline_preservation --borderline_margin 0.1 \
    --tasks "Action Antonym,Action Count,Action Localization,Action Prediction,Action Sequence,Fine-grained Action,Fine-grained Pose,Moving Attribute,Moving Count,Moving Direction,Object Interaction,Object Shuffle,State Change,Unexpected Action" \
    --max_samples 200
```

#### Temporally complex tasks (5 tasks)
```bash
# Baseline
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_temporal_baseline \
    --num_frames 16 --use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --tasks "Action Sequence,Action Prediction,Character Order,Episodic Reasoning,Scene Transition" \
    --max_samples 200

# Best config
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_temporal_entropy_borderline \
    --num_frames 16 --use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --use_entropy_adaptive --tau_entropy 0.8 \
    --use_borderline_preservation --borderline_margin 0.1 \
    --tasks "Action Sequence,Action Prediction,Character Order,Episodic Reasoning,Scene Transition" \
    --max_samples 200
```

#### Static tasks (control group, 6 tasks)
```bash
# Baseline
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_static_baseline \
    --num_frames 16 --use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --tasks "Object Existence,Object Shuffle,Scene Transition,Character Order,Episodic Reasoning,Counterfactual Inference" \
    --max_samples 200

# Best config
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_static_entropy_borderline \
    --num_frames 16 --use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --use_entropy_adaptive --tau_entropy 0.8 \
    --use_borderline_preservation --borderline_margin 0.1 \
    --tasks "Object Existence,Object Shuffle,Scene Transition,Character Order,Episodic Reasoning,Counterfactual Inference" \
    --max_samples 200
```

---

## Results Compilation

After all runs, read `upload_leaderboard.json` from each `test_results/` directory and compile:

### Final Table Template

```
| Config | MVBench | MVBench-Motion | MVBench-Temporal | MVBench-Static | EgoSchema | VideoMME | FLOPs Ratio | Retained |
|--------|---------|---------------|-----------------|---------------|-----------|----------|-------------|----------|
| Baseline (PruneVid) | 47.6 | ? | ? | ? | 49.0/42.6 | 45.3 | 0.23x | 16.2% |
| + Entropy (ablation) | ? | ? | ? | ? | ? | ? | ~0.23x | ~16% |
| + Borderline (ablation) | ? | ? | ? | ? | ? | ? | ~0.25x | ~20% |
| Full method | ? | ? | ? | ? | ? | ? | ~0.23x | ~18% |
```

### Key Metrics to Report

1. **Accuracy**: MVBench (overall + per-subset), EgoSchema, VideoMME
2. **Efficiency**: FLOPs ratio (our_flops / baseline_flops)
3. **Token count**: retained_ratio = llm_pruned_tokens / original_tokens
4. **Per-task breakdown**: Especially motion-heavy vs static tasks

---

## Execution Timeline

| Step | Task | Est. Time | GPU-hours |
|------|------|-----------|-----------|
| 1 | Smoke test | 5 min | - |
| 2 | Dev eval (3 configs × 5 tasks × 75 samples) | ~2.5 hrs | ~2.5 hrs |
| 3 | Decision gate | 5 min | - |
| 4 | Full MVBench (2 × 20 tasks × 300 samples) | ~24 hrs | ~24 hrs |
| 5 | EgoSchema (2 runs) | ~16 hrs | ~16 hrs |
| 6 | VideoMME (2 runs) | ~24 hrs | ~24 hrs |
| 7 | Motion-heavy subset analysis | ~12 hrs | ~12 hrs |
| **Total** | | | **~78.5 hrs** |

With 1 GPU: ~3.3 days continuous
With 2 GPUs: ~1.7 days continuous

---

## Decision Gates

| After Step 2 (Dev Eval) | Action |
|--------------------------|--------|
| Method > baseline by ≥1% | Proceed with confidence |
| Method > baseline by 0.5-1% | Proceed, frame as "marginal improvement" |
| Method ≈ baseline (±0.5%) | Still proceed — full benchmark may show effect |
| Method < baseline | Debug entropy threshold, check if Bug 2 fix changed behavior |

| After Step 4 (Full MVBench) | Action |
|------------------------------|--------|
| Improvement ≥ 1% | Strong result, proceed to paper |
| Improvement 0.5-1% | Moderate, include with caveats |
| Improvement < 0.5% | Focus paper on efficiency (FLOPs reduction) rather than accuracy |

---

## Notes

- **GPU requirement**: Any GPU with ≥24GB VRAM. FLOPs ratio is GPU-agnostic.
- **VCGBench skipped**: Requires OpenAI API key for GPT-based scoring. Can add later if needed.
- **Motion-adaptive dropped**: Phase 3 showed no improvement (49.87 vs 50.93). Not included in best config.
- **Weighted merge**: Kept as default (marginal effect, doesn't hurt).
