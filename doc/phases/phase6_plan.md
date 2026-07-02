# Phase 6 Plan — Full Benchmark Evaluation

> **Created**: 2026-06-28
> **Updated**: 2026-06-30 (revised plan with TFLOPs instrumentation)
> **Status**: IN PROGRESS — Step 1 (TFLOPs instrumentation)
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
- `raw_visual_tokens`: from `measure_vision_pruning()` or model forward
- `after_vision_merge_tokens`: after Stage 1 (stored on `PllavaForConditionalGeneration`)
- `llm_pruned_tokens`: after Stage 2 (stored on `VTPWindowCache`)
- `pruning_layer`: from `selected_layer` or `dynamic_selected_layer`

---

## TFLOPs Instrumentation (Step 1 — Code Changes)

Token counts are now tracked in the model and exposed via `pllava_answer()`.

### Changes Made

| File | Change |
|------|--------|
| `models/pllava/elastic_cache.py` | `VTPWindowCache.prompt_prefill()`: store `self.num_tokens_after_prune = num_tokens_left` (line 255) |
| `models/pllava/modeling_pllava.py` | `PllavaForConditionalGeneration.forward()`: store `self._last_raw_vision_tokens`, `self._last_merged_vision_tokens` after `merge_frames_dynamic()` (line ~976) |
| `models/pllava/llama.py` | `LlamaModelVTP.forward()`: after cache call, store `self.last_pruned_token_count` on model |
| `tasks/eval/model_utils.py` | `pllava_answer()`: returns 3rd element `token_info` dict |
| `tasks/eval/mvbench/pllava_eval_mvbench.py` | `infer_mvbench()`: returns token_info alongside llm_message; result_list includes token counts |
| `tasks/eval/egoshcema/pllava_eval_egoschema.py` | Same as mvbench |
| `tasks/eval/videomme/pllava_eval_videomme.py` | Same as mvbench |

### Token Info Dict Structure

```python
token_info = {
    'raw_vision_tokens': int,        # from projector output (e.g., 2304)
    'merged_vision_tokens': int,     # after vision merge Stage 1 (e.g., 627)
    'pruned_tokens': int,            # after LLM prune Stage 2 (e.g., 245)
    'pruning_layer': int,            # layer where pruning occurred (e.g., 10 or dynamic)
    'original_total_tokens': int,    # total prefill tokens including text
}
```

### FLOPs Ratio Computation (in eval results compilation)

```python
def compute_flops_ratio(token_info, total_layers=32):
    merged = token_info['merged_vision_tokens']
    pruned = token_info['pruned_tokens']
    layer = token_info['pruning_layer']
    original = 2304  # PLLaVA-7B default

    baseline_flops = original * total_layers
    our_flops = merged * layer + pruned * (total_layers - layer)
    return our_flops / baseline_flops
```

---

## Execution Plan

### Step 1: Instrument Token Count Tracking ✅
5 files modified to track and expose token counts during inference.

### Step 2: Smoke Test (~5 min)
Verify token counts work:
```bash
python scripts/infer_single_video.py \
    --video DATAS/Video-MME/25Pt1AZO9EM.mp4 \
    --model_dir MODELS/pllava-7b \
    --weight_dir MODELS/pllava-7b \
    --use_lora --lora_alpha 14 \
    --use_entropy_adaptive --tau_entropy 0.8 \
    --use_borderline_preservation --borderline_margin 0.1 \
    --alpha 0.4 --skip_generation
```
**Check**: Token counts printed: 2304 → ~600 → ~250

### Step 3: Dev Re-Eval — 5 Tasks, All Samples (~4-6 hrs)

Re-run with bug fixes + token count tracking.

#### Run 3a: Baseline (fixed layer 10)
```bash
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_dev_baseline_all \
    --num_frames 16 --use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --tasks "Action Sequence,Action Prediction,Moving Direction,Object Interaction,Unexpected Action"
```

#### Run 3b: Entropy + Borderline (best method)
```bash
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_dev_entropy_borderline_all \
    --num_frames 16 --use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --use_entropy_adaptive --tau_entropy 0.8 \
    --use_borderline_preservation --borderline_margin 0.1 \
    --tasks "Action Sequence,Action Prediction,Moving Direction,Object Interaction,Unexpected Action"
```

#### Run 3c: Entropy-only (ablation)
```bash
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_dev_entropy_only_all \
    --num_frames 16 --use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --use_entropy_adaptive --tau_entropy 0.8 \
    --tasks "Action Sequence,Action Prediction,Moving Direction,Object Interaction,Unexpected Action"
```

### Step 4: Decision Gate

Check results in `test_results/mvbench_dev_*/upload_leaderboard.json`:

| Result | Action |
|--------|--------|
| Entropy+borderline > baseline by ≥1% | Full benchmarks with confidence |
| Entropy+borderline > baseline by 0.5-1% | Full benchmarks, frame as "marginal" |
| Entropy+borderline ≈ baseline (±0.5%) | Still proceed — larger samples may reveal effect |
| Entropy+borderline < baseline | Debug further, check if Bug 2 fix changed entropy behavior |

### Step 5: Full MVBench — 20 Tasks, 300 Samples/Task (~12 hrs × 2)

#### Run 5a: Baseline
```bash
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_full_baseline \
    --num_frames 16 --use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --max_samples 300
```

//// ----- flow
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_full_flow_1.0 \
    --num_frames 16 --use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --tasks "Action Sequence,Action Prediction,Moving Direction,Object Interaction,Unexpected Action" \
    --use_motion_adaptive --motion_scale 1.0 --max_samples 300 --use_flow_pruning --flow_dynamic_ratio 0.5 > log_mvbench_full_flow_1.0.log 2>&1

#### Run 5b: Best config (entropy + borderline)
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

### Step 6: MVBench Motion-Heavy Subset Analysis (~12 hrs × 2)

#### Motion-heavy tasks (14 tasks)
```
Action Antonym, Action Count, Action Localization, Action Prediction,
Action Sequence, Fine-grained Action, Fine-grained Pose, Moving Attribute,
Moving Count, Moving Direction, Object Interaction, Object Shuffle,
State Change, Unexpected Action
```

#### Temporally complex tasks (5 tasks)
```
Action Sequence, Action Prediction, Character Order,
Episodic Reasoning, Scene Transition
```

#### Static tasks (control, 6 tasks)
```
Object Existence, Object Shuffle, Scene Transition,
Character Order, Episodic Reasoning, Counterfactual Inference
```

Run baseline and best config on each subset (200 samples/task).

### Step 7: EgoSchema + VideoMME (if dev results positive)

Same pattern: baseline vs best config, with token count tracking.

### Step 8: Results Compilation

After all runs, compile the final table:

```
| Config | MVBench | MVBench-Motion | MVBench-Temporal | MVBench-Static | EgoSchema | VideoMME | FLOPs Ratio | Retained |
|--------|---------|---------------|-----------------|---------------|-----------|----------|-------------|----------|
| Baseline (PruneVid) | 47.6 | ? | ? | ? | 49.0/42.6 | 45.3 | 0.23x | 16.2% |
| + Entropy (ablation) | ? | ? | ? | ? | ? | ? | ~0.23x | ~16% |
| + Borderline (ablation) | ? | ? | ? | ? | ? | ? | ~0.25x | ~20% |
| Full method | ? | ? | ? | ? | ? | ? | ~0.23x | ~18% |
```

---

## Execution Timeline

| Step | Task | Est. Time |
|------|------|-----------|
| 1 | Instrument token count tracking | ✅ Done |
| 2 | Smoke test | ~5 min |
| 3 | Dev re-eval (5 tasks × all samples × 3 configs) | ~4-6 hrs |
| 4 | Decision gate | ~5 min |
| 5 | Full MVBench (2 configs × 20 tasks × 300 samples) | ~24 hrs |
| 6 | Motion-heavy subset analysis (2 configs × 3 subsets) | ~24 hrs |
| 7 | EgoSchema + VideoMME (if positive) | ~40 hrs |
| 8 | Results compilation | ~1 hr |
| **Total** | | **~94-107 hrs** |

With 1 GPU: ~4-5 days continuous
With 2 GPUs: ~2-2.5 days continuous

---

## Decision Gates

| After Step 3 (Dev Re-Eval) | Action |
|-----------------------------|--------|
| Method > baseline by ≥1% | Proceed with confidence |
| Method > baseline by 0.5-1% | Proceed, frame as "marginal improvement" |
| Method ≈ baseline (±0.5%) | Still proceed — full benchmark may show effect |
| Method < baseline | Debug entropy threshold, check if Bug 2 fix changed behavior |

| After Step 5 (Full MVBench) | Action |
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
- **All existing dev eval results are pre-bug-fix**: Must re-run to get valid comparisons.
