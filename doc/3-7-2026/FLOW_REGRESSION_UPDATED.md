# Flow Pruning Regression Analysis (Updated 3-7-2026)

## Key Finding: Flow Vision Merge is Harmful

Full 988-sample ablation confirms **flow-based vision merge is worse than feature-similarity merge**. The regression is NOT primarily from LLM pruning — it's from the vision merge itself.

## Results

| Task | Baseline | Flow no LLM prune | Flow 1.0 |
|------|----------|-------------------|----------|
| Action Sequence | 55.85 | 51.06 (-4.79) | 50.53 (-5.32) |
| Action Prediction | 51.50 | 45.00 (-6.50) | 43.00 (-8.50) |
| Unexpected Action | 62.50 | 58.00 (-4.50) | 58.00 (-4.50) |
| Object Interaction | 63.50 | 56.50 (-7.00) | 55.50 (-8.00) |
| Moving Direction | 20.50 | 20.00 (-0.50) | 19.50 (-1.00) |
| **Avg** | **50.71** | **46.05 (-4.66)** | **45.24 (-5.47)** |

Sample sizes: Baseline=988, Flow no LLM prune=988, Flow 1.0=988.

## Component Analysis

| Component | Impact |
|-----------|--------|
| Flow vision merge alone | **-4.66%** (46.05 vs 50.71) |
| LLM pruning added on top | -0.81% more (45.24 vs 46.05) |
| **Total regression** | **-5.47%** |

**85% of the regression comes from the vision merge change, not from LLM pruning.**

## Token Stats

| Config | Merged tokens | CV |
|--------|--------------|-----|
| Baseline | 693-820 avg | 27-42% |
| Flow merge | 612-720 avg | 4-5% |

Baseline's feature-similarity merge produces content-adaptive token counts (CV=27-42%). Flow merge flattens this to near-uniform (CV=4-5%), losing the ability to allocate more tokens to harder content.

## Corrected Root Cause

The earlier 250-sample ablation (49.6%) was misleading due to small sample size and high variance. The full 988-sample result (46.05%) confirms:

1. **Flow vision merge is harmful** (-4.66%)
2. **LLM pruning adds marginal additional harm** (-0.81%)
3. **The feature-similarity merge in baseline is doing something right** — it allocates tokens content-adaptively

## Implications for Revised Plan

- **Do NOT modify vision merge** — keep feature-similarity (baseline behavior)
- **Use optical flow scores for LLM-level pruning** — feed flow scores into `process_attention()` for per-window alpha scaling
- **The two-tier retention (Method #1) should work at layer 10**, not at the vision merge level

## Eval Commands Used

### Flow 1.0 (original)
```bash
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_full_flow_1.0 \
    --num_frames 16 --use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --tasks "Action Sequence,Action Prediction,Moving Direction,Object Interaction,Unexpected Action" \
    --use_motion_adaptive --motion_scale 1.0 --max_samples 300 \
    --use_flow_pruning --flow_dynamic_ratio 0.5
```

### Flow no LLM prune (ablation)
```bash
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_flow_no_llm_prune_full \
    --num_frames 16 --use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 --selected_layer 999 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --tasks "Action Sequence,Action Prediction,Moving Direction,Object Interaction,Unexpected Action" \
    --use_motion_adaptive --motion_scale 1.0 --max_samples 300 \
    --use_flow_pruning --flow_dynamic_ratio 0.5
```

## Next Step

Revert flow vision merge, keep flow computation for LLM-level scoring, run baseline verification.
