# Flow Pruning Regression Analysis

## Actual Eval Command (Flow 1.0)

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

**Key parameters**:
- `--alpha 0.4 --use_motion_adaptive --motion_scale 1.0`: Motion-adaptive alpha = `0.4 * (1.0 + 1.0 * motion_ratio)`
  - Static windows: alpha=0.4 (keep 40%)
  - Dynamic windows: alpha up to 0.8 (keep up to 80%)
- `--use_flow_pruning --flow_dynamic_ratio 0.5`: Flow-based static/dynamic classification (top 50% by flow magnitude = dynamic)
- `--selected_layer 10`: LLM pruning triggers at layer 10

## Results Summary

| Task | Baseline | Flow 1.0 | Delta |
|------|----------|----------|-------|
| Action Sequence | 55.85 | 50.53 | **-5.32** |
| Action Prediction | 51.50 | 43.00 | **-8.50** |
| Unexpected Action | 62.50 | 58.00 | **-4.50** |
| Object Interaction | 63.50 | 55.50 | **-8.00** |
| Moving Direction | 20.50 | 19.50 | -1.00 |
| **Avg** | **50.71** | **45.24** | **-5.47** |

## Ablation Result: Flow Merge Only (No LLM Prune)

**Command**: Same as above with `--selected_layer 999` (skips LLM pruning)

| Task | Baseline | Flow no LLM prune | Flow 1.0 |
|------|----------|-------------------|----------|
| Action Sequence | 55.85 | 60.0 | 50.53 |
| Action Prediction | 51.50 | 32.0 | 43.00 |
| Unexpected Action | 62.50 | 68.0 | 58.00 |
| Object Interaction | 63.50 | 72.0 | 55.50 |
| Moving Direction | 20.50 | 16.0 | 19.50 |
| **Avg** | **50.71** | **49.6** | **45.24** |

**Sample sizes**: Baseline=988, Flow 1.0=988, Flow no LLM prune=250 (50 per task, noisy)

**Token stats**:
- Flow 1.0: merged=666 → pruned=508 (-24% from LLM pruning)
- Flow no LLM prune: merged=670 (same merge, no pruning)

**Conclusion**: LLM pruning at layer 10 is the primary cause of regression (-5.47%). Flow vision merge alone is roughly neutral (49.6% vs 50.71%, within noise).

## Root Cause: Two Compounding Issues

### Issue 1: LLM Pruning is New (Baseline Had None)

| Metric | Baseline | Flow 1.0 |
|--------|----------|----------|
| Avg merged tokens (entering LLM) | 693-820 | 659-684 |
| LLM pruned tokens | None (0% pruning) | 454-568 (22-31% pruning) |
| **Total tokens reaching LLM** | **693-820** | **454-568 (-35%)** |
| Pruning layer | None | 10 |

Baseline never pruned at the LLM level (`pruned_tokens: None`, `pruning_layer: None`). Flow pruning adds a second pruning pass at layer 10, removing another 22-31% of already-merged tokens. Total token reduction vs baseline is ~35%.

### Issue 2: Flow Merge Loses Content-Adaptive Variance

| Task | Baseline CV | Flow CV |
|------|-------------|---------|
| Action Prediction | 38% | 4% |
| Action Sequence | 42% | 4% |
| Moving Direction | 17% | 4% |
| Object Interaction | 41% | 5% |
| Unexpected Action | 27% | 5% |

Baseline's feature-similarity merge produces highly content-adaptive token counts (CV=27-42%). Flow-based merge produces near-uniform counts (CV=4-5%). Simple videos and complex videos get the same budget, losing the ability to allocate more tokens to harder content.

---

## Complete Method Survey

### Methods Tried (All Failed or Neutral)

| # | Method | Accuracy | Delta | Status |
|---|--------|----------|-------|--------|
| 0 | Baseline (fixed L10, alpha=0.4) | **50.71%** | 0.00 | WORKING |
| 1 | Entropy-adaptive pruning | 50.20% | -0.51 | DEAD END (uniform attention) |
| 2 | Motion-adaptive (attention variance) | 49.87% | -0.84 | DEAD END (variance ~0) |
| 3 | Borderline preservation | 50.67% | -0.26 | HARMFUL |
| 4 | Weighted token merge | 50.93% | +0.22 | NEUTRAL |
| 5 | Flow 1.0 (merge + LLM prune) | 45.24% | -5.47 | REGRESSION |
| 6 | Flow merge only (no LLM prune) | 49.6% | -1.1 | NEUTRAL (noisy) |

### Methods NOT Tried Yet

| # | Method | Source | Status |
|---|--------|--------|--------|
| 7 | Query-conditioned pruning | FRIEND_ADVICE #4 | Not implemented |
| 8 | Progressive multi-stage pruning | PRUNEVID_LIMITATION | Not implemented |
| 9 | Adaptive window sizing | FRIEND_ADVICE #2 | Not implemented |
| 10 | Temporal pivot anchoring | FRIEND_ADVICE #5 | Not implemented |
| 11 | Pixel/vision-feature variance | Phase 3 Proxy B | Not implemented |
| 12 | 12 KV-cache strategies in modify_llama.py | Reference code | Not tested |
| 13 | RAFT flow model | modeling_pllava_flow.py | Not tested |
| 14 | Hyperparameter sweep | RESEARCH_PLAN | Mostly untested |

---

## Competitive Landscape: Where Can You Beat PruneVid?

### PruneVid's Numbers (PLLaVA-7B, 16.2% retained, 0.23x FLOPs)

| Benchmark | Baseline | PruneVid | Delta |
|-----------|----------|----------|-------|
| MVBench | 46.6 | **47.6** | **+1.0** |
| VideoMME | 44.4 | **45.3** | **+0.9** |
| EgoSchema Subset | 47.8 | **49.0** | **+1.2** |
| EgoSchema Fullset | 42.6 | 42.6 | 0.0 |
| VCGBench Avg | 2.99 | 2.98 | -0.01 |

### Competing Methods (PLLaVA-7B)

| Method | Retained | FLOPs | MVBench | VideoMME | EgoSchema Sub |
|--------|----------|-------|---------|----------|---------------|
| FastV | 30.0% | 0.33x | 46.1 | 43.6 | 46.2 |
| Prumerge | 55.7% | 0.53x | 45.6 | 43.8 | 45.2 |
| Look-M | 20.0% | 1.00x | 46.6 | 44.3 | 47.0 |
| **PruneVid** | **16.2%** | **0.23x** | **47.6** | **45.3** | **49.0** |

**PruneVid is the ONLY method that improves accuracy while reducing FLOPs.** All others degrade accuracy.

### Where You Can Beat PruneVid

1. **EgoSchema Fullset** (42.6 = baseline, 0.0 improvement) — PruneVid fails here. Any improvement is novel.
2. **VCGBench** (2.98, -0.01 degradation) — PruneVid slightly hurts. Recovery = improvement.
3. **MVBench temporal subtasks** — Moving Direction at 20.5% is random. Fixing this is the biggest opportunity.
4. **Fewer retained tokens at same accuracy** — PruneVid uses 16.2%. If you match its accuracy at <16%, that's a win.
5. **Content-adaptive retention** — PruneVid uses fixed layer 10 + fixed alpha. Adaptive = novelty.

### Your Dev Eval vs Paper Numbers

Your 5-task dev eval baseline (50.71%) uses a different subset than the paper's full MVBench (47.6). The paper uses all 20 MVBench tasks. Your dev eval is a 5-task slice with ~200 samples per task. Direct comparison requires running the full 20-task MVBench.

---

## Ablation Plan: Option A (Flow Merge Alone)

### Step 1: Run full 988-sample ablation with selected_layer=999

```bash
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_flow_no_llm_prune_full \
    --num_frames 16 --use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 --selected_layer 999 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --tasks "Action Sequence,Action Prediction,Moving Direction,Object Interaction,Unexpected Action" \
    --use_motion_adaptive --motion_scale 1.0 \
    --use_flow_pruning --flow_dynamic_ratio 0.5 > log_mvbench_flow_no_llm_prune_5_tasks_full.log 2>&1
```

### Step 2: If flow merge alone is neutral/negative, sweep flow_dynamic_ratio

Test 0.3 and 0.7 to find optimal static/dynamic split.

### Step 3: If flow merge helps, run full 20-task MVBench

Compare against PruneVid's 47.6 on the same benchmark.

### Step 4: If results are competitive, run VideoMME + EgoSchema

Target PruneVid's weaknesses: EgoSchema Fullset (42.6), VCGBench (2.98).
