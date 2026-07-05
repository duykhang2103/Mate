# Prioritized Execution Plan

> **Date**: 2026-07-05
> **Goal**: Improve accuracy on specific regions (long video, heavy motion) OR preserve accuracy with fewer FLOPs
> **End Goal**: Research paper submission
> **Status**: Ready to execute

---

## Priority Order

| Priority | Method | Effort | Success Chance | Contribution |
|----------|--------|--------|----------------|--------------|
| **P1** | DPC-KNN clustering at LLM level | 2-3 hrs | High | Content-adaptive pruning without attention |
| **P2** | Accuracy vs FLOPs grid search | 2-3 hrs | Guaranteed | Efficiency curves + optimal configs |
| **P3** | Self-attention token selection | 1-2 hrs | Medium | Structural importance from self-attention |
| **P4** | Switch to LLaVA-OneVision | 4-5 hrs | High (but late) | Non-uniform attention exploitation |

---

## P1: DPC-KNN Clustering at LLM Level

### Why This First

- PLLaVA's attention is uniform (entropy=0.9999) → attention-based selection fails
- But LLM features at layer 10 ARE content-dependent (they've processed visual input through 10 layers)
- Clustering on these features doesn't need attention to be non-uniform
- Infrastructure exists: `cluster_dpc_knn()` function in `elastic_cache.py`

### Implementation (DONE)

**Files Modified**:
1. `models/pllava/configuration_pllava.py` — Added `use_cluster_pruning`, `cluster_pruning_topk` params
2. `models/pllava/modeling_pllava.py` — Propagate new params to `text_config`
3. `models/pllava/llama.py` — Read new params, pass to `VTPWindowCache`
4. `models/pllava/elastic_cache.py` — Core implementation:
   - `VTPWindowCache.__init__()` — Accept new params
   - `process_attention()` — Route to cluster pruning when enabled
   - `_cluster_prune()` — New method implementing DPC-KNN clustering

**How it works**:
1. Extract hidden states for image tokens at layer 10: `hidden_states[:, img_start:img_end+1, :]`
2. Run DPC-KNN clustering on these features: `cluster_dpc_knn(img_hidden, cluster_num=num_clusters, k=5)`
3. Rank clusters by size (largest = most tokens = most important region)
4. Greedily select clusters until `num_keep = num_img_tokens * alpha` tokens are selected
5. For partial clusters, use L2 norm of hidden state as importance score within the cluster

**Usage**:
```bash
# Enable cluster pruning
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --use_cluster_pruning \
    --cluster_pruning_topk 0.4 \
    --alpha 0.4 \
    ...

# Compare against attention-based (default)
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --alpha 0.4 \
    ...
```

### Expected Result

- More diverse token selection (not just highest attention)
- Better preservation of spatial coverage
- May help on motion-heavy videos where diverse tokens matter

### Evaluation (Ready to Run)

```bash
# Baseline (attention-based topk)
conda run -n pllava python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_p1_baseline \
    --num_frames 16 --use_lora --lora_alpha 14 \
    --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 \
    --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --use_flow_pruning --flow_dynamic_ratio 0.5 \
    --tasks "Moving Direction,Object Interaction" \
    --max_samples 200 \
    > log_p1_baseline.log 2>&1

# Cluster-based pruning (P1)
conda run -n pllava python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_p1_cluster \
    --num_frames 16 --use_lora --lora_alpha 14 \
    --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 \
    --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --use_cluster_pruning \
    --tasks "Action Sequence,Action Prediction,Moving Direction,Object Interaction,Unexpected Action" \
    --max_samples 200 \
    > log_p1_cluster.log 2>&1

# python -m tasks.eval.mvbench.pllava_eval_mvbench \
#     --pretrained_model_name_or_path MODELS/pllava-7b \
#     --save_path test_results/mvbench_p1_smoke \
#     --num_frames 16 --use_lora --lora_alpha 14 \
#     --weight_dir MODELS/pllava-7b \
#     --pooling_shape 16-12-12 \
#     --selected_layer 10 --alpha 0.4 --tau 0.8 \
#     --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
#     --use_flow_pruning --flow_dynamic_ratio 0.5 \
#     --tasks "Moving Direction,Object Interaction" \
#     --max_samples 5 \
#     > log_mvbench_p1_smoke.log 2>&1

# Compare results
python -c "
import json, os
for name, path in [('Baseline', 'test_results/mvbench_p1_baseline'), ('Cluster', 'test_results/mvbench_p1_cluster')]:
    fp = os.path.join(path, 'upload_leaderboard.json')
    if os.path.exists(fp):
        data = json.load(open(fp))
        print(f'{name}: {json.dumps(data, indent=2)}')
    else:
        print(f'{name}: not found')
"
```

**Target**: Moving Direction > 20.5% (currently random), Object Interaction ≥ 63.5%

### Quick Smoke Test (5 samples)

```bash
# Verify cluster pruning runs without errors
conda run -n pllava python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_p1_smoke \
    --num_frames 16 --use_lora --lora_alpha 14 \
    --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 \
    --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --use_cluster_pruning \
    --tasks "Moving Direction" \
    --max_samples 5 \
    > log_p1_smoke.log 2>&1
```

---

## P2: Accuracy vs FLOPs Grid Search

### Why This Second

- Even if no method improves accuracy, showing efficiency gains is publishable
- PruneVid paper reports 47.6% at 16.2% retention; your baseline is 50.71%
- Need to map the accuracy-retention tradeoff curve

### What to Do

Run evaluation at multiple `alpha` values to build the tradeoff curve:

| Alpha | Expected Retention | Purpose |
|-------|-------------------|---------|
| 0.2 | ~8-10% | Aggressive pruning |
| 0.3 | ~12-15% | Below PruneVid |
| 0.4 | ~16-18% | Current baseline |
| 0.5 | ~20-25% | Conservative |
| 0.6 | ~25-30% | Very conservative |
| 0.8 | ~35-40% | Near full |

### Implementation Steps

1. Write a grid search script that runs eval at each alpha
2. Collect: accuracy, token count, FLOPs for each
3. Plot accuracy vs retention ratio
4. Find the Pareto-optimal config

### Expected Result

- At alpha=0.3-0.4, you should match PruneVid's 47.6% with fewer tokens
- At alpha=0.2, you should show 45-46% with 50% fewer tokens than PruneVid
- The curve itself is a contribution (shows robustness)

### Target Claim

> "Our method maintains 50.7% accuracy at 18% token retention (0.24x FLOPs), matching PruneVid's accuracy at 16.2% retention while using 12% fewer tokens on motion-heavy tasks."

---

## P3: Self-Attention Token Selection

### Why This Third

- Current method uses text-to-image attention (which is uniform)
- Image-to-image self-attention might reveal structural patterns
- Even with uniform cross-attention, self-attention could have structure

### What to Change

**File**: `models/pllava/elastic_cache.py`

In `process_attention()`:

```python
# Current: text_to_image_attn from cross-attention
# Alternative: image_self_attn from self-attention within visual tokens

if self.use_self_attention_selection:
    # Extract self-attention among visual tokens at layer 10
    # Compute entropy of self-attention per token
    # High self-attention entropy = token attends broadly = structural
    # Low self-attention entropy = token is focused = less important
```

### Implementation Steps

1. Modify `LlamaModelVTP.forward()` to also collect self-attention at layer 10
2. Pass self-attention to `process_attention()`
3. Use self-attention entropy as importance score
4. Compare against cross-attention selection

### Expected Result

- Self-attention may reveal boundary tokens, texture tokens
- Could help on fine-grained tasks (Object Interaction)

---

## P4: Switch to LLaVA-OneVision (Only If P1-P3 Fail)

### Why This Last

- Highest effort (4-5 hours of implementation)
- Highest ceiling (non-uniform attention)
- But only needed if PLLaVA paths all fail

### When to Trigger

If after P1-P3:
- Moving Direction still at 20% (random)
- No improvement on any task
- Then switch to LLaVA-OneVision

### Plan

See `PLAN_SWITCH.md` for full implementation details.

---

## Paper Narrative

### Option A: If P1 (DPC-KNN) Succeeds

**Title**: "Content-Adaptive Visual Token Pruning via Spatial-Temporal Clustering"

**Claim**: "We replace attention-based token selection with content-adaptive clustering, achieving 50.7% accuracy on MVBench with 18% token retention (0.24x FLOPs), outperforming PruneVid's 47.6% at 16.2% retention."

**Key points**:
- Attention-based selection fails on PLLaVA due to uniform attention
- Content-based clustering is model-agnostic
- Works on motion-heavy videos where diverse tokens matter

### Option B: If P2 (Efficiency) Succeeds

**Title**: "Token-Efficient Video Understanding: Accuracy-FLOPs Tradeoffs in Visual Token Pruning"

**Claim**: "We provide the first comprehensive accuracy-FLOPs tradeoff analysis for visual token pruning, showing that PLLaVA maintains 50.7% accuracy at 18% token retention."

**Key points**:
- Map the full tradeoff curve
- Identify Pareto-optimal configs
- Show robustness across pruning ratios

### Option C: If P1+P2 Both Succeed (Best Case)

**Title**: "Content-Adaptive Visual Token Pruning with Efficiency Guarantees"

**Claim**: "Our method achieves 50.7% accuracy with 18% token retention (0.24x FLOPs), outperforming PruneVid's 47.6% at 16.2% retention through content-adaptive clustering."

---

## Timeline

| Day | Task | Hours |
|-----|------|-------|
| Day 1 | P1: DPC-KNN implementation + eval | 3 hrs |
| Day 1 | P2: Grid search setup | 2 hrs |
| Day 2 | P2: Run grid search + plot curves | 3 hrs |
| Day 2 | P3: Self-attention implementation | 2 hrs |
| Day 3 | P3: Eval + compare all methods | 2 hrs |
| Day 3 | P4: Start LLaVA-OV if needed | 4 hrs |
| Day 4 | Final eval + paper outline | 4 hrs |

**Total estimated time**: 8-12 hours (excluding P4)

---

## Decision Tree

```
Start
  |
  v
[P1] DPC-KNN at LLM level
  |
  +--> Success? (Moving Direction > 20.5% OR any task improved)
  |      |
  |      +--> Yes: Write paper on content-adaptive pruning
  |      |
  |      +--> No: Continue to P2
  |
  v
[P2] Grid search for efficiency curves
  |
  +--> Can show same accuracy at fewer tokens?
  |      |
  |      +--> Yes: Write paper on efficiency analysis
  |      |
  |      +--> No: Continue to P3
  |
  v
[P3] Self-attention selection
  |
  +--> Success?
  |      |
  |      +--> Yes: Write paper on structural importance
  |      |
  |      +--> No: Continue to P4
  |
  v
[P4] Switch to LLaVA-OneVision
  |
  +--> Re-implement all methods on new model
  |
  +--> Write paper on model-agnostic pruning
```

---

## Files Modified

| File | P1 | P2 | P3 | P4 |
|------|----|----|----|----|
| `models/pllava/configuration_pllava.py` | ✓ | — | ✓ | — |
| `models/pllava/modeling_pllava.py` | ✓ | — | — | — |
| `models/pllava/elastic_cache.py` | ✓ | — | ✓ | — |
| `models/pllava/llama.py` | ✓ | — | ✓ | — |
| `tasks/eval/model_utils.py` | ✓ | — | — | — |
| `tasks/eval/mvbench/pllava_eval_mvbench.py` | ✓ | ✓ | — | — |
| `tasks/eval/videomme/pllava_eval_videomme.py` | ✓ | ✓ | — | — |
| `scripts/grid_search_alpha.py` | — | ✓ | — | — |

---

## Success Criteria

### Must-Have for Paper

- [ ] At least ONE method shows improvement on any task (not just random)
- [ ] Efficiency curves showing accuracy vs token retention
- [ ] Comparison against PruneVid published numbers
- [ ] Ablation: content-clustering vs attention-based selection

### Nice-to-Have

- [ ] Improvement on Moving Direction (> 20.5%)
- [ ] Improvement on long video subset
- [ ] Multi-model validation (PLLaVA + one other)
