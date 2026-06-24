# Motion-Guided Adaptive Visual Token Pruning — Research Plan

> **Goal**: Extend PruneVid with a lightweight adaptive selection stage that preserves motion-critical and query-critical tokens, matching the same efficiency budget while improving accuracy on motion-heavy and temporally complex videos.
>
> **Target Venue**: ECCV / BMVC / WACV (B-rank conference)
>
> **Timeline**: 3–4 months to submission
>
> **Baseline**: PruneVid (ACL 2025)

---

## 1. Current Situation Assessment

### 1.1 What PruneVid Does

PruneVid is a training-free visual token pruning method for video LLMs. It operates in two stages:

- **Stage 1 (Vision-side)**: DPC-KNN clustering separates tokens into temporal windows, splits static vs dynamic regions using a similarity threshold (τ), and merges spatially similar tokens within each group.
- **Stage 2 (LLM-side)**: At a single fixed LLM layer (default: layer 10), text-to-image attention scores are used to retain the top-α fraction of visual tokens. The KV cache is pruned and subsequent layers operate on the reduced token set.

The result: 2,304 → ~200-320 tokens (~85-90% reduction), ~0.2× FLOPs, with reported accuracy improvements on MVBench, VideoMME, and Egoschema.

### 1.2 Where PruneVid Falls Short

From the paper's own results and codebase analysis:

| Benchmark           | PLLaVA Baseline | PruneVid | Delta | Assessment                             |
| ------------------- | --------------- | -------- | ----- | -------------------------------------- |
| MVBench             | 46.6            | 47.6     | +1.0  | Decent gain                            |
| VideoMME            | 44.4            | 45.3     | +0.9  | Modest gain                            |
| Egoschema (subset)  | 47.8            | 49.0     | +1.2  | Best gain                              |
| Egoschema (fullset) | 42.6            | 42.6     | 0.0   | **No improvement**                     |
| VCG-Bench (avg)     | 2.99            | 2.98     | -0.01 | **No improvement, slight degradation** |

**Key weaknesses identified from code analysis:**

1. **Fixed pruning layer (layer 10)**: The most significant architectural limitation, explicitly acknowledged in `doc/PRUNEVID_LIMITATION.md`. Simple queries resolve attention early (layer 6); complex reasoning needs deeper layers (16+). A fixed layer is structurally suboptimal.

2. **Uniform retention ratio**: The same `alpha` is applied to all tokens across all windows, regardless of whether a region contains critical motion or redundant background. Static and dynamic tokens are treated equally within each window.

3. **Collapsed attention scoring**: The current entropy computation (`_compute_attention_entropy` in `llama.py:1337-1378`) reduces attention via `max(dim=0).max(dim=0)`, collapsing across all heads and all text tokens into a single scalar per image token. This loses inter-head diversity and text-specific attention patterns.

4. **Single massive pruning drop**: One pruning event at one layer removes all marked tokens simultaneously. No opportunity for the model to "change its mind" about which tokens matter as representations become more semantic in deeper layers.

5. **Unweighted token merge**: In `TextPivotMerge_LayerWise` (`llama.py:1656-1661`), pruned tokens are merged into their nearest kept neighbor via simple 50/50 average (`(k_hh_pruned + k_hh_selected) / 2`), discarding the computed similarity score.

### 1.3 Existing Experimental Infrastructure

The codebase already contains significant experimental groundwork:

| Component                     | File                                             | Status                    | Notes                                                           |
| ----------------------------- | ------------------------------------------------ | ------------------------- | --------------------------------------------------------------- |
| Entropy-adaptive config       | `configuration_pllava.py`                        | Implemented               | `use_entropy_adaptive`, `tau_entropy`, `entropy_fallback_layer` |
| Entropy computation           | `llama.py:_compute_attention_entropy`            | Implemented but flawed    | Single-head collapse issue                                      |
| Dynamic layer selection       | `llama.py:LlamaModelVTP.forward`                 | Implemented               | Works, but only tested with broken entropy                      |
| Multi-head pruning strategies | `modify_llama.py`                                | 12 variants implemented   | H2O, PivotKV, WeightedMerge, TextPrior, etc. — mostly unused    |
| Weighted merge                | `modify_llama.py:WeightedMergeKVCache_LayerWise` | Implemented but not wired | Uses similarity-weighted merge instead of 50/50 average         |
| Progressive pruning concept   | `doc/PRUNEVID_LIMITATION.md`                     | Proposed, not implemented | "Approach A: Progressive Multi-Stage Pruning"                   |
| Single-video inference        | `scripts/infer_single_video.py`                  | Working                   | With entropy profile visualization                              |

### 1.4 Compute Resources

- **Local**: 1 GPU (available now)
- **Rented**: 2 GPU server (available on demand)
- **Estimated total compute**: ~2-3 GPU-days for full benchmark sweep across 4 benchmarks × multiple hyperparameter configs

---

## 2. Research Direction

### 2.1 Paper Title (Working)

**"Motion-Guided Adaptive Visual Token Pruning for Efficient Video Understanding"**

Alternative titles:

- "Adaptive Multi-Stage Visual Token Pruning for Video Large Language Models"
- "Query-Aware and Motion-Aware Visual Token Pruning Beyond PruneVid"

### 2.2 Core Thesis

> PruneVid's fixed-layer, uniform-ratio pruning discards motion-critical and query-relevant tokens. We introduce a lightweight adaptive mechanism that (1) uses multi-head attention entropy to decide _when_ to prune, (2) assigns per-region retention ratios based on temporal dynamics, and (3) distributes pruning across multiple layers. This preserves the same FLOPs budget while improving accuracy on motion-heavy and temporally complex videos.

### 2.3 Key Contributions (Target)

1. **Multi-head entropy-guided layer selection**: Replace PruneVid's fixed `selected_layer=10` with a dynamic trigger based on per-head attention entropy. Simple queries trigger early; complex queries trigger late.

2. **Motion-aware adaptive retention**: Replace the fixed `alpha` with per-region adaptive retention ratios. Regions with high temporal variance (motion) receive higher retention; static backgrounds receive lower retention.

3. **Progressive multi-stage pruning**: Distribute the pruning budget across 2-3 LLM layers instead of a single massive drop. This allows the model to gradually refine its visual representation as layers become more semantic.

4. **Empirical validation**: Show improvements over PruneVid on MVBench, VideoMME, Egoschema, and VCG-Bench, with particular gains on motion-heavy and temporally complex subsets.

---

## 3. Implementation Plan

### 3.1 Phase 1: Baseline Reproduction (Weeks 1–2)

**Objective**: Reproduce PruneVid paper results on all 4 benchmarks.

| Task           | Command / Action                                              | Expected Result            |
| -------------- | ------------------------------------------------------------- | -------------------------- |
| MVBench eval   | `python -m tasks.eval.mvbench.pllava_eval_mvbench [args]`     | 47.6 (paper: 47.6)         |
| VideoMME eval  | `python -m tasks.eval.videomme.pllava_eval_videomme [args]`   | 45.3 (paper: 45.3)         |
| Egoschema eval | `python -m tasks.eval.egoshcema.pllava_eval_egoschema [args]` | 49.0 subset / 42.6 fullset |
| VCGBench eval  | `python -m tasks.eval.vcgbench.pllava_eval_vcgbench [args]`   | 2.98 avg                   |

**Prerequisites**:

- Dataset in `DATAS/` (MVBench, VideoMME, Egoschema, VCGBench)
- Model weights in `MODELS/pllava-7b/`
- Dependencies installed (torch, transformers, decord, peft, safetensors, accelerate)
- `mmcv` import handled (commented out or lazy-loaded)

**Checkpoint**: All 4 baselines match paper numbers (±0.3 tolerance).

### 3.2 Phase 2: Multi-Head Entropy Fix (Weeks 3–4)

**Objective**: Fix the broken entropy computation and validate the entropy-adaptive approach.

#### 3.2.1 Changes to `models/pllava/llama.py`

**File**: `LlamaModelVTP._compute_attention_entropy()` (line 1337)

Current (broken):

```python
# Collapses all heads and text tokens into a single scalar
attn_scores = text_to_image_attn[0].max(dim=0)[0].max(dim=0)[0]  # [num_img_tokens]
attn_probs = F.softmax(attn_scores, dim=-1)
entropy = -torch.sum(attn_probs * torch.log(attn_probs))
normalized_entropy = entropy / torch.log(torch.tensor(num_img_tokens))
```

Proposed (multi-head):

```python
# Compute entropy per head, then aggregate
# text_to_image_attn shape: [batch, heads, num_text, num_img]
head_entropies = []
for h in range(text_to_image_attn.shape[1]):
    # Per-head: max across text tokens → [num_img_tokens]
    head_attn = text_to_image_attn[0, h].max(dim=0)[0]
    head_probs = F.softmax(head_attn, dim=-1)
    head_entropy = -torch.sum(head_probs * torch.log(head_probs + 1e-10))
    head_entropies.append(head_entropy)

# Aggregate: use mean across heads (or max for conservative pruning)
entropy = torch.stack(head_entropies).mean()
normalized_entropy = entropy / torch.log(torch.tensor(num_img_tokens))
```

#### 3.2.2 Validate entropy-adaptive approach

Run the existing entropy-adaptive infrastructure with the fixed entropy computation:

```bash
python scripts/infer_single_video.py \
    --video example/cooking.mp4 \
    --model_dir MODELS/pllava-7b \
    --weight_dir MODELS/pllava-7b \
    --use_lora --lora_alpha 14 \
    --use_entropy_adaptive \
    --tau_entropy 0.8 \
    --entropy_fallback_layer 20 \
    --alpha 0.4
```

Expected: Entropy profile shows decreasing entropy across layers. Trigger layer varies per query.

**Checkpoint**: Entropy-adaptive pruning runs without errors. Per-layer entropy profiles are reasonable (decreasing trend).

### 3.3 Phase 3: Motion-Aware Adaptive Retention (Weeks 5–6)

**Objective**: Replace fixed `alpha` with per-region adaptive retention ratios.

#### 3.3.1 Changes to `models/pllava/elastic_cache.py`

**File**: `VTPWindowCache.process_attention()` (line 120)

Current behavior:

```python
# Fixed alpha for all tokens in all windows
num_retain_static_tokens = int(static_size * alpha)
num_retain_dynamic_tokens = int(dynamic_size // window_size * alpha)
```

Proposed: Motion-adaptive retention based on temporal variance within each window.

Add a new method to compute motion scores:

```python
def _compute_motion_scores(self, text_to_image_attentions, static_sizes, dynamic_sizes, window_sizes):
    """
    Compute per-window motion score based on temporal variance of attention.
    Higher motion → higher retention ratio.
    """
    # For each window, compute variance of attention across frames
    # Use variance as proxy for temporal dynamics
    motion_scores = []
    start_idx = 0
    for static_size, dynamic_size, window_size in zip(static_sizes, dynamic_sizes, window_sizes):
        end_idx = start_idx + static_size + dynamic_size
        window_attn = text_to_image_attentions[start_idx:end_idx]
        # Dynamic tokens have temporal variance; static tokens don't
        dynamic_attn = window_attn[static_size:].view(window_size, -1)
        temporal_variance = dynamic_attn.var(dim=0).mean()  # scalar
        motion_scores.append(temporal_variance.item())
        start_idx = end_idx
    return motion_scores
```

Modify `process_attention` to use motion-adaptive alpha:

```python
# Instead of fixed alpha, scale alpha by motion score
for i, (static_size, dynamic_size, window_size) in enumerate(...):
    motion_factor = min(motion_scores[i] / max_motion, 1.5)  # normalize
    adaptive_alpha = alpha * (0.5 + 0.5 * motion_factor)  # range: [0.5*alpha, alpha]
    num_retain_static_tokens = int(static_size * adaptive_alpha)
    # ... rest of pruning logic
```

#### 3.3.2 Add motion score to `VTPWindowCache.__init__` and `__call__`

```python
def __init__(self, ..., use_motion_adaptive=False, motion_scale=0.5):
    self.use_motion_adaptive = use_motion_adaptive
    self.motion_scale = motion_scale

def __call__(self, ..., motion_scores=None):
    if self.use_motion_adaptive and motion_scores is not None:
        # Use motion-adaptive alpha
        ...
```

**Checkpoint**: Motion-adaptive retention runs. Dynamic regions with high temporal variance retain more tokens than static backgrounds.

### 3.4 Phase 4: Progressive Multi-Stage Pruning (Weeks 7–8)

**Objective**: Distribute pruning across 2-3 LLM layers instead of one.

#### 3.4.1 Changes to `models/pllava/llama.py`

**File**: `LlamaModelVTP.forward()` (line 1380)

Current behavior: Pruning happens once at `self.selected_layer` (or `dynamic_selected_layer`).

Proposed: Prune at multiple layers with progressively smaller retention ratios.

Add configuration:

```python
# In __init__
self.progressive_pruning = getattr(config, 'progressive_pruning', False)
self.progressive_layers = getattr(config, 'progressive_layers', [6, 12, 18])
self.progressive_alphas = getattr(config, 'progressive_alphas', [0.7, 0.5, 0.3])
```

Modify the forward pass to support multi-stage pruning:

```python
# In forward(), during prefill
if self.progressive_pruning and is_prefill:
    for stage_idx, (prune_layer, stage_alpha) in enumerate(
        zip(self.progressive_layers, self.progressive_alphas)
    ):
        if layer_idx == prune_layer:
            # Set temporary alpha for this stage
            old_alpha = self.cache.alpha
            self.cache.alpha = stage_alpha
            # Run pruning
            past_key_values, hidden_states, ... = self.cache(...)
            self.cache.alpha = old_alpha
```

#### 3.4.2 Changes to `models/pllava/configuration_pllava.py`

Add new config parameters:

```python
progressive_pruning=False,
progressive_layers=[6, 12, 18],
progressive_alphas=[0.7, 0.5, 0.3],
```

**Checkpoint**: Progressive pruning runs. Token count decreases gradually across layers.

### 3.5 Phase 5: Weighted Token Merge (Week 9)

**Objective**: Replace 50/50 average merge with similarity-weighted merge.

#### 3.5.1 Changes to `models/pllava/llama.py`

**File**: `TextPivotMerge_LayerWise.__call__()` (line 1614)

Current (line 1661):

```python
k_hh_merged = (k_hh_pruned + k_hh_selected) / 2
```

Proposed:

```python
# Use already-computed similarity weights
# similarity shape: [batch, heads, num_pruned, num_kept]
# max_values shape: [batch, heads, num_pruned] — cosine similarity to best pivot
merge_weights = max_values.unsqueeze(-1)  # [batch, heads, num_pruned, 1]
k_hh_merged = merge_weights * k_hh_selected + (1 - merge_weights) * k_hh_pruned
```

This preserves more information when the pivot is a good match (high similarity → more weight on kept token) and less when it's a poor match.

**Checkpoint**: Weighted merge runs. Token quality should be higher than naive average.

### 3.6 Phase 6: Full Benchmark Evaluation (Weeks 10–12)

**Objective**: Run complete evaluation across all 4 benchmarks with all improvements.

#### 3.6.1 Experiment Matrix

| Config                     | MVBench | VideoMME | Egoschema | VCGBench |
| -------------------------- | ------- | -------- | --------- | -------- |
| Baseline (PruneVid)        | ✓       | ✓        | ✓         | ✓        |
| + Multi-head entropy       | ✓       | ✓        | ✓         | ✓        |
| + Motion-adaptive alpha    | ✓       | ✓        | ✓         | ✓        |
| + Progressive pruning      | ✓       | ✓        | ✓         | ✓        |
| + Weighted merge           | ✓       | ✓        | ✓         | ✓        |
| Full method (all combined) | ✓       | ✓        | ✓         | ✓        |
| Ablation: entropy only     | ✓       | ✓        | ✓         | —        |
| Ablation: motion only      | ✓       | ✓        | ✓         | —        |
| Ablation: progressive only | ✓       | ✓        | ✓         | —        |

**Total evaluation runs**: ~12 configs × 4 benchmarks = ~48 runs
**Estimated time per run**: 3-6 hours (depending on benchmark size)
**Total compute**: ~12-24 GPU-days (with 2 GPUs: ~6-12 days)

#### 3.6.2 Hyperparameter Search

For each component, sweep key hyperparameters:

| Component        | Hyperparameter           | Range                                         |
| ---------------- | ------------------------ | --------------------------------------------- |
| Entropy-adaptive | `tau_entropy`            | [0.3, 0.5, 0.8, 1.0, 1.2]                     |
| Entropy-adaptive | `entropy_fallback_layer` | [15, 20, 25]                                  |
| Motion-adaptive  | `motion_scale`           | [0.3, 0.5, 0.7]                               |
| Progressive      | `progressive_layers`     | [[6,12,18], [8,16,24], [6,10,14,18,22]]       |
| Progressive      | `progressive_alphas`     | [[0.7,0.5,0.3], [0.8,0.6,0.4], [0.6,0.4,0.2]] |
| All              | `alpha`                  | [0.3, 0.4, 0.5]                               |

#### 3.6.3 Target Results

| Benchmark           | PruneVid | Our Target | Gain       |
| ------------------- | -------- | ---------- | ---------- |
| MVBench             | 47.6     | 48.5–49.5  | +1.0–2.0   |
| VideoMME            | 45.3     | 46.0–47.0  | +0.7–1.7   |
| Egoschema (subset)  | 49.0     | 50.0–51.5  | +1.0–2.5   |
| Egoschema (fullset) | 42.6     | 43.5–45.0  | +0.9–2.4   |
| VCG-Bench (avg)     | 2.98     | 3.05–3.15  | +0.07–0.17 |

**Note**: Even +1.0 on MVBench and +1.0 on Egoschema subset is a solid contribution for B-rank. The fullset and VCGBench improvements would be bonus.

### 3.7 Phase 7: Paper Writing (Weeks 13–16)

#### 3.7.1 Paper Structure

1. **Abstract** (~200 words)
   - Problem: Video LLMs process massive visual token sequences; PruneVid reduces tokens but uses fixed, uniform pruning
   - Method: Adaptive multi-stage pruning with entropy-guided layer selection, motion-aware retention, progressive dropping
   - Results: Improvements on MVBench, VideoMME, Egoschema at same FLOPs budget

2. **Introduction** (~1.5 pages)
   - Video LLMs and the token efficiency problem
   - PruneVid's contributions and limitations
   - Our observation: motion-critical and query-critical tokens need special treatment
   - Our contributions (3 bullet points)

3. **Related Work** (~1.5 pages)
   - Video understanding with LLMs (LLaVA, VideoChat, etc.)
   - Visual token pruning (PruneVid, FastV, Look-M, Prumerge)
   - Adaptive computation (early exits, dynamic networks)
   - KV cache optimization (H2O, SnapKV, StreamingLLM)

4. **Method** (~3 pages)
   - Preliminaries: PruneVid overview
   - Multi-head entropy-guided layer selection
   - Motion-aware adaptive retention
   - Progressive multi-stage pruning
   - Combined framework

5. **Experiments** (~4 pages)
   - Setup: benchmarks, metrics, implementation details
   - Main results: comparison with PruneVid and other baselines
   - Ablation studies: each component independently
   - Analysis: entropy profiles, motion scores, pruning patterns
   - Efficiency analysis: FLOPs, latency comparison

6. **Conclusion** (~0.5 pages)

**Estimated total length**: 12–14 pages (ECCV/BMVC format)

#### 3.7.2 Figures to Create

1. Framework overview diagram (method overview)
2. Entropy profile visualization (per-layer entropy for different query types)
3. Motion score heatmap (per-frame motion scores across a video)
4. Progressive pruning visualization (token count across layers)
5. Pruning pattern comparison (PruneVid vs ours — which tokens are kept)
6. Per-benchmark accuracy breakdown (bar chart)
7. Efficiency-accuracy tradeoff plot (FLOPs vs accuracy)

---

## 4. Minimal Spec vs Recommended Spec

### 4.1 Minimal Spec (Minimum Viable Paper)

| Resource                 | Requirement                         | Notes                                                              |
| ------------------------ | ----------------------------------- | ------------------------------------------------------------------ |
| **GPUs**                 | 1 GPU                               | Baseline eval: ~12 hours. Method eval: ~36 hours. Total: ~48 hours |
| **GPU VRAM**             | ≥24 GB                              | PLLaVA-7B requires ~16GB for inference. 24GB gives headroom        |
| **Storage**              | ~200 GB                             | Datasets (~150 GB) + models (~15 GB) + results (~5 GB)             |
| **Server rental**        | 3-4 days on 2-GPU server            | For final benchmark sweep                                          |
| **Benchmarks**           | 2 of 4 (MVBench + Egoschema)        | Minimum for a convincing paper                                     |
| **Hyperparameter sweep** | 3-5 configs per method              | Minimal ablation                                                   |
| **Implementation**       | Entropy fix + motion-adaptive alpha | 2 of 3 components                                                  |
| **Paper venue**          | BMVC or WACV                        | More forgiving on scope                                            |

**Timeline**: 3 months (aggressive but feasible)

### 4.2 Recommended Spec (Strong Paper)

| Resource                 | Requirement                                       | Notes                                      |
| ------------------------ | ------------------------------------------------- | ------------------------------------------ |
| **GPUs**                 | 2 GPUs                                            | Parallel evaluation, faster sweeps         |
| **GPU VRAM**             | ≥24 GB each                                       | Same as minimal                            |
| **Storage**              | ~500 GB                                           | All 4 datasets + all experiment results    |
| **Server rental**        | 7-10 days on 2-GPU server                         | Full benchmark sweep + ablations           |
| **Benchmarks**           | All 4 (MVBench, VideoMME, Egoschema, VCGBench)    | Comprehensive validation                   |
| **Hyperparameter sweep** | 8-12 configs per method                           | Thorough ablation                          |
| **Implementation**       | All 3 components (entropy + motion + progressive) | Complete method                            |
| **Paper venue**          | ECCV or BMVC                                      | Higher acceptance chance with full results |

**Timeline**: 4 months (comfortable pace)

### 4.3 Compute Budget Breakdown

| Phase     | Task                                      | GPU Hours (est.) | Calendar Days (2 GPUs) |
| --------- | ----------------------------------------- | ---------------- | ---------------------- |
| Phase 1   | Baseline reproduction (4 benchmarks)      | 24 hrs           | 0.5 day                |
| Phase 2   | Entropy fix + validation                  | 12 hrs           | 0.25 day               |
| Phase 3   | Motion-adaptive implementation + eval     | 24 hrs           | 0.5 day                |
| Phase 4   | Progressive pruning implementation + eval | 24 hrs           | 0.5 day                |
| Phase 5   | Weighted merge implementation + eval      | 12 hrs           | 0.25 day               |
| Phase 6   | Full benchmark sweep (all configs)        | 200 hrs          | 4-5 days               |
| Phase 6   | Hyperparameter search                     | 100 hrs          | 2-3 days               |
| **Total** |                                           | **~400 hrs**     | **~8-10 days**         |

---

## 5. Risk Assessment

| Risk                                                 | Likelihood | Impact | Mitigation                                                             |
| ---------------------------------------------------- | ---------- | ------ | ---------------------------------------------------------------------- |
| Improvements don't show significant gains            | Medium     | High   | Frame paper as "analysis + targeted improvement" not "beat everything" |
| Entropy doesn't correlate with optimal pruning layer | Medium     | Medium | Fall back to progressive-only method (still novel)                     |
| Motion scores don't help                             | Low-Medium | Medium | Ablation shows motion-aware is optional component                      |
| Compute budget exceeded                              | Low        | Medium | Start with 2 benchmarks, add more if time permits                      |
| Dataset issues (missing videos, etc.)                | Medium     | Low    | Already handled with skip logic; 3786/4000 MVBench samples work        |
| Paper rejected from target venue                     | Medium     | High   | Prepare backup submission to WACV (later deadline)                     |

---

## 6. Success Criteria

### Must-Have (Minimum publishable result)

- [ ] PruneVid baselines reproduced on at least 2 benchmarks
- [ ] At least 1 improvement component implemented and validated
- [ ] Improvement shows +0.5 or better on at least 1 benchmark
- [ ] Ablation study showing the component's individual contribution
- [ ] Paper draft completed

### Should-Have (Strong paper)

- [ ] PruneVid baselines reproduced on all 4 benchmarks
- [ ] All 3 improvement components implemented
- [ ] Combined method shows +1.0 or better on MVBench and Egoschema
- [ ] Comprehensive ablation (entropy, motion, progressive, weighted merge)
- [ ] Efficiency analysis (FLOPs, latency comparison)
- [ ] Qualitative analysis (pruning pattern visualization)

### Nice-to-Have (Excellent paper)

- [ ] Analysis of when/why each component helps
- [ ] Oracle experiment (does entropy correlate with optimal layer?)
- [ ] Comparison with other adaptive methods (not just PruneVid)
- [ ] Multiple base models (PLLaVA + LLaVA-OneVision)

---

## 7. File Modification Summary

| File                                        | Phase   | Changes                                                                                                      |
| ------------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------ |
| `models/pllava/llama.py`                    | 2, 4, 5 | Fix `_compute_attention_entropy`, add progressive pruning logic, weighted merge                              |
| `models/pllava/elastic_cache.py`            | 3       | Add motion-adaptive alpha to `VTPWindowCache`                                                                |
| `models/pllava/configuration_pllava.py`     | 3, 4    | Add `use_motion_adaptive`, `motion_scale`, `progressive_pruning`, `progressive_layers`, `progressive_alphas` |
| `models/pllava/modeling_pllava.py`          | 3, 4    | Propagate new config params to text_config                                                                   |
| `tasks/eval/model_utils.py`                 | 2, 3, 4 | Accept and propagate new params in `load_pllava()`                                                           |
| `scripts/infer_single_video.py`             | 2, 3    | Add CLI args for new features, visualization                                                                 |
| `tasks/eval/mvbench/__init__.py`            | 1       | Already fixed (skip missing videos, save_results bug)                                                        |
| `tasks/eval/mvbench/pllava_eval_mvbench.py` | 1       | Already fixed (skip None examples)                                                                           |

---

## 8. References

1. PruneVid paper: https://arxiv.org/abs/2412.16117v1
2. PLLaVA: https://github.com/magic-research/PLLaVA
3. H2O (Heavy-Hitter Oracle): https://arxiv.org/abs/2306.14048
4. SnapKV: https://arxiv.org/abs/2404.14469
5. FastV: https://arxiv.org/abs/2402.03308
6. Look-M: https://arxiv.org/abs/2403.20133
7. MVBench: https://arxiv.org/abs/2311.17005
8. Video-MME: https://arxiv.org/abs/2405.21075
9. EgoSchema: https://arxiv.org/abs/2308.09126

---

_Document created: 2026-06-23_
_Status: Active plan — awaiting baseline evaluation completion_
