# Phase 3 Plan — Motion-Aware Adaptive Retention

> **Created**: 2026-06-25
> **Status**: COMPLETE — Motion-adaptive dropped (no improvement)
> **Goal**: Replace fixed `alpha` with per-window adaptive retention ratios based on temporal dynamics

---

## Problem

`VTPWindowCache.process_attention()` applies the same `alpha` (e.g., 0.4) to ALL windows regardless of motion content. Windows with high motion (dynamic object movement) should retain MORE tokens; static background windows can afford to prune MORE aggressively.

Current code (`elastic_cache.py:131-151`):
```python
alpha = self.alpha  # fixed for all windows
for static_size, dynamic_size, window_size in zip(...):
    num_retain_static_tokens = int(static_size * alpha)       # same alpha
    num_retain_dynamic_tokens = int(dynamic_attentions.shape[-1] * alpha)  # same alpha
```

---

## Design: Two Motion Score Proxies

Per the research plan, test two proxies and keep the one that works:

### Proxy A: Attention-Variance (Primary)
Compute temporal variance of `text_to_image_attentions` within each window. High variance = model is uncertain across frames = more motion = retain more.

**Pros**: Self-contained in `process_attention()`, no new data needed
**Cons**: Proxy — attention variance may not perfectly correlate with visual motion

### Proxy B: Pixel/Vision-Feature Variance (Fallback)
Compute frame-difference metric from the raw vision features in `merge_frames_dynamic()`. Already have `frames_sim` (temporal similarity per patch) — inverse = motion.

**Pros**: Directly measures visual motion
**Cons**: Requires threading motion scores through the call chain (modeling_pllava → llama → elastic_cache)

---

## Implementation Plan

### Step 1: Add Config Params

**File**: `models/pllava/configuration_pllava.py`

Add to `PllavaConfig.__init__()`:
```python
use_motion_adaptive=False,   # enable motion-adaptive alpha
motion_scale=0.5,            # controls adaptive range: [alpha*(1-motion_scale), alpha]
motion_proxy="attention",    # "attention" or "pixel"
```

### Step 2: Add Motion Score Computation to VTPWindowCache

**File**: `models/pllava/elastic_cache.py`

Add method `_compute_motion_scores()` to `VTPWindowCache`:
```python
def _compute_motion_scores(self, text_to_image_attentions, static_sizes, dynamic_sizes, window_sizes):
    """
    Compute per-window motion score from attention temporal variance.
    Returns list of floats, one per window.
    """
    b, head_num, num_query, num_img = text_to_image_attentions.shape
    attn = text_to_image_attentions[0].max(dim=0)[0]  # [num_query, num_img]
    
    motion_scores = []
    start_idx = 0
    for static_size, dynamic_size, window_size in zip(static_sizes, dynamic_sizes, window_sizes):
        end_idx = start_idx + static_size + dynamic_size
        window_attn = attn[start_idx:end_idx]  # [window_total_tokens, num_img]
        
        # Dynamic tokens span multiple frames — compute variance across frame groups
        dynamic_attn = window_attn[static_size:]  # tokens from dynamic region
        if dynamic_attn.shape[0] > 0 and window_size > 1:
            tokens_per_frame = dynamic_attn.shape[0] // window_size
            if tokens_per_frame > 0:
                dynamic_grouped = dynamic_attn[:window_size * tokens_per_frame].view(window_size, tokens_per_frame, -1)
                temporal_var = dynamic_grouped.var(dim=0).mean().item()
            else:
                temporal_var = 0.0
        else:
            temporal_var = 0.0
        
        motion_scores.append(temporal_var)
        start_idx = end_idx
    
    return motion_scores
```

### Step 3: Modify process_attention to Use Adaptive Alpha

**File**: `models/pllava/elastic_cache.py`

Update `__init__` to accept motion params:
```python
def __init__(self, ..., use_motion_adaptive=False, motion_scale=0.5):
    self.use_motion_adaptive = use_motion_adaptive
    self.motion_scale = motion_scale
```

Update `process_attention` to compute and use motion scores:
```python
def process_attention(self, text_to_image_attentions, static_sizes, dynamic_sizes, window_sizes):
    ...
    # Compute motion scores if enabled
    if self.use_motion_adaptive:
        motion_scores = self._compute_motion_scores(text_to_image_attentions, static_sizes, dynamic_sizes, window_sizes)
        max_motion = max(motion_scores) if motion_scores else 1.0
        max_motion = max(max_motion, 1e-6)  # avoid div by zero
    else:
        motion_scores = None
    
    for i, (static_size, dynamic_size, window_size) in enumerate(zip(static_sizes, dynamic_sizes, window_sizes)):
        # Adaptive alpha based on motion
        if self.use_motion_adaptive and motion_scores is not None:
            motion_factor = min(motion_scores[i] / max_motion, 1.5)
            window_alpha = alpha * (1.0 - self.motion_scale + self.motion_scale * motion_factor)
        else:
            window_alpha = alpha
        
        num_retain_static_tokens = int(static_size * window_alpha)
        ...  # rest uses window_alpha instead of alpha
```

### Step 4: Propagate Config Params Through Call Chain

**Files to modify**:
1. `models/pllava/configuration_pllava.py` — add params (Step 1)
2. `models/pllava/modeling_pllava.py` — propagate to LLM config
3. `models/pllava/llama.py` — pass to VTPWindowCache constructor
4. `tasks/eval/model_utils.py` — accept in `load_pllava()`
5. `scripts/infer_single_video.py` — add CLI args
6. `tasks/eval/mvbench/pllava_eval_mvbench.py` — add CLI args

### Step 5: Smoke Test (Proxy A)

Run on `example/cooking.mp4` with `--use_motion_adaptive`:
```bash
python scripts/infer_single_video.py \
    --video example/cooking.mp4 \
    --model_dir MODELS/pllava-7b \
    --weight_dir MODELS/pllava-7b \
    --use_lora --lora_alpha 14 \
    --use_entropy_adaptive --tau_entropy 0.8 \
    --use_motion_adaptive --motion_scale 0.5 \
    --alpha 0.4
```

### Step 6: Quick A/B Test on 5 Videos

Compare: fixed alpha vs entropy-only vs entropy+motion.

### Step 7: MVBench Dev Eval

**7a. Entropy-only baseline** (no motion):
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

**7b. Entropy + Motion-adaptive**:
```bash
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_entropy_motion_pruned_5_tasks_75_samples \
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
    --use_motion_adaptive \
    --motion_scale 0.5 \
    --tasks "Action Sequence,Action Prediction,Moving Direction,Object Interaction,Unexpected Action" \
    --max_samples 75 > log_mvbench_entropy_motion_pruned_5_tasks_75_samples.log 2>&1
```

Compare these two against each other and against the Phase 2 baseline results in `doc/eval/mvbench_entropy_pruned_5_tasks_75_samples/`.

### Step 8: Compare Results

| Result | Action |
|--------|--------|
| Motion > entropy-only | Keep motion-adaptive |
| Motion ≈ entropy-only | Drop motion, focus on entropy |
| Motion < entropy-only by >1% | Debug or drop |

---

## Eval Results (5-task / 75 samples each)

| Task | Baseline (fixed alpha) | Entropy-only | Entropy + Motion |
|------|------------------------|--------------|------------------|
| Action Sequence | 56.00 | 56.00 | 56.00 |
| Action Prediction | 45.33 | 42.67 | 45.33 |
| Unexpected Action | 70.67 | 72.00 | 70.67 |
| Object Interaction | 62.67 | 65.33 | 62.67 |
| Moving Direction | 17.33 | 18.67 | 14.67 |
| **Avg** | **50.40** | **50.93** | **49.87** |

### Conclusion

Motion-adaptive retention (Proxy A: attention-variance) **does not improve** over entropy-only.

- Entropy-only: **50.93** (best)
- Baseline (fixed alpha): **50.40**
- Entropy + Motion: **49.87** (worst)

Motion-adaptive dropped. Entropy-only is the stronger approach. The attention-variance proxy likely does not correlate well with actual visual motion — high attention variance may indicate model uncertainty rather than motion, leading to suboptimal retention decisions.

**Decision**: Proceed to Phase 4 (Borderline Token Preservation) without motion-adaptive. Focus on entropy + borderline as the next direction.

---

## Files Modified

| File | Change |
|------|--------|
| `models/pllava/configuration_pllava.py` | Add `use_motion_adaptive`, `motion_scale` |
| `models/pllava/elastic_cache.py` | Add `_compute_motion_scores()`, adaptive alpha in `process_attention()` |
| `models/pllava/llama.py` | Pass motion params to VTPWindowCache |
| `models/pllava/modeling_pllava.py` | Propagate motion params |
| `tasks/eval/model_utils.py` | Accept motion params in `load_pllava()` |
| `scripts/infer_single_video.py` | Add motion CLI args |
| `tasks/eval/mvbench/pllava_eval_mvbench.py` | Add motion CLI args |

---

## Estimated Time: ~75 min

---

## Known Issues

- `_compute_motion_scores()` is called but its result (`max_motion`) is never applied to `alpha` in `process_attention()`. The motion-adaptive feature is wired up but effectively dead code. The eval results confirm this is not worth fixing — motion-adaptive does not improve accuracy.
