# Entropy-Guided Adaptive Pruning — Progress Log

> **Goal**: Replace hardcoded `selected_layer=10` with entropy-triggered dynamic layer selection
> **Target**: Publication-quality implementation with rigorous evaluation
> **Started**: 2026-06-18

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Approach | Single-stage, entropy-guided | Avoids cascading error of progressive pruning |
| Entropy trigger | First layer where H < τ_entropy | Simple, interpretable, adaptive per query |
| Fallback | Use layer 20 if no trigger | Conservative for complex queries |
| Entropy metric | Shannon entropy averaged across text tokens and heads | Stable, robust to head variation |

---

## Implementation Log

### [2026-06-18] Initial Analysis

- Read `doc/PRUNEVID_LIMITATION.md` — problem well-defined
- Explored `llama.py`, `elastic_cache.py`, `modeling_pllava.py`
- Confirmed: pruning hook is layer-agnostic, infrastructure is reusable
- Key finding: `VTPWindowCache.process_attention()` accepts any attention tensor — no modification needed for entropy-based layer selection
- Decision: Implement entropy-guided single-stage first, progressive as future work

### [2026-06-18] Step 1: Configuration

**File**: `models/pllava/configuration_pllava.py`

- [x] Added `use_entropy_adaptive: bool = False` (default off for backward compatibility)
- [x] Added `tau_entropy: float = 0.8` (entropy threshold for trigger)
- [x] Added `entropy_fallback_layer: int = 20` (conservative fallback if no trigger)
- [x] Added `entropy_computation_layers: list = None` (None = all layers)

### [2026-06-18] Step 2: VTPWindowCache Modification

**File**: `models/pllava/elastic_cache.py`

- [x] Modified `VTPWindowCache.__call__()` to accept `dynamic_selected_layer` parameter
- [x] Modified `prompt_prefill()` to use `dynamic_selected_layer` if provided, else fall back to `self.selected_layer`
- [x] KV cache pruning loop now uses the dynamic layer index

### [2026-06-18] Step 3: Entropy Computation + Dynamic Layer Selection

**File**: `models/pllava/llama.py` — `LlamaModelVTP`

- [x] Added `use_entropy_adaptive`, `tau_entropy`, `entropy_fallback_layer`, `entropy_computation_layers` to `__init__`
- [x] Added `_compute_attention_entropy()` method:
  - Extracts text-to-image attention slice from layer attention weights
  - Computes Shannon entropy: `H = -sum(p * log(p))` where `p = softmax(attn_scores)`
  - Normalizes by `log(num_img_tokens)` to get entropy in [0, 1]
  - Returns scalar entropy value
- [x] Modified `forward()`:
  - During prefill with `use_entropy_adaptive=True`: collects attention at every layer in `entropy_computation_layers`
  - Computes entropy at each collected layer
  - Finds first layer where entropy < `tau_entropy` → sets `dynamic_selected_layer`
  - If no trigger found, uses `entropy_fallback_layer`
  - Passes `dynamic_selected_layer` to `VTPWindowCache`
- [x] Backward compatible: when `use_entropy_adaptive=False`, uses original fixed-layer behavior

### [2026-06-18] Step 4: Single-Video Inference Script

**File**: `scripts/infer_single_video.py`

- [x] Added CLI args: `--use_entropy_adaptive`, `--tau_entropy`, `--entropy_fallback_layer`
- [x] Updated `RunResult` dataclass to include `use_entropy_adaptive`, `tau_entropy`, `entropy_fallback_layer`, `selected_layer_actual`, `layer_entropies`
- [x] Updated `run_once()` to extract entropy profile from `LlamaModelVTP.last_layer_entropies` after inference
- [x] Updated `print_human_report()` to display entropy profile with trigger layer marked
- [x] Updated `main()` to propagate entropy params to model config and `LlamaModelVTP`
- [x] Updated summary JSON to include entropy info
- [x] `load_pllava()` in `model_utils.py` updated to accept and pass entropy params to config

### [2026-06-18] Reverted Eval Scripts

- [x] Reverted all eval scripts (mvbench, videomme, egoshcema, vcgbench, videoqabench) to original state
- [x] Eval benchmark work will start after inference validation phase

---

## Architecture: How It Works

```
Input: Video + Query

[Original PruneVid]
  Fixed: Prune at layer 10 (hardcoded)

[Entropy-Adaptive PruneVid]
  For each layer L during prefill:
    1. Compute attention: Q_text × K_visual
    2. Compute entropy: H(L) = -sum(p * log(p))
    3. If H(L) < tau_entropy:
         -> Trigger pruning at layer L
         -> Prune KV cache for layers 0..L
         -> Continue decoding on reduced tokens
         -> BREAK
    4. If no trigger by fallback_layer:
         -> Prune at fallback_layer (conservative)

Result:
  Simple queries (H drops early) → prune at layer 6-8 → save ~40% FLOPs
  Complex queries (H drops late) → prune at layer 16-20 → preserve accuracy
```

---

## Results

| Method | τ_entropy | Avg Layer | Retained | FLOPs | MVBench | VideoMME |
|--------|-----------|-----------|----------|-------|---------|----------|
| Baseline (fixed) | — | 10.0 | 16.2% | 0.23x | 47.6 | 45.3 |
| Entropy-guided | 0.5 | — | — | — | — | — |
| Entropy-guided | 0.8 | — | — | — | — | — |
| Entropy-guided | 1.0 | — | — | — | — | — |
| Entropy-guided | 1.2 | — | — | — | — | — |
| Entropy-guided | 1.5 | — | — | — | — | — |

---

## Key Observations

(To be filled during implementation)

---

## Open Questions

1. Should entropy be computed at all layers or a subset (every 2 layers)?
2. What is the overhead of entropy computation vs FLOPs savings?
3. Does entropy correlate with optimal pruning layer? (Need oracle experiment)
4. How to handle batch inference — different queries may trigger at different layers?

---

## References

- PruneVid paper: https://arxiv.org/abs/2412.16117v1
- Original limitation doc: `doc/PRUNEVID_LIMITATION.md`
- VTP implementation: `models/pllava/llama.py:1288` (LlamaModelVTP)
- Entropy computation: `models/pllava/elastic_cache.py:108` (VTPWindowCache)

---

## Modified Files Summary

| File | Changes |
|------|---------|
| `models/pllava/configuration_pllava.py` | Added 4 new config params (`use_entropy_adaptive`, `tau_entropy`, `entropy_fallback_layer`, `entropy_computation_layers`) |
| `models/pllava/elastic_cache.py` | `VTPWindowCache` accepts `dynamic_selected_layer` parameter |
| `models/pllava/llama.py` | Added `_compute_attention_entropy()` method, modified `forward()` for dynamic layer selection, stores entropy profile on `self` |
| `models/pllava/modeling_pllava.py` | Propagate entropy params to text_config |
| `tasks/eval/model_utils.py` | `load_pllava()` accepts and propagates entropy params to config |
| `scripts/infer_single_video.py` | CLI args, entropy visualization in report, entropy profile capture |

---

## How to Run

### Single-Video Inference (current phase)

**Fixed-layer pruning (baseline):**
```bash
python scripts/infer_single_video.py \
    --video example/cooking.mp4 \
    --model_dir MODELS/pllava-7b \
    --weight_dir MODELS/pllava-7b \
    --use_lora --lora_alpha 14 \
    --selected_layer 10 --alpha 0.4
```

**Entropy-adaptive pruning:**
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

**Skip generation (token stats only):**
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

### Benchmark Evaluation (future phase)
```bash
# Will be added after inference validation
```
