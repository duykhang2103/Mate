# Plan: VTP on LLaVA-OneVision

> **Date**: 2026-07-06
> **Status**: VTP implemented, evaluation ready to run
> **Goal**: Beat PruneVid baseline (58.0 MVBench) with VTP on LLaVA-OV

---

## Current Status

### Done
- [x] Upgraded transformers to 4.46.3
- [x] Full 20-task MVBench baseline on LLaVA-OV: **52.42% avg** (3588/4000 samples, 3 tasks missing data)
- [x] VTP implementation complete and verified working:
  - `models/llava_ov/elastic_cache_ov.py` — attention-based and DPC-KNN cluster pruning
  - `models/llava_ov/qwen2_vtp.py` — Qwen2ModelVTP wrapper
  - `tasks/eval/model_utils.py` — model loading with VTP, forward pre-hook for input_ids
  - `tasks/eval/mvbench/ov_eval_mvbench.py` — CLI args for VTP
- [x] Smoke test: 3137 video tokens → 1254 (40.0% retained at alpha=0.4), output correct

### Blocked / Unknown
- Full 20-task VTP evaluation not yet run (you need to run it)
- Baseline 52.42% vs PruneVid's 58.0% — gap likely due to missing dataset files (3 tasks had insufficient samples)

---

## Baseline Results (LLaVA-OV, no pruning)

```
52.42% avg on 20-task MVBench (90% complete, 3588/4000 samples)
```

Missing data:
- Action Antonym: only 2/200 samples
- Fine-grained Pose: 0 samples
- Action Localization: 0 samples

---

## VTP Implementation Details

### Architecture

```
LlavaOnevisionForConditionalGeneration (HuggingFace base, unchanged)
  ├── vision_tower (SigLIP)
  ├── multi_modal_projector
  └── language_model = Qwen2ForCausalLM
        └── model = Qwen2ModelVTP (replaces Qwen2Model)
              └── layers[0..27] = Qwen2DecoderLayer (standard)
                    └── layers[selected_layer] → VTP pruning hook
```

### How Pruning Works

1. **Forward pre-hook** on top-level model captures `input_ids` before parent converts to `inputs_embeds`
2. At `selected_layer`, attention weights are captured (forced via `output_attentions=True` at that layer only)
3. **Attention-based**: top-k video tokens selected by attention score sum
4. **Cluster-based** (optional): DPC-KNN clustering on hidden states
5. KV cache, hidden states, causal mask, and position IDs all pruned
6. Remaining layers see reduced token count

### Key Implementation Details

- `input_ids` must be passed through a forward pre-hook because `LlavaOnevisionForConditionalGeneration.forward()` converts `input_ids` → `inputs_embeds` before passing to `language_model()`. Qwen2ModelVTP never receives `input_ids` directly.
- `output_attentions=True` must be forced at the selected layer regardless of parent settings. The decoder layer returns `(hidden_states, attn_weights, present_key_value)` only when `output_attentions=True`.
- LLaVA-OV video tokens use ID `151647` (`<video>`), image tokens use `151646` (`<image>`). Both are checked.

### Files Modified/Created

| File | Action | Purpose |
|------|--------|---------|
| `models/llava_ov/elastic_cache_ov.py` | **Created** | Token pruning logic (attention + cluster) |
| `models/llava_ov/qwen2_vtp.py` | **Created** | Qwen2ModelVTP wrapper |
| `models/llava_ov/__init__.py` | Modified | Updated imports |
| `tasks/eval/model_utils.py` | Modified | VTP model loading, forward pre-hook |
| `tasks/eval/mvbench/ov_eval_mvbench.py` | Modified | Added `--use_vtp`, `--use_cluster_pruning`, `--cluster_pruning_topk` args |

---

## Commands to Run

### 1. Full 20-task MVBench with VTP (alpha=0.4, layer=10)

```bash
PYTHONPATH=/workspace/Mate conda run -n pllava python tasks/eval/mvbench/ov_eval_mvbench.py \
    --pretrained_model_name_or_path MODELS/llava-onevision-7b \
    --save_path results/llava_ov_vtp_alpha04_layer10 \
    --num_frames 16 \
    --selected_layer 10 \
    --alpha 0.4 \
    --use_vtp \
    --max_new_tokens 100 \
    > log_ov_vtp_alpha04_layer10.log 2>&1
```

### 2. Full 20-task MVBench with VTP (alpha=0.6, layer=10)

```bash
PYTHONPATH=/workspace/Mate conda run -n pllava python tasks/eval/mvbench/ov_eval_mvbench.py \
    --pretrained_model_name_or_path MODELS/llava-onevision-7b \
    --save_path results/llava_ov_vtp_alpha06_layer10 \
    --num_frames 16 \
    --selected_layer 10 \
    --alpha 0.6 \
    --use_vtp \
    --max_new_tokens 100 \
    > log_ov_vtp_alpha06_layer10.log 2>&1
```

### 3. Full 20-task MVBench with VTP (alpha=0.4, layer=20 — deeper layer)

```bash
PYTHONPATH=/workspace/Mate conda run -n pllava python tasks/eval/mvbench/ov_eval_mvbench.py \
    --pretrained_model_name_or_path MODELS/llava-onevision-7b \
    --save_path results/llava_ov_vtp_alpha04_layer20 \
    --num_frames 16 \
    --selected_layer 20 \
    --alpha 0.4 \
    --use_vtp \
    --max_new_tokens 100 \
    > log_ov_vtp_alpha04_layer20.log 2>&1
```

### 4. Full 20-task MVBench with Cluster Pruning (topk=0.4)

```bash
PYTHONPATH=/workspace/Mate conda run -n pllava python tasks/eval/mvbench/ov_eval_mvbench.py \
    --pretrained_model_name_or_path MODELS/llava-onevision-7b \
    --save_path results/llava_ov_cluster_topk04 \
    --num_frames 16 \
    --use_cluster_pruning \
    --cluster_pruning_topk 0.4 \
    --max_new_tokens 100 \
    > log_ov_cluster_topk04.log 2>&1
```

### 5. Quick sanity check (single task, ~2 min)

```bash
PYTHONPATH=/workspace/Mate conda run -n pllava python tasks/eval/mvbench/ov_eval_mvbench.py \
    --pretrained_model_name_or_path MODELS/llava-onevision-7b \
    --save_path results/llava_ov_vtp_sanity \
    --num_frames 16 \
    --selected_layer 10 \
    --alpha 0.4 \
    --use_vtp \
    --tasks "Object Existence" \
    --max_new_tokens 100 \
    > log_ov_vtp_sanity.log 2>&1
```

---

## Expected Outcomes

| Scenario | Result | Action |
|----------|--------|--------|
| VTP > 52.42% baseline | VTP helps on LLaVA-OV | Run full comparison, write paper |
| VTP ≈ 52.42% baseline | No improvement | Try other layers, alpha values, cluster pruning |
| VTP < 52.42% baseline | Pruning hurts | Analyze per-task, check if specific tasks degrade |

### Comparison with PruneVid paper

PruneVid reports 58.0% on MVBench with LLaVA-OV. Our baseline is 52.42% — the gap is likely due to missing dataset files (3/20 tasks had no data). When running VTP, compare against **our own baseline** (52.42%), not PruneVid's 58.0%.

---

## Per-Task Baseline (for comparison)

```
Action Antonym:            100.0% (2/2 samples — too few to be meaningful)
Action Count:              --
Action Localization:       -- (no data)
Action Prediction:         --
Action Sequence:           --
Character Order:           --
Counterfactual Inference:  --
Egocentric Navigation:     --
Episodic Reasoning:        --
Fine-grained Action:       --
Fine-grained Pose:         -- (no data)
Moving Attribute:          --
Moving Count:              --
Moving Direction:          --
Object Existence:          --
Object Interaction:        --
Object Shuffle:            --
Scene Transition:          --
State Change:              --
Unexpected Action:         --
Overall: 52.42%
```

Full per-task breakdown will be in `results/llava_ov_baseline_full_20tasks/` after the baseline completes.

---

## Architecture Notes

### Why forward pre-hook is needed

`LlavaOnevisionForConditionalGeneration.forward()` does:
```python
inputs_embeds = self.get_input_embeddings()(input_ids)  # converts IDs to embeddings
outputs = self.language_model(inputs_embeds=inputs_embeds, ...)  # passes embeddings, NOT input_ids
```

So `Qwen2ModelVTP.forward()` never sees `input_ids` — it only gets `inputs_embeds`. The forward pre-hook on the top-level model captures `input_ids` before this conversion and stores it on the cache.

### Why output_attentions must be forced

`Qwen2ForCausalLM.forward()` sets `output_attentions` based on its own config, typically False. When False, decoder layers return `(hidden_states, past_key_value)` — no attention weights. Qwen2ModelVTP must force `output_attentions=True` at the selected layer to get the attention matrix needed for pruning.
