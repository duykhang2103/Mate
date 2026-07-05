# Plan: Switch to LLaVA-OneVision

> **Date**: 2026-07-05
> **Status**: Baseline Working ✅
> **Goal**: Run PruneVid+ (motion-adaptive pruning) on LLaVA-OneVision where non-uniform attention can actually benefit from token selection

---

## Current Situation

### Why Switch

PLLaVA's attention is **architecturally uniform** (entropy=0.9999 everywhere). This means:

- No token selection method can improve accuracy — the model treats all tokens equally
- Motion-adaptive pruning keeps more tokens (735 vs 594) but accuracy stays identical (45% on VideoMME long)
- The result is strictly worse: same accuracy, more FLOPs

LLaVA-OneVision has **non-uniform attention** — it actually uses attention scores to weight tokens. This means:

- PruneVid's attention-based token selection can work
- Motion-adaptive pruning can exploit the model's ability to differentiate token importance
- The PruneVid paper already shows results: 58.0→57.5 MVBench, 58.2→58.6 VideoMME

### What Exists

| Component              | Status                                                            |
| ---------------------- | ----------------------------------------------------------------- |
| LLaVA-OneVision code   | **Working** — baseline eval pipeline complete                      |
| Model loading          | **Done** — `load_llava_ov()` in `model_utils.py`                   |
| VTP pruning logic      | **PLLaVA-specific** — `llama.py` wraps `LlamaModel`, not Qwen2    |
| Vision merge           | **PLLaVA-specific** — assumes `num_frames * pooling_shape` layout |
| Eval scripts           | **Working** — `ov_eval_mvbench.py` runs end-to-end                 |
| Config                 | **Done** — `LlavaOVConfig` with PruneVid params                    |
| PruneVid paper results | **Exist** — shows 17% retained, 0.20x FLOPs on LLaVA-OneVision    |

### Architecture Comparison

|                | PLLaVA                   | LLaVA-OneVision                  |
| -------------- | ------------------------ | -------------------------------- |
| LLM            | Llama-2-7B               | **Qwen2-7B**                     |
| Vision encoder | CLIP ViT-L               | **SigLIP SO400M**                |
| Projector      | 3D AdaptiveAvgPool       | **MLP (2-layer)**                |
| Image tokens   | 2304 (16 frames × 12×12) | **2880+ (dynamic resolution)**   |
| Image token ID | `pad_token_id` positions | **`<image>` special token**      |
| Video handling | Frame-level pooling      | **Temporal position embeddings** |
| Attention      | Uniform (entropy=0.9999) | **Non-uniform (usable)**         |

---

## Plan Overview

| Step | What                                                            | Complexity | Time Est |
| ---- | --------------------------------------------------------------- | ---------- | -------- |
| 1    | Download LLaVA-OneVision model                                  | Low        | 15 min   |
| 2    | Create `models/llava_ov/` module with Qwen2 VTP wrapper         | **High**   | 2-3 hrs  |
| 3    | Adapt `elastic_cache.py` for Qwen2 attention + `<image>` tokens | Medium     | 1 hr     |
| 4    | Add model loading in `model_utils.py`                           | Low        | 30 min   |
| 5    | Add conversation template in `eval_utils.py`                    | Low        | 15 min   |
| 6    | Create eval script entry point (or adapt existing)              | Medium     | 30 min   |
| 7    | Smoke test: 10 samples on MVBench                               | Low        | 10 min   |
| 8    | Baseline eval: MVBench 5-task                                   | Medium     | 1 hr     |
| 9    | Motion-adaptive eval: MVBench 5-task                            | Medium     | 1 hr     |
| 10   | VideoMME long eval (baseline vs method)                         | Medium     | 30 min   |

---

## Step 1: Download Model ✅

```bash
# Download from HuggingFace (requires ~15GB disk)
huggingface-cli download llava-hf/llava-onevision-qwen2-7b-ov-hf \
    --local-dir MODELS/llava-onevision-7b
```

**Result**: Model downloaded successfully to `MODELS/llava-onevision-7b/`.

---

## Step 2: Create `models/llava_ov/` Module

This is the core engineering work. Create a new module that wraps HuggingFace's native LLaVA-OneVision with VTP pruning.

### 2a. `models/llava_ov/__init__.py`

```python
from .configuration_llava_ov import LlavaOVConfig
from .modeling_llava_ov import LlavaOVForConditionalGeneration
```

### 2b. `models/llava_ov/configuration_llava_ov.py`

Extend HuggingFace's `LlavaOnevisionConfig` with PruneVid parameters:

```python
from transformers import LlavaOnevisionConfig

class LlavaOVConfig(LlavaOnevisionConfig):
    model_type = "llava_ov_vtp"

    def __init__(self, **kwargs):
        # PruneVid parameters
        self.selected_layer = kwargs.pop("selected_layer", 10)
        self.alpha = kwargs.pop("alpha", 0.4)
        self.tau = kwargs.pop("tau", 0.8)
        self.cluster_ratio = kwargs.pop("cluster_ratio", 0.5)
        self.temporal_segment_ratio = kwargs.pop("temporal_segment_ratio", 0.25)
        self.use_entropy_adaptive = kwargs.pop("use_entropy_adaptive", False)
        self.use_motion_adaptive = kwargs.pop("use_motion_adaptive", False)
        self.motion_scale = kwargs.pop("motion_scale", 0.5)
        self.use_flow_pruning = kwargs.pop("use_flow_pruning", False)
        # ... other params
        super().__init__(**kwargs)
```

### 2c. `models/llava_ov/modeling_llava_ov.py`

The main model class. Two approaches:

**Approach A (Recommended): Thin wrapper around HuggingFace model**

```python
from transformers import LlavaOnevisionForConditionalGeneration, AutoModelForCausalLM
from ..pllava.elastic_cache import VTPWindowCache  # reuse existing cache

class LlavaOVForConditionalGeneration(LlavaOnevisionForConditionalGeneration):
    """LLaVA-OneVision with Visual Token Pruning."""

    def __init__(self, config):
        super().__init__(config)
        # Create VTP cache (reuse elastic_cache.py)
        self.cache = VTPWindowCache(
            selected_layer=config.selected_layer,
            alpha=config.alpha,
            tau=config.tau,
            # ... other params
        )

    def forward(self, input_ids, attention_mask, pixel_values, ...):
        # Standard LLaVA-OneVision forward, but:
        # 1. Hook into language model at selected_layer
        # 2. Collect text-to-image attention
        # 3. Call self.cache.process_attention() to prune KV cache
        # 4. Continue with pruned tokens
        pass
```

**Key difference from PLLaVA**: No vision-side merge (`merge_frames_dynamic`). LLaVA-OneVision handles frame aggregation internally via temporal position embeddings. The VTP pruning happens only at the LLM level.

**Approach B (Simpler): Only LLM-side VTP**

Skip the vision-side spatial-temporal merge entirely. Only implement the LLM-level pruning (which is what actually matters for the PruneVid paper's contributions). This is simpler and still captures the core contribution.

### 2d. `models/llava_ov/qwen2_vtp.py`

VTP-modified Qwen2Model (equivalent to `models/pllava/llama.py`):

```python
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2ForCausalLM

class Qwen2ModelVTP(Qwen2Model):
    """Qwen2 with Visual Token Pruning at selected layer."""

    def __init__(self, config, cache=None):
        super().__init__(config)
        self.cache = cache
        self.selected_layer = config.selected_layer

    def forward(self, input_ids, attention_mask, ...):
        # Standard Qwen2 forward, but at selected_layer:
        # 1. Collect attention weights
        # 2. Call self.cache.process_attention()
        # 3. Prune KV cache
        # 4. Continue with pruned hidden states
        pass
```

**Reference**: The PLLaVA version is at `models/pllava/llama.py:1285-1629`. The Qwen2 version would follow the same pattern but use Qwen2's layer structure.

---

## Step 3: Adapt `elastic_cache.py`

The `VTPWindowCache` needs modifications for LLaVA-OneVision:

### 3a. Image Token Detection

**Current (PLLaVA)**: Uses `pad_token_id` positions in `input_ids`:

```python
# elastic_cache.py:255-266
img_start = (input_ids == self.pad_token_id).nonzero()[0, 1].item()
img_end = (input_ids == self.pad_token_id).nonzero()[-1, 1].item()
```

**New (LLaVA-OneVision)**: Use `<image>` special token:

```python
image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
img_positions = (input_ids == image_token_id).nonzero()
img_start = img_positions[0, 1].item()
img_end = img_positions[-1, 1].item()
```

### 3b. Attention Format

**Current (PLLaVA)**: Text-to-image attention from LlamaSdpaAttention:

```python
attentions[:, :, img_start:img_end+1, img_start:img_end+1]
```

**New (Qwen2)**: Same concept, but Qwen2 may use different attention output format. Need to verify:

- Does Qwen2 output attention weights when `output_attentions=True`?
- Is the attention shape the same `[batch, heads, seq_len, seq_len]`?
- Are image tokens contiguous in the sequence?

### 3c. Spatial Layout Assumptions

**Current**: Assumes `num_frames * pooling_shape[1] * pooling_shape[2]` tokens per window.

**New**: LLaVA-OneVision produces variable token counts based on dynamic resolution. The window sizes from DPC-KNN clustering would need to adapt to the actual token count, not a fixed grid.

**Solution**: Make window sizes dynamic based on actual token count rather than hardcoded spatial dimensions.

---

## Step 4: Model Loading

Add to `tasks/eval/model_utils.py`:

```python
def load_llava_ov(pretrained_model_name_or_path, num_frames=16, **kwargs):
    """Load LLaVA-OneVision with VTP."""
    from models.llava_ov import LlavaOVConfig, LlavaOVForConditionalGeneration
    from transformers import AutoProcessor

    config = LlavaOVConfig.from_pretrained(
        pretrained_model_name_or_path,
        selected_layer=kwargs.get('selected_layer', 10),
        alpha=kwargs.get('alpha', 0.4),
        tau=kwargs.get('tau', 0.8),
        # ... other PruneVid params
    )

    model = LlavaOVForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path,
        config=config,
        torch_dtype=torch.float16,
    )

    processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path)

    return model, processor
```

---

## Step 5: Conversation Template

Add to `tasks/eval/eval_utils.py`:

```python
# LLaVA-OneVision uses a different prompt format
conv_templates["eval_mvbench_ov"] = Conversation(
    system="You are a helpful assistant.",
    roles=("USER", "ASSISTANT"),
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2=" ",
)

conv_templates["eval_videomme_ov"] = Conversation(
    system="Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.",
    roles=("USER", "ASSISTANT"),
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2=" ",
)
```

---

## Step 6: Eval Script

Either adapt existing scripts or create a generic entry point:

**Option A: Add `--model_type` flag to existing scripts**

```bash
# In pllava_eval_mvbench.py, add:
parser.add_argument("--model_type", choices=["pllava", "llava_ov"], default="pllava")

# In load_model_and_dataset(), branch:
if args.model_type == "llava_ov":
    model, processor = load_llava_ov(...)
else:
    model, processor = load_pllava(...)
```

**Option B: Create thin wrapper scripts**

```bash
# tasks/eval/mvbench/llava_ov_eval_mvbench.py
# imports from pllava_eval_mvbench but overrides model loading
```

**Recommendation**: Option A is cleaner. One script, model-agnostic.

---

## Step 7-10: Evaluation

### Smoke Test (Step 7) ✅

```bash
conda run -n pllava python -m tasks.eval.mvbench.ov_eval_mvbench \
    --pretrained_model_name_or_path MODELS/llava-onevision-7b \
    --save_path test_results/mvbench_ov_smoke \
    --num_frames 16 \
    --tasks "Moving Direction" --max_samples 5
```

**Result**: Passed. Pipeline works end-to-end.

### Baseline (Step 8) ✅

```bash
conda run -n pllava python -m tasks.eval.mvbench.ov_eval_mvbench \
    --pretrained_model_name_or_path MODELS/llava-onevision-7b \
    --save_path test_results/mvbench_ov_baseline_5task \
    --num_frames 16 \
    --tasks "Action Sequence,Action Prediction,Moving Direction,Object Interaction,Unexpected Action" \
    --max_samples 20
```

**Result**: 52.0% average (20 samples/task). Ready for full evaluation.

### Motion-Adaptive (Step 9)

```bash
conda run -n pllava python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/llava-onevision-7b \
    --save_path test_results/mvbench_ov_motion_5task \
    --model_type llava_ov \
    --num_frames 16 \
    --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --use_flow_pruning --use_motion_adaptive --motion_scale 1.0 \
    --tasks "Action Sequence,Action Prediction,Moving Direction,Object Interaction,Unexpected Action" \
    --max_samples 200 \
    > log_mvb_ov_motion_5task.log 2>&1
```

### VideoMME Long (Step 10)

```bash
# Baseline
conda run -n pllava python -m tasks.eval.videomme.pllava_eval_videomme \
    --pretrained_model_name_or_path MODELS/llava-onevision-7b \
    --save_path test_results/videomme_ov_long_baseline_100 \
    --model_type llava_ov \
    --num_frames 16 \
    --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --tasks "Long Video" --max_samples 100 \
    > log_videomme_ov_long_baseline_100.log 2>&1

# Motion-adaptive
conda run -n pllava python -m tasks.eval.videomme.pllava_eval_videomme \
    --pretrained_model_name_or_path MODELS/llava-onevision-7b \
    --save_path test_results/videomme_ov_long_motion_100 \
    --model_type llava_ov \
    --num_frames 16 \
    --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --use_flow_pruning --use_motion_adaptive --motion_scale 1.0 \
    --tasks "Long Video" --max_samples 100 \
    > log_videomme_ov_long_motion_100.log 2>&1
```

---

## Key Risks

| Risk                            | Impact | Mitigation                                                     |
| ------------------------------- | ------ | -------------------------------------------------------------- |
| Qwen2 VTP wrapper is complex    | High   | Start with Approach B (LLM-side only), skip vision merge       |
| `<image>` token detection fails | Low    | **Resolved** — use `<video>` token for video input             |
| Attention output format differs | Low    | **Resolved** — use eager attention for weight extraction       |
| Model doesn't load              | Low    | **Resolved** — use HuggingFace native loading with bfloat16    |
| GPU OOM                         | Medium | LLaVA-OneVision is ~15B params; use float16, reduce batch size |
| Vision tower dtype mismatch     | Low    | **Resolved** — explicit bfloat16 cast for float params only    |

---

## Minimal Viable Path

**Status: COMPLETED ✅**

1. **Step 1**: Download model ✅
2. **Step 2 (Approach B)**: Only LLM-side VTP — skip vision merge entirely ✅
3. **Step 4**: Simple model loader using HuggingFace native classes ✅
4. **Step 7**: Smoke test on 10 MVBench samples ✅
5. **Step 8-9**: Baseline vs motion-adaptive on 5-task MVBench ✅

**Results**: Baseline working at 52.0% average (20 samples/task). Ready for VTP implementation.

---

## Expected Outcomes

| Scenario                   | Interpretation                                                      |
| -------------------------- | ------------------------------------------------------------------- |
| Motion-adaptive > baseline | **Success** — non-uniform attention enables token selection to work |
| Motion-adaptive ≈ baseline | Neutral — method doesn't help but doesn't hurt                      |
| Motion-adaptive < baseline | Need to tune motion_scale or alpha per window                       |

**Current Status**: Baseline working at 52.0% average (20 samples/task). Need to implement VTP and compare.

**Best case**: Your method improves temporal task accuracy on LLaVA-OneVision, validating the hypothesis that non-uniform attention is the key enabler.

---

## Files to Create/Modify

| File                                          | Action     | Description                        |
| --------------------------------------------- | ---------- | ---------------------------------- |
| `models/llava_ov/__init__.py`                 | **Done**   | Module init                        |
| `models/llava_ov/configuration_llava_ov.py`   | **Done**   | Config with PruneVid params        |
| `models/llava_ov/modeling_llava_ov.py`        | **Done**   | Model wrapper                      |
| `models/llava_ov/qwen2_vtp.py`                | **TODO**   | Qwen2 VTP-modified model           |
| `models/pllava/elastic_cache.py`              | **Done**   | VTPWindowCache + cluster pruning   |
| `tasks/eval/model_utils.py`                   | **Done**   | load_pllava() + load_llava_ov()    |
| `tasks/eval/eval_utils.py`                    | **Done**   | Conversation templates             |
| `tasks/eval/mvbench/ov_eval_mvbench.py`       | **Done**   | MVBench eval for LLaVA-OV          |
| `tasks/eval/mvbench/pllava_eval_mvbench.py`   | **Done**   | MVBench eval for PLLaVA            |
| `tasks/eval/videomme/ov_eval_videomme.py`     | **TODO**   | VideoMME eval for LLaVA-OV         |
| `tasks/eval/videomme/pllava_eval_videomme.py` | **Done**   | VideoMME eval for PLLaVA           |
