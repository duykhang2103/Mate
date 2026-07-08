# VideoMME Eval Fix Summary

> **Date**: 2026-07-08
> **Goal**: Fix LLaVA-OV VideoMME evaluation to match PruneVid paper baseline (58.2%)

---

## Problem

LLaVA-OV VideoMME baseline was underperforming compared to PruneVid paper:

| Source | Baseline | Notes |
|--------|----------|-------|
| PruneVid paper | 58.2% | Official result |
| Our previous eval | 40% (Long Video, 20 samples) | PLLaVA, not LLaVA-OV |

Root cause: 3 bugs in `ov_eval_videomme.py` causing ~5-7% accuracy gap.

---

## Bugs Fixed

### Bug 1: Missing System Prompt (Estimated impact: ~3-5%)

**File**: `tasks/eval/videomme/ov_eval_videomme.py`

**Before (line 163-165):**
```python
conversation = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": question}]}]
text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
```

**After:**
```python
conversation = [
    {"role": "system", "content": "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question."},
    {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": question}]}
]

# Add answer_prompt as assistant pre-fill if provided
if answer_prompt is not None:
    conversation.append({"role": "assistant", "content": answer_prompt})

text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
```

**Why**: The system prompt guides the model to focus on video content and select the best option. Without it, the model generates freeform text instead of structured answers.

---

### Bug 2: Removed answer_prompt/return_prompt (Not supported by LLaVA-OV)

**File**: `tasks/eval/videomme/ov_eval_videomme.py`

LLaVA-OV's `apply_chat_template` does NOT support assistant pre-fill like PLLaVA. When we added `{"role": "assistant", "content": "Best option:("}` to the conversation, the model just echoed it without generating an answer.

**Fix**: Removed answer_prompt pre-fill and stripping entirely. The system prompt + `post_query_prompt="\nOnly give the best option."` is enough to guide the model to output `(A)`, `(B)`, etc.

---

### Bug 3: VTP Always Enabled (Missing --use_vtp argument)

**File**: `tasks/eval/videomme/ov_eval_videomme.py`

**Problem**: `load_llava_ov()` has `use_vtp=True` as default, but `ov_eval_videomme.py` had no `--use_vtp` CLI argument. This meant baseline evaluation ALWAYS ran with VTP enabled.

**Fix**:
1. Added `--use_vtp` argument to `parse_args()`
2. Added `use_vtp=False` parameter to `load_model_and_dataset()`
3. Passed `args.use_vtp` to `load_model_and_dataset()` in `run()`

---

## Files Modified

| File | Change |
|------|--------|
| `tasks/eval/videomme/ov_eval_videomme.py` | Added system prompt, fixed device placement, added --use_vtp flag |

---

## Additional Fix: Multi-GPU Device Mismatch (Bug 4)

### Problem

When running on multi-GPU setups, the error occurred:
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cuda:0!
```

### Root Cause

`model.to(torch.device(rank))` was called AFTER `device_map="auto"` already distributed the model across GPUs. This broke the device placement.

### Fix

1. **Removed `model.to(torch.device(rank))`** - Let `device_map="auto"` handle placement
2. **Changed input device placement** - Use `next(model.parameters()).device` instead of `model.device`

**Before:**
```python
model = model.to(torch.device(rank))  # BREAKS device_map="auto"
model = model.eval()

# ...

inputs = {k: v.to(model.device) ...}  # model.device may be wrong
```

**After:**
```python
# Do NOT call model.to() when using device_map="auto"
model = model.eval()

# ...

model_device = next(model.parameters()).device  # Get actual device
inputs = {k: v.to(model_device) ...}
```

### Workaround for Single GPU

If you want to force single GPU:
```bash
CUDA_VISIBLE_DEVICES=0 python -m tasks.eval.videomme.ov_eval_videomme ...
```

---

## How to Run Evaluation

### Prerequisites

1. Extract videos from zip files:
```bash
cd /workspace/Mate/DATAS/Video-MME
for zip in videos_chunked_*.zip; do
    unzip -o "$zip" -d .
done
```

2. Create JSON files from parquet (if not already done):
```python
import pandas as pd
import json

df = pd.read_parquet('videomme/test-00000-of-00001.parquet')

for duration in ['short', 'medium', 'long']:
    subset = df[df['duration'] == duration]
    records = []
    for _, row in subset.iterrows():
        records.append({
            'video': f"{row['videoID']}.mp4",
            'question': row['question'],
            'candidates': row['options'],
            'answer': row['answer'],
            'question_id': row['question_id'],
        })
    with open(f'json/{duration}.json', 'w') as f:
        json.dump(records, f, indent=2)
```

### Smoke Test (5 samples, ~2 min)

```bash
PYTHONPATH=/workspace/Mate conda run -n pllava python tasks/eval/videomme/ov_eval_videomme.py \
    --pretrained_model_name_or_path MODELS/llava-onevision-7b \
    --save_path test_results/videomme_ov_baseline_fixed_smoke \
    --num_frames 16 \
    --tasks "Long Video" \
    --max_samples 5 \
    > log_videomme_ov_baseline_fixed_smoke.log 2>&1
```

### Verify System Prompt

```bash
grep "PROMPTING LM WITH" log_videomme_ov_baseline_fixed_smoke.log
```

Should show: `Carefully watch the video...` at the start.

### Full Evaluation

```bash
PYTHONPATH=/workspace/Mate conda run -n pllava python tasks/eval/videomme/ov_eval_videomme.py \
    --pretrained_model_name_or_path MODELS/llava-onevision-7b \
    --save_path results/videomme_ov_baseline_fixed \
    --num_frames 16 \
    --max_new_tokens 100 \
    > log_videomme_ov_baseline_fixed.log 2>&1
```

### VTP Comparison (after baseline)

```bash
PYTHONPATH=/workspace/Mate conda run -n pllava python tasks/eval/videomme/ov_eval_videomme.py \
    --pretrained_model_name_or_path MODELS/llava-onevision-7b \
    --save_path results/videomme_ov_vtp_alpha04_fixed \
    --num_frames 16 \
    --selected_layer 10 \
    --alpha 0.4 \
    --use_vtp \
    --max_new_tokens 100 \
    > log_videomme_ov_vtp_alpha04_fixed.log 2>&1
```

---

## Expected Results

| Scenario | Result | Action |
|----------|--------|--------|
| Fixed baseline ≥ 57% | Bugs were the cause | Re-run VTP comparison |
| Fixed baseline ≈ 52% | Different issue | Investigate further |
| VTP > fixed baseline | VTP works on LLaVA-OV | Write paper |
| VTP ≈ fixed baseline | No improvement | Try other alpha/layer combinations |

---

## Key Differences from Previous Eval

| Aspect | Before | After |
|--------|--------|-------|
| System prompt | Missing | Added |
| answer_prompt | Pre-filled (not supported) | Removed |
| return_prompt | Used for stripping | Removed |
| Expected accuracy | ~52% (broken) | ~57-58% (matching PruneVid) |

---

## Notes

- The same 3 bugs existed in `ov_eval_mvbench.py` but were already documented in `doc/6-7-2026_1/BASELINE_GAP_ANALYSIS.md`
- VideoMME eval also handles missing videos gracefully (checks `os.path.exists()` before processing)
- LLaVA-OV processor should support system messages in chat template - verify with smoke test
