# Video-MME Evaluation Guide

> **Created**: 2026-07-01
> **Dataset**: 2700 samples (900 short + 900 medium + 900 long)
> **Video coverage**: ~98% (16 videos missing out of 760)
> **Prerequisite**: Run `conda run -n pllava pip install "huggingface-hub>=0.19.3,<1.0"` if you hit dependency errors

---

## Prerequisites

```bash
# Fix dependency if needed
conda run -n pllava pip install "huggingface-hub>=0.19.3,<1.0"

# Verify videos are extracted
ls DATAS/Video-MME/data/data/*.mp4 | wc -l  # should be ~744+
```

---

## Step 1: Smoke Test (10 samples, ~5 min)

Quick sanity check to make sure everything works.

### Baseline (no motion adaptive)

```bash
conda run -n pllava python -m tasks.eval.videomme.pllava_eval_videomme \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/videomme_smoke_baseline \
    --num_frames 16 --use_lora --lora_alpha 14 \
    --conv_mode eval_videomme \
    --tasks short --max_samples 10 > log_videomme_smoke_baseline.log 2>&1
```

### Your Method (motion adaptive, scale=1.0)

```bash
conda run -n pllava python -m tasks.eval.videomme.pllava_eval_videomme \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/videomme_smoke_motion_1.0 \
    --num_frames 16 --use_lora --lora_alpha 14 \
    --conv_mode eval_videomme \
    --use_motion_adaptive --motion_scale 1.0 \
    --tasks short --max_samples 10 > log_videomme_smoke_motion_1.log 2>&1
```

### Compare Smoke Results

```bash
python -c "
import json
b = json.load(open('test_results/videomme_smoke_baseline/all_results.json'))
m = json.load(open('test_results/videomme_smoke_motion_1.0/all_results.json'))
print('Baseline:', b.get('results', b))
print('Motion 1.0:', m.get('results', m))
"
```

---

## Step 2: Partial Eval (per split, ~30-60 min each)

Run one split at a time to get intermediate results.

### Short Videos Only

```bash
# Baseline
conda run -n pllava python -m tasks.eval.videomme.pllava_eval_videomme \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/videomme_short_baseline \
    --num_frames 16 --use_lora --lora_alpha 14 \
    --conv_mode eval_videomme \
    --tasks short

# Your method
conda run -n pllava python -m tasks.eval.videomme.pllava_eval_videomme \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/videomme_short_motion_1.0 \
    --num_frames 16 --use_lora --lora_alpha 14 \
    --conv_mode eval_videomme \
    --use_motion_adaptive --motion_scale 1.0 \
    --tasks short
```

### Medium Videos Only

```bash
# Baseline
conda run -n pllava python -m tasks.eval.videomme.pllava_eval_videomme \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/videomme_medium_baseline \
    --num_frames 16 --use_lora --lora_alpha 14 \
    --conv_mode eval_videomme \
    --tasks medium

# Your method
conda run -n pllava python -m tasks.eval.videomme.pllava_eval_videomme \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/videomme_medium_motion_1.0 \
    --num_frames 16 --use_lora --lora_alpha 14 \
    --conv_mode eval_videomme \
    --use_motion_adaptive --motion_scale 1.0 \
    --tasks medium
```

### Long Videos Only

```bash
# Baseline
conda run -n pllava python -m tasks.eval.videomme.pllava_eval_videomme \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/videomme_long_baseline \
    --num_frames 16 --use_lora --lora_alpha 14 \
    --conv_mode eval_videomme \
    --tasks long

# Your method
conda run -n pllava python -m tasks.eval.videomme.pllava_eval_videomme \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/videomme_long_motion_1.0 \
    --num_frames 16 --use_lora --lora_alpha 14 \
    --conv_mode eval_videomme \
    --use_motion_adaptive --motion_scale 1.0 \
    --tasks long
```

---

## Step 3: Full Eval (all splits, ~3-4 hrs)

Run all 2700 samples across short + medium + long.

```bash
# Baseline
conda run -n pllava python -m tasks.eval.videomme.pllava_eval_videomme \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/videomme_full_baseline \
    --num_frames 16 --use_lora --lora_alpha 14 \
    --conv_mode eval_videomme

# Your method
conda run -n pllava python -m tasks.eval.videomme.pllava_eval_videomme \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/videomme_full_motion_1.0 \
    --num_frames 16 --use_lora --lora_alpha 14 \
    --conv_mode eval_videomme \
    --use_motion_adaptive --motion_scale 1.0
```

---

## Step 4: Compare Results

```bash
python -c "
import json, os

configs = {
    'baseline': 'test_results/videomme_full_baseline',
    'motion_1.0': 'test_results/videomme_full_motion_1.0',
}

for name, path in configs.items():
    fp = os.path.join(path, 'all_results.json')
    if os.path.exists(fp):
        data = json.load(open(fp))
        results = data.get('results', data)
        print(f'{name}: {json.dumps(results, indent=2)}')
    else:
        print(f'{name}: not found')
"
```

---

## Available Arguments

| Argument                        | Default         | Description                                                       |
| ------------------------------- | --------------- | ----------------------------------------------------------------- |
| `--tasks`                       | all             | Comma-separated splits: `short`, `medium`, `long`, `short,medium` |
| `--max_samples`                 | all             | Limit samples per split (e.g., `10` for smoke test)               |
| `--use_motion_adaptive`         | off             | Enable two-tier retention (your method)                           |
| `--motion_scale`                | 0.5             | How much more dynamic tokens retain (try `0.5`, `1.0`, `2.0`)     |
| `--use_borderline_preservation` | off             | Keep tokens near pruning cutoff                                   |
| `--borderline_margin`           | 0.1             | Margin for borderline preservation                                |
| `--conv_mode`                   | `eval_videomme` | Conversation template (don't change)                              |
| `--alpha`                       | 0.1             | Base retention ratio (use `0.4` for PruneVid)                     |
| `--selected_layer`              | 10              | LLM layer for pruning (use `10` for PruneVid)                     |
| `--tau`                         | 1.0             | Similarity threshold (use `0.8` for PruneVid)                     |
| `--cluster_ratio`               | 1.0             | Spatial cluster ratio (use `0.5` for PruneVid)                    |
| `--temporal_segment_ratio`      | 1.0             | Temporal window ratio (use `0.25` for PruneVid)                   |

---

## Hyperparameter Sweep

Try different `motion_scale` values to find the sweet spot:

```bash
for SCALE in 0.5 1.0 2.0; do
    conda run -n pllava python -m tasks.eval.videomme.pllava_eval_videomme \
        --pretrained_model_name_or_path MODELS/pllava-7b \
        --save_path test_results/videomme_motion_scale_${SCALE} \
        --num_frames 16 --use_lora --lora_alpha 14 \
        --conv_mode eval_videomme \
        --use_motion_adaptive --motion_scale $SCALE \
        --tasks short --max_samples 50
done
```

---

## Troubleshooting

### "huggingface-hub" version error

```bash
conda run -n pllava pip install "huggingface-hub>=0.19.3,<1.0"
```

### Results already exist (resume behavior)

The script loads existing `all_results.json` and skips inference. To re-run:

```bash
rm -rf test_results/videomme_<your_run>
```

### Missing videos

The script gracefully skips missing videos. ~16 out of 760 videos are missing (98% coverage). Results are still reliable.

### Memory issues

Reduce `--num_frames` to 8 or use fewer samples with `--max_samples`.
