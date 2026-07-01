# Video-MME Evaluation Guide

> **Updated**: 2026-07-01
> **Dataset**: 2700 samples (900 short + 900 medium + 900 long)
> **Video coverage**: 183 videos extracted (~24% of 760 total)
> **Prerequisite**: Videos must be extracted from zip files

---

## Prerequisites

```bash
# Fix dependency if needed
conda run -n pllava pip install "huggingface-hub>=0.19.3,<1.0"

# Verify videos are extracted
ls DATAS/Video-MME/data/*.mp4 | wc -l  # should be ~183+

# Verify JSON files exist
ls DATAS/Video-MME/json/*.json  # short.json, medium.json, long.json
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
    --alpha 0.4 --selected_layer 10 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --tasks short --max_samples 10 > log_videomme_smoke_baseline.log 2>&1
```

### Your Method (motion adaptive, scale=1.0)

```bash
conda run -n pllava python -m tasks.eval.videomme.pllava_eval_videomme \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/videomme_smoke_motion_1.0 \
    --num_frames 16 --use_lora --lora_alpha 14 \
    --conv_mode eval_videomme \
    --alpha 0.4 --selected_layer 10 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --use_motion_adaptive --motion_scale 1.0 \
    --tasks short --max_samples 10 > log_videomme_smoke_motion_1.log 2>&1
```

### Compare Smoke Results

```bash
python -c "
import json
b = json.load(open('test_results/videomme_smoke_baseline/upload_leaderboard.json'))
m = json.load(open('test_results/videomme_smoke_motion_1.0/upload_leaderboard.json'))
print('Baseline:', json.dumps(b, indent=2))
print('Motion 1.0:', json.dumps(m, indent=2))
"
```

---

## Step 2: Partial Eval (per split, ~30-60 min each)

Run one split at a time to get intermediate results.

### Short Videos Only (900 samples)

```bash
# Baseline
conda run -n pllava python -m tasks.eval.videomme.pllava_eval_videomme \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/videomme_short_baseline \
    --num_frames 16 --use_lora --lora_alpha 14 \
    --conv_mode eval_videomme \
    --alpha 0.4 --selected_layer 10 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --tasks short > log_videomme_short_baseline.log 2>&1

# Your method
conda run -n pllava python -m tasks.eval.videomme.pllava_eval_videomme \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/videomme_short_motion_1.0 \
    --num_frames 16 --use_lora --lora_alpha 14 \
    --conv_mode eval_videomme \
    --alpha 0.4 --selected_layer 10 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --use_motion_adaptive --motion_scale 1.0 \
    --tasks short > log_videomme_short_motion_1.0.log 2>&1
```

### Medium Videos Only (900 samples)

```bash
# Baseline
conda run -n pllava python -m tasks.eval.videomme.pllava_eval_videomme \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/videomme_medium_baseline \
    --num_frames 16 --use_lora --lora_alpha 14 \
    --conv_mode eval_videomme \
    --alpha 0.4 --selected_layer 10 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --tasks medium > log_videomme_medium_baseline.log 2>&1

# Your method
conda run -n pllava python -m tasks.eval.videomme.pllava_eval_videomme \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/videomme_medium_motion_1.0 \
    --num_frames 16 --use_lora --lora_alpha 14 \
    --conv_mode eval_videomme \
    --alpha 0.4 --selected_layer 10 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --use_motion_adaptive --motion_scale 1.0 \
    --tasks medium > log_videomme_medium_motion_1.0.log 2>&1
```

### Long Videos Only (900 samples)

```bash
# Baseline
conda run -n pllava python -m tasks.eval.videomme.pllava_eval_videomme \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/videomme_long_baseline \
    --num_frames 16 --use_lora --lora_alpha 14 \
    --conv_mode eval_videomme \
    --alpha 0.4 --selected_layer 10 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --tasks long > log_videomme_long_baseline.log 2>&1

# Your method
conda run -n pllava python -m tasks.eval.videomme.pllava_eval_videomme \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/videomme_long_motion_1.0 \
    --num_frames 16 --use_lora --lora_alpha 14 \
    --conv_mode eval_videomme \
    --alpha 0.4 --selected_layer 10 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --use_motion_adaptive --motion_scale 1.0 \
    --tasks long > log_videomme_long_motion_1.0.log 2>&1
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
    --conv_mode eval_videomme \
    --alpha 0.4 --selected_layer 10 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    > log_videomme_full_baseline.log 2>&1

# Your method
conda run -n pllava python -m tasks.eval.videomme.pllava_eval_videomme \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/videomme_full_motion_1.0 \
    --num_frames 16 --use_lora --lora_alpha 14 \
    --conv_mode eval_videomme \
    --alpha 0.4 --selected_layer 10 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --use_motion_adaptive --motion_scale 1.0 \
    > log_videomme_full_motion_1.0.log 2>&1
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
    fp = os.path.join(path, 'upload_leaderboard.json')
    if os.path.exists(fp):
        data = json.load(open(fp))
        print(f'{name}:')
        for k, v in data.items():
            print(f'  {k}: {v:.2f}%')
    else:
        print(f'{name}: not found')
"
```

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
        --alpha 0.4 --selected_layer 10 --tau 0.8 \
        --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
        --use_motion_adaptive --motion_scale $SCALE \
        --tasks short --max_samples 50 > log_videomme_scale_${SCALE}.log 2>&1
done
```

---

## Available Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--tasks` | all | Comma-separated splits: `short`, `medium`, `long` |
| `--max_samples` | all | Limit samples per split (e.g., `10` for smoke test) |
| `--alpha` | 0.1 | Base retention ratio (use `0.4` for PruneVid) |
| `--selected_layer` | 10 | LLM layer for pruning (use `10` for PruneVid) |
| `--tau` | 1.0 | Similarity threshold (use `0.8` for PruneVid) |
| `--cluster_ratio` | 1.0 | Spatial cluster ratio (use `0.5` for PruneVid) |
| `--temporal_segment_ratio` | 1.0 | Temporal window ratio (use `0.25` for PruneVid) |
| `--use_motion_adaptive` | off | Enable motion-adaptive pruning |
| `--motion_scale` | 0.5 | How much motion affects retention (`0.5`, `1.0`, `2.0`) |
| `--use_borderline_preservation` | off | Keep tokens near pruning cutoff |
| `--borderline_margin` | 0.1 | Margin for borderline preservation |
| `--conv_mode` | `eval_videomme` | Conversation template (don't change) |

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

The script gracefully skips missing videos. Only 183 out of 760 videos are extracted (~24% coverage). Results may not be representative. To extract more videos:

```bash
cd DATAS/Video-MME
for z in videos_chunked_*.zip; do
    unzip -o "$z" -d data/
done
```

### Memory issues

Reduce `--num_frames` to 8 or use fewer samples with `--max_samples`.

---

## Output Format

Results are saved to `<save_path>/upload_leaderboard.json`:

```json
{
    "Short Video": 45.3,
    "Medium Video": 42.1,
    "Long Video": 38.7,
    "Avg": 42.0
}
```

Detailed results in `<save_path>/all_results.json`:

```json
{
    "acc_dict": {
        "Short Video": [correct, total],
        "Medium Video": [correct, total],
        "Long Video": [correct, total]
    },
    "result_list": [
        {
            "pred": "(A) ...",
            "gt": "(A) ...",
            "task_type": "Short Video",
            "video_path": "DATAS/Video-MME/data/xxx.mp4",
            "question": "Question: ...",
            "token_info": {...}
        }
    ]
}
```
