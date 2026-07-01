# Flow Correlation Sanity Check

**Goal**: Determine if optical flow magnitude correlates with PLLaVA errors on long VideoMME videos. If high-motion videos have higher error rates → flow-based adaptive pruning is worth implementing.

## Prerequisites

- conda env `pllava` with: `torch`, `transformers`, `peft`, `decord`, `cv2`, `pillow`
- Local model weights at `MODELS/pllava-7b`
- At least some extracted videos in `DATAS/Video-MME/data/`

## Run

Use whatever method works for you to run Python with CUDA in the pllava env:

```bash
cd /workspace/Mate
PYTHONPATH=/workspace/Mate:/venv/pllava/lib/python3.10/site-packages /venv/pllava/bin/python scripts/flow_correlation_sanity_check.py \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --num_frames 4 \
    --alpha 0.4 \
    --tau 0.8 \
    --selected_layer 10 \
    --save_dir doc/eval/flow_correlation
```

If CUDA fails with `random_device could not be read`, try the method you used for previous MVBench evals.

## Output

Saved to `doc/eval/flow_correlation/flow_correlation_results.json`:

```json
{
  "flow_results": { "<video>.mp4": { "avg_flow": 8.5, "std_flow": 2.1, "duration_s": 2400 } },
  "per_question": [
    { "video": "...", "question": "...", "pred": "(A)...", "gt": "(B)...", "correct": false, "flow_magnitude": 8.5 }
  ],
  "summary": {
    "overall_accuracy": 40.0,
    "correct_mean_flow": 7.2,
    "wrong_mean_flow": 9.8,
    "high_motion_accuracy": 35.0,
    "low_motion_accuracy": 45.0,
    "median_flow": 8.3
  }
}
```

## What to look for

| Signal | Meaning |
|--------|---------|
| `wrong_mean_flow` >> `correct_mean_flow` | High-motion videos are harder → flow pruning helps |
| `wrong_mean_flow` << `correct_mean_flow` | Low-motion videos are harder → flow pruning hurts |
| `wrong_mean_flow` ≈ `correct_mean_flow` | Flow doesn't correlate with errors → dead end |
| `high_motion_accuracy` << `low_motion_accuracy` | Same conclusion as first row |
| Wide spread in per-video flow (3x+) | Flow is discriminative across videos |
| Narrow spread in per-video flow (<2x) | Flow won't differentiate videos |
