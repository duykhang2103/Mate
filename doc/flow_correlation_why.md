# Why We Need the Flow Correlation Check

## Background

- Current motion-adaptive pruning doesn't improve accuracy (PLLaVA attention is uniform ~0.9999 across all layers)
- We want to know: **does optical flow correlate with PLLaVA errors?**
- If yes → build flow-based pruning
- If no → pivot to FLOPs reduction narrative (same accuracy, 84% fewer tokens)

## What the Check Does

1. Compute optical flow (Farneback) on 20 long VideoMME videos
2. Run PLLaVA inference on each question
3. Compare: do high-motion videos have higher error rates?

## Decision Table

| Result | Action |
|--------|--------|
| `wrong_mean_flow` >> `correct_mean_flow` | Build flow-based pruning |
| `wrong_mean_flow` << `correct_mean_flow` | Flow pruning hurts, pivot to FLOPs |
| `wrong_mean_flow` ≈ `correct_mean_flow` | Flow is irrelevant, pivot to FLOPs |
