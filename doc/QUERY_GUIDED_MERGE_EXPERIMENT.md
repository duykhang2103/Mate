# Query-Guided Merge Experiment

> Archived: 2026-08-17  
> Status: INACTIVE — query component removed from the proposed main method  
> Result: positive point estimate over novelty-only at the same budget, but
> exact McNemar p=0.451381; not a validated accuracy contribution

## Final outcome

The full query-guided candidate reached 44.7037% accuracy, exactly matching
original PruneVid while reducing analytical LLM prefill TFLOPs by 15.7426%.
Against the matched-budget novelty-only run, it was +0.2222 accuracy points,
but the paired result was not significant. The predefined query-component
kill gate therefore rejected the term. No further query-weight sweep is
authorized; the protocol below is retained only as historical reproduction
material. See `archive/safeprunevid/README.md` for the final combined record.

## Objective

Test whether question-conditioned, semantic-novelty routing preserves more answer
evidence than optical-flow routing at a comparable early visual-token budget.
This is an experiment gate, not yet a paper claim.

The implementation is opt-in. Without `--use_query_guided_merge`, model behavior
is unchanged.

## Method under test

After PLLaVA's multimodal projector and before `merge_frames_dynamic`:

1. Build a query vector from non-padding text tokens after the final image token.
2. Score each projected spatial location by its maximum query similarity in the
   temporal window.
3. Score semantic novelty using distance from the temporal feature center.
4. Reduce query influence when its score distribution is flat.
5. Preserve the top fixed fraction as per-frame dynamic tokens; temporally merge
   the remaining static locations through the existing PruneVid path.

The stored `token_info.query_merge_reliability` value is the mean reliability
used across the temporal windows for that sample.

Before running a model benchmark in the GPU environment, execute the tensor
invariant tests:

```bash
python -m pytest tests/test_query_guided_merge.py -q
```

## Stage 0: corrected-flow closure

Run this once to determine whether the previous optical-flow result was primarily
caused by the inverted mask:

```bash
python -m tasks.eval.videomme.pllava_eval_videomme \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path results/pllava_videomme_flow_mask_fixed_smoke \
    --num_frames 16 --use_lora --lora_alpha 14 \
    --conv_mode eval_videomme \
    --alpha 0.4 --selected_layer 10 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --use_flow_pruning --flow_dynamic_ratio 0.5 \
    --use_motion_adaptive --motion_scale 1.0 \
    --max_samples 100
```

Do not combine flow flags with the query-guided experiment.

## Stage 1: accuracy-oriented query-guided smoke test

Start at a 75% dynamic ratio. This retains more evidence than the failed 50%
flow experiment while still compressing the early visual sequence.

```bash
python -m tasks.eval.videomme.pllava_eval_videomme \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path results/pllava_videomme_query_merge_w07_r075_smoke \
    --num_frames 16 --use_lora --lora_alpha 14 \
    --conv_mode eval_videomme \
    --alpha 0.4 --selected_layer 10 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --use_query_guided_merge \
    --query_merge_weight 0.7 \
    --query_dynamic_ratio 0.75 \
    --max_samples 100
```

Compare only matched samples:

```bash
python scripts/compare_paired_results.py \
    --baseline zzresults/Video-MME/results/pllava_videomme_all_prunevid/all_results.json \
    --candidate results/pllava_videomme_query_merge_w07_r075_smoke/all_results.json \
    --json-output results/pllava_videomme_query_merge_w07_r075_smoke/paired_comparison.json
```

Estimate decoder-prefill compute from the actual per-sample sequence lengths:

```bash
python scripts/estimate_llm_prefill_flops.py \
    results/pllava_videomme_query_merge_w07_r075_smoke/all_results.json
```

The estimator reports analytical LLM prefill TFLOPs only. It must eventually be
paired with synchronized CUDA latency and peak-memory measurements.

## Full Video-MME result: weight 0.7, dynamic ratio 0.75

Run completed on 2026-08-16 with all 2,700 questions. The comparison is fully
paired against `pllava_videomme_all_prunevid`.

| Metric | PruneVid | Query merge | Change |
| --- | ---: | ---: | ---: |
| Accuracy | 44.7037% | 44.7037% | 0.0000 pp |
| Merged vision tokens | 1096.53 | 897.81 | -18.12% |
| Post-pruning tokens | 562.93 | 483.02 | -14.19% |
| Prefill tokens | 1230.40 (inferred) | 1031.68 | -16.15% |
| Analytical LLM prefill TFLOPs | 10.6476 (inferred) | 8.9714 | -15.74% |

There were 71 correct-to-wrong and 71 wrong-to-correct flips. Exact McNemar
`p=1`, and the paired bootstrap 95% accuracy-delta interval was
`[-0.8519, +0.8519]` percentage points. Thus the point estimate is a useful
accuracy/efficiency Pareto point, but it does **not** meet the strict final
acceptance criterion below: the lower confidence bound is below -0.3 points
and analytical LLM-prefill savings are below 20%.

The old baseline artifact did not record `prefill_tokens`. Its value above is
inferred per paired sample as candidate text tokens plus baseline merged vision
tokens. The FLOPs calculation uses those inferred sequence lengths and each
sample's recorded post-pruning length/layer. It is not a measured latency claim.

Duration breakdown:

| Split | Accuracy delta | Correct -> wrong | Wrong -> correct | Merged-token reduction |
| --- | ---: | ---: | ---: | ---: |
| short | +0.4444 pp | 24 | 28 | 13.99% |
| medium | -0.6667 pp | 27 | 21 | 20.14% |
| long | +0.2222 pp | 20 | 22 | 19.97% |

The stored reliability field is empty in this run because autoregressive decode
cleared the prefill diagnostic before result serialization. This has been fixed
for subsequent runs; it did not change routing or predictions.

### Next matched-budget ablation

Before changing the token budget, isolate whether question similarity improves
selection over semantic novelty alone. `--max_samples 100` selects 100 available
questions from each duration split (300 total). With a fixed dynamic ratio, this
control has the same early-token budget as the query-guided method.

```bash
python -m tasks.eval.videomme.pllava_eval_videomme \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path results/pllava_videomme_novelty_w00_r075_ablation300 \
    --num_frames 16 --use_lora --lora_alpha 14 \
    --conv_mode eval_videomme \
    --alpha 0.4 --selected_layer 10 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --use_query_guided_merge \
    --query_merge_weight 0.0 \
    --query_dynamic_ratio 0.75 \
    --max_samples 100
```

Compare novelty-only directly with the full query-guided artifact; the paired
utility will automatically restrict both inputs to the same 300 questions:

```bash
python scripts/compare_paired_results.py \
    --baseline zzresults/Video-MME/results/pllava_videomme_query_merge_w07_r075_full/pllava_videomme_query_merge_w07_r075_full/all_results.json \
    --candidate results/pllava_videomme_novelty_w00_r075_ablation300/all_results.json
```

If weight 0.7 wins at the same budget, test `query_dynamic_ratio=0.65`; this is a
safer step toward the 20% compute target than jumping directly from 0.75 to
0.50. If novelty-only ties or wins, stop calling the current selector
query-guided: first improve the query representation or use the novelty-only
method as the ablation winner.

## Decision gates

- **Go to a full run:** accuracy delta is non-negative and mean merged tokens do
  not exceed PruneVid, or the paired CI remains compatible with non-inferiority
  while merged tokens decrease by at least 15%.
- **Test stronger compression:** if the 75% run passes, repeat with
  `--query_dynamic_ratio 0.5`.
- **Diagnose the query signal:** if the accuracy delta is between -1 and 0
  percentage points, repeat with `--query_merge_weight 0.0`. If novelty-only
  wins, the direct text/vision embedding alignment is inadequate and the next
  step is a lightweight learned query adapter.
- **Kill this version:** accuracy is at least 1 percentage point worse and
  correct-to-wrong flips exceed wrong-to-correct flips. Do not tune more than
  the one novelty-only control on the smoke subset.

## Full-result acceptance criteria

A paper candidate must satisfy at least one condition on a complete benchmark:

1. Higher accuracy than PruneVid at no greater measured TFLOPs and no greater
   mean token count; or
2. A paired 95% accuracy-delta lower bound above -0.3 percentage points with at
   least 20% lower prefill latency or analytical LLM TFLOPs.

The winning configuration must then be validated on the complete MVBench set and
ported to LLaVA-OneVision. Hyperparameters are frozen before those confirmation
runs.
