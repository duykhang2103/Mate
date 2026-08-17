# SafePruneVid archive

> Archived: 2026-08-17  
> Status: rejected as the active research direction; preserved as a negative
> result and reproducibility record

## Decision

SafePruneVid is frozen. Do not run additional epsilon searches, ADBR-M/E full
evaluations, safety-router training, or SafePruneVid MVBench evaluation. The
method did not pass either predeclared acceptance rule, and the calibration
curve shows an accuracy/compute cliff rather than evidence of a hidden useful
threshold.

This archive does not define a successor method. A new plan can start
independently from this record.

## What was tested

The training-free method replaced the fixed per-window dynamic ratio with a
global distortion-per-extra-token threshold:

```text
d = max_t(1 - cosine(frame_token_t, temporal_centroid))
value = d / max(window_size - 1, 1)
dynamic = value > epsilon
```

It retained existing DPC temporal segmentation, spatial clustering, and
layer-10 attention pruning. Each window kept at least one dynamic and one
static trajectory so the original metadata and divisibility contracts stayed
valid. PLLaVA remained frozen.

## Evidence

### Query component kill gate

The full 2,700-question query-guided run matched original PruneVid accuracy,
but did not improve it:

| Metric | Original PruneVid | Query-guided, w=0.7/r=0.75 | Delta |
|---|---:|---:|---:|
| Accuracy | 44.7037% | 44.7037% | 0.0000 points |
| Correct-to-wrong / wrong-to-correct | - | 71 / 71 | McNemar p=1 |
| Merged vision tokens | 1096.53 | 897.81 | -18.1231% |
| Analytical LLM prefill | 10.6476 TFLOPs | 8.9714 TFLOPs | -15.7426% |

At the same token budget, novelty-only accuracy was 44.4815%, 0.2222 points
below query-guided, with paired 95% CI [-0.7037, 0.2593] and exact McNemar
p=0.451381. The query component therefore failed its significance gate and
was not retained as a validated contribution.

### Distortion calibration curve

Every epsilon run used the same development cohort: all 900 questions from
100 unique videos in each VideoMME duration group.

| Epsilon | Accuracy | Mean analytical TFLOPs | Mean merged tokens |
|---:|---:|---:|---:|
| 0.0025 | 44.8889% | 11.0247 | 1131.53 |
| 0.005 | 44.8889% | 10.9949 | 1128.09 |
| 0.01 | 44.6667% | 10.9495 | 1122.53 |
| 0.02 | 44.2222% | 10.7014 | 1092.26 |
| 0.029 | 44.1111% | 9.2589 | 920.74 |
| 0.04 | 40.5556% | 6.6856 | 608.30 |
| 0.08 | 38.4444% | 3.8744 | 255.93 |

The saved `adbr_m_calibration.json` predates epsilon 0.029: it lists only six
candidates, selects epsilon 0.02, and records `target_passed=false`. Treat it
as an incomplete historical artifact, not the final selection. Recomputing
over all seven artifacts selects 0.029 as closest to the 8.9714-TFLOP ADBR-M
target, but 9.2589 is still 3.20% above the target and outside its +/-1% band.

Epsilon 0.04 is below the 8.5181-TFLOP ADBR-E ceiling, but its 40.5556%
accuracy is a material collapse. No ADBR-E calibration JSON was saved.

### Paired result at epsilon 0.029

The valid preliminary comparison is against the same 900 rows of the timed
original artifact, not against the original artifact's full 2,700-question
aggregate.

| Metric | Matched original | Epsilon 0.029 | Change |
|---|---:|---:|---:|
| Accuracy | 43.5556% (392/900) | 44.1111% (397/900) | +0.5556 points |
| Correct-to-wrong / wrong-to-correct | - | 13 / 18 | McNemar p=0.47313 |
| Paired accuracy-delta 95% CI | - | - | [-0.6667, 1.7778] points |
| Mean merged tokens | 1081.06 | 920.74 | -14.8299% |
| Mean prefill tokens | 1219.36 | 1059.04 | -13.1480% |
| Mean analytical TFLOPs | 10.5836 | 9.2589 | -12.5160% |
| Mean prefill latency | 458.9209 ms | 438.7271 ms | -4.4003% |
| Mean compressor latency | 22.4187 ms | 26.2195 ms | +16.9536% |
| Mean total latency | 1441.6273 ms | 1419.4815 ms | -1.5362% |
| Mean peak CUDA memory | 14800.23 MB | 14668.97 MB | -0.8868% |

Duration accuracy moved +1.0000 point on short, -0.6667 on medium, and
+1.3333 on long questions. The oracle union was +2.0000 points, but that is
complementarity, not evidence that the proposed router would generalize.

### Acceptance outcome

SafePruneVid failed both frozen rules:

1. The accuracy direction was positive at epsilon 0.029, but exact McNemar
   p=0.47313 was not significant.
2. The paired CI lower bound was -0.6667 rather than above -0.3 points, and
   neither analytical TFLOPs (-12.5160%) nor prefill latency (-4.4003%) met
   the required 20% reduction.

No full ADBR-M, full ADBR-E, acceptance-gate, safety-router, or SafePruneVid
MVBench artifact exists. The archived evidence is calibration-stage evidence,
not a completed cross-benchmark result.

## Preserved implementation

The following remain in place so the negative result can be inspected and the
general timing/comparison infrastructure can be reused:

- `models/pllava/distortion_routing.py`
- opt-in integration in `models/pllava/modeling_pllava.py`
- opt-in evaluation flags and telemetry in the VideoMME and MVBench evaluators
- `scripts/select_distortion_epsilon.py`
- `scripts/evaluate_query_kill_gate.py`
- `scripts/evaluate_acceptance_gate.py`
- `scripts/train_safety_router.py`
- `scripts/compare_paired_results.py`
- `scripts/estimate_llm_prefill_flops.py`
- SafePruneVid routing, tooling, comparison, and archive-guard tests

The default PLLaVA path is unchanged because distortion routing remains behind
`--use_distortion_routing`. Removing the module alone would break its current
unconditional import, so code cleanup is deliberately deferred to a separate,
impact-analyzed change.

## Artifacts

Raw artifacts remain at:

```text
zzresults/Video-MME/results/safeprunevid/safeprunevid/
```

The directory contains 17 files totaling 18,220,525 bytes: seven calibration
runs, the full timed original run, and the stale ADBR-M calibration summary.
Per-sample files remain in place rather than being duplicated. Their exact
SHA-256 values are recorded in `ARTIFACTS.sha256`.

## Reproduction-only runner

The repository-root `safeprunevid-eval.sh` is now an inert archive notice, so
old shell stage commands stop instead of launching an evaluation. The original
Modal wrapper is also preserved here; the root `run_evaluation.py` now rejects
locally so an old command cannot allocate an L4 before failing. Historical
reproduction requires the explicit archive path:

```bash
modal run archive/safeprunevid/run_evaluation.py::run_pipeline \
    --stage epsilon-0029
```

The original stage orchestration is preserved with a deliberate direct-run
opt-in guard:

```bash
SAFEPRUNEVID_ALLOW_ARCHIVED=1 \
SAFEPRUNEVID_STAGE=epsilon-0029 \
bash archive/safeprunevid/safeprunevid-eval.sh
```

This is for reproduction only. Authentication, if needed, must come from a
secret-backed `HF_TOKEN` environment variable. The archived runner contains
and prints no credential. The credential previously embedded in the old
runner must still be revoked/rotated outside this repository.

## Final interpretation

The distortion threshold mostly changes token quantity; it does not improve
the retained representation. Moderate compression around epsilon 0.029 gives
a small, uncertain directional gain with limited measured latency benefit.
Higher compression quickly destroys accuracy. That evidence does not justify
another threshold search or a full evaluation, so SafePruneVid is closed as a
negative ablation rather than promoted as a paper method.
