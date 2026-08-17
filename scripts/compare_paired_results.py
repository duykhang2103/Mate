#!/usr/bin/env python3
"""Compare two PruneVid benchmark result files on their matched samples.

The benchmark JSON files contain aggregate accuracy and per-example predictions.
Aggregate scores are unsafe to compare when a dataset is incomplete, so this
utility aligns examples and computes paired accuracy, token statistics,
compression ratios, and analytical decoder-prefill FLOPs. Runtime and memory
metrics are compared only when they were explicitly recorded in `token_info`.

Example:
    python scripts/compare_paired_results.py \
        --baseline results/baseline/all_results.json \
        --candidate results/query_merge/all_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable


RUNTIME_FIELDS = (
    "prefill_latency_ms",
    "compressor_latency_ms",
    "decode_latency_ms",
    "generation_latency_ms",
    "total_latency_ms",
    "peak_gpu_memory_mb",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
        help="Number of paired bootstrap resamples used for the accuracy-delta CI.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=11008)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for a machine-readable copy of the comparison.",
    )
    return parser.parse_args()


def load_results(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("result_list")
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain a result_list array")
    return rows


def sample_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("task_type", "")),
        str(row.get("video_path", row.get("video", ""))),
        str(row.get("question", "")),
    )


def option_token(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None

    first = text.split()[0].replace(".", "")
    for letter in "abcdefg":
        if letter in first:
            return letter
    return None


def is_correct(row: dict[str, Any]) -> bool:
    pred = option_token(row.get("pred"))
    target = option_token(row.get("gt"))
    return pred is not None and target is not None and pred == target


def exact_mcnemar_pvalue(baseline_only: int, candidate_only: int) -> float:
    discordant = baseline_only + candidate_only
    if discordant == 0:
        return 1.0
    smaller = min(baseline_only, candidate_only)
    lower_tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2.0 * lower_tail)


def paired_bootstrap_ci(
    deltas: list[float], samples: int, seed: int, confidence: float = 0.95
) -> tuple[float, float]:
    if not deltas or samples <= 0:
        return (float("nan"), float("nan"))

    rng = random.Random(seed)
    count = len(deltas)
    estimates = []
    for _ in range(samples):
        estimates.append(sum(deltas[rng.randrange(count)] for _ in range(count)) / count)
    estimates.sort()

    tail = (1.0 - confidence) / 2.0
    lower_index = max(0, min(samples - 1, int(tail * samples)))
    upper_index = max(0, min(samples - 1, int((1.0 - tail) * samples) - 1))
    return estimates[lower_index], estimates[upper_index]


def numeric_values(rows: Iterable[dict[str, Any]], field: str) -> list[float]:
    values = []
    for row in rows:
        token_info = row.get("token_info") or {}
        value = token_info.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return values


def paired_numeric_values(
    baseline_rows: Iterable[dict[str, Any]],
    candidate_rows: Iterable[dict[str, Any]],
    field: str,
) -> tuple[list[float], list[float]]:
    baseline_values = []
    candidate_values = []
    for baseline_row, candidate_row in zip(baseline_rows, candidate_rows):
        baseline_value = numeric_token_value(baseline_row, field)
        candidate_value = numeric_token_value(candidate_row, field)
        if baseline_value is not None and candidate_value is not None:
            baseline_values.append(baseline_value)
            candidate_values.append(candidate_value)
    return baseline_values, candidate_values


def mean_or_none(values: list[float]) -> float | None:
    return fmean(values) if values else None


def numeric_token_value(row: dict[str, Any], field: str) -> float | None:
    value = (row.get("token_info") or {}).get(field)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def paired_prefill_values(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> tuple[list[float], list[float], list[int], dict[str, dict[str, int]]]:
    """Return aligned prefill lengths, inferring one missing side when safe.

    For a matched prompt, text length is `prefill - merged_vision_tokens`.
    Therefore a legacy artifact missing prefill length can inherit the paired
    artifact's text length and add its own recorded merged-vision length.
    """
    baseline_values = []
    candidate_values = []
    paired_indices = []
    sources = {
        "baseline": {"recorded": 0, "inferred": 0, "missing": 0},
        "candidate": {"recorded": 0, "inferred": 0, "missing": 0},
    }

    for index, (baseline_row, candidate_row) in enumerate(
        zip(baseline_rows, candidate_rows)
    ):
        baseline_prefill = numeric_token_value(baseline_row, "prefill_tokens")
        candidate_prefill = numeric_token_value(candidate_row, "prefill_tokens")
        baseline_merged = numeric_token_value(
            baseline_row, "merged_vision_tokens"
        )
        candidate_merged = numeric_token_value(
            candidate_row, "merged_vision_tokens"
        )

        baseline_source = "recorded" if baseline_prefill is not None else "missing"
        candidate_source = "recorded" if candidate_prefill is not None else "missing"

        if (
            baseline_prefill is None
            and candidate_prefill is not None
            and baseline_merged is not None
            and candidate_merged is not None
        ):
            text_tokens = candidate_prefill - candidate_merged
            if text_tokens >= 0:
                baseline_prefill = text_tokens + baseline_merged
                baseline_source = "inferred"

        if (
            candidate_prefill is None
            and baseline_prefill is not None
            and baseline_merged is not None
            and candidate_merged is not None
        ):
            text_tokens = baseline_prefill - baseline_merged
            if text_tokens >= 0:
                candidate_prefill = text_tokens + candidate_merged
                candidate_source = "inferred"

        sources["baseline"][baseline_source] += 1
        sources["candidate"][candidate_source] += 1
        if baseline_prefill is not None and candidate_prefill is not None:
            baseline_values.append(baseline_prefill)
            candidate_values.append(candidate_prefill)
            paired_indices.append(index)

    return baseline_values, candidate_values, paired_indices, sources


def relative_reduction(
    baseline_value: float | None, candidate_value: float | None
) -> float | None:
    if baseline_value in (None, 0) or candidate_value is None:
        return None
    return (baseline_value - candidate_value) / baseline_value


def retention_ratio(
    retained_value: float | None, original_value: float | None
) -> float | None:
    if original_value in (None, 0) or retained_value is None:
        return None
    return retained_value / original_value


def layer_flops(tokens: float, hidden_size: int, intermediate_size: int) -> float:
    return (
        8 * tokens * hidden_size * hidden_size
        + 4 * tokens * tokens * hidden_size
        + 6 * tokens * hidden_size * intermediate_size
    )


def sample_prefill_flops(
    row: dict[str, Any],
    prefill_tokens: float,
    hidden_size: int,
    intermediate_size: int,
    num_layers: int,
) -> float:
    pruned_tokens = numeric_token_value(row, "pruned_tokens")
    pruning_layer = numeric_token_value(row, "pruning_layer")
    if pruned_tokens is None or pruning_layer is None:
        return num_layers * layer_flops(
            prefill_tokens, hidden_size, intermediate_size
        )

    pre_prune_layers = min(num_layers, max(0, int(pruning_layer) + 1))
    post_prune_layers = num_layers - pre_prune_layers
    return (
        pre_prune_layers
        * layer_flops(prefill_tokens, hidden_size, intermediate_size)
        + post_prune_layers
        * layer_flops(pruned_tokens, hidden_size, intermediate_size)
    )


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * quantile)
    return ordered[index]


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": mean_or_none(values),
        "median": median(values) if values else None,
        "p95": percentile(values, 0.95),
    }


def compare(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
    hidden_size: int = 4096,
    intermediate_size: int = 11008,
    num_layers: int = 32,
    include_groups: bool = True,
) -> dict[str, Any]:
    baseline_map = {sample_key(row): row for row in baseline_rows}
    candidate_map = {sample_key(row): row for row in candidate_rows}
    matched_keys = sorted(baseline_map.keys() & candidate_map.keys())
    if not matched_keys:
        raise ValueError("The result files have no matched samples")

    baseline_matched = [baseline_map[key] for key in matched_keys]
    candidate_matched = [candidate_map[key] for key in matched_keys]
    baseline_correct = [is_correct(row) for row in baseline_matched]
    candidate_correct = [is_correct(row) for row in candidate_matched]

    both_correct = sum(a and b for a, b in zip(baseline_correct, candidate_correct))
    baseline_only = sum(a and not b for a, b in zip(baseline_correct, candidate_correct))
    candidate_only = sum(not a and b for a, b in zip(baseline_correct, candidate_correct))
    both_wrong = len(matched_keys) - both_correct - baseline_only - candidate_only

    deltas = [int(b) - int(a) for a, b in zip(baseline_correct, candidate_correct)]
    delta = fmean(deltas)
    ci_low, ci_high = paired_bootstrap_ci(deltas, bootstrap_samples, seed)

    token_fields = (
        "raw_vision_tokens",
        "merged_vision_tokens",
        "prefill_tokens",
        "pruned_tokens",
    )
    token_stats = {}
    for field in token_fields:
        base_values = numeric_values(baseline_matched, field)
        candidate_values = numeric_values(candidate_matched, field)
        base_mean = mean_or_none(base_values)
        candidate_mean = mean_or_none(candidate_values)
        token_stats[field] = {
            "baseline_mean": base_mean,
            "candidate_mean": candidate_mean,
            "mean_delta": (
                candidate_mean - base_mean
                if base_mean is not None and candidate_mean is not None
                else None
            ),
            "baseline_count": len(base_values),
            "candidate_count": len(candidate_values),
            "baseline": distribution(base_values),
            "candidate": distribution(candidate_values),
        }

    (
        baseline_prefills,
        candidate_prefills,
        prefill_indices,
        prefill_sources,
    ) = paired_prefill_values(baseline_matched, candidate_matched)
    baseline_prefill_mean = mean_or_none(baseline_prefills)
    candidate_prefill_mean = mean_or_none(candidate_prefills)
    token_stats["prefill_tokens"] = {
        "baseline_mean": baseline_prefill_mean,
        "candidate_mean": candidate_prefill_mean,
        "mean_delta": (
            candidate_prefill_mean - baseline_prefill_mean
            if baseline_prefill_mean is not None
            and candidate_prefill_mean is not None
            else None
        ),
        "baseline_count": len(baseline_prefills),
        "candidate_count": len(candidate_prefills),
        "baseline": distribution(baseline_prefills),
        "candidate": distribution(candidate_prefills),
        "sources": prefill_sources,
    }

    for stats in token_stats.values():
        stats["candidate_reduction"] = relative_reduction(
            stats["baseline_mean"], stats["candidate_mean"]
        )

    baseline_raw_mean = token_stats["raw_vision_tokens"]["baseline_mean"]
    candidate_raw_mean = token_stats["raw_vision_tokens"]["candidate_mean"]
    baseline_merged_mean = token_stats["merged_vision_tokens"]["baseline_mean"]
    candidate_merged_mean = token_stats["merged_vision_tokens"]["candidate_mean"]
    baseline_pruned_mean = token_stats["pruned_tokens"]["baseline_mean"]
    candidate_pruned_mean = token_stats["pruned_tokens"]["candidate_mean"]
    compression = {
        "raw_to_merged_retention": {
            "baseline": retention_ratio(baseline_merged_mean, baseline_raw_mean),
            "candidate": retention_ratio(candidate_merged_mean, candidate_raw_mean),
        },
        "raw_to_pruned_retention": {
            "baseline": retention_ratio(baseline_pruned_mean, baseline_raw_mean),
            "candidate": retention_ratio(candidate_pruned_mean, candidate_raw_mean),
        },
        "prefill_to_pruned_retention": {
            "baseline": retention_ratio(
                baseline_pruned_mean, baseline_prefill_mean
            ),
            "candidate": retention_ratio(
                candidate_pruned_mean, candidate_prefill_mean
            ),
        },
    }

    baseline_flops = []
    candidate_flops = []
    for index, baseline_prefill, candidate_prefill in zip(
        prefill_indices, baseline_prefills, candidate_prefills
    ):
        baseline_flops.append(
            sample_prefill_flops(
                baseline_matched[index],
                baseline_prefill,
                hidden_size,
                intermediate_size,
                num_layers,
            )
            / 1e12
        )
        candidate_flops.append(
            sample_prefill_flops(
                candidate_matched[index],
                candidate_prefill,
                hidden_size,
                intermediate_size,
                num_layers,
            )
            / 1e12
        )

    baseline_flops_distribution = distribution(baseline_flops)
    candidate_flops_distribution = distribution(candidate_flops)
    analytical_flops = {
        "unit": "TFLOPs",
        "baseline": baseline_flops_distribution,
        "candidate": candidate_flops_distribution,
        "candidate_mean_reduction": relative_reduction(
            baseline_flops_distribution["mean"],
            candidate_flops_distribution["mean"],
        ),
        "model_dimensions": {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_layers": num_layers,
        },
        "excludes": [
            "vision encoder",
            "multimodal projector",
            "token compressor",
            "autoregressive decode",
        ],
    }

    runtime_metrics = {}
    for field_index, field in enumerate(RUNTIME_FIELDS):
        baseline_values, candidate_values = paired_numeric_values(
            baseline_matched, candidate_matched, field
        )
        baseline_distribution = distribution(baseline_values)
        candidate_distribution = distribution(candidate_values)
        runtime_deltas = [
            candidate - baseline
            for baseline, candidate in zip(
                baseline_values, candidate_values
            )
        ]
        runtime_ci = paired_bootstrap_ci(
            runtime_deltas,
            bootstrap_samples,
            seed + field_index + 1,
        )
        runtime_metrics[field] = {
            "baseline": baseline_distribution,
            "candidate": candidate_distribution,
            "paired_mean_delta": mean_or_none(runtime_deltas),
            "paired_mean_delta_ci95": list(runtime_ci),
            "candidate_mean_reduction": relative_reduction(
                baseline_distribution["mean"], candidate_distribution["mean"]
            ),
        }

    matched_count = len(matched_keys)
    oracle_correct = both_correct + baseline_only + candidate_only
    report = {
        "samples": {
            "baseline_total": len(baseline_rows),
            "candidate_total": len(candidate_rows),
            "matched": matched_count,
            "baseline_unmatched": len(baseline_map) - matched_count,
            "candidate_unmatched": len(candidate_map) - matched_count,
        },
        "accuracy": {
            "baseline": sum(baseline_correct) / matched_count,
            "candidate": sum(candidate_correct) / matched_count,
            "delta": delta,
            "delta_ci95": [ci_low, ci_high],
        },
        "paired_outcomes": {
            "both_correct": both_correct,
            "baseline_correct_candidate_wrong": baseline_only,
            "baseline_wrong_candidate_correct": candidate_only,
            "both_wrong": both_wrong,
            "mcnemar_exact_pvalue": exact_mcnemar_pvalue(baseline_only, candidate_only),
        },
        "oracle_complementarity": {
            "correct": oracle_correct,
            "accuracy": oracle_correct / matched_count,
            "gain_over_baseline": candidate_only / matched_count,
        },
        "tokens": token_stats,
        "compression": compression,
        "analytical_llm_prefill": analytical_flops,
        "runtime": runtime_metrics,
    }
    if include_groups:
        groups = sorted({key[0] for key in matched_keys})
        report["breakdown_by_task_type"] = {}
        for group_index, group in enumerate(groups):
            group_baseline = [
                row
                for row in baseline_matched
                if str(row.get("task_type", "")) == group
            ]
            group_candidate = [
                row
                for row in candidate_matched
                if str(row.get("task_type", "")) == group
            ]
            report["breakdown_by_task_type"][group] = compare(
                group_baseline,
                group_candidate,
                bootstrap_samples=bootstrap_samples,
                seed=seed + 1000 + group_index,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_layers=num_layers,
                include_groups=False,
            )
    return report


def percent(value: float | None) -> str:
    if value is None:
        return "unavailable"
    return f"{100.0 * value:.4f}%"


def print_report(report: dict[str, Any]) -> None:
    samples = report["samples"]
    accuracy = report["accuracy"]
    outcomes = report["paired_outcomes"]

    print("Paired benchmark comparison")
    print(f"  Matched samples: {samples['matched']}")
    print(
        "  Unmatched samples: "
        f"baseline={samples['baseline_unmatched']}, candidate={samples['candidate_unmatched']}"
    )
    print(f"  Baseline accuracy:  {percent(accuracy['baseline'])}")
    print(f"  Candidate accuracy: {percent(accuracy['candidate'])}")
    print(f"  Accuracy delta:     {percent(accuracy['delta'])}")
    print(
        "  Paired 95% CI:     "
        f"[{percent(accuracy['delta_ci95'][0])}, {percent(accuracy['delta_ci95'][1])}]"
    )
    print(
        "  Prediction flips:  "
        f"correct->wrong={outcomes['baseline_correct_candidate_wrong']}, "
        f"wrong->correct={outcomes['baseline_wrong_candidate_correct']}"
    )
    print(f"  Exact McNemar p:    {outcomes['mcnemar_exact_pvalue']:.6g}")
    oracle = report["oracle_complementarity"]
    print(
        "  Oracle union:       "
        f"accuracy={percent(oracle['accuracy'])}, "
        f"gain over baseline={percent(oracle['gain_over_baseline'])}"
    )

    print("  Mean token counts:")
    for field, stats in report["tokens"].items():
        baseline_mean = stats["baseline_mean"]
        candidate_mean = stats["candidate_mean"]
        if baseline_mean is None or candidate_mean is None:
            print(f"    {field}: unavailable")
            continue
        print(
            f"    {field}: baseline={baseline_mean:.2f}, "
            f"candidate={candidate_mean:.2f}, delta={stats['mean_delta']:.2f}, "
            f"reduction={percent(stats['candidate_reduction'])}"
        )
        print(
            f"      median: baseline={stats['baseline']['median']:.2f}, "
            f"candidate={stats['candidate']['median']:.2f}; "
            f"p95: baseline={stats['baseline']['p95']:.2f}, "
            f"candidate={stats['candidate']['p95']:.2f}"
        )
        sources = stats.get("sources")
        if sources is not None:
            print(
                "      sources: "
                f"baseline recorded={sources['baseline']['recorded']}, "
                f"inferred={sources['baseline']['inferred']}; "
                f"candidate recorded={sources['candidate']['recorded']}, "
                f"inferred={sources['candidate']['inferred']}"
            )

    print("  Compression retention ratios:")
    for name, values in report["compression"].items():
        baseline_value = values["baseline"]
        candidate_value = values["candidate"]
        if baseline_value is None or candidate_value is None:
            print(f"    {name}: unavailable")
            continue
        print(
            f"    {name}: baseline={percent(baseline_value)}, "
            f"candidate={percent(candidate_value)}"
        )

    flops = report["analytical_llm_prefill"]
    baseline_flops = flops["baseline"]
    candidate_flops = flops["candidate"]
    print("  Analytical LLM prefill TFLOPs:")
    if baseline_flops["mean"] is None or candidate_flops["mean"] is None:
        print("    unavailable")
    else:
        print(
            f"    mean: baseline={baseline_flops['mean']:.4f}, "
            f"candidate={candidate_flops['mean']:.4f}, "
            f"reduction={percent(flops['candidate_mean_reduction'])}"
        )
        print(
            f"    median: baseline={baseline_flops['median']:.4f}, "
            f"candidate={candidate_flops['median']:.4f}"
        )
        print(
            f"    p95: baseline={baseline_flops['p95']:.4f}, "
            f"candidate={candidate_flops['p95']:.4f}"
        )
    print("    excludes: " + ", ".join(flops["excludes"]))

    print("  Recorded runtime and memory metrics:")
    for field, stats in report["runtime"].items():
        baseline_mean = stats["baseline"]["mean"]
        candidate_mean = stats["candidate"]["mean"]
        if baseline_mean is None or candidate_mean is None:
            print(f"    {field}: unavailable (not recorded in both artifacts)")
            continue
        print(
            f"    {field}: baseline={baseline_mean:.4f}, "
            f"candidate={candidate_mean:.4f}, "
            f"reduction={percent(stats['candidate_mean_reduction'])}"
        )
        print(
            f"      median: baseline={stats['baseline']['median']:.4f}, "
            f"candidate={stats['candidate']['median']:.4f}; "
            f"p95: baseline={stats['baseline']['p95']:.4f}, "
            f"candidate={stats['candidate']['p95']:.4f}"
        )
        runtime_ci = stats["paired_mean_delta_ci95"]
        print(
            "      paired mean-delta 95% CI: "
            f"[{runtime_ci[0]:.4f}, {runtime_ci[1]:.4f}]"
        )

    groups = report.get("breakdown_by_task_type", {})
    if groups:
        print("  Breakdown by task_type:")
        for group, group_report in groups.items():
            group_accuracy = group_report["accuracy"]
            group_flops = group_report["analytical_llm_prefill"]
            print(
                f"    {group} (n={group_report['samples']['matched']}): "
                f"baseline={percent(group_accuracy['baseline'])}, "
                f"candidate={percent(group_accuracy['candidate'])}, "
                f"delta={percent(group_accuracy['delta'])}, "
                "TFLOPs reduction="
                f"{percent(group_flops['candidate_mean_reduction'])}"
            )


def main() -> None:
    args = parse_args()
    report = compare(
        load_results(args.baseline),
        load_results(args.candidate),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_layers=args.num_layers,
    )
    print_report(report)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
