#!/usr/bin/env python3
"""Evaluate the two predeclared SafePruneVid Pareto acceptance rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.compare_paired_results import compare, load_results
except ModuleNotFoundError:  # Direct execution: python scripts/...
    from compare_paired_results import compare, load_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--fail-on-reject", action="store_true")
    return parser.parse_args()


def evaluate_acceptance(report: dict) -> dict:
    accuracy = report["accuracy"]
    outcomes = report["paired_outcomes"]
    token_stats = report["tokens"]
    prefill_runtime = report["runtime"]["prefill_latency_ms"]
    analytical_reduction = report["analytical_llm_prefill"][
        "candidate_mean_reduction"
    ]
    latency_reduction = prefill_runtime["candidate_mean_reduction"]

    token_checks = {}
    for field in ("merged_vision_tokens", "prefill_tokens", "pruned_tokens"):
        baseline_mean = token_stats[field]["baseline_mean"]
        candidate_mean = token_stats[field]["candidate_mean"]
        token_checks[field] = (
            baseline_mean is not None
            and candidate_mean is not None
            and candidate_mean <= baseline_mean
        )
    no_higher_mean_tokens = all(token_checks.values())
    no_higher_measured_compute = (
        prefill_runtime["baseline"]["mean"] is not None
        and prefill_runtime["candidate"]["mean"] is not None
        and prefill_runtime["candidate"]["mean"]
        <= prefill_runtime["baseline"]["mean"]
    )
    rule_accuracy_gain = (
        accuracy["delta"] > 0
        and outcomes["mcnemar_exact_pvalue"] < 0.05
        and no_higher_mean_tokens
        and no_higher_measured_compute
    )
    noninferiority = accuracy["delta_ci95"][0] > -0.003
    compute_reduction_20 = (
        analytical_reduction is not None and analytical_reduction >= 0.20
    ) or (
        latency_reduction is not None and latency_reduction >= 0.20
    )
    rule_efficiency_noninferiority = noninferiority and compute_reduction_20
    accepted = rule_accuracy_gain or rule_efficiency_noninferiority
    return {
        "decision": "ACCEPT" if accepted else "REJECT",
        "accepted": accepted,
        "rule_accuracy_gain": {
            "passed": rule_accuracy_gain,
            "positive_accuracy_delta": accuracy["delta"] > 0,
            "mcnemar_p_below_0_05": outcomes["mcnemar_exact_pvalue"] < 0.05,
            "no_higher_mean_tokens": no_higher_mean_tokens,
            "token_checks": token_checks,
            "no_higher_measured_prefill_latency": no_higher_measured_compute,
        },
        "rule_efficiency_noninferiority": {
            "passed": rule_efficiency_noninferiority,
            "accuracy_ci_lower_above_minus_0_3_points": noninferiority,
            "analytical_tflops_reduction_at_least_20_percent": (
                analytical_reduction is not None
                and analytical_reduction >= 0.20
            ),
            "prefill_latency_reduction_at_least_20_percent": (
                latency_reduction is not None and latency_reduction >= 0.20
            ),
        },
        "comparison": report,
    }


def main() -> None:
    args = parse_args()
    report = compare(
        load_results(args.baseline),
        load_results(args.candidate),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    result = evaluate_acceptance(report)
    print(f"SafePruneVid acceptance gate: {result['decision']}")
    print(
        "  Significant accuracy-gain rule: "
        f"{'PASS' if result['rule_accuracy_gain']['passed'] else 'FAIL'}"
    )
    print(
        "  Efficiency/noninferiority rule: "
        f"{'PASS' if result['rule_efficiency_noninferiority']['passed'] else 'FAIL'}"
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    with args.json_output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    if args.fail_on_reject and not result["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
