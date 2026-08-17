#!/usr/bin/env python3
"""Apply SafePruneVid's predeclared query-component kill gate."""

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
    parser.add_argument(
        "--novelty-only",
        type=Path,
        required=True,
        help="Full weight=0.0 result artifact.",
    )
    parser.add_argument(
        "--query-guided",
        type=Path,
        required=True,
        help="Full weight=0.7 result artifact at the same dynamic ratio.",
    )
    parser.add_argument(
        "--token-budget-tolerance",
        type=float,
        default=0.01,
        help="Maximum relative difference in mean merged tokens (default 1%%).",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def apply_gate(
    novelty_rows: list[dict],
    query_rows: list[dict],
    token_budget_tolerance: float,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    if token_budget_tolerance < 0:
        raise ValueError("token_budget_tolerance must be non-negative")

    report = compare(
        novelty_rows,
        query_rows,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        include_groups=False,
    )
    samples = report["samples"]
    accuracy = report["accuracy"]
    outcomes = report["paired_outcomes"]
    tokens = report["tokens"]["merged_vision_tokens"]
    novelty_tokens = tokens["baseline_mean"]
    query_tokens = tokens["candidate_mean"]
    if novelty_tokens in (None, 0) or query_tokens is None:
        relative_budget_difference = None
        budget_matched = False
    else:
        relative_budget_difference = abs(
            query_tokens - novelty_tokens
        ) / novelty_tokens
        budget_matched = (
            relative_budget_difference <= token_budget_tolerance
        )

    complete_pairing = (
        samples["baseline_unmatched"] == 0
        and samples["candidate_unmatched"] == 0
        and samples["matched"] == len(novelty_rows) == len(query_rows)
    )
    positive_delta = accuracy["delta"] > 0
    significant = outcomes["mcnemar_exact_pvalue"] < 0.05
    retain_query = (
        complete_pairing
        and budget_matched
        and positive_delta
        and significant
    )
    return {
        "decision": "RETAIN_QUERY" if retain_query else "REMOVE_QUERY",
        "retain_query": retain_query,
        "checks": {
            "complete_pairing": complete_pairing,
            "positive_accuracy_delta": positive_delta,
            "mcnemar_p_below_0_05": significant,
            "same_merged_token_budget": budget_matched,
        },
        "relative_merged_token_difference": relative_budget_difference,
        "token_budget_tolerance": token_budget_tolerance,
        "comparison": report,
    }


def main() -> None:
    args = parse_args()
    result = apply_gate(
        load_results(args.novelty_only),
        load_results(args.query_guided),
        token_budget_tolerance=args.token_budget_tolerance,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    comparison = result["comparison"]
    print(f"Query-component kill gate: {result['decision']}")
    print(
        "  Accuracy delta (query - novelty): "
        f"{comparison['accuracy']['delta'] * 100.0:.4f} points"
    )
    print(
        "  Exact McNemar p: "
        f"{comparison['paired_outcomes']['mcnemar_exact_pvalue']:.6g}"
    )
    budget_difference = result["relative_merged_token_difference"]
    if budget_difference is None:
        print("  Merged-token budget difference: unavailable")
    else:
        print(
            "  Merged-token budget difference: "
            f"{budget_difference * 100.0:.4f}%"
        )
    for check, passed in result["checks"].items():
        print(f"  {check}: {'PASS' if passed else 'FAIL'}")

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
