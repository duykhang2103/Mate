#!/usr/bin/env python3
"""Estimate decoder-only prefill FLOPs from PruneVid result token counts.

This estimates the dominant dense LLM operations and intentionally excludes the
vision encoder, multimodal projector, compression method, normalization, and
generation/decode steps. Report it as *analytical LLM prefill TFLOPs*, not as
end-to-end TFLOPs.

The estimate for one decoder layer with sequence length n is:

    8*n*d^2 + 4*n^2*d + 6*n*d*d_ff

which covers Q/K/V/O projections, QK/AV attention, and the three SwiGLU MLP
projections under the convention that one multiply-add is two FLOPs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="Path to all_results.json")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=11008)
    parser.add_argument("--num-layers", type=int, default=32)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("result_list")
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain a result_list array")
    return rows


def layer_flops(tokens: int, hidden_size: int, intermediate_size: int) -> int:
    return (
        8 * tokens * hidden_size * hidden_size
        + 4 * tokens * tokens * hidden_size
        + 6 * tokens * hidden_size * intermediate_size
    )


def sample_flops(
    token_info: dict[str, Any],
    hidden_size: int,
    intermediate_size: int,
    num_layers: int,
) -> int | None:
    prefill_tokens = token_info.get("prefill_tokens")
    if not isinstance(prefill_tokens, int) or prefill_tokens <= 0:
        return None

    pruned_tokens = token_info.get("pruned_tokens")
    pruning_layer = token_info.get("pruning_layer")
    if not isinstance(pruned_tokens, int) or not isinstance(pruning_layer, int):
        return num_layers * layer_flops(
            prefill_tokens, hidden_size, intermediate_size
        )

    pre_prune_layers = min(num_layers, max(0, pruning_layer + 1))
    post_prune_layers = num_layers - pre_prune_layers
    return (
        pre_prune_layers
        * layer_flops(prefill_tokens, hidden_size, intermediate_size)
        + post_prune_layers
        * layer_flops(pruned_tokens, hidden_size, intermediate_size)
    )


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * quantile)
    return ordered[index]


def main() -> None:
    args = parse_args()
    estimates = []
    missing = 0
    for row in load_rows(args.result):
        estimate = sample_flops(
            row.get("token_info") or {},
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_layers=args.num_layers,
        )
        if estimate is None:
            missing += 1
        else:
            estimates.append(estimate / 1e12)

    if not estimates:
        raise SystemExit(
            "No rows contain token_info.prefill_tokens. Re-run the benchmark "
            "with the current instrumentation."
        )

    print("Analytical LLM prefill TFLOPs")
    print(f"  Samples estimated: {len(estimates)}")
    print(f"  Samples missing:   {missing}")
    print(f"  Mean:              {fmean(estimates):.4f}")
    print(f"  Median:            {percentile(estimates, 0.5):.4f}")
    print(f"  P95:               {percentile(estimates, 0.95):.4f}")
    print("  Excludes: vision encoder, projector, compressor, decode")


if __name__ == "__main__":
    main()
