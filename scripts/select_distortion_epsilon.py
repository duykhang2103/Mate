#!/usr/bin/env python3
"""Select a distortion threshold using compute only, never labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

try:
    from scripts.estimate_llm_prefill_flops import load_rows, sample_flops
except ModuleNotFoundError:  # Direct execution: python scripts/...
    from estimate_llm_prefill_flops import load_rows, sample_flops


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--target-tflops", type=float, required=True)
    parser.add_argument(
        "--mode",
        choices=("match", "at-most"),
        default="match",
        help="Match within tolerance, or choose the least compressed result at/below target.",
    )
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--required-videos-per-group", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=11008)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def summarize_result(
    path: Path,
    hidden_size: int,
    intermediate_size: int,
    num_layers: int,
) -> dict[str, Any]:
    rows = load_rows(path)
    epsilons = {
        float((row.get("token_info") or {}).get("routing_epsilon"))
        for row in rows
        if isinstance(
            (row.get("token_info") or {}).get("routing_epsilon"),
            (int, float),
        )
    }
    if len(epsilons) != 1:
        raise ValueError(
            f"{path} must record exactly one routing_epsilon; got {epsilons}"
        )

    estimates = []
    merged_tokens = []
    videos_by_group: dict[str, set[str]] = {}
    sample_keys = []
    for row in rows:
        token_info = row.get("token_info") or {}
        estimate = sample_flops(
            token_info,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_layers=num_layers,
        )
        if estimate is None:
            raise ValueError(
                f"{path} has rows without prefill/pruning telemetry"
            )
        estimates.append(estimate / 1e12)
        merged = token_info.get("merged_vision_tokens")
        if isinstance(merged, (int, float)):
            merged_tokens.append(float(merged))
        group = str(row.get("task_type", ""))
        video = str(row.get("video_path", row.get("video", "")))
        videos_by_group.setdefault(group, set()).add(video)
        sample_keys.append((group, video, str(row.get("question", ""))))

    return {
        "path": str(path),
        "epsilon": epsilons.pop(),
        "samples": len(rows),
        "sample_keys": sorted(sample_keys),
        "mean_tflops": fmean(estimates),
        "mean_merged_tokens": fmean(merged_tokens) if merged_tokens else None,
        "unique_videos_by_group": {
            group: len(videos) for group, videos in videos_by_group.items()
        },
    }


def select_configuration(
    summaries: list[dict[str, Any]],
    target_tflops: float,
    mode: str,
    tolerance: float,
    required_videos_per_group: int,
) -> dict[str, Any]:
    if target_tflops <= 0:
        raise ValueError("target_tflops must be positive")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if not summaries:
        raise ValueError("at least one calibration result is required")

    cohort = summaries[0]["sample_keys"]
    if any(summary["sample_keys"] != cohort for summary in summaries[1:]):
        raise ValueError("all calibration artifacts must contain the same cohort")
    if required_videos_per_group:
        for summary in summaries:
            counts = summary["unique_videos_by_group"]
            if not counts or any(
                count < required_videos_per_group
                for count in counts.values()
            ):
                raise ValueError(
                    f"{summary['path']} does not contain "
                    f"{required_videos_per_group} videos in every group: {counts}"
                )

    if mode == "match":
        selected = min(
            summaries,
            key=lambda summary: abs(
                summary["mean_tflops"] - target_tflops
            ),
        )
        target_passed = (
            abs(selected["mean_tflops"] - target_tflops) / target_tflops
            <= tolerance
        )
    else:
        eligible = [
            summary
            for summary in summaries
            if summary["mean_tflops"] <= target_tflops
        ]
        if eligible:
            selected = max(eligible, key=lambda summary: summary["mean_tflops"])
            target_passed = True
        else:
            selected = min(summaries, key=lambda summary: summary["mean_tflops"])
            target_passed = False

    public_summaries = [
        {key: value for key, value in summary.items() if key != "sample_keys"}
        for summary in sorted(summaries, key=lambda item: item["epsilon"])
    ]
    public_selected = {
        key: value for key, value in selected.items() if key != "sample_keys"
    }
    return {
        "selection_uses_labels": False,
        "mode": mode,
        "target_tflops": target_tflops,
        "tolerance": tolerance,
        "target_passed": target_passed,
        "selected": public_selected,
        "candidates": public_summaries,
    }


def main() -> None:
    args = parse_args()
    summaries = [
        summarize_result(
            path,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_layers=args.num_layers,
        )
        for path in args.results
    ]
    result = select_configuration(
        summaries,
        target_tflops=args.target_tflops,
        mode=args.mode,
        tolerance=args.tolerance,
        required_videos_per_group=args.required_videos_per_group,
    )
    selected = result["selected"]
    print("Distortion epsilon calibration (accuracy labels not read)")
    print(f"  Target:   {result['target_tflops']:.4f} TFLOPs ({result['mode']})")
    print(f"  Epsilon:  {selected['epsilon']:.8g}")
    print(f"  Achieved: {selected['mean_tflops']:.4f} TFLOPs")
    print(f"  Gate:     {'PASS' if result['target_passed'] else 'FAIL'}")

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)

    if not result["target_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
