#!/usr/bin/env python3
"""Train the conditional SafePruneVid risk router with frozen PLLaVA outputs.

Run this only after both training-free ADBR configurations fail acceptance.
The script enforces the +1.0-point oracle-complementarity gate, splits by video
within each duration, fits a tiny logistic model, and chooses its threshold on
validation accuracy subject to compute remaining below original PruneVid.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.compare_paired_results import (
        compare,
        is_correct,
        load_results,
        sample_key,
    )
    from scripts.estimate_llm_prefill_flops import sample_flops
except ModuleNotFoundError:  # Direct execution: python scripts/...
    from compare_paired_results import compare, is_correct, load_results, sample_key
    from estimate_llm_prefill_flops import sample_flops


REQUIRED_FEATURE_FIELDS = (
    "routing_distortion_mean",
    "routing_distortion_std",
    "routing_distortion_q25",
    "routing_distortion_q50",
    "routing_distortion_q75",
    "routing_distortion_q95",
    "routing_num_windows",
    "routing_window_size_mean",
    "routing_window_size_std",
    "routing_window_size_min",
    "routing_window_size_max",
    "merged_vision_tokens",
    "routing_baseline_similarity_merged_tokens",
)
OPTIONAL_FEATURE_FIELDS = ("query_merge_reliability",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--routed-results-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2_000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--oracle-gain-minimum", type=float, default=0.01)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


def align_rows(
    original_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    original_map = {sample_key(row): row for row in original_rows}
    candidate_map = {sample_key(row): row for row in candidate_rows}
    if original_map.keys() != candidate_map.keys():
        raise ValueError(
            "original and candidate artifacts must contain exactly the same samples"
        )
    keys = sorted(original_map)
    return [original_map[key] for key in keys], [candidate_map[key] for key in keys]


def feature_fields(candidate_rows: list[dict[str, Any]]) -> list[str]:
    for field in REQUIRED_FEATURE_FIELDS:
        if not all(
            isinstance((row.get("token_info") or {}).get(field), (int, float))
            for row in candidate_rows
        ):
            raise ValueError(f"candidate artifact is missing router feature {field}")
    fields = list(REQUIRED_FEATURE_FIELDS)
    for field in OPTIONAL_FEATURE_FIELDS:
        if all(
            isinstance((row.get("token_info") or {}).get(field), (int, float))
            for row in candidate_rows
        ):
            fields.append(field)
    return fields


def feature_matrix(
    rows: list[dict[str, Any]], fields: list[str]
) -> np.ndarray:
    return np.asarray(
        [
            [float((row.get("token_info") or {})[field]) for field in fields]
            for row in rows
        ],
        dtype=np.float64,
    )


def stratified_video_split(
    rows: list[dict[str, Any]], seed: int
) -> dict[str, list[int]]:
    grouped_videos: dict[str, dict[str, list[int]]] = {}
    for index, row in enumerate(rows):
        group = str(row.get("task_type", ""))
        video = str(row.get("video_path", row.get("video", "")))
        grouped_videos.setdefault(group, {}).setdefault(video, []).append(index)

    rng = random.Random(seed)
    split_indices = {"train": [], "validation": [], "test": []}
    for group, video_rows in sorted(grouped_videos.items()):
        videos = sorted(video_rows)
        if len(videos) < 3:
            raise ValueError(
                f"duration/task group {group!r} needs at least three videos"
            )
        rng.shuffle(videos)
        train_count = max(1, int(0.6 * len(videos)))
        validation_count = max(1, int(0.2 * len(videos)))
        if train_count + validation_count >= len(videos):
            train_count = len(videos) - 2
            validation_count = 1
        assignments = {
            "train": videos[:train_count],
            "validation": videos[
                train_count:train_count + validation_count
            ],
            "test": videos[train_count + validation_count:],
        }
        for split, assigned_videos in assignments.items():
            for video in assigned_videos:
                split_indices[split].extend(video_rows[video])

    return {
        split: sorted(indices) for split, indices in split_indices.items()
    }


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_logistic_router(
    features: np.ndarray,
    targets: np.ndarray,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    if targets.min() == targets.max():
        raise ValueError("training split must contain both router target classes")
    means = features.mean(axis=0)
    scales = features.std(axis=0)
    scales[scales < 1e-8] = 1.0
    standardized = (features - means) / scales
    weights = np.zeros(features.shape[1], dtype=np.float64)
    bias = 0.0
    positives = float(targets.sum())
    negatives = float(len(targets) - positives)
    positive_weight = negatives / positives
    sample_weights = np.where(targets > 0.5, positive_weight, 1.0)
    normalizer = sample_weights.sum()

    for _ in range(epochs):
        probabilities = sigmoid(standardized @ weights + bias)
        residual = (probabilities - targets) * sample_weights
        gradient = standardized.T @ residual / normalizer + l2 * weights
        bias_gradient = float(residual.sum() / normalizer)
        weights -= learning_rate * gradient
        bias -= learning_rate * bias_gradient
    return weights, bias, means, scales


def predict_risk(
    features: np.ndarray,
    weights: np.ndarray,
    bias: float,
    means: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    return sigmoid(((features - means) / scales) @ weights + bias)


def row_tflops(row: dict[str, Any]) -> float:
    estimate = sample_flops(
        row.get("token_info") or {},
        hidden_size=4096,
        intermediate_size=11008,
        num_layers=32,
    )
    if estimate is None:
        raise ValueError("router artifacts require complete prefill telemetry")
    return estimate / 1e12


def choose_threshold(
    risks: np.ndarray,
    original_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, float]:
    original_compute = np.asarray([row_tflops(row) for row in original_rows])
    candidate_compute = np.asarray([row_tflops(row) for row in candidate_rows])
    original_correct = np.asarray([is_correct(row) for row in original_rows])
    candidate_correct = np.asarray([is_correct(row) for row in candidate_rows])
    baseline_mean_compute = float(original_compute.mean())
    thresholds = sorted(set(float(value) for value in risks))
    thresholds.append(max(thresholds) + 1.0)
    feasible = []
    for threshold in thresholds:
        use_original = risks >= threshold
        routed_correct = np.where(
            use_original, original_correct, candidate_correct
        )
        routed_compute = np.where(
            use_original, original_compute, candidate_compute
        )
        mean_compute = float(routed_compute.mean())
        if mean_compute < baseline_mean_compute:
            feasible.append(
                {
                    "threshold": threshold,
                    "accuracy": float(routed_correct.mean()),
                    "mean_tflops": mean_compute,
                    "original_route_fraction": float(use_original.mean()),
                }
            )
    if not feasible:
        raise ValueError(
            "no validation threshold keeps mean TFLOPs below original PruneVid"
        )
    return max(
        feasible,
        key=lambda item: (item["accuracy"], -item["mean_tflops"]),
    )


def routed_rows(
    risks: np.ndarray,
    threshold: float,
    original_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    routed = []
    for risk, original, candidate in zip(
        risks, original_rows, candidate_rows
    ):
        use_original = bool(risk >= threshold)
        selected = dict(original if use_original else candidate)
        selected["token_info"] = dict(selected.get("token_info") or {})
        selected["token_info"]["safety_router_risk"] = float(risk)
        selected["token_info"]["safety_router_used_original"] = use_original
        routed.append(selected)
    return routed


def indexed(rows: list[dict[str, Any]], indices: list[int]) -> list[dict[str, Any]]:
    return [rows[index] for index in indices]


def main() -> None:
    args = parse_args()
    original_rows, candidate_rows = align_rows(
        load_results(args.original), load_results(args.candidate)
    )
    candidate_only = sum(
        not is_correct(original) and is_correct(candidate)
        for original, candidate in zip(original_rows, candidate_rows)
    )
    oracle_gain = candidate_only / len(original_rows)
    if oracle_gain < args.oracle_gain_minimum:
        raise SystemExit(
            f"Safety router blocked: oracle gain {oracle_gain:.4%} is below "
            f"{args.oracle_gain_minimum:.4%}."
        )

    fields = feature_fields(candidate_rows)
    features = feature_matrix(candidate_rows, fields)
    targets = np.asarray(
        [
            is_correct(original) and not is_correct(candidate)
            for original, candidate in zip(original_rows, candidate_rows)
        ],
        dtype=np.float64,
    )
    splits = stratified_video_split(candidate_rows, args.seed)
    train_indices = splits["train"]
    validation_indices = splits["validation"]
    test_indices = splits["test"]
    weights, bias, means, scales = fit_logistic_router(
        features[train_indices],
        targets[train_indices],
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )
    validation_risks = predict_risk(
        features[validation_indices], weights, bias, means, scales
    )
    threshold_selection = choose_threshold(
        validation_risks,
        indexed(original_rows, validation_indices),
        indexed(candidate_rows, validation_indices),
    )
    test_risks = predict_risk(
        features[test_indices], weights, bias, means, scales
    )
    test_original = indexed(original_rows, test_indices)
    test_candidate = indexed(candidate_rows, test_indices)
    test_routed = routed_rows(
        test_risks,
        threshold_selection["threshold"],
        test_original,
        test_candidate,
    )
    test_comparison = compare(
        test_original,
        test_routed,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )

    model_artifact = {
        "type": "logistic_safety_router",
        "feature_fields": fields,
        "means": means.tolist(),
        "scales": scales.tolist(),
        "weights": weights.tolist(),
        "bias": bias,
        "threshold": threshold_selection["threshold"],
        "validation_selection": threshold_selection,
        "seed": args.seed,
    }
    report = {
        "oracle_gain": oracle_gain,
        "split_samples": {
            split: len(indices) for split, indices in splits.items()
        },
        "positive_targets": {
            split: int(targets[indices].sum())
            for split, indices in splits.items()
        },
        "validation_selection": threshold_selection,
        "test_comparison": test_comparison,
    }
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.routed_results_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    with args.model_output.open("w", encoding="utf-8") as handle:
        json.dump(model_artifact, handle, indent=2)
    with args.routed_results_output.open("w", encoding="utf-8") as handle:
        json.dump({"result_list": test_routed}, handle, indent=2)
    with args.report_output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    accuracy = test_comparison["accuracy"]
    print("Safety router test split")
    print(f"  Oracle gain gate: {oracle_gain * 100.0:.4f} points")
    print(f"  Original accuracy: {accuracy['baseline'] * 100.0:.4f}%")
    print(f"  Routed accuracy:   {accuracy['candidate'] * 100.0:.4f}%")
    print(f"  Delta:             {accuracy['delta'] * 100.0:.4f} points")


if __name__ == "__main__":
    main()
