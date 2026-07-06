#!/usr/bin/env python3
"""
MVBench Dataset Completeness Checker

Checks that all JSON entries have corresponding video files on disk.
Run before evaluation to avoid silent failures during benchmarking.

Usage:
    python scripts/check_mvbench_data.py [--json_dir DATAS/MVBench/json] [--verbose]
"""

import json
import os
import argparse
from pathlib import Path

# MVBench task definitions: (json_filename, video_prefix)
TASK_DEFS = {
    "Action Sequence":          ("action_sequence.json",          "DATAS/MVBench/video/star/Charades_v1_480/"),
    "Action Prediction":        ("action_prediction.json",        "DATAS/MVBench/video/star/Charades_v1_480/"),
    "Action Antonym":           ("action_antonym.json",           "DATAS/MVBench/video/ssv2_video/"),
    "Fine-grained Action":      ("fine_grained_action.json",      "DATAS/MVBench/video/Moments_in_Time_Raw/videos/"),
    "Unexpected Action":        ("unexpected_action.json",        "DATAS/MVBench/video/FunQA_test/test/"),
    "Object Existence":         ("object_existence.json",         "DATAS/MVBench/video/clevrer/video_validation/"),
    "Object Interaction":       ("object_interaction.json",       "DATAS/MVBench/video/star/Charades_v1_480/"),
    "Object Shuffle":           ("object_shuffle.json",           "DATAS/MVBench/video/perception/videos/"),
    "Moving Direction":         ("moving_direction.json",         "DATAS/MVBench/video/clevrer/video_validation/"),
    "Action Localization":      ("action_localization.json",      "DATAS/MVBench/video/sta/sta_video/"),
    "Scene Transition":         ("scene_transition.json",         "DATAS/MVBench/video/scene_qa/video/"),
    "Action Count":             ("action_count.json",             "DATAS/MVBench/video/perception/videos/"),
    "Moving Count":             ("moving_count.json",             "DATAS/MVBench/video/clevrer/video_validation/"),
    "Moving Attribute":         ("moving_attribute.json",         "DATAS/MVBench/video/clevrer/video_validation/"),
    "State Change":             ("state_change.json",             "DATAS/MVBench/video/perception/videos/"),
    "Fine-grained Pose":        ("fine_grained_pose.json",        "DATAS/MVBench/video/nturgbd/"),
    "Character Order":          ("character_order.json",          "DATAS/MVBench/video/perception/videos/"),
    "Egocentric Navigation":    ("egocentric_navigation.json",    "DATAS/MVBench/video/vlnqa/"),
    "Episodic Reasoning":       ("episodic_reasoning.json",       "DATAS/MVBench/video/tvqa/frames_fps3_hq/"),
    "Counterfactual Inference": ("counterfactual_inference.json", "DATAS/MVBench/video/clevrer/video_validation/"),
}


def check_dataset(json_dir="DATAS/MVBench/json", verbose=False):
    """Check completeness of MVBench dataset.

    Returns:
        dict: {task_name: {"total": int, "missing": int, "missing_files": list}}
    """
    results = {}
    total_ok = 0
    total_missing = 0

    for task_name, (json_file, prefix) in TASK_DEFS.items():
        json_path = os.path.join(json_dir, json_file)
        if not os.path.exists(json_path):
            results[task_name] = {"total": 0, "missing": -1, "missing_files": [], "error": f"JSON not found: {json_path}"}
            print(f"[MISS] {task_name}: JSON file not found ({json_path})")
            continue

        with open(json_path, "r") as f:
            data = json.load(f)

        missing_files = []
        for entry in data:
            video_path = os.path.join(prefix, entry["video"])
            if not os.path.exists(video_path):
                missing_files.append(entry["video"])

        count = len(data)
        miss_count = len(missing_files)
        results[task_name] = {
            "total": count,
            "missing": miss_count,
            "missing_files": missing_files,
        }
        total_ok += count - miss_count
        total_missing += miss_count

        status = "OK" if miss_count == 0 else f"MISS {miss_count}/{count}"
        print(f"[{status}] {task_name}: {count - miss_count}/{count} samples available")

        if verbose and missing_files:
            for f in missing_files[:5]:
                print(f"         missing: {prefix}{f}")
            if len(missing_files) > 5:
                print(f"         ... and {len(missing_files) - 5} more")

    print(f"\n{'='*50}")
    print(f"Total: {total_ok}/4000 samples available, {total_missing} missing")
    if total_missing == 0:
        print("Dataset is COMPLETE. Ready for full evaluation.")
    else:
        print(f"Dataset is INCOMPLETE. {total_missing} samples will be skipped during eval.")
        print("Re-extract the missing zips or re-download the dataset.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check MVBench dataset completeness")
    parser.add_argument("--json_dir", type=str, default="DATAS/MVBench/json", help="Path to MVBench JSON directory")
    parser.add_argument("--verbose", action="store_true", help="Print missing file names")
    args = parser.parse_args()
    check_dataset(json_dir=args.json_dir, verbose=args.verbose)
