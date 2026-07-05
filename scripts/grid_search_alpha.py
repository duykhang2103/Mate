#!/usr/bin/env python3
"""
Grid search over alpha values to build accuracy vs FLOPs tradeoff curve.
Usage: python scripts/grid_search_alpha.py
"""
import subprocess
import json
import os
from pathlib import Path

ALPHA_VALUES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
TASKS_5 = "Action Sequence,Action Prediction,Moving Direction,Object Interaction,Unexpected Action"
BASE_CMD = [
    "conda", "run", "-n", "pllava", "python", "-m", "tasks.eval.mvbench.pllava_eval_mvbench",
    "--pretrained_model_name_or_path", "MODELS/pllava-7b",
    "--num_frames", "16", "--use_lora", "--lora_alpha", "14",
    "--weight_dir", "MODELS/pllava-7b",
    "--pooling_shape", "16-12-12",
    "--selected_layer", "10", "--tau", "0.8",
    "--temporal_segment_ratio", "0.25", "--cluster_ratio", "0.5",
    "--use_flow_pruning", "--flow_dynamic_ratio", "0.5",
    "--max_samples", "200",
]

def run_alpha_search():
    results = {}
    for alpha in ALPHA_VALUES:
        save_path = f"test_results/grid_search_alpha_{alpha}"
        log_file = f"log_grid_search_alpha_{alpha}.log"
        cmd = BASE_CMD + [
            "--alpha", str(alpha),
            "--save_path", save_path,
            "--tasks", TASKS_5,
        ]
        print(f"\n{'='*60}")
        print(f"Running alpha={alpha}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}")
        
        with open(log_file, "w") as log:
            subprocess.run(cmd, stdout=log, stderr=log)
        
        # Read results
        result_file = os.path.join(save_path, "upload_leaderboard.json")
        if os.path.exists(result_file):
            with open(result_file) as f:
                data = json.load(f)
            results[alpha] = data
            print(f"Alpha {alpha}: {json.dumps(data, indent=2)}")
        else:
            print(f"Alpha {alpha}: No results found")
            results[alpha] = None
    
    # Save summary
    with open("test_results/grid_search_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to test_results/grid_search_summary.json")

if __name__ == "__main__":
    run_alpha_search()
