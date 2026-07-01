"""
Sanity check: Does optical flow magnitude correlate with PLLaVA errors?

Computes dense optical flow (Farneback) on 20 long VideoMME videos,
runs PLLaVA inference on each question, and checks if high-motion
videos have higher error rates.

Usage:
    conda run -n pllava python scripts/flow_correlation_sanity_check.py \
        --pretrained_model_name_or_path MODELS/pllava-7b \
        --use_lora --weight_dir MODELS/pllava-7b \
        --num_frames 4 --alpha 0.4 --tau 0.8 --selected_layer 10
"""

import argparse
import json
import os
import sys
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def compute_flow_magnitude(video_path, num_samples=32):
    """Compute average optical flow magnitude for a video using Farneback."""
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0))
    n_frames = len(vr)
    indices = np.linspace(0, n_frames - 1, num_samples, dtype=int)
    frames = vr.get_batch(indices).asnumpy()

    flow_mags = []
    for i in range(len(frames) - 1):
        g1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_mags.append(float(mag.mean()))
    return {
        "avg_flow": float(np.mean(flow_mags)),
        "std_flow": float(np.std(flow_mags)),
        "per_pair_flow": flow_mags,
        "duration_s": n_frames / 30.0,
    }


def load_questions_for_videos(video_names, data_dir="DATAS/Video-MME/json"):
    """Load all questions for given video names from long.json."""
    with open(os.path.join(data_dir, "long.json")) as f:
        entries = json.load(f)
    return [e for e in entries if e["video"] in video_names]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", required=True)
    parser.add_argument("--weight_dir", default=None)
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--selected_layer", type=int, default=10)
    parser.add_argument("--pooling_shape", default="16-12-12")
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--softmax", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--save_dir", default="doc/eval/flow_correlation")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Find available long videos
    with open("DATAS/Video-MME/json/long.json") as f:
        all_entries = json.load(f)
    video_names = list(set(e["video"] for e in all_entries))
    available = [v for v in video_names if os.path.exists(f"DATAS/Video-MME/data/{v}")]
    sample = sorted(available)[:20]
    print(f"Using {len(sample)} long videos")

    # 2. Compute optical flow for each video
    print("Computing optical flow...")
    flow_results = {}
    for vname in sample:
        path = f"DATAS/Video-MME/data/{vname}"
        flow_results[vname] = compute_flow_magnitude(path)
        print(f"  {vname}: flow={flow_results[vname]['avg_flow']:.4f} ± {flow_results[vname]['std_flow']:.4f}")

    # 3. Load questions
    questions = load_questions_for_videos(sample)
    print(f"\nTotal questions: {len(questions)}")

    # 4. Load model
    import torch
    from tasks.eval.model_utils import load_pllava
    from tasks.eval.videomme import check_ans

    pooling_shape = tuple(int(x) for x in args.pooling_shape.split("-"))
    weight_dir = args.weight_dir or args.pretrained_model_name_or_path

    print("Loading PLLaVA model...")
    model, processor = load_pllava(
        args.pretrained_model_name_or_path,
        num_frames=args.num_frames,
        weight_dir=weight_dir,
        pooling_shape=pooling_shape,
        selected_layer=args.selected_layer,
        alpha=args.alpha,
        softmax=args.softmax,
        head=args.head,
        tau=args.tau,
    )

    # Fix LoRA-PEFT checkpoint: merge LoRA weights into base weights
    from safetensors.torch import load_file
    import glob as globmod
    safetensors_files = sorted(globmod.glob(os.path.join(weight_dir, "*.safetensors")))
    if safetensors_files:
        all_weights = {}
        for sf in safetensors_files:
            all_weights.update(load_file(sf))
        has_lora = any('lora_A' in k for k in all_weights.keys())
        if has_lora:
            print("Detected LoRA checkpoint, merging weights...")
            lora_pairs = {}
            base_weights = {}
            for k, v in all_weights.items():
                # Remap key: language_model.base_model.model.model.X -> language_model.model.X
                # and language_model.base_model.model.lm_head -> language_model.lm_head
                nk = k
                if '.base_model.model.model.' in nk:
                    nk = nk.replace('.base_model.model.model.', '.model.')
                elif '.base_model.model.' in nk:
                    nk = nk.replace('.base_model.model.', '.')

                if 'lora_A.default.weight' in nk:
                    base_key = nk.replace('.lora_A.default.weight', '.weight')
                    lora_pairs.setdefault(base_key, {})['lora_A'] = v
                elif 'lora_B.default.weight' in nk:
                    base_key = nk.replace('.lora_B.default.weight', '.weight')
                    lora_pairs.setdefault(base_key, {})['lora_B'] = v
                elif '.base_layer.weight' in nk:
                    base_key = nk.replace('.base_layer.weight', '.weight')
                    base_weights[base_key] = v
                else:
                    base_weights[nk] = v
            # Merge LoRA: merged = base + lora_B @ lora_A
            merged_count = 0
            for base_key, lora in lora_pairs.items():
                if base_key in base_weights and 'lora_A' in lora and 'lora_B' in lora:
                    merged = base_weights[base_key].float() + (lora['lora_B'].float() @ lora['lora_A'].float())
                    base_weights[base_key] = merged.to(base_weights[base_key].dtype)
                    merged_count += 1
            msg = model.load_state_dict(base_weights, strict=False)
            print(f"Merged {merged_count} LoRA pairs, missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
            if msg.missing_keys:
                print(f"  Missing sample: {msg.missing_keys[:5]}")
        else:
            msg = model.load_state_dict(all_weights, strict=False)
            print(f"Loaded {len(all_weights)} weights directly, missing={len(msg.missing_keys)}")

    model = model.to(torch.device("cuda")).eval()
    print("Model loaded.")

    # 5. Run inference
    import torchvision
    from PIL import Image
    from decord import VideoReader, cpu as decord_cpu
    from tasks.eval.model_utils import pllava_answer
    from tasks.eval.eval_utils import conv_templates

    conv_mode = "plain"
    results = []

    for i, entry in enumerate(questions):
        vname = entry["video"]
        video_path = f"DATAS/Video-MME/data/{vname}"
        if not os.path.exists(video_path):
            continue

        # Load video frames
        def get_index(num_frames, num_segments):
            seg_size = float(num_frames - 1) / num_segments
            start = int(seg_size / 2)
            return np.array([start + int(np.round(seg_size * idx)) for idx in range(num_segments)])

        vr = VideoReader(video_path, ctx=decord_cpu(0))
        n_frames_total = len(vr)
        frame_indices = get_index(n_frames_total, args.num_frames)
        transforms = torchvision.transforms.Resize(size=336)
        video_pils = [transforms(Image.fromarray(vr[fi].asnumpy())) for fi in frame_indices]

        # Build question
        question_text = f"Question: {entry['question']}\nOptions:\n"
        answer_text = entry["answer"]
        answer_idx = -1
        for idx, c in enumerate(entry["candidates"]):
            question_text += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer_text:
                answer_idx = idx
        question_text = question_text.rstrip()
        gt = f"({chr(ord('A') + answer_idx)}) {answer_text}"

        # Inference
        conv = conv_templates[conv_mode].copy()
        conv.user_query(question_text, pre_query_prompt=None,
                        post_query_prompt="\nOnly give the best option.", is_mm=True)
        pred, _, _ = pllava_answer(
            conv=conv, model=model, processor=processor,
            img_list=video_pils, max_new_tokens=args.max_new_tokens,
            do_sample=False, top_p=0.9, temperature=1.0,
            answer_prompt="Best option:(", return_prompt="(",
        )

        correct = check_ans(pred=pred, gt=gt)
        flow = flow_results[vname]["avg_flow"]

        results.append({
            "video": vname,
            "question": entry["question"][:80],
            "pred": pred,
            "gt": gt,
            "correct": correct,
            "flow_magnitude": flow,
        })

        status = "CORRECT" if correct else "WRONG"
        print(f"  [{i+1}/{len(questions)}] {vname} flow={flow:.3f} {status} pred={pred[:30]}")

    # 6. Analyze correlation
    print("\n" + "=" * 60)
    print("ANALYSIS: Flow Magnitude vs Correctness")
    print("=" * 60)

    correct_flows = [r["flow_magnitude"] for r in results if r["correct"]]
    wrong_flows = [r["flow_magnitude"] for r in results if not r["correct"]]

    print(f"\nTotal questions: {len(results)}")
    print(f"Correct: {len(correct_flows)} ({len(correct_flows)/len(results)*100:.1f}%)")
    print(f"Wrong:   {len(wrong_flows)} ({len(wrong_flows)/len(results)*100:.1f}%)")

    if correct_flows:
        print(f"\nCorrect - mean flow: {np.mean(correct_flows):.4f}, std: {np.std(correct_flows):.4f}")
    if wrong_flows:
        print(f"Wrong   - mean flow: {np.mean(wrong_flows):.4f}, std: {np.std(wrong_flows):.4f}")

    # Per-video accuracy with flow
    video_stats = {}
    for r in results:
        v = r["video"]
        if v not in video_stats:
            video_stats[v] = {"correct": 0, "total": 0, "flow": r["flow_magnitude"]}
        video_stats[v]["total"] += 1
        if r["correct"]:
            video_stats[v]["correct"] += 1

    print("\nPer-video accuracy (sorted by flow):")
    print(f"  {'Video':<25} {'Flow':>8} {'Acc':>8} {'Correct':>8} {'Total':>6}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    for v in sorted(video_stats, key=lambda x: video_stats[x]["flow"]):
        s = video_stats[v]
        acc = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"  {v:<25} {s['flow']:>8.4f} {acc:>7.1f}% {s['correct']:>8} {s['total']:>6}")

    # Split by flow median
    all_flows = [r["flow_magnitude"] for r in results]
    median_flow = np.median(all_flows)
    high_motion = [r for r in results if r["flow_magnitude"] >= median_flow]
    low_motion = [r for r in results if r["flow_magnitude"] < median_flow]

    high_acc = sum(r["correct"] for r in high_motion) / len(high_motion) * 100 if high_motion else 0
    low_acc = sum(r["correct"] for r in low_motion) / len(low_motion) * 100 if low_motion else 0

    print(f"\nSplit by median flow ({median_flow:.4f}):")
    print(f"  HIGH motion (>= median): {len(high_motion)} questions, acc={high_acc:.1f}%")
    print(f"  LOW  motion (< median):  {len(low_motion)} questions, acc={low_acc:.1f}%")

    # 7. Save
    output = {
        "args": vars(args),
        "flow_results": {k: {kk: vv for kk, vv in v.items() if kk != "per_pair_flow"} for k, v in flow_results.items()},
        "per_question": results,
        "summary": {
            "total_questions": len(results),
            "overall_accuracy": len(correct_flows) / len(results) * 100,
            "correct_mean_flow": float(np.mean(correct_flows)) if correct_flows else None,
            "wrong_mean_flow": float(np.mean(wrong_flows)) if wrong_flows else None,
            "high_motion_accuracy": high_acc,
            "low_motion_accuracy": low_acc,
            "median_flow": float(median_flow),
        },
    }
    with open(save_dir / "flow_correlation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {save_dir / 'flow_correlation_results.json'}")


if __name__ == "__main__":
    main()
