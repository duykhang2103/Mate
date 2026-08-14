#!/usr/bin/env python3
"""
Quick single-video inference test for LLaVA-OneVision.
Verifies the model loads, processes video, and generates coherent output.

Usage:
  python scripts/infer_single_video_ov.py
  python scripts/infer_single_video_ov.py --video example/cooking.mp4 --question "What is happening?"
  python scripts/infer_single_video_ov.py --use_vtp --selected_layer 10 --alpha 0.4
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from decord import VideoReader, cpu
import torchvision

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.eval.model_utils import load_llava_ov


def parse_args():
    parser = argparse.ArgumentParser(description="LLaVA-OV single video inference test")
    parser.add_argument("--video", type=str, default="example/cooking.mp4")
    parser.add_argument("--model_dir", type=str, default="MODELS/llava-onevision-7b")
    parser.add_argument("--question", type=str, default="Describe what happens in this video in detail.")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--use_vtp", action="store_true")
    parser.add_argument("--selected_layer", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.4)
    return parser.parse_args()


def load_video_frames(video_path, num_frames=16, resolution=672):
    transform = torchvision.transforms.Resize(size=resolution)
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    seg_size = float(total_frames - 1) / num_frames
    start = int(seg_size / 2)
    indices = [start + int(round(seg_size * i)) for i in range(num_frames)]
    images = [transform(Image.fromarray(vr[idx].asnumpy())) for idx in indices]
    fps = float(vr.get_avg_fps())
    times = ", ".join(str(round(f / fps, 1)) for f in indices)
    print(f"Loaded {video_path}: {total_frames} total frames, sampled {num_frames} at [{times}] seconds")
    return images


def main():
    args = parse_args()

    if not Path(args.video).is_file():
        print(f"ERROR: Video not found: {args.video}")
        sys.exit(1)

    print(f"Loading model from {args.model_dir} (VTP={args.use_vtp})...")
    t0 = time.time()
    model, processor = load_llava_ov(
        pretrained_model_name_or_path=args.model_dir,
        num_frames=args.num_frames,
        selected_layer=args.selected_layer,
        alpha=args.alpha,
        use_vtp=args.use_vtp,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    frames = load_video_frames(args.video, args.num_frames)

    conversation = [
        {"role": "system", "content": "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question."},
        {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": args.question}]}
    ]
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], videos=frames, return_tensors="pt")
    inputs = {k: v.to(model.device) if not v.is_floating_point() else v.to(model.device, dtype=torch.bfloat16) for k, v in inputs.items()}

    print(f"\nQuestion: {args.question}")
    print("Generating...")

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            num_beams=1,
            use_cache=True,
        )
    gen_time = time.time() - t0

    output_text = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    if "assistant\n" in output_text:
        answer = output_text.split("assistant\n")[-1].strip()
    else:
        answer = output_text.strip()

    new_tokens = output_ids.shape[1] - inputs["input_ids"].shape[1]

    print(f"\n{'='*60}")
    print(f"Answer ({gen_time:.1f}s, {new_tokens} tokens):")
    print(f"{answer}")
    print(f"{'='*60}")

    if hasattr(model, 'cache'):
        cache = model.cache
        print(f"VTP stats: alpha={cache.alpha}, selected_layer={cache.selected_layer}")
    else:
        print("VTP: disabled")

    return answer


if __name__ == "__main__":
    main()
