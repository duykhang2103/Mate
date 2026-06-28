#!/usr/bin/env python3
"""
Run PruneVid on a single video and compare baseline (no pruning) vs pruned inference.

NOTE: The official PLLaVA checkpoint (ermu2001/pllava-7b) was saved under PEFT/LoRA.
      You MUST pass --use_lora --lora_alpha 14 to load the LLM weights correctly;
      otherwise all LLM weights stay randomly initialized → "<unk>" output.

Example:
  python scripts/infer_single_video.py \\
    --video example/cooking.mp4 \\
    --model_dir MODELS/pllava-7b \\
    --weight_dir MODELS/pllava-7b \\
    --question "What is happening in this video?" \\
    --use_lora --lora_alpha 14
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision
from decord import VideoReader, cpu
from PIL import Image

# Repo root on sys.path when invoked as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.eval.eval_utils import conv_templates
from tasks.eval.model_utils import load_pllava, pllava_answer

RESOLUTION = 672
LOGGER = logging.getLogger("prunevid.infer_single")


@dataclass
class VisionPruneStats:
    raw_visual_tokens: int
    after_vision_merge_tokens: int
    vision_retention_ratio: float
    static_tokens: int
    dynamic_tokens: int
    temporal_windows: list[int]
    static_per_window: list[int]
    dynamic_per_window: list[int]
    frame_prune_masks: list[np.ndarray] | None = None  # H×W bool per frame


@dataclass
class RunResult:
    mode: str
    answer: str
    wall_time_sec: float
    vision: VisionPruneStats
    estimated_llm_visual_tokens: int
    estimated_total_prefill_tokens: int | None
    alpha: float
    tau: float
    cluster_ratio: float
    temporal_segment_ratio: float
    use_entropy_adaptive: bool = False
    tau_entropy: float = 0.8
    entropy_fallback_layer: int = 20
    use_motion_adaptive: bool = False
    motion_scale: float = 0.5
    selected_layer_actual: int | None = None  # which layer was actually used for pruning
    layer_entropies: list[float] | None = None  # entropy profile across layers


def setup_logging(log_dir: Path, verbose: bool) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "infer_single_video.log"

    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="w"),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    LOGGER.info("Logging to %s", log_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-video PruneVid inference with before/after comparison.")
    parser.add_argument("--video", type=str, default="example/cooking.mp4", help="Path to input video.")
    parser.add_argument("--model_dir", type=str, default="MODELS/pllava-7b", help="PLLaVA checkpoint directory.")
    parser.add_argument(
        "--weight_dir",
        type=str,
        default=None,
        help="Local folder with LoRA/safetensors, or HF repo id (defaults to model_dir). "
        "For HF ids like ermu2001/pllava-7b, weights come from from_pretrained; omit if same as model_dir.",
    )
    parser.add_argument("--question", type=str, default="Describe what happens in this video in detail.", help="User question.")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--pooling_shape", type=str, default="16-12-12")
    parser.add_argument("--conv_mode", type=str, default="eval_mvbench")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_alpha", type=int, default=14)
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--log_dir", type=str, default="test_results/single_video_infer")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip_generation", action="store_true", help="Only report token pruning stats (no LM decode).")

    # Pruned (paper-style defaults from scripts/eval.sh)
    parser.add_argument("--selected_layer", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--temporal_segment_ratio", type=float, default=0.25)
    parser.add_argument("--cluster_ratio", type=float, default=0.5)

    # Entropy-adaptive pruning
    parser.add_argument("--use_entropy_adaptive", action="store_true", default=False,
                        help="Enable entropy-adaptive pruning (dynamic layer selection based on attention entropy)")
    parser.add_argument("--tau_entropy", type=float, default=0.8,
                        help="Entropy threshold for triggering pruning (lower = later pruning)")
    parser.add_argument("--entropy_fallback_layer", type=int, default=20,
                        help="Fallback layer if no entropy trigger found")

    # Motion-adaptive pruning
    parser.add_argument("--use_motion_adaptive", action="store_true", default=False,
                        help="Enable motion-adaptive pruning (per-window alpha based on attention variance)")
    parser.add_argument("--motion_scale", type=float, default=0.5,
                        help="Controls adaptive range: [alpha*(1-motion_scale), alpha]")

    # Borderline preservation
    parser.add_argument("--use_borderline_preservation", action="store_true", default=False,
                        help="Enable borderline token preservation (keep tokens near the pruning cutoff)")
    parser.add_argument("--borderline_margin", type=float, default=0.1,
                        help="Margin as fraction of cutoff score (0.1 = keep tokens within 10% of cutoff)")

    # Baseline disables vision merge + LLM token pruning
    parser.add_argument("--baseline_alpha", type=float, default=1.0, help="alpha=1 keeps all vision tokens in the LLM stage.")

    # Visualization output
    parser.add_argument(
        "--output_video", type=str, default=None,
        help="Path for output MP4. Default: output/{video_stem}_{timestamp}.mp4",
    )
    parser.add_argument("--save_masks", action="store_true", help="Save individual overlay frames as PNGs.")
    return parser.parse_args()


def get_frame_indices(num_frames: int, num_segments: int) -> np.ndarray:
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    return np.array([start + int(np.round(seg_size * idx)) for idx in range(num_segments)])


@dataclass
class LoadedVideo:
    images: list[Image.Image]
    frame_indices: list[int]
    total_frames: int
    fps: float
    msg: str


def load_video_frames(video_path: str, num_segments: int, resolution: int = RESOLUTION) -> LoadedVideo:
    transform = torchvision.transforms.Resize(size=resolution)
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    frame_indices = get_frame_indices(total_frames, num_segments)
    images = []
    for idx in frame_indices:
        images.append(transform(Image.fromarray(vr[int(idx)].asnumpy())))
    fps = float(vr.get_avg_fps())
    sec = ", ".join(str(round(f / fps, 1)) for f in frame_indices)
    msg = f"{len(frame_indices)} frames sampled at {sec} seconds."
    return LoadedVideo(images=images, frame_indices=list(frame_indices), total_frames=total_frames, fps=fps, msg=msg)


def estimate_llm_visual_tokens(static_sizes: list[int], dynamic_sizes: list[int], window_sizes: list[int], alpha: float) -> int:
    """Mirror VTPWindowCache.process_attention retention counts."""
    total = 0
    for static_size, dynamic_size, window_size in zip(static_sizes, dynamic_sizes, window_sizes):
        total += int(static_size * alpha)
        per_frame = dynamic_size // max(window_size, 1)
        for _ in range(window_size):
            total += int(per_frame * alpha)
    return total


@torch.no_grad()
def measure_vision_pruning(model, pixel_values: torch.Tensor, media_type: str = "video") -> VisionPruneStats:
    batch_size = 1
    vision_layer = model.config.vision_feature_layer
    strategy = model.config.vision_feature_select_strategy

    image_outputs = model.vision_tower(pixel_values, output_hidden_states=True, output_attentions=False)
    selected = image_outputs.hidden_states[vision_layer]
    if strategy == "default":
        selected = selected[:, 1:]
    elif strategy != "full":
        raise ValueError(f"Unsupported vision_feature_select_strategy: {strategy}")

    image_features = model.multi_modal_projector(
        selected,
        media_type,
        batch_size=batch_size,
        num_videos=pixel_values.shape[0] // model.config.num_frames // batch_size,
        num_frames=model.config.num_frames,
    )
    raw_tokens = int(image_features.shape[1])
    merged, static_sizes, dynamic_sizes, window_sizes = model.merge_frames_dynamic(
        image_features, threshold=model.config.tau, k=7
    )
    after_merge = int(merged.shape[1])
    return VisionPruneStats(
        raw_visual_tokens=raw_tokens,
        after_vision_merge_tokens=after_merge,
        vision_retention_ratio=after_merge / max(raw_tokens, 1),
        static_tokens=int(sum(static_sizes)),
        dynamic_tokens=int(sum(dynamic_sizes)),
        temporal_windows=list(window_sizes),
        static_per_window=[int(x) for x in static_sizes],
        dynamic_per_window=[int(x) for x in dynamic_sizes],
    )


def _baseline_merge_frames_dynamic(self, frames, threshold=0.8, k=7):
    """Identity vision path: no spatial/temporal merge (all pooled tokens kept)."""
    per_frame = self.config.pooling_shape[1] * self.config.pooling_shape[2]
    n = self.config.num_frames
    static_sizes = [0] * n
    dynamic_sizes = [per_frame] * n
    window_sizes = [1] * n
    return frames, static_sizes, dynamic_sizes, window_sizes


def set_llm_alpha(model, alpha: float) -> None:
    model.config.alpha = alpha
    # Walk through PEFT wrapper (PeftModel → base_model → model → ...) until
    # we reach LlamaModelVTP, which is the only object that has .cache.
    lm = model.language_model
    if hasattr(lm, 'base_model'):
        lm = lm.base_model
    while hasattr(lm, 'model') and not hasattr(lm, 'cache'):
        lm = lm.model
    lm.alpha = alpha
    lm.cache.alpha = alpha


def build_conversation(conv_mode: str, question: str):
    conv = conv_templates[conv_mode].copy()
    conv.user_query(question, is_mm=True)
    return conv


@torch.no_grad()
def run_once(
    model,
    processor,
    frames: list[Image.Image],
    question: str,
    conv_mode: str,
    max_new_tokens: int,
    mode: str,
    alpha: float,
    skip_generation: bool,
) -> RunResult:
    conv = build_conversation(conv_mode, question)
    inputs = processor(text=conv.get_prompt(), images=frames, return_tensors="pt")
    if inputs.get("pixel_values") is None:
        inputs.pop("pixel_values", None)
    inputs = inputs.to(model.device)

    vision_stats = measure_vision_pruning(model, inputs["pixel_values"], media_type="video")
    llm_visual = estimate_llm_visual_tokens(
        vision_stats.static_per_window,
        vision_stats.dynamic_per_window,
        vision_stats.temporal_windows,
        alpha=alpha,
    )

    answer = "[skipped generation]"
    wall = 0.0
    prefill_tokens = None

    if not skip_generation:
        if torch.cuda.is_available() and str(model.device).startswith("cuda"):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    media_type="video",
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )
            end.record()
            torch.cuda.synchronize()
            wall = start.elapsed_time(end) / 1000.0
        else:
            t0 = time.perf_counter()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    media_type="video",
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )
            wall = time.perf_counter() - t0

        text = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        text_raw = processor.batch_decode(output_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
        LOGGER.info("DEBUG: input_ids shape=%s  output_ids shape=%s", tuple(inputs["input_ids"].shape), tuple(output_ids.shape))
        LOGGER.info("DEBUG: num new tokens=%d", output_ids.shape[1] - inputs["input_ids"].shape[1])
        LOGGER.info("DEBUG: raw decoded text (no skip_special) = %s", text_raw[:500])
        LOGGER.info("DEBUG: decoded text (skip_special) = %s", text[:500])
        split_tag = conv.roles[-1]
        answer = text.split(split_tag)[-1].strip()
        if isinstance(conv.sep, list):
            answer = answer.removesuffix(conv.sep[1]).strip()
        else:
            answer = answer.removesuffix(conv.sep).strip()

        # Check for <unk> tokens (indicates bad weight loading)
        unk_count = text.count("<unk>")
        if unk_count > 5:
            LOGGER.warning(
                "!!! Detected %d <unk> tokens in output — LLM weights likely not loaded correctly.", unk_count
            )
        LOGGER.info("Output <unk> count: %d", unk_count)

        prefill_tokens = int(inputs["input_ids"].shape[1])  # text-only placeholder; image expands in forward

    # Extract entropy profile from model (set during forward pass by LlamaModelVTP)
    lm = model.language_model
    if hasattr(lm, 'base_model'):
        lm = lm.base_model
    while hasattr(lm, 'model') and not hasattr(lm, 'last_layer_entropies'):
        lm = lm.model
    layer_entropies = getattr(lm, 'last_layer_entropies', None)
    selected_layer_actual = getattr(lm, 'last_dynamic_selected_layer', None)

    return RunResult(
        mode=mode,
        answer=answer,
        wall_time_sec=wall,
        vision=vision_stats,
        estimated_llm_visual_tokens=llm_visual,
        estimated_total_prefill_tokens=prefill_tokens,
        alpha=alpha,
        tau=float(model.config.tau),
        cluster_ratio=float(model.config.cluster_ratio),
        temporal_segment_ratio=float(model.config.temporal_segment_ratio),
        use_entropy_adaptive=getattr(model.config, 'use_entropy_adaptive', False),
        tau_entropy=getattr(model.config, 'tau_entropy', 0.8),
        entropy_fallback_layer=getattr(model.config, 'entropy_fallback_layer', 20),
        selected_layer_actual=selected_layer_actual,
        layer_entropies=layer_entropies,
    )


def log_comparison(baseline: RunResult, pruned: RunResult) -> dict:
    raw = baseline.vision.raw_visual_tokens
    b_vis = baseline.vision.after_vision_merge_tokens
    p_vis = pruned.vision.after_vision_merge_tokens
    b_llm = baseline.estimated_llm_visual_tokens
    p_llm = pruned.estimated_llm_visual_tokens

    summary = {
        "raw_visual_tokens_after_projector": raw,
        "baseline": {
            "vision_tokens_after_merge": b_vis,
            "vision_retention_vs_raw": b_vis / max(raw, 1),
            "llm_visual_tokens_after_layer_prune": b_llm,
            "llm_visual_retention_vs_raw": b_llm / max(raw, 1),
            "wall_time_sec": baseline.wall_time_sec,
            "answer": baseline.answer,
        },
        "pruned": {
            "vision_tokens_after_merge": p_vis,
            "vision_retention_vs_raw": p_vis / max(raw, 1),
            "llm_visual_tokens_after_layer_prune": p_llm,
            "llm_visual_retention_vs_raw": p_llm / max(raw, 1),
            "wall_time_sec": pruned.wall_time_sec,
            "answer": pruned.answer,
        },
        "delta_pruned_vs_baseline": {
            "vision_tokens_removed": b_vis - p_vis,
            "vision_token_reduction_pct": (1 - p_vis / max(b_vis, 1)) * 100,
            "llm_visual_tokens_removed": b_llm - p_llm,
            "llm_visual_token_reduction_pct": (1 - p_llm / max(b_llm, 1)) * 100,
            "overall_visual_retention_vs_raw_pct": (p_llm / max(raw, 1)) * 100,
            "speedup_baseline_over_pruned": (
                baseline.wall_time_sec / pruned.wall_time_sec
                if pruned.wall_time_sec > 0 and baseline.wall_time_sec > 0
                else None
            ),
            "answers_match": baseline.answer.strip() == pruned.answer.strip(),
        },
        "pruning_hyperparameters": {
            "alpha": pruned.alpha,
            "tau": pruned.tau,
            "cluster_ratio": pruned.cluster_ratio,
            "temporal_segment_ratio": pruned.temporal_segment_ratio,
        },
    }
    return summary


def build_frame_prune_masks(
    static_per_window: list[int],
    dynamic_per_window: list[int],
    window_sizes: list[int],
    alpha: float,
    H: int = 12,
    W: int = 12,
) -> list[np.ndarray]:
    """Build per-frame H×W uint8 masks: 0=kept, 1=static-pruned, 2=dynamic-pruned, 3=both.

    Within each temporal window:
    - Static-pruned cells are the SAME across all frames (shared pattern).
    - Dynamic-pruned cells differ per frame (per-frame pattern), and are
      drawn from cells NOT already static-pruned.
    """
    rng = np.random.default_rng(2026)
    frame_masks: list[np.ndarray] = []
    for static_sz, dynamic_sz, win_sz in zip(static_per_window, dynamic_per_window, window_sizes):
        tokens_per_frame = dynamic_sz // win_sz

        num_static_pruned = min(static_sz - int(static_sz * alpha), H * W)
        static_pruned = np.zeros(H * W, dtype=bool)
        if num_static_pruned > 0:
            idx = rng.choice(H * W, size=num_static_pruned, replace=False)
            static_pruned[idx] = True

        for _ in range(win_sz):
            num_dynamic_pruned = min(tokens_per_frame - int(tokens_per_frame * alpha), H * W)
            available = np.where(~static_pruned)[0]
            count = min(num_dynamic_pruned, len(available))
            dynamic_pruned = np.zeros(H * W, dtype=bool)
            if count > 0:
                idx = rng.choice(available, size=count, replace=False)
                dynamic_pruned[idx] = True

            mask = np.zeros(H * W, dtype=np.uint8)
            mask[static_pruned] = 1
            mask[dynamic_pruned] = 2
            mask[static_pruned & dynamic_pruned] = 3
            frame_masks.append(mask.reshape(H, W))
    return frame_masks


def print_human_report(baseline: RunResult, pruned: RunResult, summary: dict, video: str, question: str) -> None:
    LOGGER.info("=" * 72)
    LOGGER.info("Video: %s", video)
    LOGGER.info("Question: %s", question)
    LOGGER.info("-" * 72)
    LOGGER.info(
        "Visual tokens (projector output): %d  (%d sampled frames)",
        baseline.vision.raw_visual_tokens,
        len(baseline.vision.temporal_windows),
    )
    LOGGER.info("")
    LOGGER.info("BASELINE (no vision merge, alpha=%.2f)", baseline.alpha)
    LOGGER.info("  Vision tokens after merge : %d (%.1f%% of raw)", baseline.vision.after_vision_merge_tokens, summary["baseline"]["vision_retention_vs_raw"] * 100)
    LOGGER.info("  LLM visual tokens (est.)  : %d (%.1f%% of raw)", baseline.estimated_llm_visual_tokens, summary["baseline"]["llm_visual_retention_vs_raw"] * 100)
    LOGGER.info("  Static / dynamic (vision) : %d / %d", baseline.vision.static_tokens, baseline.vision.dynamic_tokens)
    LOGGER.info("  Generate time             : %.3f s", baseline.wall_time_sec)
    LOGGER.info("  Answer:\n%s", baseline.answer)
    LOGGER.info("")
    LOGGER.info("PRUNED (PruneVid)")
    LOGGER.info("  tau=%.2f  cluster_ratio=%.2f  temporal_segment_ratio=%.2f  alpha=%.2f",
                pruned.tau, pruned.cluster_ratio, pruned.temporal_segment_ratio, pruned.alpha)
    if pruned.use_entropy_adaptive:
        LOGGER.info("  ENTROPY-ADAPTIVE: tau_entropy=%.2f  fallback_layer=%d", pruned.tau_entropy, pruned.entropy_fallback_layer)
        if pruned.selected_layer_actual is not None:
            LOGGER.info("  Selected pruning layer    : %d", pruned.selected_layer_actual)
        if pruned.layer_entropies:
            LOGGER.info("  Entropy profile (layer, entropy):")
            for layer_idx, entropy in pruned.layer_entropies:
                marker = " <-- TRIGGER" if layer_idx == pruned.selected_layer_actual else ""
                LOGGER.info("    Layer %2d: %.4f%s", layer_idx, entropy, marker)
    else:
        LOGGER.info("  Fixed pruning layer       : %d", pruned.selected_layer_actual if pruned.selected_layer_actual is not None else -1)
    LOGGER.info("  Vision tokens after merge : %d (%.1f%% of raw)", pruned.vision.after_vision_merge_tokens, summary["pruned"]["vision_retention_vs_raw"] * 100)
    LOGGER.info("  LLM visual tokens (est.)  : %d (%.1f%% of raw)", pruned.estimated_llm_visual_tokens, summary["pruned"]["llm_visual_retention_vs_raw"] * 100)
    LOGGER.info("  Static / dynamic (vision) : %d / %d", pruned.vision.static_tokens, pruned.vision.dynamic_tokens)
    LOGGER.info("  Temporal windows (frames) : %s", pruned.vision.temporal_windows)
    LOGGER.info("  Static per window         : %s", pruned.vision.static_per_window)
    LOGGER.info("  Dynamic per window        : %s", pruned.vision.dynamic_per_window)
    LOGGER.info("  Generate time             : %.3f s", pruned.wall_time_sec)
    LOGGER.info("  Answer:\n%s", pruned.answer)
    LOGGER.info("")
    LOGGER.info("DIFFERENCE (pruned vs baseline)")
    d = summary["delta_pruned_vs_baseline"]
    LOGGER.info("  Vision tokens removed     : %d (%.1f%% reduction)", d["vision_tokens_removed"], d["vision_token_reduction_pct"])
    LOGGER.info("  LLM visual tokens removed : %d (%.1f%% reduction)", d["llm_visual_tokens_removed"], d["llm_visual_token_reduction_pct"])
    LOGGER.info("  Overall retention vs raw  : %.1f%%", d["overall_visual_retention_vs_raw_pct"])
    if d["speedup_baseline_over_pruned"] is not None:
        LOGGER.info("  Time ratio (baseline/pruned): %.2fx", d["speedup_baseline_over_pruned"])
    LOGGER.info("  Answers identical         : %s", d["answers_match"])
    LOGGER.info("=" * 72)


def _map_frames_to_windows(num_frames: int, window_sizes: list[int]) -> dict[int, int]:
    """Return {global_frame_index: window_index} mapping."""
    mapping: dict[int, int] = {}
    cum = 0
    for w_idx, w_size in enumerate(window_sizes):
        for fi in range(w_size):
            mapping[cum + fi] = w_idx
        cum += w_size
    return mapping


def create_visualization_video(
    loaded: LoadedVideo,
    pruned_stats: VisionPruneStats,
    output_path: str,
    save_masks: bool = False,
) -> None:
    """Create an MP4 with overlay showing pruning statistics per temporal window."""
    try:
        import cv2
    except ImportError:
        LOGGER.warning("opencv-python not installed; skipping visualization video.")
        LOGGER.warning("Install with: pip install opencv-python")
        return

    num_frames = len(loaded.images)
    if num_frames == 0:
        return

    w, h = loaded.images[0].size
    timeline_h = 80
    total_h = h + timeline_h

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    window_sizes = pruned_stats.temporal_windows
    static_pw = pruned_stats.static_per_window
    dynamic_pw = pruned_stats.dynamic_per_window
    f2w = _map_frames_to_windows(num_frames, window_sizes)

    # Build video writer (2 fps — each frame shown for 0.5 s)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, 2.0, (w, total_h))

    mask_dir: Path | None = None
    if save_masks:
        mask_dir = out_path.parent / (out_path.stem + "_masks")
        mask_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_frames):
        frame = cv2.cvtColor(np.array(loaded.images[i]), cv2.COLOR_RGB2BGR)
        display = frame.copy()

        # --- prune mask overlay (H×W grid resized to frame) ---
        if pruned_stats.frame_prune_masks is not None and i < len(pruned_stats.frame_prune_masks):
            mask = pruned_stats.frame_prune_masks[i].astype(np.uint8)  # H×W, 0=kept 1=static 2=dynamic 3=both
            mask_big = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)  # (h,w) uint8
            # Color per pixel type: static=red, dynamic=blue, both=magenta
            overlay = np.zeros((h, w, 3), dtype=np.uint8)
            static_mask = (mask_big == 1) | (mask_big == 3)
            dynamic_mask = (mask_big == 2) | (mask_big == 3)
            overlay[static_mask] = (0, 0, 255)    # BGR red
            overlay[dynamic_mask] = (255, 0, 0)   # BGR blue
            overlay[static_mask & dynamic_mask] = (255, 0, 255)  # BGR magenta
            # Blend at 0.5 opacity only where pruned
            blended = cv2.addWeighted(frame, 1.0, overlay, 0.5, 0, dtype=cv2.CV_8U)
            display = np.where((mask_big > 0)[..., None], blended, frame)

        w_idx = f2w.get(i, -1)

        # --- overlay text ---
        lines = [f"Frame {i + 1} / {num_frames}  |  orig idx {loaded.frame_indices[i]}"]
        if w_idx >= 0:
            total_t = static_pw[w_idx] + dynamic_pw[w_idx]
            kept_pct = static_pw[w_idx] / max(total_t, 1) * 100
            lines.append(
                f"Window {w_idx + 1}/{len(window_sizes)}  ({window_sizes[w_idx]} frames)"
            )
            lines.append(f"Static {static_pw[w_idx]}  Dynamic {dynamic_pw[w_idx]}  ({kept_pct:.0f}% kept)")

        y = 28
        for line in lines:
            cv2.putText(display, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 2, cv2.LINE_AA)
            y += 26

        # --- timeline bar ---
        timeline = np.zeros((timeline_h, w, 3), dtype=np.uint8)
        margin = 24
        usable = w - 2 * margin
        if window_sizes:
            cum_f = 0
            for wj, ws in enumerate(window_sizes):
                seg_px = int(ws / num_frames * usable)
                x1 = margin + int(cum_f / num_frames * usable)
                x2 = x1 + seg_px
                total_t = static_pw[wj] + dynamic_pw[wj]
                ret = static_pw[wj] / max(total_t, 1)
                r = int(255 * (1 - ret))
                g = int(255 * ret)
                # cv2.rectangle(timeline, (x1, 6), (x2, timeline_h - 6), (r, g, b), -1)
                cv2.rectangle(timeline, (x1, 6), (x2, timeline_h - 6), (r, g, 0), -1)
                if f2w.get(i, -1) == wj:
                    cv2.rectangle(timeline, (x1, 2), (x2, timeline_h - 2), (255, 255, 255), 2)
                cum_f += ws
            cv2.putText(timeline, "red=static-pruned  blue=dynamic-pruned  green=kept (timeline)",
                        (margin, timeline_h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (180, 180, 180), 1)

        combined = np.vstack([display, timeline])
        writer.write(combined)

        if mask_dir is not None:
            cv2.imwrite(str(mask_dir / f"frame_{i + 1:04d}.png"), display)

    writer.release()
    LOGGER.info("Visualization video saved  ->  %s", out_path)


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    setup_logging(log_dir, args.verbose)

    video_path = Path(args.video)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    weight_dir = args.weight_dir or args.model_dir
    pooling_shape = tuple(int(x) for x in args.pooling_shape.split("-"))

    LOGGER.info("Loading model from %s (weight_dir=%s)", args.model_dir, weight_dir)
    LOGGER.info("LoRA: %s | lora_alpha=%s", args.use_lora, args.lora_alpha)
    LOGGER.info("Entropy-adaptive: %s | tau_entropy=%s | fallback_layer=%s", args.use_entropy_adaptive, args.tau_entropy, args.entropy_fallback_layer)
    model, processor = load_pllava(
        args.model_dir,
        num_frames=args.num_frames,
        use_lora=args.use_lora,
        weight_dir=weight_dir,
        lora_alpha=args.lora_alpha,
        pooling_shape=pooling_shape,
        selected_layer=args.selected_layer,
        alpha=args.alpha,
        tau=args.tau,
        cluster_ratio=args.cluster_ratio,
        temporal_segment_ratio=args.temporal_segment_ratio,
        use_entropy_adaptive=args.use_entropy_adaptive,
        tau_entropy=args.tau_entropy,
        entropy_fallback_layer=args.entropy_fallback_layer,
        use_motion_adaptive=args.use_motion_adaptive,
        motion_scale=args.motion_scale,
        use_borderline_preservation=args.use_borderline_preservation,
        borderline_margin=args.borderline_margin,
    )
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = model.to(device).eval()

    loaded = load_video_frames(str(video_path), num_segments=args.num_frames)
    LOGGER.info("Loaded video: %s", loaded.msg)

    original_merge = model.merge_frames_dynamic

    # --- Baseline: disable vision merge + full LLM retention ---
    model.merge_frames_dynamic = types.MethodType(_baseline_merge_frames_dynamic, model)
    set_llm_alpha(model, args.baseline_alpha)
    LOGGER.info("Running BASELINE inference...")
    baseline = run_once(
        model,
        processor,
        loaded.images,
        args.question,
        args.conv_mode,
        args.max_new_tokens,
        mode="baseline",
        alpha=args.baseline_alpha,
        skip_generation=args.skip_generation,
    )

    # --- Pruned: restore merge + paper hyperparameters ---
    model.merge_frames_dynamic = original_merge
    set_llm_alpha(model, args.alpha)
    model.config.tau = args.tau
    model.config.cluster_ratio = args.cluster_ratio
    model.config.temporal_segment_ratio = args.temporal_segment_ratio
    model.config.use_entropy_adaptive = args.use_entropy_adaptive
    model.config.tau_entropy = args.tau_entropy
    model.config.entropy_fallback_layer = args.entropy_fallback_layer
    model.config.use_motion_adaptive = args.use_motion_adaptive
    model.config.motion_scale = args.motion_scale
    # Propagate to LlamaModelVTP
    lm = model.language_model
    if hasattr(lm, 'base_model'):
        lm = lm.base_model
    while hasattr(lm, 'model') and not hasattr(lm, 'cache'):
        lm = lm.model
    lm.use_entropy_adaptive = args.use_entropy_adaptive
    lm.tau_entropy = args.tau_entropy
    lm.entropy_fallback_layer = args.entropy_fallback_layer
    lm.use_motion_adaptive = args.use_motion_adaptive
    lm.motion_scale = args.motion_scale
    LOGGER.info("Running PRUNED inference (entropy_adaptive=%s)...", args.use_entropy_adaptive)
    pruned = run_once(
        model,
        processor,
        loaded.images,
        args.question,
        args.conv_mode,
        args.max_new_tokens,
        mode="pruned",
        alpha=args.alpha,
        skip_generation=args.skip_generation,
    )

    summary = log_comparison(baseline, pruned)
    summary["video"] = str(video_path)
    summary["question"] = args.question
    summary["frame_sampling"] = loaded.msg
    summary["baseline"].update(asdict(baseline.vision))
    summary["pruned"].update(asdict(pruned.vision))
    summary["pruning_hyperparameters"] = {
        "selected_layer": args.selected_layer,
        "alpha": args.alpha,
        "tau": args.tau,
        "cluster_ratio": args.cluster_ratio,
        "temporal_segment_ratio": args.temporal_segment_ratio,
        "baseline_alpha": args.baseline_alpha,
        "use_entropy_adaptive": args.use_entropy_adaptive,
        "tau_entropy": args.tau_entropy,
        "entropy_fallback_layer": args.entropy_fallback_layer,
        "use_motion_adaptive": args.use_motion_adaptive,
        "motion_scale": args.motion_scale,
    }
    if pruned.selected_layer_actual is not None:
        summary["pruning_hyperparameters"]["selected_layer_actual"] = pruned.selected_layer_actual
    if pruned.layer_entropies:
        summary["layer_entropies"] = {str(k): round(v, 4) for k, v in pruned.layer_entropies}

    out_json = log_dir / "comparison.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    LOGGER.info("Wrote JSON report to %s", out_json)

    print_human_report(baseline, pruned, summary, str(video_path), args.question)

    # Build per-frame pruning masks from the pruned run's vision stats
    H, W = pooling_shape[1], pooling_shape[2]
    if pruned.vision.temporal_windows and H > 0 and W > 0:
        pruned.vision.frame_prune_masks = build_frame_prune_masks(
            pruned.vision.static_per_window,
            pruned.vision.dynamic_per_window,
            pruned.vision.temporal_windows,
            pruned.alpha,
            H=H,
            W=W,
        )
        LOGGER.info("Built prune masks for %d frames (%dx%d)", len(pruned.vision.frame_prune_masks), H, W)

    # Generate visualization video
    if pruned.vision.temporal_windows:
        output_video = args.output_video
        if output_video is None:
            stem = video_path.stem
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_video = f"output/{stem}_{ts}.mp4"
        create_visualization_video(loaded, pruned.vision, output_video, args.save_masks)
    else:
        LOGGER.info("Skipping visualization (no temporal window data).")


if __name__ == "__main__":
    main()

###
# python scripts/infer_single_video.py \
#     --video example/cooking.mp4 \
#     --model_dir MODELS/pllava-7b \
#     --weight_dir MODELS/pllava-7b \
#     --question "What is happening in this video?" \
#     --num_frames 16 \
#     --use_lora --lora_alpha 14
