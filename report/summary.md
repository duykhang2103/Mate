# PruneVid — Project Flow Summary

## Overview

PruneVid is a **vision token pruning and merging framework** for video LLMs. It reduces the computational cost of video-language models by eliminating redundancy across both the **vision encoder** side and the **LLM decoder** side — achieving ~2× speedup with minimal accuracy loss. The model is built on **PLLaVA** (Phi-4 + CLIP ViT-L/14 + multimodal projector) with a two-stage pruning pipeline.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Vision Encoder | CLIP ViT-L/14 (custom `modeling_clip.py` with spatial pooling, anyres) |
| Language Model | Llama / Phi-4 (custom `llama.py` with KV-cache pruning hooks) |
| Multimodal Projector | 2-layer MLP (CLIP vision tokens → Llama embedding space) |
| Training | PyTorch 2.2, HuggingFace Transformers, Accelerate, DeepSpeed, PEFT (LoRA) |
| Video Loading | decord (primary), pyav (fallback), imageio (GIF) |
| Configuration | OmegaConf-style Python/YAML + CLI overrides (`Config` + `EasyDict`) |

---

## Directory Structure

```
PruneVid/
├── scripts/
│   ├── infer_single_video.py    # Single-video inference with prune vs. baseline comparison
│   └── eval.sh                  # Shell script for multi-benchmark evaluation
│
├── models/
│   ├── pllava/
│   │   ├── modeling_pllava.py           # PllavaForConditionalGeneration (main model)
│   │   ├── pllava_prumerge.py           # Vision-side pruning: CLIPVisionTower, DPC-KNN, merge_frames_dynamic
│   │   ├── elastic_cache.py             # DPC-KNN clustering: FastVTSPruner, ElasticCachePruner, VTPWindowCache
│   │   ├── llama.py                     # Custom Llama: LlamaForCausalLMVTP, TextPivotMerge_LayerWise
│   │   ├── modify_llama.py              # Attention variants: H2O, MixMer, PixelPrunMerge, PivotMerge, Weighted, TextPrior
│   │   ├── modeling_clip.py             # Custom CLIP vision encoder
│   │   ├── llava_arch.py                # LlavaMetaModel, multimodal projector, video frame processing
│   │   ├── configuration_pllava.py      # PllavaConfig (pruning hyperparameters)
│   │   ├── processing_pllava.py         # PllavaProcessor (tokenizer + image processor)
│   │   ├── modeling_pllava_flow.py      # Optical-flow guided variant
│   │   └── modeling_pllava_SF.py        # Static/flow variant
│   └── __init__.py
│
├── dataset/
│   ├── base_dataset.py          # ImageVideoBaseDataset (optical flow extraction, RAFT)
│   ├── it_dataset.py            # ITImgTrainDataset / ITVidTrainDataset
│   ├── video_utils.py           # Frame indexing, video loading (decord/pyav/gif/hdfs)
│   └── utils.py                 # load_image_from_path, pre_text
│
├── tasks/
│   ├── shared_utils.py          # get_media_types, optimizer/scheduler creation
│   ├── train/                   # LoRA fine-tuning scripts + configs
│   └── eval/                    # Evaluation benchmarks (MVBench, VideoMME, EgoSchema, VCGBench, demo)
│       ├── eval_utils.py        # Conversation, EvalDataset, ChatPllava
│       ├── model_utils.py       # load_pllava, pllava_answer, weight loading
│       ├── mvbench/
│       ├── videomme/
│       ├── egoshcema/
│       └── vcgbench/
│
└── utils/
    ├── config.py                # Config class (OmegaConf-style from .py/.yaml/.json + CLI overrides)
    ├── config_utils.py          # setup_main (config + logging + distributed setup)
    ├── distributed.py           # init_distributed_mode (DDP/multi-GPU)
    ├── optimizer.py / scheduler.py / logger.py / basic_utils.py
```

---

## End-to-End Data Flow

```
Video File
    │
    ▼
[1] Frame Sampling
    │   decord/pyav: uniformly sample N frames (default: 16)
    │   Resize to 672×672 (inference) or 336×336 (training)
    │
    ▼
[2] Vision Encoding  ───────────── models/pllava_prumerge.py
    │   CLIP ViT-L/14 patch embedding
    │   Per frame: 257 patches (including [CLS])
    │   Remove [CLS] → 256 patches/frame
    │   Spatial pooling → (16, 12, 12) = 144 pooled tokens/frame
    │
    ▼
[3] Multimodal Projector  ─────── models/pllava/llava_arch.py
    │   2-layer MLP: CLIP hidden dim → Llama hidden dim
    │
    ▼
[4] STAGE 1: Vision-Side Pruning  ── models/pllava/pllava_prumerge.py
    │   merge_frames_dynamic():
    │
    │   a) Concatenate pooled tokens across frames
    │      Shape: (B, num_frames × pooled_tokens, D)
    │
    │   b) Split into temporal windows (temporal_segment_ratio=0.25)
    │      e.g., 16 frames → 4 windows of 4 frames each
    │
    │   c) DPC-KNN clustering per window (elastic_cache.py):
    │      - Compute N×N cosine similarity matrix
    │      - Local density ρ = sum of top-k neighbor similarities
    │      - Distance δ = distance to nearest higher-density point
    │      - Cluster centers = points with high ρ × δ
    │      - Assign each point to nearest higher-density neighbor
    │
    │   d) Cross-window cluster alignment:
    │      - Match cluster centers across windows by position proximity
    │
    │   e) Static vs. Dynamic separation:
    │      - STATIC tokens: cluster centers appearing at same position
    │        across multiple windows (redundant across time)
    │      - DYNAMIC tokens: unmatched centers or position-varying tokens
    │
    │   f) Merge:
    │      - Static: keep top cluster_ratio (0.5) of centroids
    │      - Dynamic: keep top cluster_ratio per frame per window
    │
    │   Output: reduced vision token count (e.g., 2304 → ~500-800)
    │
    ▼
[5] Text Preprocessing  ─────────── tasks/eval/eval_utils.py
    │   Conversation template:
    │     "<|system|> ... <|end|> <|user|> <video>\\n QUESTION <|end|> <|assistant|>"
    │   Tokenization via PllavaProcessor (tokenizer + image processor)
    │
    ▼
[6] LLM Forward Pass  ───────────── models/pllava/modeling_pllava.py
    │   Embedding lookup → Positional encoding
    │   Pass through LlamaDecoderLayers 0..N
    │
    ▼
[7] STAGE 2: LLM-Side Pruning  ──── models/pllava/llama.py
    │   At selected_layer (default=10):
    │
    │   a) Compute multi-head attention: Q_text × K_visual
    │   b) Extract text-to-image attention scores
    │   c) Dual-threshold pruning:
    │      - alpha (0.4): keep top-40% scoring visual tokens
    │      - tau (0.8): keep tokens with score > 0.8
    │   d) Update KV cache: remove pruned token entries
    │   e) Downstream layers (layer 11..N) operate on reduced cache
    │
    ▼
[8] Autoregressive Decoding
    │   Standard LLM generate() — token by token
    │   Until max_new_tokens or stop condition
    │
    ▼
Answer Text
```

---

## Stage 1: Vision-Side Pruning (DPC-KNN)

**File**: `models/pllava/pllava_prumerge.py` + `models/pllava/elastic_cache.py`

### Algorithm Detail

For each temporal window (subset of video frames):

```
1. Get window tokens: shape (B, N_window, D)
2. Compute N×N cosine similarity matrix
3. DPC-KNN clustering:
   a) ρ_i = sum of similarities to top-k neighbors (k=7)
   b) δ_i = min distance to any point with higher ρ
   c) Cluster centers = points with γ = ρ × δ above threshold
   d) Assign each point to nearest higher-ρ neighbor
4. Cross-window alignment:
   - Track cluster center IDs across windows
   - Match centers by position proximity
5. Static/Dynamic split:
   - STATIC: centers appearing at same position across windows
   - DYNAMIC: unmatched or position-varying centers
6. Merge:
   - Static: keep top cluster_ratio (0.5) centroids
   - Dynamic: keep top cluster_ratio per frame per window
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau` | 0.8 | Cosine similarity threshold for DPC-KNN neighbor selection |
| `k` | 7 | Number of KNN neighbors for density estimation |
| `temporal_segment_ratio` | 0.25 | Window size as fraction of total frames |
| `cluster_ratio` | 0.5 | Fraction of tokens retained per cluster/window |
| `pooling_shape` | (16, 12, 12) | Spatial pooling: (pool_size, height, width) |

---

## Stage 2: LLM-Side Pruning (Text-Pivot)

**File**: `models/pllava/llama.py` (class `TextPivotMerge_LayerWise`)

### Algorithm Detail

```
At selected_layer (default = 10th decoder layer):

1. Compute attention: Q_text × K_visual → attention scores
2. If head=0: average attention across all heads
3. Score each visual token by its attention from text tokens
4. Apply dual threshold:
   a) Alpha pruning (α=0.4): keep only top 40% scoring tokens
   b) Tau pruning (τ=0.8): keep only tokens with score > 0.8
5. Update KV cache:
   - Remove k/v entries for pruned visual tokens
6. All subsequent layers (11..N) operate on the reduced cache
```

---

## Inference Pipeline

**File**: `scripts/infer_single_video.py`

```
1. Load model with pruning config
2. BASELINE run:
   - Disable merge_frames_dynamic (identity)
   - Set alpha=1.0 (no LLM pruning)
3. PRUNE run:
   - Enable merge_frames_dynamic
   - Set alpha=0.4
4. Report comparison:
   - Vision tokens: original → after merge → after LLM prune
   - Per-window static/dynamic counts
   - Wall-clock time speedup
   - Answer comparison
```

---

## Training Pipeline

**File**: `tasks/train/train_pllava_nframe_accel.py`

### Flow

```
1. Load config (config_pllava_nframe.py)
2. Initialize Accelerator (mixed precision, gradient accumulation)
3. Create model:
   - Load pretrained PLLaVA checkpoint
   - Freeze base Llama + CLIP
   - Apply LoRA (q_proj, v_proj; rank=128, alpha=32, dropout=0.05)
4. Create datasets:
   - ITVidTrainDataset (video QA pairs from annotation JSONs)
   - ITImgTrainDataset (image QA pairs)
   - Weighted random sampling across media types
5. Preprocessing (tokenization):
   - Replace <Video> tags with <image>
   - Split roles (system/user/assistant)
   - Create input_ids / labels / attention_mask
   - Labels: -100 for input tokens (ignored), token IDs for completion
   - Pad/truncate to max_txt_l=512
6. Training loop:
   - Forward pass → cross-entropy loss (completion tokens only)
   - Backward pass → gradient accumulation
   - Update only LoRA + projector weights
   - Save checkpoints every save_steps
   - Resume from checkpoint (optimizer, scheduler, step)
7. Logging: wandb / tensorboard
```

### Training Config (default)

| Parameter | Value |
|-----------|-------|
| num_frames | 16 |
| batch_size | 8 |
| learning_rate | 2e-5 |
| optimizer | AdamW |
| scheduler | Cosine (2 epochs) |
| LoRA rank | 128 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| pooling_shape | (16, 8, 8) |

---

## Evaluation Benchmarks

| Benchmark | File | Description |
|-----------|------|-------------|
| **MVBench** | `tasks/eval/mvbench/` | 20 subtasks (action sequence, object interaction, etc.) |
| **VideoMME** | `tasks/eval/videomme/` | Short/medium/long video categories |
| **EgoSchema** | `tasks/eval/egoshcema/` | FullSet (500 videos) |
| **VCGBench** | `tasks/eval/vcgbench/` | VideoChatGPT-Bench (GPT-based scoring) |

### Default Eval Command

```bash
python -m tasks.eval.mvbench.pllava_eval_mvbench \
  --selected_layer 10 --alpha 0.4 --tau 0.8 \
  --temporal_segment_ratio 0.25 --cluster_ratio 0.5
```

---

## Key Module Responsibilities (Detailed)

### `models/pllava/pllava_prumerge.py`
- `CLIPVisionTower`: wraps CLIP vision model; adds `merge_frames_dynamic()`, `expand_tower()` for anyres, spatial pooling (`pooling_shape`)
- `merge_frames_dynamic()`: splits tokens into temporal windows, calls DPC-KNN pruner, returns merged tokens with per-window statistics (static/dynamic counts)

### `models/pllava/elastic_cache.py`
- `FastVTSPruner`: DPC-KNN clustering engine — computes cosine similarity, finds cluster centers, merges tokens
- `ElasticCachePruner`: alternative elastic cache pruning (merge + merge)
- `VTPWindowCache`: manages per-window clustering, static/dynamic separation, cross-window alignment
- DPC-KNN core: density-based peak clustering with KNN

### `models/pllava/llama.py`
- `LlamaForCausalLMVTP`: custom Llama with VTP (Visual Token Pruning) hook at `selected_layer`
- `TextPivotMerge_LayerWise`: attention-based visual token pruning — computes text→image attention scores, applies alpha+tau thresholds, updates KV cache

### `models/pllava/modify_llama.py`
- Alternative attention variants: `H2O` (heavy-hitter), `MixMer`, `PixelPrunMerge` (resampling), `PivotMerge` (weighted average), `Weighted`, `TextPrior`

### `models/pllava/modeling_pllava.py`
- `PllavaForConditionalGeneration`: main model class wrapping CLIP vision + Llama LLM + projector; supports `.generate()` with pruning hooks

### `utils/config.py`
- `Config` class: loads from `.py`/`.yaml`/`.json` with `_base_` inheritance, CLI overrides via dot-separated keys, `${variable}` interpolation, `eval()` dynamic evaluation

---

## Summary of Pruning Effect

| Stage | Input Tokens | Output Tokens | Reduction |
|-------|-------------|---------------|-----------|
| Raw vision (16 frames × 144 pooled) | 2,304 | — | — |
| After Stage 1 (DPC-KNN merge) | 2,304 | ~500-800 | ~65-75% |
| After Stage 2 (text-pivot prune) | ~500-800 | ~200-320 | ~60% |
| **Total** | **2,304** | **~200-320** | **~85-90%** |

The combined pipeline achieves ~2× wall-clock speedup with <1% accuracy loss on standard video QA benchmarks.
