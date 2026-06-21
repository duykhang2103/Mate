# PruneVid Repository Overview

## What this repository contributes

PruneVid implements a training-free visual token pruning method for video large language models (video LLMs). It integrates with PLLaVA and adds a two-stage pruning and merging pipeline that reduces visual token count while preserving accuracy on video QA benchmarks. The main contributions visible in the code are:

- A temporal and spatial token merging strategy that separates static vs dynamic visual regions, then merges tokens within each group to reduce redundancy.
- A question-aware pruning step inside the language model that keeps the most relevant visual tokens based on text-to-image attention.
- End-to-end evaluation scripts for MVBench, VideoMME, and EgoSchema with configurable pruning hyperparameters.

The README and code indicate the method is evaluated on multiple benchmarks and compares against other pruning/merging baselines, with large FLOPs reductions at comparable accuracy.

## High-level method summary

PruneVid reduces visual tokens in two stages:

1) Vision-side spatial-temporal merging
   - Split video tokens into temporal segments.
   - For each segment, separate static regions (high similarity across frames) from dynamic regions.
   - Merge spatially similar tokens within static and dynamic groups using clustering.

2) LLM-side question-aware pruning
   - After the multimodal tokens are inserted into the text sequence, compute attention from text tokens to image tokens at a selected LLM layer.
   - Keep top tokens based on attention (parameterized by alpha) for both static and dynamic regions.
   - Update the KV cache and hidden states so subsequent decoding attends to the reduced visual token set.

This is training-free: it operates at inference time on pretrained models.

## Visual overview

```mermaid
flowchart LR
  A[Video frames] --> B[CLIP vision encoder]
  B --> C[Patch tokens per frame]
  C --> D[Temporal segmentation]
  D --> E[Static vs dynamic split
  (tau threshold)]
  E --> F[Spatial token merge
  (DPC-KNN, cluster_ratio)]
  F --> G[Merged visual tokens
  + metadata]
  G --> H[Insert into text sequence]
  H --> I[LLM layer selected_layer
  attention to image tokens]
  I --> J[Keep top tokens
  (alpha)]
  J --> K[Pruned KV cache
  + final decoding]
```

Legend:

- tau: static vs dynamic similarity threshold inside each temporal window.
- temporal_segment_ratio: number of temporal windows relative to frames.
- cluster_ratio: spatial merge strength inside static/dynamic groups.
- alpha: fraction of visual tokens kept after question-aware pruning.

## Where the method lives in the code

### 1) Vision-side spatial-temporal merging

File: models/pllava/modeling_pllava.py

Key logic:

- merge_frames_dynamic(...):
  - Uses frame-level clustering to determine temporal windows (temporal_segment_ratio controls the number of temporal segments).
  - Within each window, computes per-patch similarity across frames to separate static vs dynamic tokens using a similarity threshold tau.
  - Merges spatial tokens for static and dynamic groups using DPC-KNN clustering (cluster_ratio controls spatial merge ratio).

- spatial_merge_tokens(...):
  - Runs DPC-KNN clustering and computes cluster centroids for token merging.

- merge_frames_dynamic(...) output:
  - Returns merged visual features plus metadata: static_sizes, dynamic_sizes, window_sizes. These are later used by the LLM-side pruning to select tokens per window.

Relevant hyperparameters (configurable in evaluation scripts and config):

- tau: similarity threshold for static vs dynamic separation.
- temporal_segment_ratio: fraction of temporal segments used when clustering frames.
- cluster_ratio: fraction of spatial clusters retained in each region.

### 2) LLM-side question-aware pruning

File: models/pllava/llama.py

Key logic:

- LlamaModelVTP and VTPWindowCache:
  - The LLM runs normally, but at a selected layer (selected_layer), it exposes attention weights.
  - VTPWindowCache processes text-to-image attention and selects top visual tokens.
  - For each temporal window, it keeps a fraction alpha of static tokens and dynamic tokens, based on attention scores.
  - The KV cache and hidden states are then pruned so subsequent decoding attends only to the retained tokens.

Relevant hyperparameters:

- selected_layer: which LLM layer provides attention for pruning.
- alpha: ratio of tokens kept per static and dynamic group.
- head, softmax: control attention aggregation in pruning.

### 3) Multimodal integration and forward flow

File: models/pllava/modeling_pllava.py

Key logic:

- PllavaForConditionalGeneration.forward(...):
  - Extracts CLIP visual features.
  - Applies the multimodal projector.
  - Calls merge_frames_dynamic(...) to reduce tokens.
  - Merges visual tokens into the text sequence.
  - Runs the LLM with pruning-aware cache (LlamaForCausalLMVTP).

### 4) Optional vision pruning prototype

File: models/pllava/pllava_prumerge.py

This file contains an alternative token pruning and merging strategy directly inside the CLIP vision tower. It uses attention from the CLIP CLS token to select important patches and merges non-topk patches into weighted cluster centers. It appears to be an experimental or legacy path rather than the main PruneVid pipeline.

## Configuration and evaluation entry points

- Configuration: models/pllava/configuration_pllava.py
  - Defines pruning parameters (selected_layer, alpha, tau, temporal_segment_ratio, cluster_ratio).

- Model loading: tasks/eval/model_utils.py
  - Passes pruning parameters into PllavaConfig for evaluation.

- Evaluation scripts:
  - tasks/eval/mvbench/pllava_eval_mvbench.py
  - tasks/eval/videomme/pllava_eval_videomme.py
  - tasks/eval/egoshcema/pllava_eval_egoschema.py

The scripts allow sweeps over alpha, tau, temporal_segment_ratio, and cluster_ratio (see scripts/eval.sh in README).

## Method details in plain language

- Temporal segmentation:
  - Frames are clustered into a smaller number of temporal segments. This is controlled by temporal_segment_ratio. The method then processes each segment as a window.

- Static vs dynamic separation:
  - Inside each window, patch tokens are marked as static if they are similar across frames, and dynamic otherwise. Similarity is computed per patch across frames and thresholded by tau.

- Spatial merging:
  - Tokens in static and dynamic groups are clustered spatially and replaced by cluster centroids. This merges redundant tokens while keeping structure. The number of clusters is controlled by cluster_ratio.

- Question-aware pruning:
  - After visual tokens are inserted into the text stream, attention from text to image tokens is computed at a chosen LLM layer. Tokens with the highest attention are kept. The ratio alpha controls how many static and dynamic tokens survive per temporal window.

## Suggested reading order

1) README.md for overall motivation and evaluation tables.
2) models/pllava/modeling_pllava.py for the core temporal and spatial merge logic.
3) models/pllava/llama.py for the attention-based pruning inside the LLM.
4) tasks/eval/model_utils.py and tasks/eval/* for parameter wiring and evaluation usage.
