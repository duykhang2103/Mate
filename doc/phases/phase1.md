# Phase 1: Baseline Reproduction

> **Goal**: Reproduce PruneVid paper results on all 4 benchmarks.
> **Duration**: Weeks 1–2
> **Status**: IN PROGRESS

---

## Progress Tracker

| Benchmark | Status      | Result | Target      | Match? |
| --------- | ----------- | ------ | ----------- | ------ |
| MVBench   | RUNNING     | —      | 47.6        | —      |
| VideoMME  | NOT STARTED | —      | 45.3        | —      |
| EgoSchema | NOT STARTED | —      | 49.0 / 42.6 | —      |
| VCGBench  | NOT STARTED | —      | 2.98        | —      |

---

## What You Need To Do

### Step 0: Verify Dataset Structure

Each benchmark expects a specific directory layout under `DATAS/`. Download from HuggingFace and place them as follows:

#### MVBench (ALREADY DONE — partially)

```
DATAS/MVBench/
├── json/                    # 20 JSON files ✓
│   ├── action_sequence.json
│   ├── action_prediction.json
│   └── ... (20 total)
└── video/                   # Videos — zip files need extraction
    ├── star.zip             # → extract to star/
    ├── ssv2_video.zip       # → extract to ssv2_video/
    ├── clevrer.zip          # → extract to clevrer/
    ├── perception.zip       # → extract to perception/
    ├── Moments_in_Time_Raw.zip
    ├── FunQA_test.zip
    ├── sta.zip
    ├── scene_qa.zip
    ├── tvqa.zip
    ├── vlnqa.zip
    └── MVBench_videos_ntu.txt  # NTU videos need manual download (optional)

Download: https://huggingface.co/datasets/OpenGVLab/MVBench
```

**Action needed**: Extract all zip files:

```bash
cd /workspace/Mate/DATAS/MVBench/video
for z in *.zip; do unzip -o "$z" -d .; done
```

**Note**: 14 videos are missing from the extracted star.zip (12 for Action Sequence, 2 for Object Existence). The NTU videos (200 samples, Fine-grained Pose task) require manual download from ROSE Lab due to licensing. Skip if you can't get them — the eval will skip missing videos gracefully.

#### VideoMME (NOT DOWNLOADED)

```
DATAS/Video-MME/
├── json/
│   ├── short.json
│   ├── medium.json
│   └── long.json
└── data/
    └── (video files — organized by the dataset's structure)
```

Download: https://huggingface.co/datasets/lmms-lab/Video-MME

#### EgoSchema (NOT DOWNLOADED)

```
DATAS/ego_schema/
├── json/
│   └── egoschema_fullset.json
└── videos/
    └── (video files)
```

Download: https://huggingface.co/datasets/lmms-lab/egoschema

#### VCGBench (NOT DOWNLOADED)

```
DATAS/VCGBench/
├── Zero_Shot_QA/
│   └── (JSON annotation files)
└── Videos/
    └── (video files)
```

Download: https://huggingface.co/datasets/lmms-lab/VideoChatGPT

**Note**: VCGBench requires an **OpenAI API key** for GPT-based scoring. Set it before running:

```bash
export OPENAI_API_KEY="your-key-here"
```

---

### Step 1: Wait for MVBench to Finish

MVBench is currently running. Check progress:

```bash
tail -f log-mvbench.log
```

When finished, results will be in `test_results/mvbench/`:

- `all_results.json` — full results
- `upload_leaderboard.json` — per-task accuracy + average

**Expected**: ~47.6 average accuracy (paper result). Without NTU videos, you'll have ~3800/4000 samples across 19 tasks.

Record your result in the Progress Tracker above.

---

## Quick Eval (Fast Iteration)

Full MVBench takes ~4 hours. For rapid testing during development, use the quick eval flags:

### Available flags (MVBench only)

| Flag            | Description                            | Example                                  |
| --------------- | -------------------------------------- | ---------------------------------------- |
| `--tasks`       | Comma-separated task names to evaluate | `--tasks "Action Sequence,State Change"` |
| `--max_samples` | Max samples per task                   | `--max_samples 20`                       |

### Example: Quick sanity check (~2 min)

Run 1 task, 20 samples each → ~40 samples, ~2 minutes:

```bash
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_quick \
    --num_frames 16 \
    --use_lora --lora_alpha 14 \
    --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 \
    --selected_layer 10 \
    --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 \
    --cluster_ratio 0.5 \
    --tasks "Action Sequence" \
    --max_samples 20
```

### Example: Motion-heavy tasks only (~30-45 min)

Run 3 motion tasks, 100 samples each → ~300 samples, ~30-45 minutes:

```bash
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_motion \
    --num_frames 16 \
    --use_lora --lora_alpha 14 \
    --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 \
    --selected_layer 10 \
    --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 \
    --cluster_ratio 0.5 \
    --tasks "Action Sequence,Action Prediction,State Change" \
    --max_samples 100
```

### Available MVBench tasks (20 total)

| Task                     | Type       | Samples                    |
| ------------------------ | ---------- | -------------------------- |
| Action Sequence          | Motion     | 200                        |
| Action Prediction        | Motion     | 200                        |
| Action Antonym           | Motion     | 200                        |
| Fine-grained Action      | Motion     | 200                        |
| Unexpected Action        | Motion     | 200                        |
| Object Existence         | Perception | 200                        |
| Object Interaction       | Motion     | 200                        |
| Object Shuffle           | Perception | 200                        |
| Moving Direction         | Motion     | 200                        |
| Action Localization      | Motion     | 200                        |
| Scene Transition         | Perception | 200                        |
| Action Count             | Perception | 200                        |
| Moving Count             | Motion     | 200                        |
| Moving Attribute         | Motion     | 200                        |
| State Change             | Motion     | 200                        |
| Fine-grained Pose        | Motion     | 200 (NTU - license needed) |
| Character Order          | Perception | 200                        |
| Egocentric Navigation    | Motion     | 200                        |
| Episodic Reasoning       | Perception | 200                        |
| Counterfactual Inference | Perception | 200                        |

### Recommended iteration workflow

1. **Code change** → run with `--tasks "Action Sequence" --max_samples 20` (~2 min)
2. **Looks promising** → run with `--tasks "Action Sequence,Action Prediction,State Change" --max_samples 100` (~30-45 min)
3. **Confirm improvement** → run full MVBench without `--tasks`/`--max_samples` (~4 hours)

### Note on save_path

Each run saves to `--save_path`. Use different paths for quick vs full evals to avoid overwriting:

- Quick: `test_results/mvbench_quick`
- Motion subset: `test_results/mvbench_motion`
- Full: `test_results/mvbench`

---

### Step 2: Run VideoMME

```bash
cd /workspace/Mate

python -m tasks.eval.videomme.pllava_eval_videomme \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/videomme \
    --num_frames 16 \
    --use_lora --lora_alpha 14 \
    --top_p 1.0 --temperature 1.0 \
    --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 \
    --conv_mode eval_videomme \
    --selected_layer 10 \
    --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 \
    --cluster_ratio 0.5
```

**Expected**: ~45.3 average accuracy (paper result).
**Time estimate**: ~4-8 hours depending on GPU (VideoMME has ~900 samples with longer videos).

Check result:

```bash
cat test_results/videomme/upload_leaderboard.json
```

---

### Step 3: Run EgoSchema

```bash
cd /workspace/Mate

python -m tasks.eval.egoshcema.pllava_eval_egoschema \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/egoschema \
    --num_frames 16 \
    --use_lora --lora_alpha 14 \
    --top_p 1.0 --temperature 1.0 \
    --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 \
    --conv_mode eval_mvbench \
    --selected_layer 10 \
    --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 \
    --cluster_ratio 0.5
```

**Expected**: ~49.0 (subset) / ~42.6 (fullset) accuracy (paper result).
**Time estimate**: ~3-6 hours (500 samples).

Check result:

```bash
cat test_results/egoschema/upload_leaderboard.json
```

---

### Step 4: Run VCGBench

**Requires OpenAI API key** for GPT-based scoring:

```bash
export OPENAI_API_KEY="your-key-here"
```

```bash
cd /workspace/Mate

python -m tasks.eval.vcgbench.pllava_eval_vcgbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/vcgbench \
    --num_frames 16 \
    --use_lora --lora_alpha 4 \
    --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 \
    --selected_layer 5 \
    --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 \
    --cluster_ratio 0.5
```

**IMPORTANT**: VCGBench uses **different LoRA settings**: `lora_alpha=4` and `selected_layer=5` (not 14/10 like the other benchmarks).

**Expected**: ~2.98 average score (paper result).
**Time estimate**: ~2-4 hours for inference + GPT scoring time.

Check result:

```bash
cat test_results/vcgbench/upload_leaderboard.json
```

---

## Code Fixes Applied (Phase 1)

The following bugs were fixed before starting evaluation:

| File                                            | Fix                                                  | Why                               |
| ----------------------------------------------- | ---------------------------------------------------- | --------------------------------- |
| `tasks/eval/mvbench/__init__.py`                | Fixed `save_results()` double-counting bug           | Avg accuracy was inflated         |
| `tasks/eval/videomme/__init__.py`               | Fixed `save_results()` double-counting bug           | Same bug as MVBench               |
| `tasks/eval/egoshcema/__init__.py`              | Fixed `save_results()` double-counting bug           | Same bug as MVBench               |
| `tasks/eval/mvbench/__init__.py`                | Added skip logic for missing videos in `__getitem__` | Prevents crash on missing files   |
| `tasks/eval/mvbench/pllava_eval_mvbench.py`     | Added `if example is None: continue` in eval loop    | Skips None from `__getitem__`     |
| `tasks/eval/mvbench/pllava_eval_mvbench.py`     | Added `--tasks` and `--max_samples` flags            | Fast iteration during development |
| `tasks/eval/videomme/__init__.py`               | Added skip logic for missing videos in `__getitem__` | Same as MVBench                   |
| `tasks/eval/videomme/pllava_eval_videomme.py`   | Added `if example is None: continue` in eval loop    | Same as MVBench                   |
| `tasks/eval/egoshcema/__init__.py`              | Added skip logic for missing videos in `__getitem__` | Same as MVBench                   |
| `tasks/eval/egoshcema/pllava_eval_egoschema.py` | Added `if example is None: continue` in eval loop    | Same as MVBench                   |
| `tasks/eval/vcgbench/pllava_eval_vcgbench.py`   | Added `if example is None: continue` in eval loop    | Same as MVBench                   |

---

## Dependencies Check

Before running each benchmark, verify these are installed:

```bash
python -c "
import torch, transformers, decord, peft, safetensors, accelerate
print('All dependencies OK')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}, {torch.cuda.device_count()} GPU(s)')
print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
"
```

If `mmcv` is needed (for optical flow — optional), you'll need to handle the import. The current code has a bare `from mmcv.runner import load_checkpoint` in `model_utils.py`. Either:

- Install mmcv: `pip install mmcv` (heavy dependency)
- Or comment out the import and set `load_checkpoint = None` (the optical flow model is optional and loads via try/except)

---

## Troubleshooting

### "No such file or directory" for video

→ Video not extracted or not downloaded. Run the unzip commands in Step 0.

### "RuntimeError: Error reading ..."

→ Corrupt video file. The skip logic should handle this — the sample will be skipped.

### CUDA out of memory

→ Reduce `--num_frames` to 8 temporarily, or ensure no other process is using the GPU.

### Import errors

→ Ensure you're running from `/workspace/Mate` and all dependencies are installed.

### Results don't match paper

→ Small differences (±0.3) are normal due to floating-point precision and missing videos. Larger gaps suggest a weight loading or configuration issue.

---

## After Phase 1

Once all 4 baselines are recorded, you're ready for Phase 2 (Multi-Head Entropy Fix). Update the Progress Tracker above with your actual results and proceed to `doc/phases/phase2.md`.
