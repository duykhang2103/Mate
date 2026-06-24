# Datasets for Motion-Heavy & Temporally Complex Videos

## Your Goal

> Benchmark to check if method improves accuracy on **motion-heavy and temporally complex videos**

---

## ✅ Current Benchmarks (Best for Video LLMs)

### 1. **MVBench** — **BEST for Motion-Heavy** ⭐⭐⭐⭐⭐

| Aspect                         | Details                                                                                                                                                                                                                                                                           |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Motion-Heavy Subsets**       | `action_antonym`, `action_count`, `action_localization`, `action_prediction`, `action_sequence`, `fine_grained_action`, `fine_grained_pose`, `moving_attribute`, `moving_count`, `moving_direction`, `object_interaction`, `object_shuffle`, `state_change`, `unexpected_actions` |
| **Temporally Complex Subsets** | `action_sequence`, `action_prediction`, `character_order`, `episodic_reasoning`, `scene_transition`                                                                                                                                                                               |
| **Video Length**               | Short clips (~5-30s)                                                                                                                                                                                                                                                              |
| **Tasks**                      | 20 temporal tasks (action, localization, prediction, sequence)                                                                                                                                                                                                                    |
| **Why It's Good**              | Specifically designed for temporal understanding that single-frame can't solve                                                                                                                                                                                                    |
| **Motion-Heavy Count**         | ~8-10 subsets are motion-heavy (40-50% of total)                                                                                                                                                                                                                                  |
| **PruneVid Used It**           | Yes, baseline 46.6 → 47.6                                                                                                                                                                                                                                                         |

**Actionable Code:**

```python
motion_heavy_subsets = [
    "action_antonym", "action_count", "action_localization",
    "action_prediction", "action_sequence", "fine_grained_action",
    "fine_grained_pose", "moving_attribute", "moving_count",
    "moving_direction", "object_interaction", "object_shuffle",
    "state_change", "unexpected_actions"
]

temporal_complex_subsets = [
    "action_sequence", "action_prediction", "character_order",
    "episodic_reasoning", "scene_transition"
]
```

---

### 2. **VideoMME** — **Good for Temporal Complexity** ⭐⭐⭐⭐

| Aspect                   | Details                                                                              |
| ------------------------ | ------------------------------------------------------------------------------------ |
| **Motion-Heavy Domains** | `Sports competitions`, `Film & Television` (action scenes), `Artistic Performance`   |
| **Temporally Complex**   | `Knowledge clips` (long reasoning), `Film & Television` (complex storytelling)       |
| **Video Length**         | Short (<2 min), Medium (4-15 min), Long (30-60 min)                                  |
| **Tasks**                | 12 task types: action recognition, temporal reasoning, event understanding, synopsis |
| **Why It's Good**        | Explicitly targets long-context and multimodal capabilities                          |
| **Duration Breakdown**   | Short videos: less temporal complexity; Long videos: MORE temporal complexity        |
| **PruneVid Used It**     | Yes, baseline 44.4 → 45.3                                                            |

**Actionable Code:**

```python
# VideoMME reports accuracy by duration
short_videos = "<2 minutes"  # Less temporal complexity
medium_videos = "4-15 minutes"  # Medium temporal complexity
long_videos = "30-60 minutes"  # MORE temporal complexity
```

---

### 3. **EgoSchema** — **BEST for Very Long Temporal** ⭐⭐⭐⭐

| Aspect                        | Details                                                                   |
| ----------------------------- | ------------------------------------------------------------------------- |
| **Motion-Heavy**              | Ego-centric human activities (daily actions, walking, cooking)            |
| **Temporally Complex**        | **EXTREMELY long** (3-minute clips, 5.7x longer than 2nd closest dataset) |
| **Video Length**              | 3 minutes per clip (very long for videoQA)                                |
| **Tasks**                     | Multiple-choice QA on long video understanding                            |
| **Why It's Good**             | **10x-100x longer temporal reasoning** than almost all other datasets     |
| **Intrinsic Temporal Length** | **5.7x longer than 2nd closest dataset**                                  |
| **PruneVid Used It**          | Yes, subset 47.8 → 49.0, fullset 42.6 → 42.6 (no improvement!)            |

**Actionable Code:**

```python
# EgoSchema is PERFECT for "temporally complex" goal (10x-100x harder temporal reasoning)
# Your method should improve EgoSchema fullset (where PruneVid failed)
# EgoSchema has "temporal certificate sets" for measuring intrinsic temporal difficulty
```

---

### 4. **VCG-Bench (VideoChatGPT-Bench)** — **Weak for Your Goal** ⭐

| Aspect                 | Details                                          |
| ---------------------- | ------------------------------------------------ |
| **Motion-Heavy**       | Mixed (some motion, mostly general videoQA)      |
| **Temporally Complex** | Low-Medium (shorter videos, less temporal depth) |
| **Video Length**       | Short clips                                      |
| **Tasks**              | General videoQA, captioning                      |
| **PruneVid Used It**   | Yes, baseline 2.99 → 2.98 (SLIGHT DEGRADATION!)  |

**Recommendation:** Less important for your goal; keep but don't focus.

---

## 🆕 New Datasets to ADD (Strong for Motion/Temporal)

### 5. **TemporalBench** — **BEST for Fine-Grained Temporal** ⭐⭐⭐⭐⭐ (NEW!)

| Aspect                 | Details                                                         |
| ---------------------- | --------------------------------------------------------------- |
| **Motion-Heavy**       | YES: `motion magnitude`, `action frequency`                     |
| **Temporally Complex** | **SPECIFICALLY for fine-grained temporal**                      |
| **Video Length**       | Short to medium clips                                           |
| **Tasks**              | Action frequency, motion magnitude, event order                 |
| **Why It's Good**      | **Dedicated benchmark for fine-grained temporal understanding** |
| **SOTA Performance**   | GPT-4o: 38.5% (huge human-AI gap ~30%)                          |
| **PruneVid Used It**   | **NO** (new benchmark, 2024) ✅                                 |
| **Size**               | ~10K video QA pairs, ~2K human annotations                      |

**Actionable Code:**

```python
# TemporalBench tasks:
motion_tasks = ["action_frequency", "motion_magnitude"]
temporal_tasks = ["event_order", "temporal_dynamics"]
```

---

### 6. **PerceptionTest** — **Good for Temporal + Motion** ⭐⭐⭐⭐

| Aspect                 | Details                                            |
| ---------------------- | -------------------------------------------------- |
| **Motion-Heavy**       | YES: 73.5K temporal action segments                |
| **Temporally Complex** | YES: temporal action segments, object tracks       |
| **Video Length**       | Real-world videos (mixed lengths)                  |
| **Tasks**              | Memory, Abstraction, Physics, Semantics            |
| **Why It's Good**      | 190K object tracks, 73.5K temporal action segments |
| **Size**               | 11.6K videos, 190K object tracks                   |
| **PruneVid Used It**   | **NO** (newer) ✅                                  |

---

### 7. **LongVideoBench** — **BEST for Long-Form Temporal** ⭐⭐⭐⭐⭐

| Aspect                 | Details                                              |
| ---------------------- | ---------------------------------------------------- |
| **Motion-Heavy**       | Medium (mixed content)                               |
| **Temporally Complex** | **EXTREMELY long** (long visual inputs)              |
| **Video Length**       | Very long (10+ minutes)                              |
| **Tasks**              | Long-context interleaved video understanding         |
| **Why It's Good**      | **Essentially evaluates LMMs on long visual inputs** |
| **SOTA Performance**   | T\* improves GPT-4o: 50.5% → 53.1%                   |
| **PruneVid Used It**   | **NO** ✅                                            |

---

## 📊 Final Recommendation: Which Datasets to Use

### **Minimum Viable (2-3 months)**

```
Primary: MVBench (motion-heavy subsets) + EgoSchema fullset (very long temporal)
- MVBench for motion-heavy validation
- EgoSchema fullset for temporal complexity (where PruneVid FAILED)
```

### **Recommended (4 months)**

```
Primary: MVBench + VideoMME (long duration) + EgoSchema fullset
Secondary: TemporalBench + PerceptionTest

- MVBench: motion-heavy (40-50% subsets)
- VideoMME: long temporal (30-60 min videos)
- EgoSchema: EXTREMELY long temporal (10x-100x harder)
- TemporalBench: fine-grained temporal (NEW, PERFECT for your goal)
- PerceptionTest: temporal action + motion
```

### **Excellence (6 months)**

```
Full suite: MVBench + VideoMME + EgoSchema + TemporalBench + PerceptionTest + LongVideoBench + VCG-Bench
```

---

## 🎯 Specific Goal: "Motion-Heavy & Temporally Complex"

### **Best for Motion-Heavy:**

1. **MVBench** (action subsets: action_antonym, action_count, fine_grained_action, moving_attribute, moving_direction)
2. **TemporalBench** (motion magnitude, action frequency)
3. **PerceptionTest** (73.5K temporal action segments)

### **Best for Temporally Complex:**

1. **EgoSchema fullset** (5.7x longer intrinsic temporal length, 10x-100x harder than other datasets)
2. **VideoMME long duration** (30-60 min videos)
3. **TemporalBench** (fine-grained temporal understanding)
4. **LongVideoBench** (very long visual inputs)

---

## ✅ Action Plan for Your Research (Phase 6 Additions)

```python
# Add these to your experiment matrix:

# 1. MVBench motion-heavy subset breakdown
MVBench_motion_heavy = [
    "action_antonym", "action_count", "action_localization",
    "action_prediction", "action_sequence", "fine_grained_action",
    "fine_grained_pose", "moving_attribute", "moving_count",
    "moving_direction", "object_interaction", "object_shuffle",
    "state_change", "unexpected_actions"
]

# 2. VideoMME long duration breakdown
VideoMME_long = "30-60 minutes"  # Most temporal complexity

# 3. EgoSchema fullset (where PruneVid failed)
EgoSchema_fullset = "5000+ questions, 3-minute clips"  # 10x-100x harder temporal

# 4. NEW: TemporalBench (fine-grained temporal)
TemporalBench_motion = ["action_frequency", "motion_magnitude"]
TemporalBench_temporal = ["event_order"]

# 5. NEW: PerceptionTest temporal action
PerceptionTest_action = "73.5K temporal action segments"
```

### **Priority Order:**

| Priority | Dataset                      | Why                                                     |
| -------- | ---------------------------- | ------------------------------------------------------- |
| 1        | **MVBench** (motion subsets) | HAVE IT, motion-heavy, PruneVid baseline                |
| 2        | **EgoSchema fullset**        | HAVE IT, EXTREMELY long temporal, PruneVid failed here  |
| 3        | **VideoMME** (long duration) | HAVE IT, 30-60 min videos                               |
| 4        | **TemporalBench** (NEW)      | PERFECT for fine-grained temporal, no PruneVid baseline |
| 5        | **PerceptionTest** (NEW)     | Good for temporal action, no baseline                   |
| 6        | VCG-Bench                    | Keep, but less important                                |

---

## Bottom Line

**Your current 4 datasets are good, but add TemporalBench** (perfect for your goal) for strongest validation of "motion-heavy & temporally complex" improvements.
