Below is a **concrete, focused experiment plan** that uses **3–5 strongest, paper-backed methods** you can apply to **PruneVid**, with:

- **What to change in PruneVid**
- **Which benchmarks to focus on**
- **Which metrics to report**
- **How to show you beat PruneVid clearly**

All ideas are backed by recent pruning papers (PruneVid itself, FastVID, SharpV, SToP, Information-Aware Pruning, etc.).

***

## Core Goal

Design **“PruneVid+”** that:

- Uses **the same or lower token budget** as PruneVid,
- But shows:
  - **Better accuracy on at least one benchmark family** (e.g., fine-grained or long-video),
  - **Better robustness across pruning ratios**,
  - Or **better efficiency metrics** (TTFT, memory) at similar accuracy.

You do not need to beat PruneVid everywhere; a clear win in one dimension is enough.

***

## Selected Methods (3–5 strongest, paper-backed)

We pick the following 4 methods:

1. **Adaptive / per-frame pruning ratios (motion/entropy-based)**  
   - Backed by: **FastVID** (2025) – dynamic density pruning with motion-awareness improves accuracy under comparable retention vs. PruneVid [FastVID].

2. **Information-aware / entropy-aware merging (instead of plain average)**  
   - Backed by: **Information-Aware Visual Token Pruning** (2025) – information-guided merging improves accuracy under aggressive pruning, especially on fine-grained tasks [Info-Aware].

3. **Sink-token suppression (penalize sink tokens in pruning score)**  
   - Backed by: **SToP** (2026) – sink-aware pruning reduces hallucination and improves fine-grained performance, often outperforming PruneVid on those metrics [SToP].

4. **Uptraining / robustness training with random pruning rates**  
   - Backed by: **EVS** (2025) and **SharpV** (2025) – training with random pruning rates makes models robust to aggressive pruning, often matching full accuracy at very low token budgets, outperforming PruneVid (training-free) in robustness [EVS][SharpV].

(If you want a 5th, you can add **multi-layer cross-attention for token selection** or **multi-objective optimization for efficiency**, but 4 is already rich.)

***

## 1. What to Change in PruneVid

### A. Adaptive / Per-frame Pruning Ratios

**Current PruneVid:**  
- Uses a fixed pruning ratio for all frames and static/dynamic regions.

**Change:**
1. **Compute a motion score per frame**:
   - Simple option: frame difference (norm of pixel difference between consecutive frames).
   - Or: token variance across frames in each region.
2. **Map motion score to pruning ratio**:
   - Low motion → higher pruning ratio (e.g., 80–90%).
   - High motion → lower pruning ratio (e.g., 30–50%).
   - Ensure the *average* retention is similar to PruneVid’s original setting (e.g., 12–20%).
3. Apply this per-frame ratio **before** PruneVid’s temporal and spatial merging.

**Implementation hint:**
- Keep PruneVid’s merging and selection logic.
- Only change the threshold/ratio used to decide how many tokens to keep per frame.

***

### B. Information-aware / Entropy-aware Merging

**Current PruneVid:**  
- Merges static tokens by simple **average** in time and space.

**Change:**
1. For each token:
   - Compute **entropy** or **variance** of its feature vector.
2. For temporal merging:
   - Instead of plain mean, use **weighted average**:
     - Weight tokens by their entropy/variance.
3. For spatial merging (DPC-KNN):
   - Modify the merging step inside each cluster:
     - Keep a small number of high-entropy tokens unmerged.
     - Or weight the cluster average by token entropy.

**Implementation hint:**
- This is a small change in the merging function.
- Keep DPC-KNN clustering; only change how cluster representatives are computed.

***

### C. Sink-token Suppression

**Current PruneVid:**  
- Uses cross-attention scores at layer `M` as the main importance signal.
- Does not explicitly handle “sink tokens”.

**Change:**
1. Define a **sink score** for each token:
   - `sink_score = attention_magnitude − α · token_variance` (or similar).
   - High attention but low variance → high sink score.
2. Modify the pruning score:
   - `final_score = cross_attention_score − λ · sink_score`.
3. Optionally:
   - Ensure at least X% of kept tokens have low sink score.

**Implementation hint:**
- Add a small pre-processing step before the LLM-guided selection.
- Tune λ to balance between accuracy and sink suppression.

***

### D. Uptraining / Robustness Training

**Current PruneVid:**  
- Training-free; no adaptation of the video LLM.

**Change:**
1. Add a **lightweight training phase** (e.g., 1–3 epochs) on your video LLM:
   - Randomly vary pruning ratio per sample (e.g., 10–30%).
   - Use PruneVid’s pipeline with your modifications (A–C).
2. Optionally add a **hidden-state reconstruction loss**:
   - `L_recon = ||h_pruned − h_full||²` (for a small subset of layers).
   - Train only:
     - Last 1–2 LLM layers, or
     - A small adapter on visual tokens.
3. Keep most of the model frozen.

**Implementation hint:**
- Use a small learning rate.
- Only a few epochs to avoid overfitting.
- This is a small “uptraining” over PruneVid, not full retraining.

***

## 2. Benchmarks to Focus On

You should focus on **two complementary benchmark families**:

### A. Main Video Benchmarks (for overall accuracy)

- **MVBench**
- **VideoMME**
- **EgoSchema**

These are the same benchmarks PruneVid uses.  
You need to show:

- Your method is **not worse** than PruneVid here (within ±0.5–1 point).
- Or clearly better on at least one of them.

### B. Fine-grained / Long-video Benchmarks (for targeted win)

These are where you aim to **clearly beat PruneVid**:

1. **Fine-grained / compositional benchmarks** (if available):
   - Any benchmark that tests:
     - Hallucination
     - Compositional reasoning
     - Object-level or temporal detail

2. **Long-video subsets**:
   - E.g., VideoMME long subsets, or any long-video benchmark you have.
   - PruneVid already has issues in very long videos; you can exploit this.

You can also create your own **long-video subset** from an existing benchmark if you need more controlled evaluation.

***

## 3. Metrics to Report

To clearly show you beat PruneVid, report these metrics:

### A. Accuracy Metrics

- **Main benchmarks**:
  - MVBench score
  - VideoMME score (overall + long subsets if available)
  - EgoSchema score (full + subset if available)

- **Fine-grained / long-video**:
  - Accuracy on fine-grained tasks (hallucination, compositional, temporal QA).
  - Accuracy on long-video subsets.

### B. Efficiency Metrics

For each method (PruneVid baseline vs. PruneVid+):

- **Token retention ratio** (% of original tokens kept).
- **FLOPs** (relative to baseline).
- **TTFT** (time-to-first-token).
- **Prefill time** (LLM prefill stage).
- **GPU memory usage** (peak).

### C. Robustness Metrics

- Plot **accuracy vs. token budget** curves:
  - Try multiple pruning ratios: e.g., 10%, 15%, 20%, 25%, 30%.
  - For each, report accuracy on MVBench and one fine-grained/long benchmark.
- Show that your method’s curve is **flatter** (less degradation) than PruneVid.

***

## 4. How to Show You Beat PruneVid

Design your experiment and results so that at least one of the following is clearly visible:

### Scenario 1: Better Accuracy at Same Token Budget

- Fix token budget (e.g., 12%).
- Show:
  - Your method has **higher accuracy** on fine-grained or long-video benchmarks.
  - On main benchmarks, it is **within ±0.5–1 point** of PruneVid.
- Claim: “PruneVid+ improves fine-grained/long-video performance at same efficiency.”

### Scenario 2: Same Accuracy with Lower Token Budget

- Find the token budget where your method matches PruneVid’s main benchmark accuracy.
- Show that this budget is **lower** (e.g., 10% vs. 12%).
- Claim: “PruneVid+ achieves same accuracy with fewer tokens, reducing FLOPs and memory.”

### Scenario 3: Better Robustness / Stability Across Pruning Ratios

- Show accuracy vs. token budget curves for both methods.
- Demonstrate that your method degrades **more slowly** as you prune more.
- Claim: “PruneVid+ is more robust to aggressive pruning, especially on long videos / fine-grained tasks.”

### Scenario 4: Better Efficiency at Similar Accuracy

- At similar main benchmark accuracy:
  - Show lower TTFT, prefill time, and memory for your method.
- Claim: “PruneVid+ achieves same accuracy with better efficiency.”

Even if you only achieve one of these, it’s a valid and strong contribution.

***

## 5. Concrete Experiment Plan (Step-by-step)

### Step 1: Baseline Reproduction

- Run **original PruneVid** on:
  - MVBench, VideoMME, EgoSchema.
  - One fine-grained/long subset.
- Record:
  - Accuracy.
  - Token ratio, FLOPs, TTFT, memory.
- Ensure numbers are close to the paper (or at least consistent).

### Step 2: Add Method A Only (Adaptive Ratios)

- Modify PruneVid to use **adaptive per-frame ratios**.
- Run on the same benchmarks.
- Compare:
  - Accuracy changes.
  - Token distribution across frames.
- If accuracy improves or stays similar, keep this change.

### Step 3: Add Method B (Info-aware Merging)

- Combine:
  - Adaptive ratios + info-aware merging.
- Run again.
- Check if:
  - Fine-grained performance improves more than main benchmarks.

### Step 4: Add Method C (Sink Suppression)

- Combine:
  - Adaptive ratios + info-aware merging + sink suppression.
- Run again.
- Expect:
  - Main benchmarks: similar to PruneVid.
  - Fine-grained / long-video: better.

### Step 5: Add Method D (Uptraining)

- Freeze the main model.
- Train a small adapter / last 1–2 layers with:
  - Random pruning ratios.
  - Optional reconstruction loss.
- Run full evaluation again.
- Expect:
  - Robustness curve: your method degrades more slowly.
  - At very low token budgets (10–12%), your method may even match or exceed PruneVid.

### Step 6: Robustness and Efficiency Analysis

- For each variant (A, AB, ABC, ABCD):
  - Run multiple token budgets (10%, 15%, 20%, 25%).
  - Plot accuracy vs. token budget for:
    - MVBench.
    - One fine-grained/long benchmark.
  - Compare TTFT, memory, prefill time.

### Step 7: Final Claim

Based on results, choose the strongest claim:

- “PruneVid+ improves fine-grained/long-video performance at same token budget.”
- “PruneVid+ is more robust to aggressive pruning.”
- “PruneVid+ achieves same accuracy with lower token budget and better efficiency.”

These are all valid, paper-backed improvements over PruneVid.

***
