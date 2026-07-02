
## 1. Adaptive / per-frame pruning ratios (motion or entropy based)

**What to do:**  
Instead of a fixed pruning ratio for all frames:
- Compute a motion score (frame difference, optical flow, or token variance) per frame.
- Use higher pruning ratio for low-motion frames, lower for high-motion frames, while keeping the *average* token budget similar to PruneVid’s.

**Why you should try it:**  
- **FastVID** (2025) shows that **dynamic density pruning** based on token importance and motion-aware metrics can significantly improve accuracy under similar retention ratios compared to PruneVid. [papers.nips](https://papers.nips.cc/paper_files/paper/2025/file/b2e63e36c57e153b9015fece2352a9f9-Paper-Conference.pdf)
- PruneVid’s limitations section notes that “achieving high accuracy with drastically reduced token budgets remains a fundamental challenge”; adaptive ratios help preserve more tokens when needed, addressing this directly. [studocu](https://www.studocu.com/row/document/makerere-university/computer-science/prunevid-enhancing-video-understanding-efficiency-through-token-pruning/146016029)

***

## 2. Information-aware or entropy-aware merging (instead of plain average)

**What to do:**  
Replace PruneVid’s simple average merging of static tokens with:
- Weighted merging based on token variance or entropy.
- Or use clustering that explicitly preserves high-information tokens.

**Why you should try it:**  
- **Information-Aware Visual Token Pruning** (2025) explicitly shows that pruning/merging guided by **information content** (not just similarity) improves accuracy under aggressive pruning and especially helps on fine-grained tasks. [arxiv](https://arxiv.org/html/2511.08003v1)
- PruneVid’s merging is purely similarity-based; adding information-awareness can reduce loss of important but redundant tokens, improving accuracy at the same token budget.

***

## 3. Importance re-scoring after PruneVid’s main stages

**What to do:**  
After PruneVid does:
- Temporal merge → spatial merge → LLM-guided selection,
add a **re-scoring step**:
- Pass merged tokens through a tiny MLP to produce an importance score.
- Combine this with PruneVid’s original selection scores to decide which tokens to keep.

**Why you should try it:**  
- **SharpV** (2025) uses **adaptive, information-aware thresholds** and re-scoring to improve token selection, achieving higher accuracy at very low token budgets (e.g., 12%) than other methods. [arxiv](https://arxiv.org/html/2511.08003v1)
- PruneVid’s selection only uses cross-attention at one layer; adding a re-scoring module can capture additional importance signals and improve robustness.

***

## 4. Sink-token or low-quality token suppression

**What to do:**  
Define a “sink score” for each token (high attention but low semantic content):
- Penalize these tokens in the pruning score: `final_score = original_score − λ · sink_score`.
- Optionally enforce a minimum non-sink token count.

**Why you should try it:**  
- **SToP (Sink-Token-Aware Pruning)** (2026) explicitly shows that adding a sink penalty reduces hallucination and improves fine-grained performance, often with **lower performance drop than PruneVid** at aggressive pruning (10–15%) on fine-grained tasks. [arxiv](https://arxiv.org/html/2604.20937v1)
- PruneVid does not explicitly handle sink tokens; adding this can beat PruneVid on fine-grained metrics even if main benchmarks are similar.

***

## 5. Temporal attention / weighted temporal fusion instead of simple averaging

**What to do:**  
Replace PruneVid’s plain average temporal merge with:
- A small temporal attention module that weights frames by motion or question relevance.
- Merge tokens using attention-weighted fusion instead of mean.

**Why you should try it:**  
- **FastVID** uses **dynamic density pruning** that considers temporal dynamics and token density, showing large gains over PruneVid under comparable retention ratios. [papers.nips](https://papers.nips.cc/paper_files/paper/2025/file/b2e63e36c57e153b9015fece2352a9f9-Paper-Conference.pdf)
- Simple averaging can dilute important changes; attention-weighted fusion preserves more critical temporal information, improving temporal reasoning benchmarks.

***

## 6. Motion-guided temporal pruning and segment-wise adaptive pruning

**What to do:**  
- Use motion maps to decide which frames to keep more tokens.
- Apply different pruning strategies per scene segment (e.g., intro, action, ending).

**Why you should try it:**  
- **FastVID** and **DivPrune** (2025) both emphasize **motion-aware and diversity-aware** pruning to better preserve dynamic content, improving performance on action-heavy and long-video benchmarks. [cvpr.thecvf](https://cvpr.thecvf.com/virtual/2025/poster/34849)
- PruneVid’s temporal merging is static-region focused; motion-guided strategies can better handle dynamic scenes and improve long-video performance.

***

## 7. Multi-scale / region-aware spatial merging

**What to do:**  
- Merge tokens at multiple spatial scales (coarse + fine).
- Detect object regions (via a small detector or segmentation hint) and preserve more tokens in those regions.

**Why you should try it:**  
- **DivPrune** (CVPR 2025) shows that **diversity-based pruning** that preserves diverse tokens across regions improves fine-grained understanding and semantic coverage. [cvpr.thecvf](https://cvpr.thecvf.com/virtual/2025/poster/34849)
- PruneVid’s spatial merging is purely similarity-based; region-aware merging can preserve object-level details better, improving object-level QA benchmarks.

***

## 8. Multi-layer cross-attention for token selection

**What to do:**  
Instead of using cross-attention only at layer `M`:
- Use cross-attention from multiple layers (e.g., `M-1`, `M`, `M+1`).
- Combine scores to get a more robust importance measure.

**Why you should try it:**  
- **SharpV** uses **self-calibrated, adaptive thresholds** and leverages information from multiple layers to improve token selection, achieving better accuracy at low token budgets. [arxiv](https://arxiv.org/html/2511.08003v1)
- PruneVid’s single-layer selection can be noisy; multi-layer selection can stabilize token importance and reduce pruning errors.

***

## 9. Weighted KV merging instead of simple drop

**What to do:**  
Instead of deleting pruned tokens in KV cache:
- Merge their K/V with kept tokens using weighted average.
- Optionally add a small correction term to preserve attention flow.

**Why you should try it:**  
- **SToP** shows that refined KV handling (not just dropping) reduces performance drop on fine-grained tasks under aggressive pruning, often outperforming PruneVid in those metrics. [arxiv](https://arxiv.org/html/2604.20937v1)
- Simple drop discards information from pruned tokens; weighted merging preserves some of it, improving accuracy.

***

## 10. Uptraining / robustness training with random pruning rates

**What to do:**  
Add a small training phase:
- Randomly vary pruning ratio per sample during a few epochs.
- Optionally add a loss that encourages pruned hidden states to approximate full hidden states.

**Why you should try it:**  
- **EVS (Efficient Video Sampling)** and **SharpV** both show that **training with random pruning rates** makes models robust to aggressive pruning, often maintaining near-baseline accuracy at very low token budgets, outperforming training-free methods like PruneVid. [arxiv](https://arxiv.org/abs/2510.14624)
- PruneVid is training-free; adding a small uptraining step can significantly improve robustness and accuracy under extreme pruning.

***

## 11. Multi-objective optimization of pruning (accuracy + efficiency)

**What to do:**  
Define a combined objective:
- e.g., `L = −accuracy + λ₁ · token_budget + λ₂ · TTFT + λ₃ · memory`.
- Tune thresholds to minimize this.

**Why you should try it:**  
- **Efficiency-focused methods** (e.g., FastVID, SharpV) explicitly optimize for both accuracy and efficiency (TTFT, FLOPs), showing that you can beat PruneVid on efficiency metrics even at similar accuracy. [papers.nips](https://papers.nips.cc/paper_files/paper/2025/file/b2e63e36c57e153b9015fece2352a9f9-Paper-Conference.pdf)
- PruneVid mainly focuses on accuracy vs. FLOPs; adding multi-objective optimization can yield better TTFT/memory at similar accuracy.

***

## 12. Task-specific or benchmark-specific tuning

**What to do:**  
- Tune pruning ratios, thresholds, and layer choices specifically for MVBench or VideoMME.
- For temporal QA: keep more tokens around event boundaries.
- For object QA: keep more tokens in object regions.

**Why you should try it:**  
- Many pruning papers (e.g., **FastVID**, **SharpV**) report **benchmark-specific gains**, showing that with careful tuning you can exceed PruneVid on specific benchmarks even if generic performance is similar. [papers.nips](https://papers.nips.cc/paper_files/paper/2025/file/b2e63e36c57e153b9015fece2352a9f9-Paper-Conference.pdf)
- PruneVid uses fixed settings; task-specific tuning can exploit structure in your target benchmark and improve performance there.

***

## 13. Hybrid methods: combine PruneVid with ideas from FastVID, SharpV, SToP, etc.

**What to do:**  
- **PruneVid + adaptive thresholds (like SharpV)**.
- **PruneVid + sink suppression (like SToP)**.
- **PruneVid + uptraining (like EVS/SharpV)**.

**Why you should try it:**  
- **FastVID** directly claims **3.5× improvement** under comparable retention ratios compared to PruneVid, using dynamic density and motion-aware pruning. [papers.nips](https://papers.nips.cc/paper_files/paper/2025/file/b2e63e36c57e153b9015fece2352a9f9-Paper-Conference.pdf)
- **SharpV** shows that adaptive thresholds and information-aware merging can outperform PruneVid at low token budgets on PLLaVA and LLaVA-OneVision. [arxiv](https://arxiv.org/html/2511.08003v1)
- **SToP** shows that combining with sink-aware pruning can beat PruneVid on fine-grained tasks. [arxiv](https://arxiv.org/html/2604.20937v1)
- By taking PruneVid as base and adding these ideas, you can claim “PruneVid+X” and empirically show it outperforms plain PruneVid.

***

## 14. Evaluation strategy to explicitly show you beat PruneVid

**What to do:**  
- Focus on a clear metric: fine-grained benchmarks, long-video subsets, or robustness curves.
- Show accuracy vs. token budget curves for your method and PruneVid.
- Measure TTFT, prefill time, and memory; show you have same accuracy but lower efficiency cost.

**Why you should try it:**  
- Recent pruning papers (e.g., **SharpV**, **FastVID**, **SToP**) all emphasize **robustness curves** and **efficiency metrics** as key evidence of beating prior methods. [arxiv](https://arxiv.org/html/2604.20937v1)
- Even if overall accuracy is similar, showing better robustness or efficiency is a valid and strong way to claim improvement over PruneVid.
