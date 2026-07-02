Here are concrete modifications, ordered from safest/cheapest to most ambitious. All hook into points where PruneVid already has a seam, so you're not rewriting the architecture.

**1. Replace the static/dynamic threshold with true optical-flow magnitude (lowest risk, do this first)**

PruneVid already splits tokens into static vs. dynamic groups using a similarity threshold τ on frame-to-frame token similarity — that's already a cheap motion proxy, just a weak one (similarity in feature space, not pixel space). Swap it for real motion:

- Compute dense optical flow (RAFT, or even classic Farneback for speed) between consecutive frames at low resolution, before the vision encoder.
- Pool flow magnitude into the same patch grid PruneVid uses for tokens.
- Use *flow magnitude* instead of *feature similarity* to decide static vs. dynamic, and to weight retention within the dynamic group.

This directly fixes the conflation problem from before (attention volatility ≠ motion) and it's a drop-in replacement for an existing component rather than a new pipeline. Cheap to implement, cheap to ablate (just compare τ-on-feature-similarity vs. τ-on-flow), and it's a clean, explainable contribution: "PruneVid's static/dynamic split is a similarity heuristic; we replace it with ground-truth motion and show this matters specifically on temporal benchmarks."

**2. Frame-level adaptive window sizing based on motion density**

PruneVid uses fixed-size temporal windows for clustering. In motion-heavy segments, a fixed window conflates several distinct sub-events into one cluster, while in static segments the same window size wastes clustering effort on near-duplicate frames. Make window size inversely proportional to local motion density (computed from the same flow signal in #1): dense-motion stretches get finer-grained windows (less temporal compression, preserving event boundaries), static stretches get coarser windows (more aggressive merging). This targets exactly the "temporally complex" half of your claim — benchmarks like Action Sequence/State Change/Moving Direction fail in current methods often because rapid event boundaries get blurred into one window.

**3. Two-tier retention budget keyed to motion, not one continuous adaptive scalar**

Rather than the continuous `alpha * (0.5 + 0.5*motion_factor)` formula (which I flagged as possibly too narrow), use a simpler, more aggressive two-bucket policy: classify each window as motion or static (binary, from flow magnitude), give motion windows a meaningfully higher fixed retention ratio (e.g., 1.5-2x baseline α) and static windows a meaningfully lower one (e.g., 0.5x), keeping the *global* token budget constant by solving for the two ratios given the static/dynamic ratio across the video. Binary + wide spread is easier to reason about, easier to tune, and more likely to produce a measurable effect than a narrow continuous range — and it's a direct, legible story for a paper: "we don't prune motion and static regions the same way."

**4. Query-conditioned motion weighting (ties into your existing entropy work)**

Not every question needs the motion regions emphasized — "what color is the shirt" doesn't care about motion, "what direction did the person move" does. You already have infrastructure for attention-based query analysis (the entropy work). Use the query embedding (or early-layer text-to-image attention pattern) to gate how strongly the motion-adaptive retention kicks in — if the question's attention pattern is already spatially/object-focused rather than spread across frames, fall back closer to PruneVid's uniform behavior; if it's temporally diffuse, lean harder into motion-weighted retention. This is your most novel and most defensible contribution if it works, because it explains *why* motion-adaptivity sometimes won't help (and your earlier worry — flat aggregate scores — becomes itself a finding: "naive motion-weighting helps on motion subsets but hurts/neutral elsewhere; query-gating recovers the best of both").

**5. Temporal pivot anchoring in the merge step**

When PruneVid merges pruned tokens into kept ones (`TextPivotMerge_LayerWise`), the merge target is chosen by feature similarity within a window, with no explicit temporal-order awareness. For motion-heavy windows, a pruned token from frame *t* could get merged into a kept token from frame *t+3*, losing exactly the directional/sequential information your "temporally complex" claim is about. A targeted fix: in dynamic-classified windows only, bias the pivot selection toward temporally-adjacent kept tokens (not just feature-similar ones) so that motion trajectories aren't smeared across non-adjacent frames during merge. This is a small, surgical change to an existing function, easy to ablate in isolation, and speaks directly to "temporally complex" rather than generic accuracy.

**Suggested combo for your actual paper**: ship #1 (flow-based static/dynamic split) as your foundation since it fixes a real flaw and is cheap, #3 (two-tier budget) as your primary mechanism since it's the most legible "this is why it helps on motion-heavy videos" story, and #4 (query gating) as your secondary contribution that explains the failure mode and turns a potential null result into an interesting one. Skip #2 and #5 for the first paper unless #1+#3+#4 land flat — they're good follow-up-paper material but add more implementation risk than your 3-4 month window probably wants to absorb on a first pass.

One more thing worth doing before you commit: run a 15-minute sanity check — pull flow magnitude on 20-30 MVBench videos split by their motion-heavy vs. static sub-task labels, and just look at whether flow actually separates them the way you'd expect. If it doesn't (e.g., MVBench's "motion-heavy" categories turn out to be more about object/spatial reasoning than visual motion), that tells you immediately whether this whole direction has a shot before you build anything.