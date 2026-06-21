## 1. Problem Definition & Hypothesis

**The Limitation:** PruneVid hardcodes the token pruning hook at `selected_layer = 10`.
**The Flaw:** Neural networks process information hierarchically. Early layers capture low-level spatial features, while deeper layers handle high-level semantic reasoning. Forcing all videos and queries to prune at exactly layer 10 is structurally rigid. A simple query ("Is it raining?") might resolve its visual attention by layer 6, meaning layers 7-10 are wasting compute. A complex reasoning query ("Why did the car crash?") might not fully align text and video until layer 16; pruning at layer 10 prematurely destroys critical visual evidence.

**The Hypothesis:** By implementing a **Layer-Adaptive** or **Progressive** pruning mechanism driven by the model's internal attention states, we can achieve higher accuracy on complex reasoning tasks and lower latency on simple tasks, establishing a superior Pareto frontier.

---

## 2. Proposed Methodologies

To solve this, you should explore two primary architectural designs.

### Approach A: Progressive Multi-Stage Pruning (Easier to Implement, High Baseline Beater)

Instead of a single massive token drop (e.g., keeping 40% at layer 10), distribute the pruning across multiple network depths.

- **Mechanism:** Prune 20% at layer $L_{early}$ (e.g., 6), another 20% at $L_{mid}$ (e.g., 12), and a final 20% at $L_{late}$ (e.g., 18).
- **Why it works:** It allows the model to slowly discard obvious background tokens early on, while retaining potentially ambiguous tokens deeper into the network where semantic understanding is sharper.
- **System Design Benefit:** This mirrors a scalable funnel architecture, progressively shedding load rather than relying on a single bottleneck gate.

### Approach B: Attention-Entropy Triggered Pruning (Novel, High Conference Appeal)

Allow the model to dynamically decide _when_ to prune based on its own uncertainty.

- **Mechanism:** At each LLM layer, calculate the **Information Entropy ($H$)** of the cross-attention matrix (Text-to-Vision).
  - High Entropy = The text tokens are attending to _all_ visual tokens equally (the model is uncertain/still searching).
  - Low Entropy = The text tokens have strongly focused on specific visual tokens (the model has "found" the answer).
- **The Trigger:** Once the attention entropy drops below a threshold $\tau_{entropy}$, trigger the pruning hook.
- **Why it works:** It is completely self-adaptive. Simple queries trigger early (saving massive FLOPs); complex queries trigger late (preserving accuracy).
