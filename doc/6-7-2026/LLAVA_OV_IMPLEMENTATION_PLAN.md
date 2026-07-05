# LLaVA-OneVision Implementation Plan

> **Date**: 2026-07-06
> **Goal**: Implement VTP on LLaVA-OneVision to exploit non-uniform attention
> **Status**: Baseline working, VTP pending

---

## Progress

### ✅ Completed

1. **Model Download**: `llava-hf/llava-onevision-qwen2-7b-ov-hf` → `MODELS/llava-onevision-7b`
2. **Module Created**: `models/llava_ov/` with config, modeling, init
3. **Model Loader**: `load_llava_ov()` in `tasks/eval/model_utils.py`
4. **Answer Function**: `ov_answer()` using native `apply_chat_template`
5. **Eval Script**: `tasks/eval/mvbench/ov_eval_mvbench.py`
6. **Smoke Test**: Passed (5 samples)
7. **Partial Eval**: 20 samples/task → 52.0% average

### 🔲 Pending

1. Full MVBench evaluation (all samples)
2. VTP wrapper for Qwen2 (`qwen2_vtp.py`)
3. Token merge + pruning (`elastic_cache_ov.py`)
4. Hook into LLaVA-OV forward pass
5. VideoMME evaluation

---

## Background

### Why LLaVA-OneVision?

PLLaVA's attention is uniformly distributed (entropy=0.9999), making ALL token selection methods fail. LLaVA-OneVision has:
- Non-uniform attention that can actually benefit from token selection
- PruneVid paper already reports results: 58.0 MVBench, 58.2 VideoMME
- Different architecture (Qwen2 instead of LLaMA) requiring new VTP wrapper

### Architecture Comparison

| Feature | PLLaVA | LLaVA-OneVision |
|---------|--------|-----------------|
| Vision Encoder | SigLIP | SigLIP |
| LLM Backbone | LLaMA-2-7B | Qwen2-7B |
| Attention | Uniform (entropy=0.9999) | Non-uniform |
| Token Merge | feature-similarity | MLP projection |
| Hidden Size | 4096 | 4096 |
| Layers | 32 | 32 |
| Attention Heads | 32 | 32 |

---

## Implementation Steps

### Phase 1: Model Loading (1-2 hours)

#### Step 1.1: Add LLaVA-OV Config

**File**: `models/pllava/configuration_pllava.py`

Add `LlavaOnevisionConfig` class:
```python
class LlavaOnevisionConfig(PretrainedConfig):
    model_type = "llava_onevision"
    
    def __init__(
        self,
        vision_tower="openai/siglip-so400m-patch14-384",
        vision_feature_layer=-2,
        vision_feature_select_strategy="full",
        image_token_index=151646,
        video_token_index=151647,
        mm_vision_select_layer=-2,
        mm_use_im_start_end=False,
        mm_use_im_patch_token=False,
        image_grid_pinpoints="(1,1)",
        image_newline=0,
        **kwargs,
    ):
```

#### Step 1.2: Add Model Loading Function

**File**: `tasks/eval/model_utils.py`

Add `load_llava_ov()` function:
```python
def load_llava_ov(
    pretrained_model_name_or_path,
    lora_checkpoint_path=None,
    lora_alpha=14,
    weight_dir=None,
    bf16=True,
    device_map="auto",
    **kwargs
):
    from transformers import LlavaOnevisionForConditionalGeneration
    
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        device_map=device_map,
        attn_implementation="eager",  # Required for attention weights
        **kwargs
    )
    
    if lora_checkpoint_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model,
            lora_checkpoint_path,
            adapter_name="prunevid",
        )
        model.merge_and_unload()
    
    return model
```

#### Step 1.3: Download LLaVA-OneVision Weights

```bash
# Download from HuggingFace
huggingface-cli download llava-hf/llava-onevision-qwen2-7b-ov-siglip-384 \
    --local-dir MODELS/llava-onevision-7b

# Or use existing PLLaVA LoRA weights if compatible
```

---

### Phase 2: VTP Wrapper for Qwen2 (2-3 hours)

#### Step 2.1: Create Qwen2 VTP Wrapper

**File**: `models/llava_ov/qwen2_vtp.py` (new file)

```python
class Qwen2VTPWrapper(nn.Module):
    """
    VTP wrapper for Qwen2 model in LLaVA-OneVision.
    Wraps Qwen2Model (without lm_head) to capture attention weights.
    """
    
    def __init__(self, qwen2_model, config):
        super().__init__()
        self.model = qwen2_model
        self.config = config
        self.selected_layer = getattr(config, 'selected_layer', 10)
        
    def forward(
        self,
        inputs_embeds,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=True,
        output_hidden_states=False,
        return_dict=False,
    ):
        # Forward through Qwen2
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=True,  # Always get attention weights at selected layer
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        return outputs
```

#### Step 2.2: Extract Image Tokens from Inputs

**File**: `models/llava_ov/utils.py` (new file)

```python
def extract_image_tokens(inputs_embeds, input_ids, image_token_index=151646):
    """
    Extract visual token embeddings from inputs_embeds.
    
    In LLaVA-OneVision, image tokens replace <image> placeholders via masked_scatter.
    After processing, image tokens are contiguous at the positions of <image> in input_ids.
    
    Returns:
        image_embeds: [batch, num_image_tokens, hidden_size]
        image_mask: [batch, seq_len] (True at image token positions)
    """
    image_mask = (input_ids == image_token_index)
    return inputs_embeds[:, image_mask, :], image_mask
```

---

### Phase 3: Adapt Vision Token Merge (1-2 hours)

#### Step 3.1: Adapt Feature-Similarity Merge

**File**: `models/llava_ov/elastic_cache_ov.py` (new file)

```python
class ElasticCacheOV:
    """
    Vision token merge + VTP for LLaVA-OneVision.
    
    Adapted from PLLaVA's ElasticCache to work with LLaVA-OV's architecture:
    - Vision merge happens in LlavaOnevisionModel (outer layer)
    - VTP wraps Qwen2Model (not LlamaModel)
    """
    
    def __init__(
        self,
        selected_layer=10,
        alpha=0.4,
        tau=0.8,
        cluster_ratio=0.5,
        use_cluster_pruning=False,
        temporal_segment_ratio=0.25,
    ):
        self.selected_layer = selected_layer
        self.alpha = alpha
        self.tau = tau
        self.cluster_ratio = cluster_ratio
        self.use_cluster_pruning = use_cluster_pruning
        self.temporal_segment_ratio = temporal_segment_ratio
        
        self.all_img_tokens = []
        self.step = 0
        self.num_img_tokens = 0
        
    def merge_tokens(self, image_embeds, num_frames, spatial_pool=2):
        """
        Merge visual tokens across frames using feature-similarity.
        
        LLaVA-OneVision layout: [batch, num_frames * spatial_pool, hidden_size]
        After merge: [batch, num_merged_tokens, hidden_size]
        """
        # Implement feature-similarity merge for LLaVA-OV
        # Similar to PLLaVA but adapted for different token layout
        pass
    
    def prune_tokens(self, hidden_states, image_mask):
        """
        Prune tokens at selected layer using attention or clustering.
        """
        if self.use_cluster_pruning:
            return self._cluster_prune(hidden_states, image_mask)
        else:
            return self._attention_prune(hidden_states, image_mask)
```

#### Step 3.2: Implement Token Merge Logic

The merge logic needs to handle LLaVA-OV's video token layout:
- Input: `[batch, num_frames * spatial_pool, hidden_size]`
- Merge: Feature-similarity across frames
- Output: `[batch, num_merged_tokens, hidden_size]`

---

### Phase 4: Integrate into LLaVA-OV Forward Pass (2-3 hours)

#### Step 4.1: Modify LlavaOnevisionModel

**File**: `models/llava_ov/modeling_llava_ov.py` (new file)

```python
class LlavaOnevisionModelOV(LlavaOnevisionModel):
    """
    Modified LlavaOnevisionModel with VTP support.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.elastic_cache = ElasticCacheOV(
            selected_layer=config.selected_layer,
            alpha=config.alpha,
            tau=config.tau,
        )
        
    def forward(self, ...):
        # 1. Get image features
        image_features = self.get_image_features(pixel_values, image_sizes)
        
        # 2. Merge tokens
        image_features = self.elastic_cache.merge_tokens(
            image_features, 
            num_frames=self.config.num_frames,
            spatial_pool=self.config.spatial_pool,
        )
        
        # 3. Replace <image> placeholders
        special_image_mask = (input_ids == self.config.image_token_index)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
        
        # 4. Forward through language model with VTP
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=True,  # Required for VTP
            output_hidden_states=True,
            return_dict=True,
        )
        
        # 5. Prune at selected layer
        if self.elastic_cache.step == self.elastic_cache.selected_layer:
            hidden_states = outputs.hidden_states[-1]
            pruned_hidden = self.elastic_cache.prune_tokens(
                hidden_states, 
                special_image_mask
            )
            # Replace hidden states with pruned version
            outputs = outputs._replace(hidden_states=pruned_hidden)
        
        return outputs
```

#### Step 4.2: Update Forward Pass

Modify the forward pass to:
1. Extract image tokens from inputs_embeds
2. Merge tokens using feature-similarity
3. Forward through Qwen2 with VTP
4. Prune at selected layer

---

### Phase 5: Evaluation Setup (1-2 hours)

#### Step 5.1: Update Eval Scripts

**File**: `tasks/eval/mvbench/ov_eval_mvbench.py` (new file)

```python
# Add CLI args for LLaVA-OneVision
parser.add_argument("--use_llava_ov", action="store_true")
parser.add_argument("--llava_ov_model_path", type=str, default="MODELS/llava-onevision-7b")
```

#### Step 5.2: Smoke Test

```bash
# Smoke test with 5 samples
python -m tasks.eval.mvbench.ov_eval_mvbench \
    --model_path MODELS/llava-onevision-7b \
    --num_frames 16 \
    --alpha 0.4 \
    --tau 0.8 \
    --tasks "Moving Direction" \
    --max_samples 5 \
    > log_ov_smoke.log 2>&1
```

#### Step 5.3: Full Evaluation

```bash
# Full evaluation on MVBench
python -m tasks.eval.mvbench.ov_eval_mvbench \
    --model_path MODELS/llava-onevision-7b \
    --save_path test_results/mvbench_ov \
    --num_frames 16 \
    --alpha 0.4 \
    --tau 0.8 \
    --tasks "Action Sequence,Action Prediction,Moving Direction,Object Interaction,Unexpected Action" \
    > log_ov_full.log 2>&1
```

---

## File Structure

```
models/llava_ov/
├── __init__.py
├── configuration_ov.py          # LlavaOnevisionConfig
├── modeling_ov.py               # Modified LlavaOnevisionModel
├── qwen2_vtp.py                 # VTP wrapper for Qwen2
├── elastic_cache_ov.py          # Token merge + pruning
└── utils.py                     # Helper functions

tasks/eval/mvbench/
└── ov_eval_mvbench.py           # MVBench eval for LLaVA-OV

tasks/eval/videomme/
└── ov_eval_videomme.py          # VideoMME eval for LLaVA-OV
```

---

## Key Differences from PLLaVA

| Aspect | PLLaVA | LLaVA-OneVision |
|--------|--------|-----------------|
| LLM Wrapper | `LlamaModelVTP` | `Qwen2VTPWrapper` |
| Token Merge | Inside LLM | Outside (in LlavaOnevisionModel) |
| Image Token ID | Dynamic | Fixed (151646) |
| Video Token ID | N/A | Fixed (151647) |
| Output Format | `CausalLMOutputWithPast` | `LlavaOnevisionCausalLMOutputWithPast` |
| Attention | Uniform | Non-uniform |

---

## Expected Results

### Based on PruneVid Paper

| Metric | Baseline | PruneVid (17% retention) |
|--------|----------|--------------------------|
| MVBench | 58.0 | 57.5 (-0.5) |
| VideoMME | 58.2 | 58.6 (+0.4) |

### Our Targets

| Metric | Target | Notes |
|--------|--------|-------|
| MVBench | ≥ 58.0 | Match or exceed PruneVid |
| VideoMME | ≥ 58.5 | Outperform PruneVid |
| Moving Direction | > 25% | Exploit non-uniform attention |
| Object Interaction | > 70% | Fine-grained understanding |
| Token Retention | ≤ 17% | Match PruneVid efficiency |
| FLOPs Reduction | ≥ 30% | Significant speedup |

---

## Success Criteria

### Must-Have

- [ ] LLaVA-OneVision loads without errors
- [ ] VTP wrapper captures attention weights from Qwen2
- [ ] Token merge works with LLaVA-OV's token layout
- [ ] Evaluation runs end-to-end on MVBench
- [ ] Results match or exceed PruneVid paper numbers

### Nice-to-Have

- [ ] Improvement on Moving Direction (> 25%)
- [ ] Improvement on long video subset
- [ ] Ablation: attention-based vs clustering selection
- [ ] Efficiency curves (accuracy vs FLOPs)

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLaVA-OV weights unavailable | High | Use HuggingFace hub or existing PLLaVA weights |
| Qwen2 VTP wrapper incompatible | High | Test with small model first |
| Token merge layout differs | Medium | Verify with debugging prints |
| Evaluation scripts need updates | Medium | Start with smoke test |

---

## Timeline

| Phase | Task | Hours |
|-------|------|-------|
| 1 | Model Loading | 1-2 |
| 2 | VTP Wrapper | 2-3 |
| 3 | Token Merge | 1-2 |
| 4 | Integration | 2-3 |
| 5 | Evaluation | 1-2 |
| **Total** | | **7-12 hours** |

---

## Next Steps

1. **Download LLaVA-OneVision weights**
2. **Implement `load_llava_ov()` function**
3. **Create Qwen2 VTP wrapper**
4. **Adapt token merge for LLaVA-OV layout**
5. **Run smoke test**
6. **Run full evaluation**
7. **Compare against PruneVid paper numbers**

---

## References

- [PruneVid Paper](https://arxiv.org/abs/2502.07972): Visual Token Pruning for Video LLMs
- [LLaVA-OneVision](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-siglip-384): Model weights
- [Qwen2 Architecture](https://huggingface.co/Qwen/Qwen2-7B): Qwen2-7B details
