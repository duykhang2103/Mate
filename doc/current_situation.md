# Current Situation

## What Works

- Flow computation on VideoMME works (4x spread between high/low motion videos)
- 183/900 VideoMME videos extracted on disk
- Script written at `scripts/flow_correlation_sanity_check.py`

## What's Broken

### 1. Model Weights: LoRA Not Merged

`MODELS/pllava-7b` contains a **LoRA PEFT checkpoint**, not merged weights.

Safetensors keys look like:
```
language_model.base_model.model.model.layers.X.self_attn.q_proj.base_layer.weight
language_model.base_model.model.model.layers.X.self_attn.q_proj.lora_A.default.weight
language_model.base_model.model.model.layers.X.self_attn.q_proj.lora_B.default.weight
```

Model expects:
```
language_model.model.layers.X.self_attn.q_proj.weight
```

The LoRA merge logic in the script now correctly remaps the prefix (64 pairs merged, 0 unexpected). But **64 keys still missing**: `q_proj.weight` and `v_proj.weight` for all 32 layers — these are exactly the LoRA-adapted layers where the merged weight needs `base + lora_B @ lora_A`. The merge IS happening but the resulting tensor names don't match because the checkpoint has `.base_layer.weight` for LoRA-wrapped layers, and the merge logic loads those as `base_key` but the model expects `language_model.model.layers.X.self_attn.q_proj.weight`.

**Root cause**: For LoRA-wrapped layers (q_proj, v_proj), the checkpoint has `.base_layer.weight` not `.weight`. The merge logic stripped `.base_layer.weight` to get the key without `.weight`, but `load_state_dict` expects `.weight` suffix.

**Fix applied**: Changed `base_key = nk.replace('.base_layer.weight', '')` to `base_key = nk.replace('.base_layer.weight', '.weight')`. Same for lora_A/lora_B keys. This ensures the merged tensor is stored under `language_model.model.layers.X.self_attn.q_proj.weight` which matches what the model expects.

### 2. CUDA Initialization Fails

`conda run -n pllava --no-capture-output bash -c 'python ...'` causes:
```
RuntimeError: random_device could not be read: Invalid argument
```

Direct execution also fails with the same error. The previous evals (which produced MVBench results at 50.71%) must have been run in a different environment or session. Need to figure out the user's working invocation method.

## Next Steps

1. Debug the 64 missing keys — the LoRA merge should produce the right tensors
2. Figure out the correct way to invoke Python with CUDA (the user's working method)
3. Once model loads correctly, run the flow correlation check
