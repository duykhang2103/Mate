# Quick-Start: Optical Flow Implementation

> **Goal**: Get optical-flow-based static/dynamic split working in < 2 hours
> **Prereq**: Read `doc/phases/phase8_plan.md` Change 1 + 2

---

## Step 1: Add Flow Methods to `modeling_pllava.py` (~30 min)

Add these two methods to `PllavaForConditionalGeneration` class, after `merge_frames_dynamic()`:

```python
def _compute_flow_magnitude(self, frames):
    """
    Compute optical flow magnitude between consecutive frames using Farneback.
    Args:
        frames: [B, W, L, C] — window of feature frames
    Returns:
        flow_mag: [B, W-1, H, W_grid] — flow magnitude per frame pair
    """
    import cv2
    
    B, W, L, C = frames.shape
    # Reshape to: [B*W, C] → need pixel-level frames
    # Actually, we need the original pixel frames, not feature frames
    # The flow should be computed on the input pixel_values, not on features
    # 
    # SOLUTION: Compute flow on pixel_values BEFORE vision encoder
    # Store pixel_values on self during forward pass
    pass

def _pool_flow_to_tokens(self, flow_mag, H=12, W=12):
    """
    Pool dense flow magnitude to match the token grid.
    Args:
        flow_mag: [B, T, h, w] — dense flow magnitude
    Returns:
        pooled: [B, T, H*W] — per-token motion score
    """
    import torch.nn.functional as F
    # Adaptive average pooling
    pooled = F.adaptive_avg_pool2d(
        flow_mag.unsqueeze(0),  # [1, B, T, h, w] → wrong shape
        (H, W)
    )
    return pooled.reshape(flow_mag.shape[0], flow_mag.shape[1], H * W)
```

**IMPORTANT**: The flow needs to be computed on **pixel values**, not feature maps. So we need to:

1. Store `pixel_values` on the model during `forward()`
2. Compute flow on the stored pixel values inside `merge_frames_dynamic()`

### Better approach: Compute flow in forward() before vision encoder

```python
# In forward(), before calling vision tower:
if self.config.use_flow_pruning:
    # Compute flow on pixel values
    # pixel_values: [B*T, C, H, W] — all frames stacked
    flow_mag = self._compute_flow_from_pixels(pixel_values, num_frames)
    self._last_flow_mag = flow_mag  # store for merge_frames_dynamic

# Then in merge_frames_dynamic(), use self._last_flow_mag instead of feature similarity
```

---

## Step 2: Modify `merge_frames_dynamic()` (~30 min)

Replace the feature-similarity threshold with flow-based classification:

```python
def merge_frames_dynamic(self, frames, threshold=0.8, k=7):
    B, L, C = frames.shape
    frames = frames.view(B, self.config.num_frames, self.config.pooling_shape[1]*self.config.pooling_shape[2], C)
    
    # DPC-KNN windowing (unchanged)
    idx_clusters, _ = cluster_dpc_knn(frames.mean(dim=2), cluster_num=int(self.config.num_frames*self.config.temporal_segment_ratio), k=k)
    idx_clusters = refine_clusters(idx_clusters)
    window_list = segment_lengths(idx_clusters)
    
    L = self.config.pooling_shape[1]*self.config.pooling_shape[2]
    
    # NEW: Use flow-based motion scores for static/dynamic classification
    flow_mag = getattr(self, '_last_flow_mag', None)  # [B, T, H, W]
    
    static_features = []
    dynamic_features = []
    static_sizes = []
    dynamic_sizes = []
    
    start_idx = 0
    for window_size in window_list[0]:
        current_frames = frames[:, start_idx:start_idx+window_size, :, :]
        
        if flow_mag is not None:
            # Flow-based classification
            window_flow = flow_mag[:, start_idx:start_idx+window_size, :, :]  # [B, W, H, W_grid]
            avg_flow = window_flow.mean(dim=1)  # [B, H, W_grid]
            
            # Pool to token grid
            avg_flow_tokens = F.adaptive_avg_pool2d(
                avg_flow.unsqueeze(1), 
                (self.config.pooling_shape[1], self.config.pooling_shape[2])
            ).flatten(1)  # [B, 144]
            
            # Top cluster_ratio% by flow magnitude are dynamic
            k_dynamic = max(1, int(L * self.config.cluster_ratio))
            _, topk_idx = avg_flow_tokens.topk(k_dynamic, dim=-1)
            mask = torch.zeros(B, L, dtype=torch.bool, device=frames.device)
            mask.scatter_(1, topk_idx, True)
            mask_expand = mask.view(B, 1, L, 1).expand(-1, window_size, -1, C)
        else:
            # Fallback: feature similarity (original behavior)
            frames_normed = F.normalize(current_frames, p=2, dim=-1)
            frames_sim = einsum('b w l c, b t l c -> b w t l', frames_normed, frames_normed)
            frames_sim = (frames_sim.sum(dim=-2) - 1).sum(dim=-2) / (window_size*(window_size-1))
            mask = frames_sim > threshold
            mask_expand = mask.view(B, 1, L, 1).expand(-1, window_size, -1, C)
        
        # Rest is unchanged: static = mask, dynamic = ~mask
        static_mask = mask_expand
        static_feat = torch.masked_select(current_frames, static_mask).view(B, window_size, -1, C).mean(dim=1)
        if static_feat.shape[1] > 14:
            static_feat = self.spatial_merge_tokens(static_feat, num_cluster=int(static_feat.shape[1]*self.config.cluster_ratio), k=7)
        static_features.append(static_feat)
        static_sizes.append(static_feat.shape[1])
        
        dynamic_mask = ~mask_expand
        dynamic_feat = torch.masked_select(current_frames, dynamic_mask).view(B, window_size, -1, C)
        dynamic_window_list = []
        for i in range(window_size):
            dynamic_feat_window = dynamic_feat[:,i,:,:]
            if dynamic_feat_window.shape[1] > 14:
                dynamic_feat_window = self.spatial_merge_tokens(dynamic_feat_window, num_cluster=int(dynamic_feat_window.shape[1]*self.config.cluster_ratio), k=7)
            dynamic_window_list.append(dynamic_feat_window)
        dynamic_feat = torch.cat(dynamic_window_list, dim=1)
        dynamic_features.append(dynamic_feat)
        dynamic_sizes.append(dynamic_feat.shape[1])
        
        start_idx += window_size
    
    # Combine
    final_features = []
    for static_feature, dynamic_feature in zip(static_features, dynamic_features):
        final_features.append(static_feature)
        final_features.append(dynamic_feature)
    final_features = torch.cat(final_features, dim=1)
    window_sizes = window_list[0].tolist()
    
    return final_features, static_sizes, dynamic_sizes, window_sizes
```

---

## Step 3: Add Flow Computation from Pixels (~30 min)

Add this method to compute flow from raw pixel values:

```python
def _compute_flow_from_pixels(self, pixel_values, num_frames):
    """
    Compute optical flow magnitude from raw pixel values.
    Args:
        pixel_values: [B*T, C, H, W] — all frames stacked
        num_frames: int — number of frames
    Returns:
        flow_mag: [B, num_frames-1, H_grid, W_grid] — pooled flow magnitude
    """
    import cv2
    
    B = pixel_values.shape[0] // num_frames
    # pixel_values is normalized; denormalize for flow computation
    # Mean and std from PLLaVA training:
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(pixel_values.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(pixel_values.device)
    
    frames = pixel_values.view(B, num_frames, *pixel_values.shape[1:])
    frames = frames * std.view(1, 1, 3, 1, 1) + mean.view(1, 1, 3, 1, 1)
    frames = (frames * 255).clamp(0, 255).to(torch.uint8)
    
    flow_mags = []
    for b in range(B):
        pair_mags = []
        for t in range(num_frames - 1):
            # Convert to grayscale
            f1 = frames[b, t].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
            f2 = frames[b, t+1].permute(1, 2, 0).cpu().numpy()
            g1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
            g2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)
            
            # Farneback flow
            flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            pair_mags.append(torch.from_numpy(mag).float())
        
        flow_mags.append(torch.stack(pair_mags))  # [T-1, H, W]
    
    return torch.stack(flow_mags)  # [B, T-1, H, W]
```

---

## Step 4: Store Pixel Values in Forward Pass (~5 min)

In `PllavaForConditionalGeneration.forward()`, store pixel_values before the vision encoder:

```python
def forward(self, input_ids=None, pixel_values=None, ...):
    # Store for flow computation
    if self.config.use_flow_pruning and pixel_values is not None:
        self._stored_pixel_values = pixel_values
    
    # ... existing code ...
    image_outputs = self.vision_tower(pixel_values, ...)
    # ... existing code ...
```

---

## Step 5: Smoke Test (~15 min)

```bash
conda run -n pllava python scripts/infer_single_video.py \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --use_lora --weight_dir MODELS/pllava-7b \
    --num_frames 16 --alpha 0.4 --selected_layer 10 \
    --use_motion_adaptive --motion_scale 1.0 \
    --video DATAS/Video-MME/25Pt1AZO9EM.mp4 \
    --skip_generation
```

**Check**: Motion scores should now be non-zero and vary across windows.

---

## Step 6: 5-Task Eval (~4-6 hrs)

```bash
python -m tasks.eval.mvbench.pllava_eval_mvbench \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path test_results/mvbench_flow_motion_scale_1.0 \
    --num_frames 16 --use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b \
    --pooling_shape 16-12-12 --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --use_motion_adaptive --motion_scale 1.0 \
    --tasks "Action Sequence,Action Prediction,Moving Direction,Object Interaction,Unexpected Action"
```

Compare with baseline (50.71%).
