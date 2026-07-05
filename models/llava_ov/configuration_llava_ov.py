from transformers import LlavaOnevisionConfig


class LlavaOVConfig(LlavaOnevisionConfig):
    model_type = "llava_ov_vtp"

    def __init__(self, **kwargs):
        # PruneVid parameters
        self.selected_layer = kwargs.pop("selected_layer", 10)
        self.alpha = kwargs.pop("alpha", 0.4)
        self.tau = kwargs.pop("tau", 0.8)
        self.cluster_ratio = kwargs.pop("cluster_ratio", 0.5)
        self.temporal_segment_ratio = kwargs.pop("temporal_segment_ratio", 0.25)
        self.use_entropy_adaptive = kwargs.pop("use_entropy_adaptive", False)
        self.use_motion_adaptive = kwargs.pop("use_motion_adaptive", False)
        self.motion_scale = kwargs.pop("motion_scale", 0.5)
        self.motion_invert = kwargs.pop("motion_invert", False)
        self.use_flow_pruning = kwargs.pop("use_flow_pruning", False)
        self.flow_dynamic_ratio = kwargs.pop("flow_dynamic_ratio", 0.5)
        self.use_cluster_pruning = kwargs.pop("use_cluster_pruning", False)
        self.cluster_pruning_topk = kwargs.pop("cluster_pruning_topk", 0.4)
        super().__init__(**kwargs)
