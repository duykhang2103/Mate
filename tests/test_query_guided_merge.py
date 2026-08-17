import pytest
from types import SimpleNamespace


torch = pytest.importorskip("torch")

from models.pllava.modeling_pllava import compute_query_guided_static_mask
import models.pllava.modeling_pllava as modeling_pllava


def test_query_guided_mask_preserves_exact_dynamic_budget():
    generator = torch.Generator().manual_seed(7)
    frames = torch.randn(2, 3, 8, 4, generator=generator)
    query = torch.randn(2, 4, generator=generator)

    static_mask, reliability = compute_query_guided_static_mask(
        frames, query, dynamic_ratio=0.25, query_weight=0.7
    )

    assert static_mask.shape == (2, 8)
    assert (~static_mask).sum(dim=-1).tolist() == [2, 2]
    assert reliability.shape == (2, 1)
    assert torch.all((0.0 <= reliability) & (reliability <= 1.0))


def test_query_relevance_selects_aligned_token_when_novelty_is_flat():
    frames = torch.eye(4).view(1, 1, 4, 4)
    query = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    static_mask, reliability = compute_query_guided_static_mask(
        frames, query, dynamic_ratio=0.25, query_weight=1.0
    )

    assert not static_mask[0, 0]
    assert static_mask[0, 1:].all()
    assert reliability.item() > 0.99


def test_semantic_novelty_is_used_when_query_scores_are_flat():
    frames = torch.zeros(1, 2, 4, 2)
    frames[:, :, :, 0] = 1.0
    frames[0, 1, 0] = torch.tensor([0.0, 1.0])
    flat_query = torch.zeros(1, 2)

    static_mask, reliability = compute_query_guided_static_mask(
        frames, flat_query, dynamic_ratio=0.25, query_weight=1.0
    )

    assert not static_mask[0, 0]
    assert static_mask[0, 1:].all()
    assert reliability.item() == pytest.approx(0.0)


@pytest.mark.parametrize("ratio", [0.0, 1.0, -0.1, 1.1])
def test_query_guided_mask_rejects_invalid_dynamic_ratio(ratio):
    frames = torch.randn(1, 2, 4, 3)
    query = torch.randn(1, 3)

    with pytest.raises(ValueError, match="dynamic_ratio"):
        compute_query_guided_static_mask(
            frames, query, dynamic_ratio=ratio, query_weight=0.7
        )


def test_distortion_feature_flag_disabled_preserves_legacy_output(monkeypatch):
    monkeypatch.setattr(
        modeling_pllava,
        "cluster_dpc_knn",
        lambda frames, cluster_num, k: (
            torch.zeros(frames.shape[:2], dtype=torch.long),
            None,
        ),
    )
    monkeypatch.setattr(
        modeling_pllava, "refine_clusters", lambda clusters: clusters
    )
    monkeypatch.setattr(
        modeling_pllava,
        "segment_lengths",
        lambda clusters: torch.tensor([[2]]),
    )
    monkeypatch.setattr(
        modeling_pllava,
        "compute_distortion_controlled_static_masks",
        lambda *args, **kwargs: pytest.fail(
            "disabled distortion router must not be invoked"
        ),
    )

    fake_model = SimpleNamespace(
        config=SimpleNamespace(
            num_frames=2,
            pooling_shape=(2, 1, 4),
            temporal_segment_ratio=0.5,
            cluster_ratio=0.5,
            use_query_guided_merge=False,
        ),
        spatial_merge_tokens=lambda features, num_cluster, k: features,
        _last_flow_mag=None,
    )
    merge = modeling_pllava.PllavaForConditionalGeneration.merge_frames_dynamic
    frames = torch.tensor(
        [[
            [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
            [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0],
        ]]
    )

    implicit_disabled = merge(fake_model, frames, threshold=0.8)
    fake_model.config.use_distortion_routing = False
    explicit_disabled = merge(fake_model, frames, threshold=0.8)

    assert torch.equal(implicit_disabled[0], explicit_disabled[0])
    assert implicit_disabled[1:] == explicit_disabled[1:]
