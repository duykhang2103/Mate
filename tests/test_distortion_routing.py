import unittest


try:
    import torch
except ImportError:  # pragma: no cover - exercised on lightweight CI images
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DistortionRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from models.pllava.distortion_routing import (
            apply_learned_safety_router,
            compute_distortion_controlled_static_masks,
            estimate_post_cluster_tokens,
        )

        cls.route = staticmethod(compute_distortion_controlled_static_masks)
        cls.estimate_tokens = staticmethod(estimate_post_cluster_tokens)
        cls.apply_router = staticmethod(apply_learned_safety_router)

    @staticmethod
    def _constant_window(batch_size=1):
        window = torch.zeros(batch_size, 2, 4, 2)
        window[..., 0] = 1.0
        return window

    @staticmethod
    def _changing_window(batch_size=1):
        first = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]
        )
        second = torch.tensor(
            [[0.0, 1.0], [0.0, -1.0], [0.0, 1.0], [0.0, -1.0]]
        )
        return torch.stack([first, second]).unsqueeze(0).repeat(
            batch_size, 1, 1, 1
        )

    def test_global_allocation_favors_high_distortion_window(self):
        masks, diagnostics = self.route(
            [self._constant_window(), self._changing_window()],
            epsilon=0.1,
        )

        dynamic_counts = [int((~mask).sum()) for mask in masks]
        self.assertEqual(dynamic_counts[0], 1)
        self.assertEqual(dynamic_counts[1], 3)
        self.assertEqual(int(diagnostics["dynamic_tracks"].item()), 4)

    def test_higher_threshold_never_increases_dynamic_tracks(self):
        windows = [self._constant_window(), self._changing_window()]
        counts = []
        for epsilon in (0.0, 0.1, 1.0):
            masks, _ = self.route(windows, epsilon=epsilon)
            counts.append(sum(int((~mask).sum()) for mask in masks))

        self.assertGreaterEqual(counts[0], counts[1])
        self.assertGreaterEqual(counts[1], counts[2])

    def test_every_window_preserves_static_and_dynamic_coverage(self):
        masks, _ = self.route(
            [self._constant_window(), self._changing_window()],
            epsilon=0.0,
        )

        for mask in masks:
            self.assertTrue(mask.any(dim=-1).all())
            self.assertTrue((~mask).any(dim=-1).all())

    def test_batch_masks_and_diagnostics_have_expected_shapes(self):
        masks, diagnostics = self.route(
            [self._constant_window(2), self._changing_window(2)],
            epsilon=0.1,
        )

        self.assertEqual([tuple(mask.shape) for mask in masks], [(2, 4), (2, 4)])
        self.assertEqual(tuple(diagnostics["dynamic_tracks"].shape), (2,))
        self.assertTrue(torch.equal(masks[0][0], masks[0][1]))
        self.assertTrue(torch.equal(masks[1][0], masks[1][1]))

    def test_dynamic_token_count_remains_divisible_by_window_size(self):
        windows = [self._constant_window(), self._changing_window()]
        masks, _ = self.route(windows, epsilon=0.1)

        for window, static_mask in zip(windows, masks):
            dynamic_tokens = int((~static_mask).sum()) * window.shape[1]
            self.assertEqual(dynamic_tokens % window.shape[1], 0)

    def test_post_cluster_token_estimate_matches_metadata_contract(self):
        windows = [self._constant_window(), self._changing_window()]
        masks, _ = self.route(windows, epsilon=0.1)

        estimated = self.estimate_tokens(masks, windows, cluster_ratio=0.5)

        # Four spatial tracks, with 1 and 3 dynamic tracks in 2-frame windows:
        # (3 static + 1*2 dynamic) + (1 static + 3*2 dynamic) = 12.
        self.assertEqual(int(estimated.item()), 12)

    def test_learned_router_can_restore_baseline_masks_per_sample(self):
        adaptive_masks = [torch.tensor([[True, True, False, False]])]
        baseline_masks = [torch.tensor([[True, False, True, False]])]
        diagnostics = {"distortion_mean": torch.tensor([1.0])}
        router = {
            "feature_fields": [
                "routing_distortion_mean",
                "merged_vision_tokens",
            ],
            "weights": [10.0, 0.0],
            "means": [0.0, 0.0],
            "scales": [1.0, 1.0],
            "bias": 0.0,
            "threshold": 0.5,
        }

        selected, risks, used_baseline = self.apply_router(
            adaptive_masks,
            baseline_masks,
            diagnostics,
            predicted_merged_tokens=torch.tensor([6]),
            router=router,
        )

        self.assertTrue(torch.equal(selected[0], baseline_masks[0]))
        self.assertGreater(float(risks.item()), 0.99)
        self.assertTrue(bool(used_baseline.item()))

    def test_query_relevance_can_raise_only_aligned_trajectories(self):
        window = self._changing_window()
        query = torch.tensor([[1.0, 0.0]])
        plain_masks, _ = self.route([window], epsilon=0.4)
        query_masks, diagnostics = self.route(
            [window],
            epsilon=0.4,
            query_embedding=query,
            query_weight=1.0,
        )

        self.assertGreater(
            int((~query_masks[0]).sum()),
            int((~plain_masks[0]).sum()),
        )
        self.assertGreater(float(diagnostics["query_reliability"].item()), 0.9)

    def test_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self.route([self._constant_window()], epsilon=-0.1)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.route(
                [self._constant_window()],
                epsilon=0.1,
                query_embedding=torch.zeros(1, 3),
            )


if __name__ == "__main__":
    unittest.main()
