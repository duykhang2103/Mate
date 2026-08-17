import unittest

from scripts.evaluate_query_kill_gate import apply_gate
from scripts.evaluate_acceptance_gate import evaluate_acceptance
from scripts.compare_paired_results import compare
from scripts.select_distortion_epsilon import select_configuration


def result_row(index, pred, merged=50):
    return {
        "task_type": "short",
        "video_path": f"video_{index}.mp4",
        "question": f"Question {index}?",
        "pred": pred,
        "gt": "(A)",
        "token_info": {
            "raw_vision_tokens": 100,
            "merged_vision_tokens": merged,
            "prefill_tokens": merged + 20,
            "pruned_tokens": 30,
            "pruning_layer": 10,
        },
    }


class QueryKillGateTests(unittest.TestCase):
    def test_retains_only_significant_positive_matched_budget_gain(self):
        novelty = [result_row(i, "(B)") for i in range(6)]
        query = [result_row(i, "(A)") for i in range(6)]

        result = apply_gate(
            novelty,
            query,
            token_budget_tolerance=0.01,
            bootstrap_samples=20,
            seed=0,
        )

        self.assertEqual(result["decision"], "RETAIN_QUERY")
        self.assertTrue(all(result["checks"].values()))

    def test_removes_query_when_accuracy_does_not_improve(self):
        novelty = [result_row(i, "(A)") for i in range(6)]
        query = [result_row(i, "(A)") for i in range(6)]

        result = apply_gate(
            novelty,
            query,
            token_budget_tolerance=0.01,
            bootstrap_samples=20,
            seed=0,
        )

        self.assertEqual(result["decision"], "REMOVE_QUERY")
        self.assertFalse(result["checks"]["positive_accuracy_delta"])


class EpsilonSelectionTests(unittest.TestCase):
    @staticmethod
    def summary(epsilon, tflops):
        return {
            "path": f"epsilon_{epsilon}.json",
            "epsilon": epsilon,
            "samples": 3,
            "sample_keys": [
                ("short", "short.mp4", "q"),
                ("medium", "medium.mp4", "q"),
                ("long", "long.mp4", "q"),
            ],
            "mean_tflops": tflops,
            "mean_merged_tokens": 100,
            "unique_videos_by_group": {
                "short": 1,
                "medium": 1,
                "long": 1,
            },
        }

    def test_match_selects_closest_compute_without_labels(self):
        result = select_configuration(
            [self.summary(0.01, 9.2), self.summary(0.02, 8.95)],
            target_tflops=8.9714,
            mode="match",
            tolerance=0.01,
            required_videos_per_group=1,
        )

        self.assertEqual(result["selected"]["epsilon"], 0.02)
        self.assertTrue(result["target_passed"])
        self.assertFalse(result["selection_uses_labels"])

    def test_at_most_selects_least_compressed_eligible_result(self):
        result = select_configuration(
            [
                self.summary(0.01, 8.7),
                self.summary(0.02, 8.5),
                self.summary(0.04, 8.1),
            ],
            target_tflops=8.5181,
            mode="at-most",
            tolerance=0.01,
            required_videos_per_group=1,
        )

        self.assertEqual(result["selected"]["epsilon"], 0.02)
        self.assertTrue(result["target_passed"])


class AcceptanceGateTests(unittest.TestCase):
    def test_accepts_noninferior_candidate_with_twenty_percent_compute_gain(self):
        baseline = [result_row(0, "(A)", merged=90)]
        candidate = [result_row(0, "(A)", merged=20)]
        baseline[0]["token_info"].update(
            prefill_tokens=110,
            pruned_tokens=90,
            prefill_latency_ms=100,
        )
        candidate[0]["token_info"].update(
            prefill_tokens=40,
            pruned_tokens=20,
            prefill_latency_ms=60,
        )
        report = compare(
            baseline, candidate, bootstrap_samples=20, seed=0
        )

        result = evaluate_acceptance(report)

        self.assertTrue(result["accepted"])
        self.assertTrue(result["rule_efficiency_noninferiority"]["passed"])


if __name__ == "__main__":
    unittest.main()
