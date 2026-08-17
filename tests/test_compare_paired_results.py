import io
import unittest
from contextlib import redirect_stdout

from scripts.compare_paired_results import compare, print_report


def row(pred, merged, pruned, prefill=None, **metrics):
    token_info = {
        "raw_vision_tokens": 100,
        "merged_vision_tokens": merged,
        "pruned_tokens": pruned,
        "pruning_layer": 0,
        **metrics,
    }
    if prefill is not None:
        token_info["prefill_tokens"] = prefill
    return {
        "task_type": "short",
        "video_path": "video.mp4",
        "question": "Question?",
        "pred": pred,
        "gt": "(A)",
        "token_info": token_info,
    }


class ComparePairedResultsTests(unittest.TestCase):
    def test_infers_legacy_prefill_and_reports_compression_and_flops(self):
        baseline = [row("(A)", merged=60, pruned=40)]
        candidate = [row("(A)", merged=50, pruned=35, prefill=70)]

        report = compare(
            baseline,
            candidate,
            bootstrap_samples=10,
            seed=0,
            hidden_size=8,
            intermediate_size=16,
            num_layers=2,
        )

        prefill = report["tokens"]["prefill_tokens"]
        self.assertEqual(prefill["baseline_mean"], 80)
        self.assertEqual(prefill["candidate_mean"], 70)
        self.assertEqual(prefill["candidate_reduction"], 0.125)
        self.assertEqual(prefill["sources"]["baseline"]["inferred"], 1)
        self.assertEqual(prefill["sources"]["candidate"]["recorded"], 1)
        self.assertEqual(
            report["compression"]["raw_to_merged_retention"],
            {"baseline": 0.6, "candidate": 0.5},
        )
        self.assertIsNotNone(report["analytical_llm_prefill"]["baseline"]["mean"])
        self.assertGreater(
            report["analytical_llm_prefill"]["candidate_mean_reduction"], 0
        )
        self.assertEqual(prefill["baseline"]["median"], 80)
        self.assertIn("short", report["breakdown_by_task_type"])
        self.assertEqual(report["oracle_complementarity"]["accuracy"], 1.0)

    def test_runtime_metrics_are_compared_only_when_recorded(self):
        baseline = [
            row(
                "(A)",
                merged=60,
                pruned=40,
                prefill=80,
                total_latency_ms=100,
                peak_gpu_memory_mb=1000,
            )
        ]
        candidate = [
            row(
                "(A)",
                merged=50,
                pruned=35,
                prefill=70,
                total_latency_ms=75,
                peak_gpu_memory_mb=900,
            )
        ]

        report = compare(baseline, candidate, bootstrap_samples=10, seed=0)

        self.assertEqual(report["runtime"]["total_latency_ms"]["candidate_mean_reduction"], 0.25)
        self.assertEqual(report["runtime"]["peak_gpu_memory_mb"]["candidate_mean_reduction"], 0.1)
        self.assertEqual(
            report["runtime"]["total_latency_ms"]["paired_mean_delta"],
            -25,
        )
        self.assertIsNone(report["runtime"]["prefill_latency_ms"]["baseline"]["mean"])

    def test_text_report_marks_unrecorded_runtime_metrics_unavailable(self):
        report = compare(
            [row("(A)", merged=60, pruned=40, prefill=80)],
            [row("(A)", merged=50, pruned=35, prefill=70)],
            bootstrap_samples=10,
            seed=0,
        )
        output = io.StringIO()

        with redirect_stdout(output):
            print_report(report)

        rendered = output.getvalue()
        self.assertIn("Analytical LLM prefill TFLOPs", rendered)
        self.assertIn("total_latency_ms: unavailable", rendered)


if __name__ == "__main__":
    unittest.main()
