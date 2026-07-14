from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts import run_benchmark_e2e


class BenchmarkE2ETests(unittest.TestCase):
    def run_quietly(self, argv: list[str]) -> int:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return run_benchmark_e2e.run(argv)

    def test_smoke_phase_writes_manifest_and_report_bundles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            code = self.run_quietly(
                [
                    "--phases",
                    "smoke",
                    "--output-root",
                    tmpdir,
                    "--run-id",
                    "smoke-test",
                    "--smoke-limit",
                    "1",
                ]
            )

            self.assertEqual(code, 0)
            run_root = Path(tmpdir) / "smoke-test"
            manifest = json.loads((run_root / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "passed")
            self.assertEqual(manifest["phases_requested"], ["smoke"])
            self.assertEqual(manifest["phases"][0]["status"], "passed")
            self.assertIn("head", manifest["git"])

            for name in ["smoke_pairbench", "smoke_verifiers", "smoke_external_jsonl"]:
                report_dir = run_root / name
                self.assertTrue((report_dir / "summary.json").exists())
                self.assertTrue((report_dir / "per_case.jsonl").exists())
                self.assertTrue((report_dir / "curves.json").exists())
                self.assertTrue((report_dir / "experiment_report.md").exists())
                self.assertTrue((report_dir / "trace_examples").exists())
                self.assertTrue((report_dir / "loom_traces").exists())

    def test_assets_phase_generates_artifact_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            code = self.run_quietly(
                [
                    "--phases",
                    "smoke,assets",
                    "--output-root",
                    tmpdir,
                    "--run-id",
                    "assets-test",
                    "--smoke-limit",
                    "1",
                ]
            )

            self.assertEqual(code, 0)
            assets_dir = Path(tmpdir) / "assets-test" / "artifacts"
            manifest = json.loads((assets_dir / "manifest.json").read_text())
            self.assertIn("benchmark_snapshot.md", manifest["files"])
            self.assertIn("architecture.svg", manifest["files"])
            self.assertIn("policy_curve.svg", manifest["files"])

    def test_learned_phase_trains_evaluates_and_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            code = self.run_quietly(
                [
                    "--phases",
                    "learned",
                    "--output-root",
                    tmpdir,
                    "--run-id",
                    "learned-test",
                    "--learned-train-limit",
                    "1",
                    "--learned-eval-limit",
                    "1",
                    "--external-limit",
                    "1",
                    "--learned-ruler-path",
                    str(Path(__file__).resolve().parent / "fixtures" / "external-benchmark-mini.jsonl"),
                    "--learned-babilong-path",
                    str(Path(__file__).resolve().parent / "fixtures" / "external-benchmark-mini.jsonl"),
                ]
            )

            self.assertEqual(code, 0)
            run_root = Path(tmpdir) / "learned-test"
            manifest = json.loads((run_root / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "passed")
            result = manifest["phases"][0]["result"]
            model_path = Path(result["training"]["model_path"])
            self.assertTrue(model_path.exists())
            learned_report = Path(result["learned_report"]["report_path"])
            self.assertTrue(learned_report.exists())
            self.assertIn("Verdict:", learned_report.read_text())
            self.assertIn("Training source: `traces`", learned_report.read_text())
            self.assertEqual(result["learned_report"]["non_toy_wins"], 0)
            self.assertEqual(result["learned_report"]["acceptance_rule"]["minimum_examples"], 8)
            pairbench_report = next(report for report in result["reports"] if report["name"] == "learned_pairbench")
            self.assertEqual(pairbench_report["start_index"], 1)
            self.assertEqual(
                set(pairbench_report["policies"]),
                {
                    "direct_full_context",
                    "keep_recent",
                    "single_critic_topk",
                    "pairwise_tournament",
                    "learned_retention",
                },
            )
            datasets = {report["dataset"] for report in result["reports"]}
            self.assertIn("ruler_external", datasets)
            self.assertIn("babilong_external", datasets)
            smoke_comparisons = result["learned_report"]["comparisons"]
            self.assertFalse(any(row["acceptance_eligible"] for row in smoke_comparisons))

    def test_learned_acceptance_gate_uses_unrounded_deltas(self) -> None:
        learned = {
            "reward_score": 0.01,
            "answer_accuracy": 1.0,
            "provenance_score": 1.0,
            "results": [
                {"name": f"case-{index}", "reward_score": reward, "answer_accuracy": 1.0, "provenance_score": 1.0}
                for index, reward in enumerate((0.01, 0.01, 0.01, 0.01, 0.008))
            ],
        }
        pairwise = {
            "reward_score": 0.0,
            "answer_accuracy": 1.0,
            "provenance_score": 1.0,
            "results": [
                {"name": f"case-{index}", "reward_score": 0.0, "answer_accuracy": 1.0, "provenance_score": 1.0}
                for index in range(5)
            ],
        }
        deltas = run_benchmark_e2e._learned_acceptance_deltas(learned, pairwise)
        self.assertIsNotNone(deltas)
        reward_delta, answer_delta, provenance_delta = deltas or (0.0, 0.0, 0.0)
        self.assertAlmostEqual(reward_delta, 0.0096)
        pairwise["results"][0]["name"] = "different-case"
        self.assertIsNone(run_benchmark_e2e._learned_acceptance_deltas(learned, pairwise))
        self.assertFalse(
            run_benchmark_e2e._is_learned_acceptance_win(
                acceptance_eligible=True,
                reward_delta=reward_delta,
                answer_delta=answer_delta,
                provenance_delta=provenance_delta,
            )
        )
        self.assertFalse(
            run_benchmark_e2e._is_learned_acceptance_win(
                acceptance_eligible=True,
                reward_delta=0.02,
                answer_delta=-0.0004,
                provenance_delta=0.0,
            )
        )
        self.assertTrue(
            run_benchmark_e2e._is_learned_acceptance_win(
                acceptance_eligible=True,
                reward_delta=0.01,
                answer_delta=0.0,
                provenance_delta=0.0,
            )
        )

    def test_smoke_phase_resolves_fixture_defaults_from_non_repo_cwd(self) -> None:
        previous_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as launch_cwd:
            try:
                os.chdir(launch_cwd)
                code = self.run_quietly(
                    [
                        "--phases",
                        "smoke",
                        "--output-root",
                        tmpdir,
                        "--run-id",
                        "outside-cwd-test",
                        "--smoke-limit",
                        "1",
                    ]
                )
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(code, 0)
            run_root = Path(tmpdir) / "outside-cwd-test"
            manifest = json.loads((run_root / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "passed")
            self.assertTrue((run_root / "smoke_verifiers" / "summary.json").exists())

    def test_real_model_phase_rejects_unknown_hosted_cost_model_before_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            code = self.run_quietly(
                [
                    "--phases",
                    "real_model",
                    "--output-root",
                    tmpdir,
                    "--run-id",
                    "real-model-safety-test",
                    "--real-model",
                    "unknown-hosted-model",
                    "--real-api-key",
                    "test-key",
                ]
            )

            self.assertEqual(code, 1)
            manifest = json.loads((Path(tmpdir) / "real-model-safety-test" / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("no cost table entry", manifest["phases"][0]["error"])


if __name__ == "__main__":
    unittest.main()
