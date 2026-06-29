from __future__ import annotations

import contextlib
import io
import json
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
