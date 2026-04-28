from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bench import extract_anchor_blocks
from showcases.planning import PlanningTask, run_planning_suite


class ShowcaseTests(unittest.TestCase):
    def test_planning_suite_returns_grounded_plan_shape(self) -> None:
        repo_root = Path(__file__).resolve().parent / "fixtures" / "verifiers-mini"
        context = []
        context.extend(extract_anchor_blocks(repo_root / "docs" / "evaluation.md", ["endpoint_id", "results.jsonl"], window=6))
        context.extend(extract_anchor_blocks(repo_root / "verifiers" / "scripts" / "eval.py", ["endpoint_id", "TOML"], window=6))
        task = PlanningTask(
            name="mini-plan",
            problem="Make endpoint_id validation clearer and keep resume documentation aligned. Propose the minimal patch plan.",
            context=context,
            repo_root=str(repo_root),
            evidence_files=["docs/evaluation.md", "verifiers/scripts/eval.py"],
            expected_files=["docs/evaluation.md", "verifiers/scripts/eval.py"],
            expected_keywords=["validation", "docs"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_planning_suite(
                [task],
                budget=80,
                max_depth=2,
                model="demo/heuristic",
                base_url="http://localhost:11434/v1",
                api_key=None,
                use_openai_backend=False,
                seed=0,
                output_dir=tmpdir,
            )
            self.assertEqual(summary["tasks"], 1)
            self.assertTrue(summary["results"][0]["steps"])
            self.assertTrue(summary["results"][0]["citations"])
            self.assertIn("uv run python -m unittest", summary["results"][0]["steps"][-1])
            self.assertTrue((Path(tmpdir) / "summary.json").exists())
            self.assertTrue((Path(tmpdir) / "plans" / "mini-plan.md").exists())


if __name__ == "__main__":
    unittest.main()
