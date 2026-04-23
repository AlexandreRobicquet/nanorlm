from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bench import build_pairbench, extract_anchor_blocks, run_dataset
from nanorlm import HeuristicBackend, RLM, RLMConfig


class NanoRLMTests(unittest.TestCase):
    def test_completion_enforces_budget_and_emits_trace(self) -> None:
        example = build_pairbench(n=1, seed=0)[0]
        engine = RLM(
            RLMConfig(
                model="demo/heuristic",
                base_url="http://localhost:11434/v1",
                max_depth=2,
                memory_budget_tokens=60,
                retention_policy="pairwise_tournament",
                seed=0,
            ),
            backend=HeuristicBackend(seed=0),
        )
        result = engine.completion(example.query, example.context)
        self.assertLessEqual(sum(item.tokens for item in result.kept_items), 60)
        self.assertGreater(len(result.kept_items), 0)
        self.assertIn("[split]", result.trace.tree)
        self.assertIn("[retain]", result.trace.tree)
        self.assertIn("[final_answer]", result.trace.tree)
        self.assertTrue(result.answer)

    def test_all_policies_run_end_to_end(self) -> None:
        examples = build_pairbench(n=2, seed=0)
        for policy in ["keep_recent", "summary_only", "single_critic_topk", "pairwise_tournament"]:
            summary = run_dataset(examples, policy, budget=60, max_depth=2)
            self.assertEqual(summary["examples"], 2)
            self.assertEqual(summary["policy"], policy)
            for row in summary["results"]:
                self.assertTrue(row["answer"])

    def test_run_dataset_is_deterministic_under_seed(self) -> None:
        examples = build_pairbench(n=1, seed=0)
        first = run_dataset(examples, "pairwise_tournament", budget=60, max_depth=2)
        second = run_dataset(examples, "pairwise_tournament", budget=60, max_depth=2)
        self.assertEqual(first["results"][0]["answer"], second["results"][0]["answer"])
        self.assertEqual(first["results"][0]["retained"], second["results"][0]["retained"])

    def test_extract_anchor_blocks_finds_local_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.md"
            path.write_text("alpha\nbeta\nneedle line\nomega\n")
            blocks = extract_anchor_blocks(path, ["needle line"], window=1)
            self.assertTrue(blocks)
            self.assertIn("needle line", blocks[0].text)


if __name__ == "__main__":
    unittest.main()
