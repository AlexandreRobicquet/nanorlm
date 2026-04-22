from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bench import BenchmarkExample, build_needlepairs, build_pairbench, extract_anchor_blocks, run_dataset
from nanorlm import ContextBlock, HeuristicBackend, RLM, RLMConfig


class NanoRLMTests(unittest.TestCase):
    def test_completion_returns_answer_trace_and_budgeted_memory(self) -> None:
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
        self.assertIn("amber", result.answer)
        self.assertIn("orbit", result.answer)
        self.assertIn("[split]", result.trace.tree)
        self.assertLessEqual(sum(item.tokens for item in result.kept_items), 60)

    def test_pairwise_beats_single_critic_on_pairbench(self) -> None:
        examples = build_pairbench(n=10, seed=0)
        single = run_dataset(examples, "single_critic_topk", budget=60, max_depth=2)
        pairwise = run_dataset(examples, "pairwise_tournament", budget=60, max_depth=2)
        self.assertGreater(pairwise["accuracy"], single["accuracy"])

    def test_pairwise_beats_single_critic_on_needlepairs(self) -> None:
        examples = build_needlepairs(n=6, seed=0)
        single = run_dataset(examples, "single_critic_topk", budget=60, max_depth=2)
        pairwise = run_dataset(examples, "pairwise_tournament", budget=60, max_depth=2)
        self.assertGreater(pairwise["accuracy"], single["accuracy"])

    def test_extract_anchor_blocks_finds_local_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.md"
            path.write_text("alpha\nbeta\nneedle line\nomega\n")
            blocks = extract_anchor_blocks(path, ["needle line"], window=1)
            self.assertTrue(blocks)
            self.assertIn("needle line", blocks[0].text)


if __name__ == "__main__":
    unittest.main()
