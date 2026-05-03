from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from statistics import fmean
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import (
    build_dossierbench,
    build_needlepairs,
    build_pairbench,
    extract_anchor_blocks,
    generate_curves,
    load_verifiers_smoke,
    policy_sweep,
    run_dataset,
    write_report_bundle,
)
from nanorlm import AnswerResult, ContextBlock, HeuristicBackend, InspectionResult, RLM, RLMConfig, Usage


class NoScoreBackend(HeuristicBackend):
    def __init__(self) -> None:
        super().__init__(seed=0)
        self.score_calls = 0

    def inspect(self, query: str, documents: list[ContextBlock], depth: int, branch: str) -> InspectionResult:  # type: ignore[override]
        return InspectionResult(
            summary="compact evidence",
            evidence=[],
            answer_candidate="compact evidence",
            confidence=0.5,
            metadata={},
            usage=Usage(calls=1),
        )

    def answer(self, query: str, memory):  # type: ignore[override]
        return AnswerResult(answer="compact evidence", confidence=0.5, usage=Usage(calls=1))

    def score_candidate(self, query: str, item):  # type: ignore[override]
        self.score_calls += 1
        raise AssertionError("score_candidate should not run during leaf creation for keep_recent")


class NanoRLMTests(unittest.TestCase):
    def test_completion_returns_answer_trace_and_budgeted_memory(self) -> None:
        example = build_pairbench(n=1, seed=0)[0]
        engine = RLM(
            RLMConfig(
                model="demo/heuristic",
                base_url="http://localhost:11434/v1",
                max_depth=4,
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
        self.assertEqual(result.retention_stats["policy"], "pairwise_tournament")
        self.assertGreaterEqual(result.retention_stats["total_retention_steps"], 1)
        self.assertTrue(result.per_step_budget)
        self.assertIsInstance(result.drop_reasons, list)

    def test_pairwise_beats_single_critic_on_pairbench(self) -> None:
        examples = build_pairbench(n=10, seed=0)
        single = run_dataset(examples, "single_critic_topk", budget=60, max_depth=2)
        pairwise = run_dataset(examples, "pairwise_tournament", budget=60, max_depth=2)
        self.assertGreater(pairwise["accuracy"], single["accuracy"])

    def test_pairwise_beats_single_critic_on_needlepairs(self) -> None:
        examples = build_needlepairs(n=6, seed=0)
        single = run_dataset(examples, "single_critic_topk", budget=60, max_depth=3)
        pairwise = run_dataset(examples, "pairwise_tournament", budget=60, max_depth=3)
        self.assertGreater(pairwise["accuracy"], single["accuracy"])

    def test_pairwise_beats_single_critic_on_dossierbench_launch_budget(self) -> None:
        single_scores = []
        pairwise_scores = []
        for seed in [0, 1, 2]:
            examples = build_dossierbench(n=6, seed=seed)
            rows = policy_sweep(
                examples,
                ["single_critic_topk", "pairwise_tournament"],
                budget=80,
                max_depth=4,
                seed=seed,
                dataset_name="dossierbench",
            )
            by_policy = {row["policy"]: row for row in rows}
            single_scores.append(by_policy["single_critic_topk"]["answer_accuracy"])
            pairwise_scores.append(by_policy["pairwise_tournament"]["answer_accuracy"])
        self.assertGreater(fmean(pairwise_scores), fmean(single_scores))

    def test_extract_anchor_blocks_finds_local_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.md"
            path.write_text("alpha\nbeta\nneedle line\nomega\n")
            blocks = extract_anchor_blocks(path, ["needle line"], window=1)
            self.assertTrue(blocks)
            self.assertIn("needle line", blocks[0].text)

    def test_keep_recent_does_not_score_leaf_items_eagerly(self) -> None:
        backend = NoScoreBackend()
        engine = RLM(
            RLMConfig(
                model="demo/heuristic",
                base_url="http://localhost:11434/v1",
                max_depth=4,
                memory_budget_tokens=80,
                retention_policy="keep_recent",
                seed=0,
            ),
            backend=backend,
        )
        result = engine.completion(
            "What changed?",
            [ContextBlock(name="note-a.txt", text="alpha"), ContextBlock(name="note-b.txt", text="beta")],
        )
        self.assertEqual(result.answer, "compact evidence")
        self.assertEqual(backend.score_calls, 0)

    def test_small_multi_block_context_stays_leaf(self) -> None:
        engine = RLM(
            RLMConfig(
                model="demo/heuristic",
                base_url="http://localhost:11434/v1",
                max_depth=4,
                memory_budget_tokens=80,
                retention_policy="keep_recent",
                seed=0,
            ),
            backend=HeuristicBackend(seed=0),
        )
        result = engine.completion(
            "What changed?",
            [ContextBlock(name="note-a.txt", text="alpha"), ContextBlock(name="note-b.txt", text="beta")],
        )
        self.assertNotIn("[split]", result.trace.tree)

    def test_summary_only_does_not_report_rewritten_items_as_dropped(self) -> None:
        long_line = "alpha " * 60
        engine = RLM(
            RLMConfig(
                model="demo/heuristic",
                base_url="http://localhost:11434/v1",
                max_depth=2,
                memory_budget_tokens=200,
                retention_policy="summary_only",
                seed=0,
            ),
            backend=HeuristicBackend(seed=0),
        )
        result = engine.completion(
            "What changed?",
            [
                ContextBlock(name="left.txt", text=long_line + "left"),
                ContextBlock(name="right.txt", text=long_line + "right"),
            ],
        )
        self.assertEqual(result.retention_stats["total_dropped_items"], 0)
        self.assertEqual(result.per_step_budget[-1]["before_count"], result.per_step_budget[-1]["after_count"])

    def test_report_bundle_writes_schema_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            examples = build_pairbench(n=4, seed=0)
            summaries = policy_sweep(
                examples,
                ["single_critic_topk", "pairwise_tournament"],
                budget=60,
                max_depth=2,
                output_dir=tmpdir,
                dataset_name="pairbench",
            )
            curves = generate_curves(
                "pairbench",
                lambda seed: build_pairbench(n=4, seed=seed),
                policies=["single_critic_topk", "pairwise_tournament"],
                budgets=[60],
                depths=[2],
                seeds=[0],
            )
            write_report_bundle(
                tmpdir,
                dataset_name="pairbench",
                summaries=summaries,
                curves=curves,
                command="python bench.py --dataset pairbench --limit 4 --budget 60 --depth 2",
            )
            self.assertTrue((Path(tmpdir) / "summary.json").exists())
            self.assertTrue((Path(tmpdir) / "per_case.jsonl").exists())
            self.assertTrue((Path(tmpdir) / "curves.json").exists())
            self.assertTrue((Path(tmpdir) / "trace_examples" / "pairwise_tournament").exists())

    def test_verifiers_smoke_fixture_loads_and_runs(self) -> None:
        repo_root = Path(__file__).resolve().parent / "fixtures" / "verifiers-mini"
        examples = load_verifiers_smoke(repo_root)
        self.assertEqual(len(examples), 2)
        summary = run_dataset(
            examples,
            "pairwise_tournament",
            budget=80,
            max_depth=2,
            dataset_name="verifiers_smoke",
        )
        self.assertEqual(summary["examples"], 2)


if __name__ == "__main__":
    unittest.main()
