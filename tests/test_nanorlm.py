from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import (
    build_dossierbench,
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
        context = [
            ContextBlock(name="incident-1.txt", text="api gateway rollout blocker is stale endpoint registry cache after deploy validation retry failure on startup."),
            ContextBlock(name="incident-2.txt", text="fix is to invalidate the endpoint registry cache on reload before the gateway re-reads route metadata."),
            ContextBlock(name="incident-3.txt", text="owner is infra and the patch is queued for the next deploy window after review."),
            ContextBlock(name="incident-4.txt", text="unrelated background notes describe another service and an old dashboard migration that is already closed."),
        ]
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
        result = engine.completion("What is blocking the api gateway rollout?", context)
        self.assertIn("cache", result.answer.lower())
        self.assertIn("[split]", result.trace.tree)
        self.assertLessEqual(sum(item.tokens for item in result.kept_items), 60)
        self.assertEqual(result.retention_stats["policy"], "pairwise_tournament")
        self.assertGreaterEqual(result.retention_stats["total_retention_steps"], 1)
        self.assertTrue(result.per_step_budget)
        self.assertIsInstance(result.drop_reasons, list)

    def test_pairwise_differs_from_single_critic_on_dossierbench_internal_regression(self) -> None:
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
            single_results = by_policy["single_critic_topk"]["results"]
            pairwise_results = by_policy["pairwise_tournament"]["results"]
            differences = sum(
                1
                for single_result, pairwise_result in zip(single_results, pairwise_results)
                if single_result["retained_summaries"] != pairwise_result["retained_summaries"]
            )
            self.assertGreater(differences, 0)

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
