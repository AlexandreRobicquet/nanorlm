from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from inspection_replay import InspectionReplayBackend
from loom_trace import memory_artifact_id
from nanorlm import ContextBlock, HeuristicBackend, InspectionResult, RLM, RLMConfig


class CountingInspectionBackend(HeuristicBackend):
    def __init__(self, *, fail_inspection: bool = False) -> None:
        super().__init__(seed=0)
        self.fail_inspection = fail_inspection
        self.inspect_calls = 0

    def inspect(self, query, documents, depth, branch) -> InspectionResult:  # type: ignore[override]
        self.inspect_calls += 1
        if self.fail_inspection:
            raise AssertionError("inspection should have been replayed")
        return super().inspect(query, documents, depth, branch)


class InspectionReplayTests(unittest.TestCase):
    def context(self) -> list[ContextBlock]:
        return [
            ContextBlock(name="a.txt", text="alpha release key is amber " * 30),
            ContextBlock(name="b.txt", text="beta rollout owner is infra " * 30),
            ContextBlock(name="c.txt", text="gamma notes are unrelated " * 30),
            ContextBlock(name="d.txt", text="delta confirms amber " * 30),
        ]

    def config(self) -> RLMConfig:
        return RLMConfig(
            model="demo/heuristic",
            provider="heuristic",
            max_depth=3,
            memory_budget_tokens=40,
            retention_policy="pairwise_tournament",
            retention_judge="heuristic",
            seed=0,
        )

    def test_capture_then_replay_avoids_duplicate_inspections(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = Path(tmpdir) / "case.json"
            capture_backend = CountingInspectionBackend()
            first = RLM(
                self.config(),
                backend=InspectionReplayBackend(capture_backend, store),
            ).completion("What is the release key?", self.context())

            replay_backend = CountingInspectionBackend(fail_inspection=True)
            second = RLM(
                self.config(),
                backend=InspectionReplayBackend(replay_backend, store, mode="replay_only"),
            ).completion("What is the release key?", self.context())

        first_stats = first.retention_stats["inspection_replay"]
        second_stats = second.retention_stats["inspection_replay"]
        self.assertGreater(capture_backend.inspect_calls, 0)
        self.assertEqual(replay_backend.inspect_calls, 0)
        self.assertEqual(first_stats["captured"], capture_backend.inspect_calls)
        self.assertEqual(second_stats["replayed"], capture_backend.inspect_calls)
        self.assertEqual(first.answer, second.answer)
        self.assertEqual(
            {key: first.stage_budgets["inspect"][key] for key in ("prompt_tokens", "completion_tokens", "calls")},
            {key: second.stage_budgets["inspect"][key] for key in ("prompt_tokens", "completion_tokens", "calls")},
        )
        self.assertEqual(
            [memory_artifact_id(item) for item in first.kept_items],
            [memory_artifact_id(item) for item in second.kept_items],
        )

    def test_replay_only_fails_closed_on_missing_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = InspectionReplayBackend(
                CountingInspectionBackend(),
                Path(tmpdir) / "missing.json",
                mode="replay_only",
            )
            with self.assertRaisesRegex(RuntimeError, "inspection replay miss"):
                backend.inspect("query", [ContextBlock(name="a.txt", text="alpha")], 0, "root")

    def test_replay_fails_closed_when_capture_is_modified(self) -> None:
        documents = [ContextBlock(name="a.txt", text="alpha")]
        with tempfile.TemporaryDirectory() as tmpdir:
            store = Path(tmpdir) / "capture.json"
            capture = InspectionReplayBackend(CountingInspectionBackend(), store)
            capture.inspect("query", documents, 0, "root")
            payload = json.loads(store.read_text())
            record = next(iter(payload["records"].values()))
            record["result"]["summary"] = "tampered"
            store.write_text(json.dumps(payload))

            replay = InspectionReplayBackend(CountingInspectionBackend(), store, mode="replay_only")
            with self.assertRaisesRegex(ValueError, "integrity check failed"):
                replay.inspect("query", documents, 0, "root")

    def test_replay_is_scoped_to_model_namespace(self) -> None:
        documents = [ContextBlock(name="a.txt", text="alpha")]
        with tempfile.TemporaryDirectory() as tmpdir:
            store = Path(tmpdir) / "capture.json"
            capture = InspectionReplayBackend(
                CountingInspectionBackend(),
                store,
                namespace={"model": "model-a"},
            )
            capture.inspect("query", documents, 0, "root")

            replay = InspectionReplayBackend(
                CountingInspectionBackend(),
                store,
                mode="replay_only",
                namespace={"model": "model-b"},
            )
            with self.assertRaisesRegex(RuntimeError, "inspection replay miss"):
                replay.inspect("query", documents, 0, "root")


if __name__ == "__main__":
    unittest.main()
