from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from loom_trace import build_loom_trace, memory_artifact_id, write_loom_trace
from nanorlm import ContextBlock, HeuristicBackend, RLM, RLMConfig


class LoomTraceBridgeTests(unittest.TestCase):
    def make_result(self, policy: str = "pairwise_tournament"):
        context = [
            ContextBlock(name="a.txt", text="The release key is amber and the owner is infra. " * 4),
            ContextBlock(name="b.txt", text="The deploy window opens Tuesday after review. " * 4),
            ContextBlock(name="c.txt", text="Unrelated migration notes describe a closed task. " * 4),
            ContextBlock(name="d.txt", text="The rollback procedure retains the amber key. " * 4),
        ]
        engine = RLM(
            RLMConfig(
                model="demo/heuristic",
                provider="heuristic",
                max_depth=3,
                memory_budget_tokens=24,
                retention_policy=policy,
                retention_judge="heuristic",
                seed=0,
            ),
            backend=HeuristicBackend(seed=0),
        )
        return engine.completion("What is the release key?", context)

    def build_events(self, result):
        return build_loom_trace(
            result,
            dataset="bridge-test",
            case_name="case-1",
            query="What is the release key?",
            policy="pairwise_tournament",
            provider="heuristic",
            model="demo/heuristic",
            seed=0,
            budget_tokens=24,
            answer_score=1.0,
            provenance_score=1.0,
            expected_answer="amber",
            expected_provenance=["a.txt"],
        )

    def test_export_has_complete_envelopes_and_resolved_references(self) -> None:
        result = self.make_result()
        events = self.build_events(result)

        self.assertTrue(events)
        self.assertEqual(events[-1]["event_type"], "outcome")
        self.assertEqual({event["schema_version"] for event in events}, {"0.1"})
        self.assertEqual(len({event["event_id"] for event in events}), len(events))
        self.assertEqual(len({event["task_id"] for event in events}), 1)
        self.assertEqual(len({event["run_id"] for event in events}), 1)

        artifacts = {
            event["artifact_id"]
            for event in events
            if event["event_type"] == "artifact_emit"
        }
        for event in events:
            if event["event_type"] == "retrieval_query":
                self.assertTrue(set(event["candidate_ids"]).issubset(artifacts))
            if event["event_type"] == "retrieval_use":
                self.assertIn(event["artifact_id"], artifacts)
            if event["event_type"] == "artifact_emit":
                self.assertTrue(set(event["parents"]).issubset(artifacts))
            if event["event_type"] == "stage_end":
                self.assertEqual(
                    set(event["budget_used"]),
                    {"prompt_tokens", "completion_tokens", "calls", "wall_ms"},
                )

    def test_task_and_artifact_ids_are_stable_but_run_ids_are_unique(self) -> None:
        first = self.make_result()
        second = self.make_result()
        first_events = self.build_events(first)
        second_events = self.build_events(second)

        self.assertEqual(
            [event["event_id"] for event in first_events],
            [event["event_id"] for event in second_events],
        )
        self.assertEqual(first_events[0]["task_id"], second_events[0]["task_id"])
        self.assertNotEqual(first_events[0]["run_id"], second_events[0]["run_id"])
        self.assertEqual(
            [memory_artifact_id(item) for item in first.kept_items],
            [memory_artifact_id(item) for item in second.kept_items],
        )

    def test_jsonl_writer_round_trips(self) -> None:
        events = self.build_events(self.make_result())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_loom_trace(Path(tmpdir) / "trace.jsonl", events)
            loaded = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual(loaded, events)

    def test_summary_rewrite_is_a_parented_retention_artifact(self) -> None:
        result = self.make_result(policy="summary_only")
        events = build_loom_trace(
            result,
            dataset="bridge-test",
            case_name="case-summary",
            query="What is the release key?",
            policy="summary_only",
            provider="heuristic",
            model="demo/heuristic",
            seed=0,
            budget_tokens=24,
            answer_score=0.0,
            provenance_score=0.0,
            expected_answer="amber",
            expected_provenance=["a.txt"],
        )

        derived = [
            event
            for event in events
            if event["event_type"] == "artifact_emit"
            and event["artifact_type"] == "nanorlm.memory_item.retained"
        ]
        self.assertTrue(derived)
        self.assertTrue(all(event["parents"] for event in derived))
        emitted_ids = {
            event["artifact_id"]
            for event in events
            if event["event_type"] == "artifact_emit"
        }
        self.assertTrue(all(set(event["parents"]).issubset(emitted_ids) for event in derived))


if __name__ == "__main__":
    unittest.main()
