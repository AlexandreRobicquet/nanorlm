from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanorlm import MemoryItem
from learned_retention import LearnedRetentionModel, LearnedRetentionPolicy, train_linear_retention_model
from policies import KeepRecentPolicy, PairwiseTournamentPolicy, SingleCriticTopKPolicy, SummaryOnlyPolicy


class DummyJudge:
    def score_candidate(self, query: str, item: MemoryItem) -> float:
        score = 0.0
        if "cache" in item.summary:
            score += 1.5
        if "owner" in item.summary:
            score += 0.5
        return score

    def compare_candidates(self, query: str, left: MemoryItem, right: MemoryItem) -> int:
        left_score = self.score_candidate(query, left)
        right_score = self.score_candidate(query, right)
        if left_score == right_score:
            return 0
        return 1 if left_score > right_score else -1


class CountingJudge(DummyJudge):
    def __init__(self) -> None:
        self.score_calls = 0
        self.compare_calls = 0

    def score_candidate(self, query: str, item: MemoryItem) -> float:
        self.score_calls += 1
        return super().score_candidate(query, item)

    def compare_candidates(self, query: str, left: MemoryItem, right: MemoryItem) -> int:
        self.compare_calls += 1
        return super().compare_candidates(query, left, right)


def item(timestamp: float, summary: str, provenance: str, tokens: int = 12) -> MemoryItem:
    return MemoryItem(
        summary=summary,
        provenance=provenance,
        raw_pointer=provenance,
        tokens=tokens,
        depth=1,
        timestamp=timestamp,
        metadata={},
    )


class PolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = [
            item(1.0, "root cause is stale cache", "incidents/cache.txt"),
            item(2.0, "owner is infra", "incidents/owner.txt"),
            item(3.0, "unrelated notes", "notes/misc.txt"),
            item(4.0, "cache fix is reload", "incidents/fix.txt"),
        ]
        self.query = "What is the rollout blocker and cache fix?"

    def test_keep_recent_respects_budget(self) -> None:
        kept = KeepRecentPolicy().select(self.query, self.candidates, budget=24)
        self.assertLessEqual(sum(candidate.tokens for candidate in kept), 24)

    def test_summary_only_drops_metadata(self) -> None:
        kept = SummaryOnlyPolicy().select(self.query, self.candidates, budget=24)
        self.assertTrue(all(not candidate.metadata for candidate in kept))

    def test_single_critic_prefers_higher_scoring_candidates(self) -> None:
        policy = SingleCriticTopKPolicy(judge=DummyJudge())
        kept = policy.select(self.query, self.candidates, budget=24)
        self.assertTrue(any("cache" in candidate.summary for candidate in kept))
        self.assertEqual(len(policy.decision_candidates()), len(self.candidates))
        self.assertTrue(any(candidate.score > 0.0 for candidate in policy.decision_candidates() if candidate not in kept))

    def test_pairwise_respects_budget(self) -> None:
        kept = PairwiseTournamentPolicy(judge=DummyJudge(), seed=0).select(self.query, self.candidates, budget=24)
        self.assertLessEqual(sum(candidate.tokens for candidate in kept), 24)
        self.assertTrue(kept)

    def test_pairwise_prefers_higher_scored_candidates(self) -> None:
        policy = PairwiseTournamentPolicy(judge=DummyJudge(), seed=0)
        kept = policy.select(self.query, self.candidates, budget=24)
        self.assertTrue(any("cache" in candidate.summary for candidate in kept))
        self.assertEqual(len(policy.decision_candidates()), len(self.candidates))
        self.assertTrue(all(candidate.wins or candidate.losses for candidate in policy.decision_candidates()))

    def test_pairwise_keeps_complementary_evidence_under_tight_budget(self) -> None:
        class TieJudge:
            def score_candidate(self, query: str, candidate: MemoryItem) -> float:
                return 1.0

            def compare_candidates(self, query: str, left: MemoryItem, right: MemoryItem) -> int:
                return 0

        candidates = [
            item(1.0, "alpha cache duplicate detail", "facts/alpha-a.txt", tokens=10),
            item(2.0, "alpha cache duplicate detail", "facts/alpha-b.txt", tokens=10),
            item(3.0, "beta owner complementary detail", "facts/beta.txt", tokens=10),
            item(4.0, "alpha cache duplicate extra", "facts/alpha-c.txt", tokens=10),
        ]
        kept = PairwiseTournamentPolicy(judge=TieJudge(), seed=0).select(
            "Need alpha cache and beta owner facts",
            candidates,
            budget=20,
        )
        self.assertLessEqual(sum(candidate.tokens for candidate in kept), 20)
        self.assertTrue(any("alpha" in candidate.summary for candidate in kept))
        self.assertTrue(any("beta" in candidate.summary for candidate in kept))

    def test_critic_policies_skip_judge_when_candidates_fit_budget(self) -> None:
        single_judge = CountingJudge()
        pairwise_judge = CountingJudge()

        single = SingleCriticTopKPolicy(judge=single_judge).select(self.query, self.candidates, budget=100)
        pairwise = PairwiseTournamentPolicy(judge=pairwise_judge, seed=0).select(self.query, self.candidates, budget=100)

        self.assertEqual(len(single), len(self.candidates))
        self.assertEqual(len(pairwise), len(self.candidates))
        self.assertEqual(single_judge.score_calls, 0)
        self.assertEqual(pairwise_judge.score_calls, 0)
        self.assertEqual(pairwise_judge.compare_calls, 0)

    def test_learned_retention_prefers_relevant_durable_evidence(self) -> None:
        model = LearnedRetentionModel.default()
        candidates = [
            item(
                1.0,
                "CASE_ID: incident-001 FACT_KIND: root_cause FACT_VALUE: stale cache SLOT: durable",
                "dossiers/incident-001-root-cause.md",
                tokens=12,
            ),
            item(
                2.0,
                "CASE_ID: incident-099 FACT_KIND: root_cause FACT_VALUE: wrong cache SLOT: distractor belongs to another investigation",
                "dossiers/incident-099-root-cause.md",
                tokens=12,
            ),
            item(
                3.0,
                "Duplicate: incident-001 stale cache but no patch file SLOT: duplicate",
                "dossiers/incident-001-duplicate.md",
                tokens=12,
            ),
        ]
        kept = LearnedRetentionPolicy(model=model).select(
            "For incident-001, what root cause should be retained?",
            candidates,
            budget=12,
        )
        self.assertEqual(kept[0].provenance, "dossiers/incident-001-root-cause.md")

    def test_learned_retention_model_round_trips(self) -> None:
        model = LearnedRetentionModel.default()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            model.save(path)
            loaded = LearnedRetentionModel.load(path)
        self.assertEqual(loaded.weights["query_all_overlap"], model.weights["query_all_overlap"])
        self.assertEqual(loaded.metadata["source"], "built_in_default")

    def test_learned_retention_model_rejects_schema_mismatch(self) -> None:
        model = LearnedRetentionModel.default()
        payload = model.to_payload()
        payload["version"] = -1
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "version mismatch"):
                LearnedRetentionModel.load(path)

    def test_pairwise_training_records_ranking_diagnostics(self) -> None:
        rows = [
            {
                "dataset": "mini",
                "seed": 0,
                "case": "case-1",
                "decision_id": "case-1:0",
                "label": True,
                "features": {"confidence": 1.0},
            },
            {
                "dataset": "mini",
                "seed": 0,
                "case": "case-1",
                "decision_id": "case-1:0",
                "label": False,
                "features": {"confidence": 0.0},
            },
        ]
        model = train_linear_retention_model(rows, objective="pairwise", epochs=2, learning_rate=0.05)
        self.assertEqual(model.metadata["objective"], "pairwise")
        self.assertEqual(model.metadata["training_pairs"], 1)
        self.assertEqual(model.metadata["pairwise_accuracy_after"], 1.0)

    def test_pointwise_training_does_not_materialize_decision_pairs(self) -> None:
        rows = [
            {"dataset": "mini", "case": "case-1", "label": True, "features": {"confidence": 1.0}},
            {"dataset": "mini", "case": "case-1", "label": False, "features": {"confidence": 0.0}},
        ]
        model = train_linear_retention_model(rows, objective="pointwise", epochs=1)
        self.assertEqual(model.metadata["training_pairs"], 0)
        self.assertIsNone(model.metadata["pairwise_accuracy_before"])

    def test_pairwise_training_requires_decision_keys(self) -> None:
        rows = [
            {"dataset": "mini", "case": "case-1", "label": True, "features": {"confidence": 1.0}},
            {"dataset": "mini", "case": "case-1", "label": False, "features": {"confidence": 0.0}},
        ]
        with self.assertRaisesRegex(ValueError, "require decision_id or step"):
            train_linear_retention_model(rows, objective="pairwise", epochs=1)


if __name__ == "__main__":
    unittest.main()
