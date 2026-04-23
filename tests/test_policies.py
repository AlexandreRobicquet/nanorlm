from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanorlm import MemoryItem
from policies import KeepRecentPolicy, PairwiseTournamentPolicy, SingleCriticTopKPolicy, SummaryOnlyPolicy


class DummyJudge:
    def score_candidate(self, query: str, item: MemoryItem) -> float:
        score = 1.0 if item.metadata.get("pair_id") == "pair-000" else 0.0
        if item.metadata.get("fact_kind") == "right":
            score += 0.5
        return score

    def compare_candidates(self, query: str, left: MemoryItem, right: MemoryItem) -> int:
        left_score = self.score_candidate(query, left)
        right_score = self.score_candidate(query, right)
        if left_score == right_score:
            return 0
        return 1 if left_score > right_score else -1


def item(timestamp: float, pair_id: str, fact_kind: str, tokens: int = 12) -> MemoryItem:
    return MemoryItem(
        summary=f"{pair_id} {fact_kind}",
        provenance=f"{pair_id}/{fact_kind}",
        raw_pointer=f"{pair_id}/{fact_kind}",
        tokens=tokens,
        depth=1,
        timestamp=timestamp,
        metadata={"pair_id": pair_id, "fact_kind": fact_kind},
    )


class PolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = [
            item(1.0, "pair-000", "left"),
            item(2.0, "pair-000", "right"),
            item(3.0, "pair-111", "left"),
            item(4.0, "pair-111", "right"),
        ]
        self.query = "What is the code for pair-000? Combine left and right."

    def test_keep_recent_respects_budget(self) -> None:
        kept = KeepRecentPolicy().select(self.query, self.candidates, budget=24)
        self.assertLessEqual(sum(candidate.tokens for candidate in kept), 24)

    def test_summary_only_drops_metadata(self) -> None:
        kept = SummaryOnlyPolicy().select(self.query, self.candidates, budget=24)
        self.assertTrue(all(not candidate.metadata for candidate in kept))

    def test_single_critic_prefers_higher_scoring_candidates(self) -> None:
        kept = SingleCriticTopKPolicy(judge=DummyJudge()).select(self.query, self.candidates, budget=24)
        self.assertTrue(any(candidate.metadata["pair_id"] == "pair-000" for candidate in kept))

    def test_pairwise_respects_budget(self) -> None:
        kept = PairwiseTournamentPolicy(judge=DummyJudge(), seed=0).select(self.query, self.candidates, budget=24)
        self.assertLessEqual(sum(candidate.tokens for candidate in kept), 24)
        self.assertTrue(kept)

    def test_pairwise_prefers_higher_scored_candidates(self) -> None:
        kept = PairwiseTournamentPolicy(judge=DummyJudge(), seed=0).select(self.query, self.candidates, budget=24)
        self.assertTrue(any(candidate.metadata["pair_id"] == "pair-000" for candidate in kept))


if __name__ == "__main__":
    unittest.main()
