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
        kept = SingleCriticTopKPolicy(judge=DummyJudge()).select(self.query, self.candidates, budget=24)
        self.assertTrue(any("cache" in candidate.summary for candidate in kept))

    def test_pairwise_respects_budget(self) -> None:
        kept = PairwiseTournamentPolicy(judge=DummyJudge(), seed=0).select(self.query, self.candidates, budget=24)
        self.assertLessEqual(sum(candidate.tokens for candidate in kept), 24)
        self.assertTrue(kept)

    def test_pairwise_prefers_higher_scored_candidates(self) -> None:
        kept = PairwiseTournamentPolicy(judge=DummyJudge(), seed=0).select(self.query, self.candidates, budget=24)
        self.assertTrue(any("cache" in candidate.summary for candidate in kept))


if __name__ == "__main__":
    unittest.main()
