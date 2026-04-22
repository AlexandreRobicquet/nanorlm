from __future__ import annotations

import random
from dataclasses import replace
from typing import Protocol, Sequence

from nanorlm import MemoryItem, estimate_tokens, query_terms, truncate_words


class Judge(Protocol):
    def score_candidate(self, query: str, item: MemoryItem) -> float:
        ...

    def compare_candidates(self, query: str, left: MemoryItem, right: MemoryItem) -> int:
        ...


class RetentionPolicy:
    name = "base"

    def select(self, root_query: str, candidates: Sequence[MemoryItem], budget: int) -> list[MemoryItem]:
        raise NotImplementedError


def _fit_budget(items: Sequence[MemoryItem], budget: int, reserve_recent: bool = False) -> list[MemoryItem]:
    if not items:
        return []
    kept: list[MemoryItem] = []
    used = 0
    recent = max(items, key=lambda item: item.timestamp) if reserve_recent else None
    for item in items:
        if used + item.tokens > budget and kept:
            continue
        kept.append(item)
        used += item.tokens
    if reserve_recent and recent is not None and recent not in kept:
        if recent.tokens <= budget:
            while kept and sum(item.tokens for item in kept) + recent.tokens > budget:
                kept.pop()
            kept.append(recent)
    if not kept:
        smallest = min(items, key=lambda item: item.tokens)
        return [smallest]
    seen: set[tuple[str, str]] = set()
    deduped: list[MemoryItem] = []
    for item in kept:
        key = (item.provenance, item.summary)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


class KeepRecentPolicy(RetentionPolicy):
    name = "keep_recent"

    def select(self, root_query: str, candidates: Sequence[MemoryItem], budget: int) -> list[MemoryItem]:
        ranked = sorted(candidates, key=lambda item: item.timestamp, reverse=True)
        return _fit_budget(ranked, budget, reserve_recent=True)


class SummaryOnlyPolicy(RetentionPolicy):
    name = "summary_only"

    def select(self, root_query: str, candidates: Sequence[MemoryItem], budget: int) -> list[MemoryItem]:
        if not candidates:
            return []
        per_item_words = max(8, budget // max(1, len(candidates)) // 2)
        compressed = [
            replace(
                item,
                summary=truncate_words(item.summary, per_item_words),
                answer_candidate="",
                metadata={},
                tokens=estimate_tokens(truncate_words(item.summary, per_item_words)),
            )
            for item in candidates
        ]
        ranked = sorted(compressed, key=lambda item: (item.depth, item.timestamp), reverse=True)
        return _fit_budget(ranked, budget, reserve_recent=True)


class SingleCriticTopKPolicy(RetentionPolicy):
    name = "single_critic_topk"

    def __init__(self, judge: Judge) -> None:
        self.judge = judge

    def select(self, root_query: str, candidates: Sequence[MemoryItem], budget: int) -> list[MemoryItem]:
        ranked = sorted(
            (replace(item, score=self.judge.score_candidate(root_query, item)) for item in candidates),
            key=lambda item: (-item.score, item.depth, -item.timestamp),
        )
        return _fit_budget(ranked, budget, reserve_recent=True)


class PairwiseTournamentPolicy(RetentionPolicy):
    name = "pairwise_tournament"

    def __init__(self, judge: Judge, seed: int = 0, rounds: int = 3) -> None:
        self.judge = judge
        self.random = random.Random(seed)
        self.rounds = rounds

    def select(self, root_query: str, candidates: Sequence[MemoryItem], budget: int) -> list[MemoryItem]:
        if not candidates:
            return []
        ranked = [
            replace(item, score=self.judge.score_candidate(root_query, item), wins=0, losses=0)
            for item in candidates
        ]
        ranked.sort(key=lambda item: (-item.score, -item.timestamp))
        for round_index in range(min(self.rounds, max(1, len(ranked) - 1))):
            if round_index % 2 == 1:
                self.random.shuffle(ranked)
            pairs = [ranked[index : index + 2] for index in range(0, len(ranked), 2)]
            for pair in pairs:
                if len(pair) == 1:
                    pair[0].wins += 1
                    continue
                left, right = pair
                winner = self.judge.compare_candidates(root_query, left, right)
                if winner > 0:
                    left.wins += 1
                    right.losses += 1
                elif winner < 0:
                    right.wins += 1
                    left.losses += 1
                else:
                    left.wins += 1
                    right.wins += 1
            ranked.sort(key=lambda item: (-item.wins, item.losses, -self._diversity_bonus(root_query, item), -item.score, -item.timestamp))
        return self._select_with_budget(ranked, budget)

    def _diversity_bonus(self, root_query: str, item: MemoryItem) -> float:
        tags = set(query_terms(item.provenance))
        return float(len(tags & query_terms(root_query)))

    def _select_with_budget(self, ranked: Sequence[MemoryItem], budget: int) -> list[MemoryItem]:
        kept: list[MemoryItem] = []
        used = 0
        recent = max(ranked, key=lambda item: item.timestamp)
        for item in ranked:
            if used + item.tokens > budget:
                continue
            kept.append(item)
            used += item.tokens
        if recent not in kept and recent.tokens <= budget - used:
            kept.append(recent)
        return kept or [min(ranked, key=lambda item: item.tokens)]


def build_policy(name: str, judge: Judge, seed: int = 0) -> RetentionPolicy:
    normalized = name.strip().lower()
    if normalized == "keep_recent":
        return KeepRecentPolicy()
    if normalized == "summary_only":
        return SummaryOnlyPolicy()
    if normalized == "single_critic_topk":
        return SingleCriticTopKPolicy(judge=judge)
    if normalized == "pairwise_tournament":
        return PairwiseTournamentPolicy(judge=judge, seed=seed)
    raise ValueError(f"unknown retention policy: {name}")
