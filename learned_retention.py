from __future__ import annotations

import json
import math
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from nanorlm import MemoryItem, estimate_tokens, normalize_text, query_terms


MODEL_ENV_VAR = "NANORLM_LEARNED_RETENTION_MODEL"
MODEL_VERSION = 1
IDENTIFIER_RE = re.compile(r"\b[a-z][a-z0-9_/-]*-\d{2,}\b", flags=re.IGNORECASE)

FEATURE_NAMES = [
    "query_summary_overlap",
    "query_answer_overlap",
    "query_provenance_overlap",
    "query_all_overlap",
    "identifier_overlap",
    "fact_marker",
    "durable_slot",
    "distractor_slot",
    "duplicate_slot",
    "confidence",
    "answer_candidate_present",
    "token_efficiency",
    "token_pressure",
    "depth",
    "source_path_count",
    "novelty",
    "redundancy",
]

DEFAULT_WEIGHTS = {
    "query_summary_overlap": 1.6,
    "query_answer_overlap": 1.2,
    "query_provenance_overlap": 0.9,
    "query_all_overlap": 2.2,
    "identifier_overlap": 2.6,
    "fact_marker": 0.45,
    "durable_slot": 0.7,
    "distractor_slot": -2.2,
    "duplicate_slot": -0.6,
    "confidence": 0.6,
    "answer_candidate_present": 0.35,
    "token_efficiency": 0.5,
    "token_pressure": -1.1,
    "depth": -0.1,
    "source_path_count": 0.15,
    "novelty": 0.65,
    "redundancy": -0.75,
}


def _clip(value: float, upper: float) -> float:
    return max(0.0, min(upper, value))


def _scaled_count(value: int, upper: int) -> float:
    if upper <= 0:
        return 0.0
    return _clip(float(value), float(upper)) / float(upper)


def _metadata_text(metadata: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            parts.append(f"{key} {value}")
        elif isinstance(value, list):
            parts.append(f"{key} {' '.join(str(item) for item in value[:8])}")
    return " ".join(parts)


def _identifier_overlap(query: str, item: MemoryItem) -> int:
    query_ids = {match.group(0).lower() for match in IDENTIFIER_RE.finditer(query)}
    item_ids = {
        match.group(0).lower()
        for match in IDENTIFIER_RE.finditer(f"{item.provenance} {item.summary} {item.answer_candidate}")
    }
    return len(query_ids & item_ids)


def _has_any(text: str, needles: Iterable[str]) -> bool:
    lower = text.lower()
    return any(needle in lower for needle in needles)


def _source_path_count(item: MemoryItem) -> int:
    paths = item.metadata.get("source_paths", [])
    if isinstance(paths, list):
        return len(paths)
    return 0


def retention_features(
    query: str,
    item: MemoryItem,
    budget: int,
    covered_terms: set[str] | None = None,
) -> dict[str, float]:
    covered = covered_terms or set()
    metadata_blob = _metadata_text(item.metadata)
    summary_terms = query_terms(item.summary)
    answer_terms = query_terms(item.answer_candidate)
    provenance_terms = query_terms(item.provenance)
    metadata_terms = query_terms(metadata_blob)
    all_terms = summary_terms | answer_terms | provenance_terms | metadata_terms
    query_set = query_terms(query)
    item_blob = f"{item.provenance}\n{item.summary}\n{item.answer_candidate}\n{metadata_blob}"
    tokens = max(1, item.tokens or estimate_tokens(item.summary))
    budget_floor = max(1, budget)
    return {
        "query_summary_overlap": _scaled_count(len(query_set & summary_terms), 8),
        "query_answer_overlap": _scaled_count(len(query_set & answer_terms), 8),
        "query_provenance_overlap": _scaled_count(len(query_set & provenance_terms), 8),
        "query_all_overlap": _scaled_count(len(query_set & all_terms), 12),
        "identifier_overlap": _scaled_count(_identifier_overlap(query, item), 3),
        "fact_marker": 1.0
        if _has_any(item_blob, ["fact_kind", "fact_value", "pair_id", "case_id", "ruler_id", "babilong_id"])
        else 0.0,
        "durable_slot": 1.0 if _has_any(item_blob, ["slot: durable", "slot durable"]) else 0.0,
        "distractor_slot": 1.0
        if _has_any(item_blob, ["slot: distractor", "slot distractor", "belongs to another"])
        else 0.0,
        "duplicate_slot": 1.0 if _has_any(item_blob, ["slot: duplicate", "slot duplicate", "duplicate:"]) else 0.0,
        "confidence": _clip(float(item.confidence or 0.0), 1.0),
        "answer_candidate_present": 1.0 if item.answer_candidate else 0.0,
        "token_efficiency": min(1.0, 1.0 / math.sqrt(float(tokens))),
        "token_pressure": min(2.0, float(tokens) / float(budget_floor)) / 2.0,
        "depth": min(1.0, float(max(0, item.depth)) / 6.0),
        "source_path_count": _scaled_count(_source_path_count(item), 4),
        "novelty": _scaled_count(len(all_terms - covered), 12),
        "redundancy": _scaled_count(len(all_terms & covered), 12),
    }


@dataclass(slots=True)
class LearnedRetentionModel:
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    intercept: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "LearnedRetentionModel":
        return cls(metadata={"source": "built_in_default", "version": MODEL_VERSION})

    @classmethod
    def load(cls, path: str | Path) -> "LearnedRetentionModel":
        payload = json.loads(Path(path).read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"learned retention model must be a JSON object: {path}")
        version = int(payload.get("version", -1))
        if version != MODEL_VERSION:
            raise ValueError(
                f"learned retention model version mismatch for {path}: expected {MODEL_VERSION}, got {version}"
            )
        feature_names = payload.get("feature_names")
        if feature_names != FEATURE_NAMES:
            raise ValueError(f"learned retention model feature set mismatch: {path}")
        weights = {str(key): float(value) for key, value in payload.get("weights", {}).items()}
        if not weights:
            raise ValueError(f"learned retention model has no weights: {path}")
        missing_weights = [name for name in FEATURE_NAMES if name not in weights]
        if missing_weights:
            raise ValueError(
                f"learned retention model is missing weight(s): {', '.join(missing_weights)}"
            )
        metadata = dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), dict) else {}
        metadata.setdefault("model_path", str(path))
        return cls(weights=weights, intercept=float(payload.get("intercept", 0.0)), metadata=metadata)

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": MODEL_VERSION,
            "feature_names": FEATURE_NAMES,
            "intercept": self.intercept,
            "weights": {name: round(float(self.weights.get(name, 0.0)), 8) for name in FEATURE_NAMES},
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_payload(), indent=2, sort_keys=True) + "\n")

    def score_features(self, features: dict[str, float]) -> float:
        return self.intercept + sum(self.weights.get(name, 0.0) * features.get(name, 0.0) for name in FEATURE_NAMES)

    def score(self, query: str, item: MemoryItem, budget: int, covered_terms: set[str] | None = None) -> float:
        return self.score_features(retention_features(query, item, budget, covered_terms))


def load_learned_retention_model(path: str | Path | None = None) -> LearnedRetentionModel:
    resolved_path = str(path or os.environ.get(MODEL_ENV_VAR, "")).strip()
    if resolved_path:
        return LearnedRetentionModel.load(resolved_path)
    return LearnedRetentionModel.default()


def train_linear_retention_model(
    rows: Sequence[dict[str, Any]],
    *,
    epochs: int = 20,
    learning_rate: float = 0.15,
    l2: float = 0.0005,
    seed: int = 0,
) -> LearnedRetentionModel:
    weights = dict(DEFAULT_WEIGHTS)
    intercept = 0.0
    rng = random.Random(seed)
    training_rows = [row for row in rows if "label" in row and isinstance(row.get("features"), dict)]
    if not training_rows:
        raise ValueError("cannot train learned retention model without labeled feature rows")

    for _epoch in range(max(1, epochs)):
        shuffled = list(training_rows)
        rng.shuffle(shuffled)
        for row in shuffled:
            features = {name: float(row["features"].get(name, 0.0)) for name in FEATURE_NAMES}
            label = 1.0 if row.get("label") else 0.0
            linear = intercept + sum(weights.get(name, 0.0) * features[name] for name in FEATURE_NAMES)
            prediction = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, linear))))
            class_weight = 2.0 if label else 1.0
            error = (label - prediction) * class_weight
            intercept += learning_rate * error
            for name in FEATURE_NAMES:
                weights[name] = (weights.get(name, 0.0) * (1.0 - learning_rate * l2)) + learning_rate * error * features[name]

    positives = sum(1 for row in training_rows if row.get("label"))
    negatives = len(training_rows) - positives
    return LearnedRetentionModel(
        weights=weights,
        intercept=intercept,
        metadata={
            "source": "offline_trace_training",
            "version": MODEL_VERSION,
            "training_rows": len(training_rows),
            "positive_rows": positives,
            "negative_rows": negatives,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "l2": l2,
            "seed": seed,
        },
    )


def item_terms(item: MemoryItem) -> set[str]:
    return query_terms(f"{item.provenance} {item.summary} {item.answer_candidate} {_metadata_text(item.metadata)}")


class LearnedRetentionPolicy:
    name = "learned_retention"

    def __init__(self, model: LearnedRetentionModel | None = None, model_path: str | Path | None = None) -> None:
        self.model = model or load_learned_retention_model(model_path)

    def select(self, root_query: str, candidates: Sequence[MemoryItem], budget: int) -> list[MemoryItem]:
        if not candidates:
            return []
        if sum(item.tokens for item in candidates) <= budget:
            return self._dedupe(candidates)

        kept: list[MemoryItem] = []
        used = 0
        covered_terms: set[str] = set()
        remaining = self._dedupe(candidates)
        while remaining:
            feasible = [item for item in remaining if used + item.tokens <= budget]
            if not feasible:
                break
            best = max(
                feasible,
                key=lambda item: (
                    self.model.score(root_query, item, budget, covered_terms),
                    -item.tokens,
                    item.timestamp,
                ),
            )
            kept.append(best)
            used += best.tokens
            covered_terms.update(item_terms(best))
            remaining.remove(best)

        if kept:
            return kept
        return [min(candidates, key=lambda item: item.tokens)]

    def _dedupe(self, candidates: Sequence[MemoryItem]) -> list[MemoryItem]:
        seen: set[tuple[str, str]] = set()
        deduped: list[MemoryItem] = []
        for item in candidates:
            key = (item.provenance, normalize_text(item.summary))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped
