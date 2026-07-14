"""Capture leaf inspections once and replay them across retention policies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from nanorlm import AnswerResult, Backend, ContextBlock, InspectionResult, MemoryItem, Usage


STORE_FORMAT = "nanorlm-inspection-replay"
STORE_VERSION = 1
REPLAY_MODES = ("capture_or_replay", "replay_only")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def inspection_request(
    query: str,
    documents: Sequence[ContextBlock],
    depth: int,
    branch: str,
    namespace: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a portable request fingerprint without storing raw benchmark context."""

    return {
        "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        "depth": depth,
        "branch": branch,
        "namespace": dict(namespace or {}),
        "documents": [
            {
                "name": document.name,
                "text_sha256": hashlib.sha256(document.text.encode("utf-8")).hexdigest(),
            }
            for document in documents
        ],
    }


def inspection_key(
    query: str,
    documents: Sequence[ContextBlock],
    depth: int,
    branch: str,
    namespace: Mapping[str, Any] | None = None,
) -> str:
    return f"inspect_{_sha256(inspection_request(query, documents, depth, branch, namespace))[:24]}"


def _usage_payload(usage: Usage) -> dict[str, int]:
    return {
        "prompt_tokens": int(usage.prompt_tokens),
        "completion_tokens": int(usage.completion_tokens),
        "calls": int(usage.calls),
    }


def _result_payload(result: InspectionResult) -> dict[str, Any]:
    metadata = dict(result.metadata)
    metadata.pop("inspection_replay", None)
    return {
        "summary": result.summary,
        "evidence": list(result.evidence),
        "answer_candidate": result.answer_candidate,
        "confidence": float(result.confidence),
        "metadata": metadata,
        "usage": _usage_payload(result.usage),
    }


class InspectionReplayBackend:
    """Backend adapter with content-addressed, integrity-checked inspection replay."""

    def __init__(
        self,
        backend: Backend,
        store_path: str | Path,
        *,
        mode: str = "capture_or_replay",
        namespace: Mapping[str, Any] | None = None,
    ) -> None:
        if mode not in REPLAY_MODES:
            raise ValueError(f"unknown inspection replay mode: {mode}")
        self.backend = backend
        self.store_path = Path(store_path)
        self.mode = mode
        self.namespace = dict(namespace or {"backend_type": type(backend).__name__})
        self._records = self._load_records()
        self._captured = 0
        self._replayed = 0

    def inspect(
        self,
        query: str,
        documents: Sequence[ContextBlock],
        depth: int,
        branch: str,
    ) -> InspectionResult:
        request = inspection_request(query, documents, depth, branch, self.namespace)
        key = f"inspect_{_sha256(request)[:24]}"
        record = self._records.get(key)
        if record is not None:
            if record.get("request") != request:
                raise ValueError(f"inspection replay request mismatch for {key}")
            self._replayed += 1
            return self._materialize(key, record, replayed=True)
        if self.mode == "replay_only":
            raise RuntimeError(f"inspection replay miss for {key} in {self.store_path}")

        result = self.backend.inspect(query, documents, depth, branch)
        result_payload = _result_payload(result)
        record_without_hash = {"request": request, "result": result_payload}
        record = {**record_without_hash, "record_sha256": _sha256(record_without_hash)}
        self._records[key] = record
        self._captured += 1
        self._write_records()
        return self._materialize(key, record, replayed=False)

    def answer(self, query: str, memory: Sequence[MemoryItem]) -> AnswerResult:
        return self.backend.answer(query, memory)

    def score_candidate(self, query: str, item: MemoryItem) -> float:
        return self.backend.score_candidate(query, item)

    def compare_candidates(self, query: str, left: MemoryItem, right: MemoryItem) -> int:
        return self.backend.compare_candidates(query, left, right)

    def drain_usage(self) -> Usage:
        drain = getattr(self.backend, "drain_usage", None)
        if not callable(drain):
            return Usage()
        return drain()

    def response_model_identifiers(self) -> list[str]:
        identifiers = getattr(self.backend, "response_model_identifiers", None)
        if not callable(identifiers):
            return []
        return list(identifiers())

    def replay_stats(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "captured": self._captured,
            "replayed": self._replayed,
            "records": len(self._records),
            "namespace": dict(self.namespace),
            "store_file": self.store_path.name,
            "store_sha256": self._store_sha256(),
        }

    def _materialize(self, key: str, record: Mapping[str, Any], *, replayed: bool) -> InspectionResult:
        expected_hash = str(record.get("record_sha256", ""))
        record_without_hash = {name: value for name, value in record.items() if name != "record_sha256"}
        actual_hash = _sha256(record_without_hash)
        if not expected_hash or actual_hash != expected_hash:
            raise ValueError(f"inspection replay integrity check failed for {key}")
        payload = record.get("result")
        if not isinstance(payload, Mapping):
            raise ValueError(f"inspection replay result is malformed for {key}")
        usage_payload = payload.get("usage")
        usage_values = usage_payload if isinstance(usage_payload, Mapping) else {}
        metadata_payload = payload.get("metadata")
        metadata = dict(metadata_payload) if isinstance(metadata_payload, Mapping) else {}
        metadata["inspection_replay"] = {
            "key": key,
            "replayed": replayed,
            "record_sha256": expected_hash,
            "store_version": STORE_VERSION,
        }
        evidence = payload.get("evidence")
        return InspectionResult(
            summary=str(payload.get("summary", "")),
            evidence=[str(item) for item in evidence] if isinstance(evidence, list) else [],
            answer_candidate=str(payload.get("answer_candidate", "")),
            confidence=float(payload.get("confidence", 0.0)),
            metadata=metadata,
            usage=Usage(
                prompt_tokens=int(usage_values.get("prompt_tokens", 0)),
                completion_tokens=int(usage_values.get("completion_tokens", 0)),
                calls=int(usage_values.get("calls", 0)),
            ),
        )

    def _load_records(self) -> dict[str, dict[str, Any]]:
        if not self.store_path.exists():
            return {}
        payload = json.loads(self.store_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"inspection replay store must be a JSON object: {self.store_path}")
        if payload.get("format") != STORE_FORMAT or payload.get("version") != STORE_VERSION:
            raise ValueError(f"unsupported inspection replay store: {self.store_path}")
        raw_records = payload.get("records")
        if not isinstance(raw_records, Mapping):
            raise ValueError(f"inspection replay store has no records object: {self.store_path}")
        records: dict[str, dict[str, Any]] = {}
        for key, value in raw_records.items():
            if not isinstance(value, Mapping):
                raise ValueError(f"inspection replay record is malformed: {key}")
            records[str(key)] = dict(value)
        return records

    def _write_records(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": STORE_FORMAT,
            "version": STORE_VERSION,
            "records": self._records,
        }
        temporary_path = self.store_path.with_suffix(f"{self.store_path.suffix}.tmp")
        temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary_path.replace(self.store_path)

    def _store_sha256(self) -> str | None:
        if not self.store_path.exists():
            return None
        return hashlib.sha256(self.store_path.read_bytes()).hexdigest()
