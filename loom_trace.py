"""Export nanoRLM benchmark runs to the frozen LOOM trace contract v0.1."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from nanorlm import MemoryItem, RLMResult, memory_item_record


SCHEMA_VERSION = "0.1"
BRIDGE_VERSION = "0.1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_id(prefix: str, value: Any) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def _memory_content(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the deterministic, policy-independent content of one memory item."""

    metadata = record.get("metadata")
    metadata_dict = dict(metadata) if isinstance(metadata, Mapping) else {}
    replay = metadata_dict.get("inspection_replay")
    replay_dict = dict(replay) if isinstance(replay, Mapping) else {}
    return {
        "summary": str(record.get("summary", "")),
        "provenance": str(record.get("provenance", "")),
        "raw_pointer": str(record.get("raw_pointer", "")),
        "tokens": int(record.get("tokens", 0)),
        "depth": int(record.get("depth", 0)),
        "answer_candidate": str(record.get("answer_candidate", "")),
        "confidence": float(record.get("confidence", 0.0)),
        "source_paths": sorted(str(path) for path in metadata_dict.get("source_paths", [])),
        "block_names": [str(name) for name in metadata_dict.get("block_names", [])],
        "inspection_replay_key": str(replay_dict.get("key", "")),
    }


def memory_artifact_id(record: Mapping[str, Any] | MemoryItem) -> str:
    """Return a stable artifact id that excludes timestamps and policy scores."""

    raw_record = memory_item_record(record) if isinstance(record, MemoryItem) else dict(record)
    return _stable_id("art_memory", _memory_content(raw_record))


def _budget(value: Mapping[str, Any] | None = None) -> dict[str, int]:
    source = value or {}
    return {
        "prompt_tokens": max(0, int(source.get("prompt_tokens", 0))),
        "completion_tokens": max(0, int(source.get("completion_tokens", 0))),
        "calls": max(0, int(source.get("calls", 0))),
        "wall_ms": max(0, int(round(float(source.get("wall_ms", 0))))),
    }


class _TraceBuilder:
    def __init__(self, *, task_id: str, run_id: str, started_at: datetime | None = None) -> None:
        self.task_id = task_id
        self.run_id = run_id
        self.started_at = started_at or datetime.now(timezone.utc)
        if self.started_at.tzinfo is None:
            self.started_at = self.started_at.replace(tzinfo=timezone.utc)
        self.events: list[dict[str, Any]] = []

    def emit(self, event_type: str, **payload: Any) -> dict[str, Any]:
        index = len(self.events) + 1
        timestamp = self.started_at.astimezone(timezone.utc) + timedelta(microseconds=index)
        event = {
            "event_id": f"evt_{index:06d}_{event_type}",
            "event_type": event_type,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
            "schema_version": SCHEMA_VERSION,
            **payload,
        }
        self.events.append(event)
        return event


def build_loom_trace(
    result: RLMResult,
    *,
    dataset: str,
    case_name: str,
    query: str,
    policy: str,
    provider: str,
    model: str,
    seed: int,
    budget_tokens: int,
    answer_score: float,
    provenance_score: float,
    expected_answer: str,
    expected_provenance: Sequence[str],
    started_at: datetime | None = None,
) -> list[dict[str, Any]]:
    """Build one complete, cross-reference-safe LOOM v0.1 event stream."""

    task_id = _stable_id("task", {"dataset": dataset, "case": case_name, "query": query})
    trace_started_at = started_at or datetime.now(timezone.utc)
    if trace_started_at.tzinfo is None:
        trace_started_at = trace_started_at.replace(tzinfo=timezone.utc)

    inspection_records: dict[str, dict[str, Any]] = {}
    decision_candidates: list[list[dict[str, Any]]] = []
    available_artifact_ids: set[str] = set()
    for decision in result.retention_decisions:
        candidates: list[dict[str, Any]] = []
        for raw_candidate in decision.get("candidates", []):
            if not isinstance(raw_candidate, Mapping):
                continue
            raw_input = raw_candidate.get("input_item")
            input_record = raw_input if isinstance(raw_input, Mapping) else raw_candidate
            input_content = _memory_content(input_record)
            input_id = memory_artifact_id(input_record)
            if input_id not in available_artifact_ids:
                inspection_records.setdefault(input_id, input_content)
                available_artifact_ids.add(input_id)

            selected = bool(raw_candidate.get("selected"))
            output_content = _memory_content(raw_candidate) if selected else None
            output_id = memory_artifact_id(raw_candidate) if selected else None
            if output_id is not None:
                available_artifact_ids.add(output_id)
            candidates.append(
                {
                    "input_id": input_id,
                    "output_id": output_id,
                    "output_content": output_content,
                    "candidate": raw_candidate,
                }
            )
        decision_candidates.append(candidates)

    for item in result.kept_items:
        artifact_id = memory_artifact_id(item)
        if artifact_id not in available_artifact_ids:
            inspection_records.setdefault(artifact_id, _memory_content(memory_item_record(item)))
            available_artifact_ids.add(artifact_id)

    final_parent_ids = list(dict.fromkeys(memory_artifact_id(item) for item in result.kept_items))
    run_id = _stable_id(
        "run",
        {
            "task_id": task_id,
            "policy": policy,
            "provider": provider,
            "model": model,
            "seed": seed,
            "budget_tokens": budget_tokens,
            "answer": result.answer,
            "memory_artifact_ids": sorted(available_artifact_ids),
            "started_at": trace_started_at.astimezone(timezone.utc).isoformat(),
        },
    )
    builder = _TraceBuilder(task_id=task_id, run_id=run_id, started_at=trace_started_at)

    inspect_stage_id = "stage_inspect_000"
    replay_keys = sorted(
        {
            str(record.get("inspection_replay_key"))
            for record in inspection_records.values()
            if record.get("inspection_replay_key")
        }
    )
    replay_stats = result.retention_stats.get("inspection_replay", {})
    replay_payload = {
        "keys": replay_keys,
        "enabled": bool(replay_keys),
    }
    if isinstance(replay_stats, Mapping) and replay_stats:
        replay_payload.update(
            {
                "mode": replay_stats.get("mode"),
                "captured": int(replay_stats.get("captured", 0)),
                "replayed": int(replay_stats.get("replayed", 0)),
                "store_sha256": replay_stats.get("store_sha256"),
            }
        )
    builder.emit(
        "stage_start",
        stage_id=inspect_stage_id,
        round_index=0,
        stage_name="inspect",
        inputs={"query": query, "context_policy": "recursive_leaf_inspection"},
        budget_request={"memory_tokens": budget_tokens},
        seed=seed,
        replay=replay_payload,
    )
    for artifact_id, content in inspection_records.items():
        builder.emit(
            "artifact_emit",
            stage_id=inspect_stage_id,
            round_index=0,
            artifact_id=artifact_id,
            artifact_type="nanorlm.memory_item",
            content=content,
            parents=[],
        )
    stage_budgets = result.stage_budgets if isinstance(result.stage_budgets, Mapping) else {}
    builder.emit(
        "stage_end",
        stage_id=inspect_stage_id,
        round_index=0,
        stage_name="inspect",
        outputs={"artifact_ids": list(inspection_records)},
        budget_used=_budget(stage_budgets.get("inspect")),
        status="ok",
        error=None,
    )
    emitted_artifact_ids = set(inspection_records)

    for decision_index, decision in enumerate(result.retention_decisions):
        stage_id = f"stage_retain_{decision_index:03d}"
        candidates = decision_candidates[decision_index]
        candidate_ids = list(dict.fromkeys(str(row["input_id"]) for row in candidates))
        selected = [
            row
            for row in candidates
            if bool(row["candidate"].get("selected"))
        ]
        selected.sort(
            key=lambda row: (
                int(row["candidate"].get("selection_rank"))
                if row["candidate"].get("selection_rank") is not None
                else len(candidates)
            )
        )
        selected_ids = list(
            dict.fromkeys(str(row["output_id"] or row["input_id"]) for row in selected)
        )
        round_index = int(decision.get("decision_index", decision_index))
        builder.emit(
            "stage_start",
            stage_id=stage_id,
            round_index=round_index,
            stage_name="retain",
            inputs={
                "candidate_ids": candidate_ids,
                "policy": policy,
                "branch": str(decision.get("branch", "")),
                "depth": int(decision.get("depth", 0)),
            },
            budget_request={"memory_tokens": int(decision.get("budget", budget_tokens))},
            seed=seed,
        )
        builder.emit(
            "retrieval_query",
            stage_id=stage_id,
            round_index=round_index,
            query={"text": query, "policy": policy},
            candidate_ids=candidate_ids,
            retriever={"name": policy, "version": BRIDGE_VERSION},
        )
        for rank, row in enumerate(selected):
            candidate = row["candidate"]
            event: dict[str, Any] = {
                "stage_id": stage_id,
                "round_index": round_index,
                "artifact_id": row["input_id"],
                "usage": "selected_for_memory",
                "rank": rank,
            }
            score = candidate.get("score")
            if isinstance(score, int | float) and not isinstance(score, bool):
                event["score"] = float(score)
            builder.emit("retrieval_use", **event)
        for row in selected:
            input_id = str(row["input_id"])
            output_id = str(row["output_id"] or input_id)
            if output_id == input_id or output_id in emitted_artifact_ids:
                continue
            builder.emit(
                "artifact_emit",
                stage_id=stage_id,
                round_index=round_index,
                artifact_id=output_id,
                artifact_type="nanorlm.memory_item.retained",
                content=row["output_content"],
                parents=[input_id],
            )
            emitted_artifact_ids.add(output_id)
        builder.emit(
            "stage_end",
            stage_id=stage_id,
            round_index=round_index,
            stage_name="retain",
            outputs={"selected_artifact_ids": selected_ids},
            budget_used=_budget(decision.get("budget_used")),
            status="ok",
            error=None,
        )

    answer_stage_id = "stage_final_answer_000"
    builder.emit(
        "stage_start",
        stage_id=answer_stage_id,
        round_index=0,
        stage_name="final_answer",
        inputs={"query": query, "retained_artifact_ids": final_parent_ids},
        seed=seed,
    )
    answer_artifact_id = _stable_id(
        "art_answer",
        {"task_id": task_id, "policy": policy, "answer": result.answer, "parents": final_parent_ids},
    )
    builder.emit(
        "artifact_emit",
        stage_id=answer_stage_id,
        round_index=0,
        artifact_id=answer_artifact_id,
        artifact_type="nanorlm.final_answer",
        content={"answer": result.answer},
        parents=final_parent_ids,
    )
    builder.emit(
        "stage_end",
        stage_id=answer_stage_id,
        round_index=0,
        stage_name="final_answer",
        outputs={"answer_artifact_id": answer_artifact_id},
        budget_used=_budget(stage_budgets.get("final_answer")),
        status="ok",
        error=None,
    )
    builder.emit(
        "outcome",
        target_id=task_id,
        score=float(answer_score),
        source="verifier",
        metadata={
            "bridge_version": BRIDGE_VERSION,
            "dataset": dataset,
            "case_name": case_name,
            "policy": policy,
            "provider": provider,
            "model": model,
            "seed": seed,
            "memory_budget_tokens": budget_tokens,
            "expected_answer": expected_answer,
            "expected_provenance": list(expected_provenance),
            "exact_match": float(answer_score),
            "provenance_score": float(provenance_score),
            "answer_artifact_id": answer_artifact_id,
        },
    )
    return builder.events


def write_loom_trace(path: str | Path, events: Sequence[Mapping[str, Any]]) -> Path:
    """Write a LOOM event stream as deterministic-key-order JSONL."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(f"{json.dumps(dict(event), sort_keys=True)}\n" for event in events),
        encoding="utf-8",
    )
    return output_path
