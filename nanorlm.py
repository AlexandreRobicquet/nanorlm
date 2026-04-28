from __future__ import annotations

import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence


WORD_RE = re.compile(r"[A-Za-z0-9_./:-]+")
OPENAI_COMPATIBLE_DEFAULT_BASE_URL = "https://api.openai.com/v1"
ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com"


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(WORD_RE.findall(text)) * 1.3))


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[: max(1, max_words - 1)]) + " ..."


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def query_terms(text: str) -> set[str]:
    return {token.lower() for token in WORD_RE.findall(text) if len(token) >= 3}


def score_overlap(query: str, text: str) -> float:
    q = query_terms(query)
    t = query_terms(text)
    if not q or not t:
        return 0.0
    return float(len(q & t))


def slugify(text: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return clean or "item"


def extract_target_id(text: str) -> str | None:
    match = re.search(r"\b((?:pair|case|incident|migration|release|plan)[-_ ]?\d+)\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).replace(" ", "-").lower()


def item_target_id(item: "MemoryItem") -> str:
    for key in ("pair_id", "case_id"):
        value = str(item.metadata.get(key, "")).strip().lower()
        if value:
            return value
    return ""


def item_source_paths(item: "MemoryItem") -> list[str]:
    paths = item.metadata.get("source_paths", [])
    if isinstance(paths, list):
        return [str(path) for path in paths]
    return []


def memory_signature(item: "MemoryItem") -> tuple[str, str, str]:
    return (item.raw_pointer, item.provenance, item.summary)


def item_facts(item: "MemoryItem") -> dict[str, str]:
    facts = item.metadata.get("facts", {})
    if isinstance(facts, dict):
        return {
            str(key).lower(): str(value)
            for key, value in facts.items()
            if str(key).strip() and str(value).strip()
        }
    fact_kind = str(item.metadata.get("fact_kind", "")).strip().lower()
    fact_value = str(item.metadata.get("fact_value", "")).strip()
    if fact_kind and fact_value:
        return {fact_kind: fact_value}
    return {}


def query_fact_priority(query: str, item: "MemoryItem") -> float:
    query_lower = query.lower()
    priority = 0.0
    for fact_kind in item_facts(item):
        if fact_kind == "root_cause" and ("root cause" in query_lower or "blocker" in query_lower or "blocks" in query_lower):
            priority += 3.0
        elif fact_kind == "file" and "file" in query_lower:
            priority += 3.0
        elif fact_kind == "fix" and ("fix" in query_lower or "change" in query_lower or "made first" in query_lower):
            priority += 3.0
        elif fact_kind == "owner" and ("who" in query_lower or "owner" in query_lower):
            priority += 3.0
        elif fact_kind == "service" and ("service" in query_lower or "system" in query_lower):
            priority += 2.0
        elif fact_kind == "left" and "left" in query_lower:
            priority += 2.0
        elif fact_kind == "right" and "right" in query_lower:
            priority += 2.0
    return priority


def extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start == -1:
        raise ValueError("response did not contain JSON")
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])
    raise ValueError("response contained an unterminated JSON object")


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def normalize_provider_name(provider: str) -> str:
    return provider.strip().lower().replace("-", "_")


def is_local_base_url(base_url: str | None) -> bool:
    if not base_url:
        return False
    parsed = urllib.parse.urlparse(base_url)
    host = (parsed.hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "0.0.0.0"}


def resolve_provider(config: "RLMConfig") -> Literal["heuristic", "openai_compatible", "anthropic"]:
    provider = normalize_provider_name(config.provider)
    if provider != "auto":
        if provider not in {"heuristic", "openai_compatible", "anthropic"}:
            raise ValueError(f"unknown provider: {config.provider}")
        return provider  # type: ignore[return-value]
    if config.model.startswith("demo/"):
        return "heuristic"
    base_url = (config.base_url or "").strip().lower()
    if base_url:
        if "anthropic.com" in base_url:
            return "anthropic"
        return "openai_compatible"
    if config.model.lower().startswith("claude"):
        return "anthropic"
    return "openai_compatible"


def resolved_base_url(config: "RLMConfig", provider: str) -> str | None:
    if provider == "heuristic":
        return None
    if config.base_url:
        return config.base_url
    if provider == "anthropic":
        return ANTHROPIC_DEFAULT_BASE_URL
    return OPENAI_COMPATIBLE_DEFAULT_BASE_URL


def resolved_api_key(config: "RLMConfig", provider: str, base_url: str | None) -> str | None:
    if config.api_key is not None:
        return config.api_key
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY")
    if provider == "openai_compatible" and not is_local_base_url(base_url):
        return os.environ.get("OPENAI_API_KEY")
    return None


def extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text", "")
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        return "\n".join(parts).strip()
    if content is None:
        return ""
    return str(content)


def validate_required_keys(data: dict[str, Any], required_keys: Sequence[str]) -> dict[str, Any]:
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise ValueError(f"response JSON missing keys: {', '.join(missing)}")
    return data


def chunk_lines(text: str, max_lines: int = 48) -> list[str]:
    lines = text.splitlines()
    if not lines:
        return [text]
    chunks: list[str] = []
    for start in range(0, len(lines), max_lines):
        chunks.append("\n".join(lines[start : start + max_lines]))
    return chunks


@dataclass(slots=True)
class ContextBlock:
    name: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)


@dataclass(slots=True)
class MemoryItem:
    summary: str
    provenance: str
    raw_pointer: str
    tokens: int
    depth: int
    timestamp: float
    answer_candidate: str = ""
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    wins: int = 0
    losses: int = 0
    score: float = 0.0

    def clone(self, **updates: Any) -> "MemoryItem":
        return replace(self, **updates)


@dataclass(slots=True)
class TraceEvent:
    kind: str
    depth: int
    label: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


@dataclass(slots=True)
class TraceArtifact:
    events: list[TraceEvent]
    tree: str
    jsonl: str

    def write_jsonl(self, path: str | Path) -> None:
        Path(path).write_text(self.jsonl)

    def write_tree(self, path: str | Path) -> None:
        Path(path).write_text(self.tree)


@dataclass(slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    def add(self, prompt_tokens: int = 0, completion_tokens: int = 0, calls: int = 0) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.calls += calls


@dataclass(slots=True)
class InspectionResult:
    summary: str
    evidence: list[str]
    answer_candidate: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)


@dataclass(slots=True)
class AnswerResult:
    answer: str
    confidence: float
    usage: Usage = field(default_factory=Usage)


@dataclass(slots=True)
class RLMResult:
    answer: str
    trace: TraceArtifact
    usage: Usage
    cost_estimate: float
    kept_items: list[MemoryItem]
    retention_stats: dict[str, Any] = field(default_factory=dict)
    provenance_hits: list[str] = field(default_factory=list)
    drop_reasons: list[dict[str, Any]] = field(default_factory=list)
    per_step_budget: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class RLMConfig:
    model: str
    provider: Literal["auto", "heuristic", "openai_compatible", "anthropic"] = "auto"
    base_url: str | None = None
    api_key: str | None = None
    max_depth: int = 1
    max_steps: int = 64
    memory_budget_tokens: int = 320
    retention_policy: str = "pairwise_tournament"
    sandbox: str | None = None
    seed: int = 0


class RetentionPolicy(Protocol):
    name: str

    def select(self, root_query: str, candidates: Sequence[MemoryItem], budget: int) -> list[MemoryItem]:
        ...


class Backend(Protocol):
    def inspect(self, query: str, documents: Sequence[ContextBlock], depth: int, branch: str) -> InspectionResult:
        ...

    def answer(self, query: str, memory: Sequence[MemoryItem]) -> AnswerResult:
        ...

    def score_candidate(self, query: str, item: MemoryItem) -> float:
        ...

    def compare_candidates(self, query: str, left: MemoryItem, right: MemoryItem) -> int:
        ...


class TraceRecorder:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def emit(self, kind: str, depth: int, label: str, **payload: Any) -> None:
        self.events.append(TraceEvent(kind=kind, depth=depth, label=label, payload=payload))

    def artifact(self) -> TraceArtifact:
        tree_lines = []
        for event in self.events:
            indent = "  " * event.depth
            detail = ""
            if event.payload:
                compact = ", ".join(f"{key}={value}" for key, value in event.payload.items())
                detail = f" ({compact})"
            tree_lines.append(f"{indent}- [{event.kind}] {event.label}{detail}")
        jsonl = "\n".join(event.to_json() for event in self.events)
        return TraceArtifact(events=list(self.events), tree="\n".join(tree_lines), jsonl=jsonl)


def materialize_context(context: str | Sequence[ContextBlock | dict[str, Any] | tuple[str, str]]) -> list[ContextBlock]:
    if isinstance(context, str):
        return [ContextBlock(name="context.txt", text=context)]
    blocks: list[ContextBlock] = []
    for index, item in enumerate(context):
        if isinstance(item, ContextBlock):
            blocks.append(item)
        elif isinstance(item, dict):
            blocks.append(
                ContextBlock(
                    name=str(item.get("name", f"context-{index}")),
                    text=str(item.get("text", "")),
                    metadata=dict(item.get("metadata", {})),
                )
            )
        elif isinstance(item, tuple) and len(item) == 2:
            blocks.append(ContextBlock(name=str(item[0]), text=str(item[1])))
        else:
            raise TypeError(f"unsupported context item: {item!r}")
    return blocks


class HeuristicBackend:
    """Deterministic offline backend for tests, demos, and local development."""

    def __init__(self, seed: int = 0) -> None:
        self.random = random.Random(seed)

    def inspect(self, query: str, documents: Sequence[ContextBlock], depth: int, branch: str) -> InspectionResult:
        snippets: list[tuple[float, str]] = []
        metadata: dict[str, Any] = {}
        evidence: list[str] = []
        marker_groups: dict[str, dict[str, Any]] = {}
        query_target = extract_target_id(query)
        for document in documents:
            matches = self._salient_lines(query, document.text)
            document_score = matches[0][0] if matches else 0.0
            for score, line in matches[:2]:
                snippets.append((score, f"{document.name}: {line}"))
                evidence.append(f"{document.name}: {line}")
            markers = self._extract_markers(document.text)
            if markers:
                marker_target = (markers.get("pair_id") or markers.get("case_id") or document.name).strip().lower()
                target_bonus = 0.0
                if query_target and marker_target == query_target:
                    target_bonus += 8.0
                group = marker_groups.setdefault(marker_target, {"score": 0.0, "markers": []})
                group["score"] += document_score + target_bonus
                group["markers"].append(markers)
        snippets.sort(key=lambda item: (-item[0], item[1]))
        if marker_groups:
            ranked_groups = sorted(
                marker_groups.items(),
                key=lambda item: (
                    item[0] == (query_target or ""),
                    item[1]["score"],
                    len(self._merge_marker_group(item[1]["markers"]).get("facts", {})),
                ),
                reverse=True,
            )
            metadata = self._merge_marker_group(ranked_groups[0][1]["markers"])
        summary_parts = [snippet for _, snippet in snippets[:3]]
        if not summary_parts:
            joined_names = ", ".join(document.name for document in documents[:3])
            summary_parts.append(f"{joined_names}: no strong lexical match, returning leading context")
        summary = " | ".join(summary_parts)
        answer_candidate = metadata.get("answer", "")
        facts = metadata.get("facts", {})
        if not answer_candidate and isinstance(facts, dict) and facts:
            answer_candidate = self._answer_candidate_from_facts(facts)
        if not answer_candidate and summary_parts:
            answer_candidate = summary_parts[0].split(": ", 1)[-1]
        confidence = min(0.95, 0.2 + 0.15 * len(summary_parts))
        usage = Usage(prompt_tokens=sum(document.tokens for document in documents), completion_tokens=estimate_tokens(summary), calls=1)
        return InspectionResult(summary=summary, evidence=evidence[:6], answer_candidate=answer_candidate, confidence=confidence, metadata=metadata, usage=usage)

    def answer(self, query: str, memory: Sequence[MemoryItem]) -> AnswerResult:
        pair_answer = self._solve_pair_query(query, memory)
        if pair_answer:
            answer = pair_answer
        else:
            ranked = sorted(memory, key=lambda item: (-self.score_candidate(query, item), -item.timestamp))
            lines: list[str] = []
            for item in ranked[:3]:
                snippet = item.answer_candidate or item.summary
                if snippet:
                    lines.append(f"{item.provenance}: {snippet}")
            answer = "\n".join(lines) if lines else "I do not have enough retained evidence."
        usage = Usage(prompt_tokens=sum(item.tokens for item in memory), completion_tokens=estimate_tokens(answer), calls=1)
        return AnswerResult(answer=answer, confidence=0.65 if memory else 0.1, usage=usage)

    def score_candidate(self, query: str, item: MemoryItem) -> float:
        score = score_overlap(query, item.summary + " " + item.answer_candidate)
        target_id = item_target_id(item)
        query_target = extract_target_id(query)
        if target_id and query_target and target_id == query_target:
            score += 8.0
        facts = item_facts(item)
        if facts:
            score += min(4.0, 1.25 * len(facts))
            if target_id and query_target and target_id == query_target:
                score += 0.5 * len(facts)
        if item.metadata.get("fact_kind"):
            score += 1.0
        if item.metadata.get("answer"):
            score += 2.0
        if item.confidence:
            score += item.confidence
        return score

    def compare_candidates(self, query: str, left: MemoryItem, right: MemoryItem) -> int:
        left_score = self.score_candidate(query, left)
        right_score = self.score_candidate(query, right)
        query_target = extract_target_id(query)
        left_target = item_target_id(left)
        right_target = item_target_id(right)
        left_facts = item_facts(left)
        right_facts = item_facts(right)
        if query_target:
            if left_target == query_target and right_target != query_target:
                left_score += 6.0
            elif right_target == query_target and left_target != query_target:
                right_score += 6.0
            elif left_target == query_target and right_target == query_target:
                left_score += 0.4 * len(left_facts)
                right_score += 0.4 * len(right_facts)
        left_score += 1.5 * query_fact_priority(query, left)
        right_score += 1.5 * query_fact_priority(query, right)
        if item_target_id(left) and item_target_id(left) == item_target_id(right):
            if left.metadata.get("fact_kind") != right.metadata.get("fact_kind"):
                if left.tokens < right.tokens:
                    left_score += 0.25
                elif right.tokens < left.tokens:
                    right_score += 0.25
            if len(left_facts) != len(right_facts):
                if len(left_facts) > len(right_facts):
                    left_score += 0.5
                else:
                    right_score += 0.5
        if abs(left_score - right_score) < 0.1:
            if left.timestamp > right.timestamp:
                return 1
            if right.timestamp > left.timestamp:
                return -1
            return 0
        return 1 if left_score > right_score else -1

    def _merge_marker_group(self, markers_list: Sequence[dict[str, str]]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        facts: dict[str, str] = {}
        unknowns: list[str] = []
        for markers in markers_list:
            for key in ("pair_id", "case_id", "slot"):
                value = str(markers.get(key, "")).strip()
                if value and key not in merged:
                    merged[key] = value
            fact_kind = str(markers.get("fact_kind", "")).strip().lower()
            fact_value = str(markers.get("fact_value", "")).strip()
            if fact_kind and fact_value and fact_kind not in facts:
                facts[fact_kind] = fact_value
            answer = str(markers.get("answer", "")).strip()
            if answer:
                merged["answer"] = answer
            unknown = str(markers.get("unknown", "")).strip()
            if unknown:
                unknowns.append(unknown)
        if facts:
            merged["facts"] = facts
            if len(facts) == 1:
                fact_kind, fact_value = next(iter(facts.items()))
                merged["fact_kind"] = fact_kind
                merged["fact_value"] = fact_value
        if unknowns:
            merged["unknowns"] = unknowns
        return merged

    def _answer_candidate_from_facts(self, facts: dict[str, str]) -> str:
        if "left" in facts and "right" in facts:
            return f"{facts['left']} {facts['right']}"
        if "service" in facts and "root_cause" in facts:
            response = f"{facts['service']}: {facts['root_cause']}"
            if "fix" in facts:
                response += f" | fix: {facts['fix']}"
            if "file" in facts:
                response += f" | file: {facts['file']}"
            return response
        ordered_keys = ["root_cause", "fix", "file", "owner", "deadline", "risk"]
        ordered_values = [facts[key] for key in ordered_keys if key in facts]
        if ordered_values:
            return " | ".join(ordered_values)
        return " | ".join(facts.values())

    def _salient_lines(self, query: str, text: str) -> list[tuple[float, str]]:
        query_set = query_terms(query)
        matches: list[tuple[float, str]] = []
        for line in text.splitlines():
            clean = line.strip()
            if not clean:
                continue
            overlap = len(query_set & query_terms(clean))
            marker_bonus = 0.0
            lower = clean.lower()
            if "answer:" in lower:
                marker_bonus += 3.0
            if "fact_value" in lower or "fact_kind" in lower:
                marker_bonus += 1.5
            if "pair_id" in lower:
                marker_bonus += 1.0
            if "endpoint_id" in lower or "api_key_var" in lower or "api_base_url" in lower:
                marker_bonus += 1.0
            if overlap == 0 and marker_bonus == 0.0:
                continue
            matches.append((float(overlap) + marker_bonus + len(clean) / 500.0, clean))
        return sorted(matches, key=lambda item: (-item[0], item[1]))

    def _extract_markers(self, text: str) -> dict[str, str]:
        markers: dict[str, str] = {}
        patterns = {
            "pair_id": r"\bPAIR_ID\s*[:=]\s*([A-Za-z0-9._-]+)",
            "case_id": r"\bCASE_ID\s*[:=]\s*([A-Za-z0-9._-]+)",
            "fact_kind": r"\bFACT_KIND\s*[:=]\s*([A-Za-z0-9._-]+)",
            "fact_value": r"\bFACT_VALUE\s*[:=]\s*(.+)",
            "answer": r"\bANSWER\s*[:=]\s*(.+)",
            "slot": r"\bSLOT\s*[:=]\s*([A-Za-z0-9._-]+)",
            "unknown": r"\bUNKNOWN\s*[:=]\s*(.+)",
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, text, flags=re.MULTILINE)
            if match:
                markers[key] = match.group(1).strip()
        return markers

    def _solve_pair_query(self, query: str, memory: Sequence[MemoryItem]) -> str | None:
        target_case = extract_target_id(query)
        facts: dict[str, dict[str, str]] = {}
        for item in memory:
            pair_id = item_target_id(item)
            kind = str(item.metadata.get("fact_kind", "")).lower()
            value = str(item.metadata.get("fact_value", ""))
            if pair_id and kind and value:
                facts.setdefault(pair_id, {})[kind] = value
            for fact_kind, fact_value in item_facts(item).items():
                if pair_id:
                    facts.setdefault(pair_id, {})[fact_kind] = fact_value
            answer = item.metadata.get("answer")
            if answer and (not target_case or target_case == pair_id):
                return str(answer)
        if target_case and target_case in facts:
            pair_facts = facts[target_case]
            if "left" in pair_facts and "right" in pair_facts:
                return f"{pair_facts['left']} {pair_facts['right']}"
            if "service" in pair_facts and "root_cause" in pair_facts:
                response = f"{pair_facts['service']}: {pair_facts['root_cause']}"
                if "fix" in pair_facts:
                    response += f" | fix: {pair_facts['fix']}"
                if "file" in pair_facts:
                    response += f" | file: {pair_facts['file']}"
                return response
            ordered_keys = ["root_cause", "fix", "file", "owner", "deadline", "risk"]
            ordered_values = [pair_facts[key] for key in ordered_keys if key in pair_facts]
            if ordered_values:
                return " | ".join(ordered_values)
        return None


class StructuredOutputBackend:
    provider_name = "backend"

    def __init__(self, config: RLMConfig) -> None:
        self.config = config

    def inspect(self, query: str, documents: Sequence[ContextBlock], depth: int, branch: str) -> InspectionResult:
        joined = "\n\n".join(f"### {document.name}\n{document.text}" for document in documents)
        data, usage = self._chat_json(
            "inspect",
            (
                "You are a recursive language model worker. "
                "Read the provided branch context and return strict JSON with keys "
                "summary, evidence, answer_candidate, confidence. "
                "The summary should be terse and preserve only facts that help answer the root query."
            ),
            (
                f"Root query:\n{query}\n\n"
                f"Branch: {branch}\nDepth: {depth}\n\n"
                f"Context:\n{joined}\n\n"
                "Return JSON only."
            ),
            required_keys=["summary", "evidence", "answer_candidate", "confidence"],
        )
        evidence_payload = data.get("evidence", [])
        if isinstance(evidence_payload, list):
            evidence = [str(item) for item in evidence_payload][:6]
        elif evidence_payload:
            evidence = [str(evidence_payload)]
        else:
            evidence = []
        return InspectionResult(
            summary=str(data.get("summary", "")).strip(),
            evidence=evidence,
            answer_candidate=str(data.get("answer_candidate", "")).strip(),
            confidence=float(data.get("confidence", 0.5)),
            metadata={},
            usage=usage,
        )

    def answer(self, query: str, memory: Sequence[MemoryItem]) -> AnswerResult:
        memory_blob = "\n".join(f"- {item.provenance}: {item.summary}" for item in memory)
        payload = self._chat_text(
            (
                "Answer the root query using only the retained memory. "
                "Be concise and cite provenance inline."
            ),
            f"Query:\n{query}\n\nRetained memory:\n{memory_blob}",
        )
        return AnswerResult(answer=payload["content"], confidence=0.8, usage=payload["usage"])

    def score_candidate(self, query: str, item: MemoryItem) -> float:
        data, _ = self._chat_json(
            "score_candidate",
            "Return strict JSON with a single numeric field named score in [0, 10].",
            (
                f"Root query:\n{query}\n\n"
                f"Candidate summary:\n{item.summary}\n\n"
                f"Candidate provenance: {item.provenance}"
            ),
            required_keys=["score"],
        )
        return float(data.get("score", 0.0))

    def compare_candidates(self, query: str, left: MemoryItem, right: MemoryItem) -> int:
        data, _ = self._chat_json(
            "compare_candidates",
            "Return strict JSON with winner set to left, right, or tie.",
            (
                f"Root query:\n{query}\n\n"
                f"Left candidate:\n{left.summary}\n\n"
                f"Right candidate:\n{right.summary}\n\n"
                "Which candidate is more important to retain under tight memory?"
            ),
            required_keys=["winner"],
        )
        winner = str(data.get("winner", "tie")).lower()
        if winner == "left":
            return 1
        if winner == "right":
            return -1
        return 0

    def _chat_json(
        self,
        operation: str,
        system_prompt: str,
        user_prompt: str,
        *,
        required_keys: Sequence[str],
    ) -> tuple[dict[str, Any], Usage]:
        payload = self._chat_text(system_prompt, user_prompt)
        usage = payload["usage"]
        try:
            return self._parse_json_payload(payload["content"], required_keys), usage
        except ValueError as exc:
            repair = self._chat_text(
                (
                    "You repair malformed structured replies. "
                    "Return valid JSON only with the required keys."
                ),
                (
                    f"Operation: {operation}\n"
                    f"Required keys: {', '.join(required_keys)}\n\n"
                    f"Previous response:\n{payload['content']}\n\n"
                    "Return corrected JSON only."
                ),
            )
            usage.add(repair["usage"].prompt_tokens, repair["usage"].completion_tokens, repair["usage"].calls)
            try:
                return self._parse_json_payload(repair["content"], required_keys), usage
            except ValueError as repair_exc:
                raise RuntimeError(
                    f"{self.provider_name} returned invalid JSON for {operation}: {repair_exc}"
                ) from exc

    def _parse_json_payload(self, content: str, required_keys: Sequence[str]) -> dict[str, Any]:
        clean = strip_code_fences(content)
        data = validate_required_keys(extract_json_object(clean), required_keys)
        confidence = data.get("confidence")
        if confidence is not None:
            float(confidence)
        score = data.get("score")
        if score is not None:
            float(score)
        winner = data.get("winner")
        if winner is not None and str(winner).lower() not in {"left", "right", "tie"}:
            raise ValueError("winner must be left, right, or tie")
        return data

    def _chat_text(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        raise NotImplementedError


class OpenAICompatibleBackend(StructuredOutputBackend):
    """Tiny OpenAI-compatible client that only depends on the stdlib."""

    provider_name = "openai_compatible"

    def _chat_text(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        base_url = (self.config.base_url or OPENAI_COMPATIBLE_DEFAULT_BASE_URL).rstrip("/")
        url = f"{base_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI-compatible request failed: {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach {url}: {exc.reason}") from exc
        content = extract_text_content(raw["choices"][0]["message"]["content"])
        usage_payload = raw.get("usage", {})
        usage = Usage(
            prompt_tokens=int(usage_payload.get("prompt_tokens", 0)),
            completion_tokens=int(usage_payload.get("completion_tokens", 0)),
            calls=1,
        )
        return {"content": content, "usage": usage}


class AnthropicMessagesBackend(StructuredOutputBackend):
    provider_name = "anthropic"
    anthropic_version = "2023-06-01"

    def _chat_text(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        base_url = (self.config.base_url or ANTHROPIC_DEFAULT_BASE_URL).rstrip("/")
        url = f"{base_url}/v1/messages"
        payload = {
            "model": self.config.model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": 1024,
            "temperature": 0.0,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": self.anthropic_version,
        }
        if self.config.api_key:
            headers["x-api-key"] = self.config.api_key
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Anthropic request failed: {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach {url}: {exc.reason}") from exc
        content = extract_text_content(raw.get("content", []))
        usage_payload = raw.get("usage", {})
        usage = Usage(
            prompt_tokens=int(usage_payload.get("input_tokens", 0)),
            completion_tokens=int(usage_payload.get("output_tokens", 0)),
            calls=1,
        )
        return {"content": content, "usage": usage}


OpenAIChatBackend = OpenAICompatibleBackend


def build_backend(config: RLMConfig) -> Backend:
    provider = resolve_provider(config)
    base_url = resolved_base_url(config, provider)
    api_key = resolved_api_key(config, provider, base_url)
    resolved = replace(config, provider=provider, base_url=base_url, api_key=api_key)
    if provider == "heuristic":
        return HeuristicBackend(seed=config.seed)
    if provider == "anthropic":
        return AnthropicMessagesBackend(resolved)
    return OpenAICompatibleBackend(resolved)


class RLM:
    def __init__(self, config: RLMConfig, backend: Backend | None = None, policy: RetentionPolicy | None = None) -> None:
        self.config = config
        self.backend = backend or self._make_backend(config)
        if policy is None:
            from policies import build_policy

            self.policy = build_policy(config.retention_policy, judge=self.backend, seed=config.seed)
        else:
            self.policy = policy
        self.random = random.Random(config.seed)

    def completion(self, query: str, context: str | Sequence[ContextBlock | dict[str, Any] | tuple[str, str]]) -> RLMResult:
        blocks = materialize_context(context)
        recorder = TraceRecorder()
        usage = Usage()
        drop_reasons: list[dict[str, Any]] = []
        per_step_budget: list[dict[str, Any]] = []
        step_counter = [0]
        recorder.emit("inspect", 0, "root query", query=truncate_words(query, 16), blocks=len(blocks))
        kept_items = self._walk(
            query=query,
            blocks=blocks,
            depth=0,
            branch="root",
            recorder=recorder,
            usage=usage,
            step_counter=step_counter,
            drop_reasons=drop_reasons,
            per_step_budget=per_step_budget,
        )
        final = self.backend.answer(query, kept_items)
        usage.add(final.usage.prompt_tokens, final.usage.completion_tokens, final.usage.calls)
        recorder.emit("final_answer", 0, "compose answer", retained=len(kept_items), answer_preview=truncate_words(final.answer, 24))
        trace = recorder.artifact()
        retention_stats = {
            "policy": self.policy.name,
            "steps_used": step_counter[0],
            "total_retention_steps": len(per_step_budget),
            "total_dropped_items": len(drop_reasons),
            "final_retained_items": len(kept_items),
            "final_retained_tokens": sum(item.tokens for item in kept_items),
            "max_memory_depth": max((item.depth for item in kept_items), default=0),
        }
        return RLMResult(
            answer=final.answer,
            trace=trace,
            usage=usage,
            cost_estimate=self._estimate_cost(usage),
            kept_items=kept_items,
            retention_stats=retention_stats,
            provenance_hits=[],
            drop_reasons=drop_reasons,
            per_step_budget=per_step_budget,
        )

    def _walk(
        self,
        query: str,
        blocks: Sequence[ContextBlock],
        depth: int,
        branch: str,
        recorder: TraceRecorder,
        usage: Usage,
        step_counter: list[int],
        drop_reasons: list[dict[str, Any]],
        per_step_budget: list[dict[str, Any]],
    ) -> list[MemoryItem]:
        step_counter[0] += 1
        if step_counter[0] > self.config.max_steps:
            recorder.emit("retain", depth, "max steps reached", kept=0)
            return []
        if self._is_leaf(blocks, depth):
            recorder.emit("inspect", depth, f"{branch} leaf", tokens=sum(block.tokens for block in blocks), blocks=len(blocks))
            result = self.backend.inspect(query, blocks, depth, branch)
            usage.add(result.usage.prompt_tokens, result.usage.completion_tokens, result.usage.calls)
            provenance = ", ".join(block.name for block in blocks[:3])
            if len(blocks) > 3:
                provenance += ", ..."
            source_paths = sorted(
                {
                    str(block.metadata.get("path", block.name))
                    for block in blocks
                }
            )
            timestamp = time.time() + step_counter[0]
            metadata = {**result.metadata, "source_paths": source_paths, "block_names": [block.name for block in blocks]}
            item = MemoryItem(
                summary=result.summary,
                provenance=provenance,
                raw_pointer=branch,
                tokens=estimate_tokens(result.summary),
                depth=depth,
                timestamp=timestamp,
                answer_candidate=result.answer_candidate,
                confidence=result.confidence,
                metadata=metadata,
                score=0.0,
            )
            return [item]

        groups = self._split_blocks(blocks)
        recorder.emit("split", depth, f"{branch} split", groups=len(groups), blocks=len(blocks))
        memory: list[MemoryItem] = []
        for index, group in enumerate(groups):
            child_branch = f"{branch}.{index}"
            recorder.emit("recurse", depth + 1, child_branch, blocks=len(group), tokens=sum(block.tokens for block in group))
            memory.extend(self._walk(query, group, depth + 1, child_branch, recorder, usage, step_counter, drop_reasons, per_step_budget))
            before = len(memory)
            before_items = list(memory)
            memory = self.policy.select(query, memory, self.config.memory_budget_tokens)
            kept_signatures = {memory_signature(item) for item in memory}
            dropped = [item for item in before_items if memory_signature(item) not in kept_signatures]
            step_budget = {
                "step": step_counter[0],
                "depth": depth,
                "branch": branch,
                "policy": self.policy.name,
                "budget": self.config.memory_budget_tokens,
                "before_count": before,
                "after_count": len(memory),
                "before_tokens": sum(item.tokens for item in before_items),
                "after_tokens": sum(item.tokens for item in memory),
            }
            per_step_budget.append(step_budget)
            for item in dropped:
                drop_reasons.append(
                    {
                        "step": step_counter[0],
                        "depth": depth,
                        "branch": branch,
                        "policy": self.policy.name,
                        "reason": "policy_budget_trim",
                        "provenance": item.provenance,
                        "summary": truncate_words(item.summary, 18),
                        "tokens": item.tokens,
                        "target_id": item_target_id(item),
                    }
                )
            recorder.emit(
                "retain",
                depth,
                f"{branch} policy={self.policy.name}",
                before=before,
                after=len(memory),
                budget=self.config.memory_budget_tokens,
                dropped=len(dropped),
                kept_tokens=sum(item.tokens for item in memory),
            )
        return memory

    def _is_leaf(self, blocks: Sequence[ContextBlock], depth: int) -> bool:
        if depth >= self.config.max_depth:
            return True
        if len(blocks) <= 1:
            return True
        total_tokens = sum(block.tokens for block in blocks)
        return total_tokens <= max(64, self.config.memory_budget_tokens // 2)

    def _split_blocks(self, blocks: Sequence[ContextBlock]) -> list[list[ContextBlock]]:
        midpoint = max(1, len(blocks) // 2)
        return [list(blocks[:midpoint]), list(blocks[midpoint:])]

    def _make_backend(self, config: RLMConfig) -> Backend:
        return build_backend(config)

    def _estimate_cost(self, usage: Usage) -> float:
        provider = resolve_provider(self.config)
        if provider != "openai_compatible" or is_local_base_url(resolved_base_url(self.config, provider)):
            return 0.0
        price_table = {
            "gpt-5-mini": (0.00000025, 0.000002),
            "gpt-4.1-mini": (0.0000004, 0.0000016),
            "gpt-4.1": (0.000002, 0.000008),
        }
        if self.config.model not in price_table:
            return 0.0
        prompt_price, completion_price = price_table[self.config.model]
        return round(usage.prompt_tokens * prompt_price + usage.completion_tokens * completion_price, 6)


def load_text_blocks(path: str | Path, chunk_size_lines: int = 48) -> list[ContextBlock]:
    file_path = Path(path)
    text = file_path.read_text()
    chunks = chunk_lines(text, max_lines=chunk_size_lines)
    blocks = []
    for index, chunk in enumerate(chunks):
        suffix = f":{index + 1}" if len(chunks) > 1 else ""
        blocks.append(ContextBlock(name=f"{file_path.name}{suffix}", text=chunk, metadata={"path": str(file_path)}))
    return blocks


def write_trace(result: RLMResult, path: str | Path) -> None:
    result.trace.write_jsonl(path)
