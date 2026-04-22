from __future__ import annotations

import json
import math
import os
import random
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, Sequence


WORD_RE = re.compile(r"[A-Za-z0-9_./:-]+")


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


@dataclass(slots=True)
class RLMConfig:
    model: str
    base_url: str = "https://api.openai.com/v1"
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
        evidence: list[str] = []
        for document in documents:
            matches = self._salient_lines(query, document.text)
            for score, line in matches[:2]:
                snippets.append((score, f"{document.name}: {line}"))
                evidence.append(f"{document.name}: {line}")
        snippets.sort(key=lambda item: (-item[0], item[1]))
        summary_parts = [snippet for _, snippet in snippets[:3]]
        if not summary_parts:
            joined_names = ", ".join(document.name for document in documents[:3])
            summary_parts.append(f"{joined_names}: no strong lexical match, returning leading context")
        summary = " | ".join(summary_parts)
        answer_candidate = summary_parts[0].split(": ", 1)[-1] if summary_parts else ""
        confidence = min(0.95, 0.2 + 0.15 * len(summary_parts))
        usage = Usage(prompt_tokens=sum(document.tokens for document in documents), completion_tokens=estimate_tokens(summary), calls=1)
        return InspectionResult(summary=summary, evidence=evidence[:6], answer_candidate=answer_candidate, confidence=confidence, metadata={}, usage=usage)

    def answer(self, query: str, memory: Sequence[MemoryItem]) -> AnswerResult:
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
        if item.confidence:
            score += item.confidence
        return score

    def compare_candidates(self, query: str, left: MemoryItem, right: MemoryItem) -> int:
        left_score = self.score_candidate(query, left)
        right_score = self.score_candidate(query, right)
        if abs(left_score - right_score) < 0.1:
            if left.timestamp > right.timestamp:
                return 1
            if right.timestamp > left.timestamp:
                return -1
            return 0
        return 1 if left_score > right_score else -1

    def _salient_lines(self, query: str, text: str) -> list[tuple[float, str]]:
        query_set = query_terms(query)
        matches: list[tuple[float, str]] = []
        for line in text.splitlines():
            clean = line.strip()
            if not clean:
                continue
            overlap = len(query_set & query_terms(clean))
            if overlap == 0:
                continue
            matches.append((float(overlap) + len(clean) / 500.0, clean))
        return sorted(matches, key=lambda item: (-item[0], item[1]))


class OpenAIChatBackend:
    """Tiny OpenAI-compatible client that only depends on the stdlib."""

    def __init__(self, config: RLMConfig) -> None:
        self.config = config

    def inspect(self, query: str, documents: Sequence[ContextBlock], depth: int, branch: str) -> InspectionResult:
        joined = "\n\n".join(f"### {document.name}\n{document.text}" for document in documents)
        payload = self._chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a recursive language model worker. "
                        "Read the provided branch context and return strict JSON with keys "
                        "summary, evidence, answer_candidate, confidence. "
                        "The summary should be terse and preserve only facts that help answer the root query."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Root query:\n{query}\n\n"
                        f"Branch: {branch}\nDepth: {depth}\n\n"
                        f"Context:\n{joined}\n\n"
                        "Return JSON only."
                    ),
                },
            ]
        )
        data = extract_json_object(payload["content"])
        return InspectionResult(
            summary=str(data.get("summary", "")).strip(),
            evidence=[str(item) for item in data.get("evidence", [])][:6],
            answer_candidate=str(data.get("answer_candidate", "")).strip(),
            confidence=float(data.get("confidence", 0.5)),
            metadata={},
            usage=payload["usage"],
        )

    def answer(self, query: str, memory: Sequence[MemoryItem]) -> AnswerResult:
        memory_blob = "\n".join(f"- {item.provenance}: {item.summary}" for item in memory)
        payload = self._chat(
            messages=[
                {
                    "role": "system",
                    "content": "Answer the root query using only the retained memory. Be concise and cite provenance inline.",
                },
                {
                    "role": "user",
                    "content": f"Query:\n{query}\n\nRetained memory:\n{memory_blob}",
                },
            ]
        )
        return AnswerResult(answer=payload["content"], confidence=0.8, usage=payload["usage"])

    def score_candidate(self, query: str, item: MemoryItem) -> float:
        payload = self._chat(
            messages=[
                {
                    "role": "system",
                    "content": "Return strict JSON with a single numeric field named score in [0, 10].",
                },
                {
                    "role": "user",
                    "content": (
                        f"Root query:\n{query}\n\n"
                        f"Candidate summary:\n{item.summary}\n\n"
                        f"Candidate provenance: {item.provenance}"
                    ),
                },
            ]
        )
        data = extract_json_object(payload["content"])
        return float(data.get("score", 0.0))

    def compare_candidates(self, query: str, left: MemoryItem, right: MemoryItem) -> int:
        payload = self._chat(
            messages=[
                {
                    "role": "system",
                    "content": "Return strict JSON with winner set to left, right, or tie.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Root query:\n{query}\n\n"
                        f"Left candidate:\n{left.summary}\n\n"
                        f"Right candidate:\n{right.summary}\n\n"
                        "Which candidate is more important to retain under tight memory?"
                    ),
                },
            ]
        )
        data = extract_json_object(payload["content"])
        winner = str(data.get("winner", "tie")).lower()
        if winner == "left":
            return 1
        if winner == "right":
            return -1
        return 0

    def _chat(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        base_url = self.config.base_url.rstrip("/")
        url = f"{base_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": messages,
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

        content = raw["choices"][0]["message"]["content"]
        usage_payload = raw.get("usage", {})
        usage = Usage(
            prompt_tokens=int(usage_payload.get("prompt_tokens", 0)),
            completion_tokens=int(usage_payload.get("completion_tokens", 0)),
            calls=1,
        )
        return {"content": content, "usage": usage}


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
        recorder.emit("inspect", 0, "root query", query=truncate_words(query, 16), blocks=len(blocks))
        kept_items = self._walk(query=query, blocks=blocks, depth=0, branch="root", recorder=recorder, usage=usage, step_counter=[0])
        final = self.backend.answer(query, kept_items)
        usage.add(final.usage.prompt_tokens, final.usage.completion_tokens, final.usage.calls)
        recorder.emit("final_answer", 0, "compose answer", retained=len(kept_items), answer_preview=truncate_words(final.answer, 24))
        trace = recorder.artifact()
        return RLMResult(
            answer=final.answer,
            trace=trace,
            usage=usage,
            cost_estimate=self._estimate_cost(usage),
            kept_items=kept_items,
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
            item = MemoryItem(
                summary=result.summary,
                provenance=provenance,
                raw_pointer=branch,
                tokens=estimate_tokens(result.summary),
                depth=depth,
                timestamp=float(step_counter[0]),
                answer_candidate=result.answer_candidate,
                confidence=result.confidence,
                metadata=result.metadata,
            )
            return [item]

        groups = self._split_blocks(blocks)
        recorder.emit("split", depth, f"{branch} split", groups=len(groups), blocks=len(blocks))
        memory: list[MemoryItem] = []
        for index, group in enumerate(groups):
            child_branch = f"{branch}.{index}"
            recorder.emit("recurse", depth + 1, child_branch, blocks=len(group), tokens=sum(block.tokens for block in group))
            memory.extend(self._walk(query, group, depth + 1, child_branch, recorder, usage, step_counter))
            before = len(memory)
            memory = self.policy.select(query, memory, self.config.memory_budget_tokens)
            recorder.emit("retain", depth, f"{branch} policy={self.policy.name}", before=before, after=len(memory), budget=self.config.memory_budget_tokens)
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
        if config.api_key or os.environ.get("OPENAI_API_KEY") or "localhost" not in config.base_url:
            api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
            return OpenAIChatBackend(replace(config, api_key=api_key))
        return HeuristicBackend(seed=config.seed)

    def _estimate_cost(self, usage: Usage) -> float:
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
