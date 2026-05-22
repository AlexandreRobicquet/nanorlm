from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} must be a JSON object")
                rows.append(row)
        return rows
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ["data", "examples", "rows"]:
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    raise ValueError(f"{path} does not contain a JSON object list")


def first_present(row: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def as_answer_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def infer_task_name(row: dict[str, Any], source_path: Path) -> str:
    explicit = first_present(row, ["task", "task_name", "task_class"])
    if explicit is not None:
        return str(explicit)
    if source_path.parent.name and source_path.parent.name not in {"data", "synthetic"}:
        return source_path.parent.name
    return "ruler"


def infer_query(context: Any) -> str:
    text = str(context)
    question_matches = re.findall(r"Question:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if question_matches:
        return question_matches[-1].strip()
    question_sentences = re.findall(r"([^.\n!?]*\?)(?:\s|$)", text)
    if question_sentences:
        return question_sentences[-1].strip()
    return "Answer the benchmark query using only the provided context."


def infer_name(row: dict[str, Any], *, task_name: str, context_length: Any, index: int) -> str:
    for key in ["name", "id"]:
        if key in row:
            return str(row[key])
    if "index" in row:
        length_part = str(context_length) if context_length is not None else "unknown"
        return f"{task_name}-{length_part}-{row['index']}"
    return f"{task_name}-{index:04d}"


def chunk_context_blocks(text: str, *, example_name: str, source_path: Path, source_index: int) -> list[dict[str, Any]]:
    lines = text.splitlines()
    chunks: list[str] = []
    if len(lines) > 1:
        chunk_size = 96
        for start in range(0, len(lines), chunk_size):
            chunks.append("\n".join(lines[start : start + chunk_size]).strip())
    else:
        words = text.split()
        chunk_size = 320
        for start in range(0, len(words), chunk_size):
            chunks.append(" ".join(words[start : start + chunk_size]).strip())
    return [
        {
            "name": f"{example_name}/context-{chunk_index:03d}.txt",
            "text": chunk,
            "metadata": {
                "benchmark": "RULER",
                "source_path": str(source_path),
                "source_index": source_index,
                "chunk_index": chunk_index,
            },
        }
        for chunk_index, chunk in enumerate(chunks, start=1)
        if chunk
    ]


def convert_row(row: dict[str, Any], *, source_path: Path, index: int) -> dict[str, Any]:
    context = first_present(row, ["context", "input", "prompt"])
    answers = as_answer_list(first_present(row, ["outputs", "answer", "answers", "expected", "output"]))
    if context is None:
        raise ValueError(f"{source_path}:{index} is missing context/input/prompt")
    query = first_present(row, ["query", "question"]) or infer_query(context)
    if query is None:
        query = "Answer the benchmark query using only the provided context."
    if not answers:
        raise ValueError(f"{source_path}:{index} is missing a non-empty answer/outputs field")
    task_name = infer_task_name(row, source_path)
    context_length = first_present(row, ["context_length", "length", "seq_length", "tokens"])
    name = infer_name(row, task_name=task_name, context_length=context_length, index=index)
    metadata = {
        "benchmark": "RULER",
        "source_path": str(source_path),
        "source_index": index,
        "task_name": task_name,
    }
    if context_length is not None:
        metadata["context_length"] = context_length
    return {
        "name": name,
        "query": str(query),
        "context": chunk_context_blocks(str(context), example_name=name, source_path=source_path, source_index=index),
        "answer": " | ".join(answers),
        "must_contain": answers,
        "expected_provenance": [],
        "task_class": f"ruler/{task_name}",
        "metadata": metadata,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert RULER-generated rows into nanoRLM external_jsonl rows.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_path = Path(args.input)
    output_path = Path(args.output)
    rows = load_rows(source_path)
    if args.limit:
        rows = rows[: args.limit]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            handle.write(json.dumps(convert_row(row, source_path=source_path, index=index), sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
