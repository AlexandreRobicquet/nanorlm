from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import CLI_PROVIDER_CHOICES, extract_anchor_blocks, resolve_provider_choice
from nanorlm import ContextBlock, RLM, RLMConfig, RLMResult, item_source_paths

@dataclass(slots=True)
class PlanningTask:
    name: str
    problem: str
    context: list[ContextBlock]
    repo_root: str
    evidence_files: list[str]
    expected_files: list[str]
    expected_keywords: list[str]
    anchors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GroundedPlan:
    name: str
    problem: str
    retained_answer: str
    steps: list[str]
    citations: list[str]
    unknowns: list[str]
    file_recall: float
    file_hits: list[str]
    keyword_coverage: float
    keyword_hits: list[str]

    def to_markdown(self) -> str:
        lines = [f"# {self.name}", "", self.problem, "", "## Plan"]
        for index, step in enumerate(self.steps, start=1):
            lines.append(f"{index}. {step}")
        lines.extend(["", "## Citations"])
        for citation in self.citations:
            lines.append(f"- `{citation}`")
        lines.extend(["", "## Unknowns"])
        if self.unknowns:
            for unknown in self.unknowns:
                lines.append(f"- {unknown}")
        else:
            lines.append("- None surfaced in retained evidence.")
        lines.extend(
            [
                "",
                "## Metrics",
                f"- file_recall: {self.file_recall:.3f}",
                f"- keyword_coverage: {self.keyword_coverage:.3f}",
            ]
        )
        return "\n".join(lines) + "\n"


def load_planning_tasks(repo_root: str | Path, tasks_path: str | Path | None = None) -> list[PlanningTask]:
    repo_root = Path(repo_root)
    task_rows = json.loads((Path(tasks_path) if tasks_path else ROOT / "showcases" / "planning_tasks.json").read_text())
    tasks: list[PlanningTask] = []
    for row in task_rows:
        context: list[ContextBlock] = []
        anchors = list(row.get("anchors", []))
        for rel_path in row["evidence_files"]:
            context.extend(extract_anchor_blocks(repo_root / rel_path, anchors, window=10))
        tasks.append(
            PlanningTask(
                name=row["name"],
                problem=row["problem"],
                context=context,
                repo_root=str(repo_root),
                evidence_files=list(row["evidence_files"]),
                expected_files=list(row.get("expected_files", row["evidence_files"])),
                expected_keywords=list(row.get("expected_keywords", [])),
                anchors=anchors,
            )
        )
    return tasks


def _dedupe(paths: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for path in paths:
        normalized = str(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _normalize_citation(path: str, repo_root: str) -> str:
    candidate = Path(path)
    root = Path(repo_root)
    try:
        return str(candidate.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def _category(path: str) -> str:
    normalized = path.lower()
    if normalized.startswith("tests/"):
        return "tests"
    if normalized.startswith("docs/") or normalized.endswith("readme.md") or normalized.endswith("pyproject.toml"):
        return "docs"
    return "impl"


def _score_hits(actual_paths: Sequence[str], expected_paths: Sequence[str]) -> tuple[float, list[str]]:
    actual = [path.lower() for path in actual_paths]
    hits = [
        path
        for path in expected_paths
        if any(candidate == path.lower() or candidate.endswith(path.lower()) for candidate in actual)
    ]
    if not expected_paths:
        return 0.0, []
    return round(len(hits) / len(expected_paths), 3), hits


def _score_keywords(text: str, keywords: Sequence[str]) -> tuple[float, list[str]]:
    haystack = text.lower()
    hits = [keyword for keyword in keywords if keyword.lower() in haystack]
    if not keywords:
        return 0.0, []
    return round(len(hits) / len(keywords), 3), hits


def _unittest_module_target(path: str) -> str:
    target = Path(path)
    if target.suffix == ".py":
        target = target.with_suffix("")
    return ".".join(part for part in target.parts if part not in ("", "."))


def _validation_command(test_files: Sequence[str], impl_files: Sequence[str], task: PlanningTask) -> str:
    if test_files:
        targets = " ".join(_unittest_module_target(path) for path in test_files[:2])
        return f"uv run python -m unittest {targets}"
    if impl_files:
        target = Path(impl_files[0]).stem.replace("-", "_")
        return f"uv run python -m unittest discover -s tests -p '*{target}*.py'"
    target = task.name.replace("-", "_")
    return f"uv run python -m unittest discover -s tests -p '*{target}*.py'"


def synthesize_plan(task: PlanningTask, result: RLMResult) -> GroundedPlan:
    citations = _dedupe([_normalize_citation(path, task.repo_root) for item in result.kept_items for path in item_source_paths(item)])
    impl_files = [path for path in citations if _category(path) == "impl"]
    test_files = [path for path in citations if _category(path) == "tests"]
    doc_files = [path for path in citations if _category(path) == "docs"]
    steps: list[str] = []
    unknowns: list[str] = []

    if impl_files:
        steps.append(
            f"Patch `{', '.join(impl_files[:2])}` first and keep the change as local as possible to the behavior described in the problem statement."
        )
    else:
        unknowns.append("No implementation file surfaced in retained evidence.")

    if test_files:
        steps.append(
            f"Add or tighten regression coverage in `{', '.join(test_files[:2])}` so the behavior is pinned down before broadening the change."
        )
    else:
        unknowns.append("No dedicated regression test file surfaced in retained evidence.")

    if doc_files:
        steps.append(
            f"Refresh developer-facing guidance in `{', '.join(doc_files[:2])}` so examples and docs stay aligned with the implementation."
        )
    else:
        unknowns.append("Docs impact is not explicitly confirmed by retained evidence.")

    steps.append(
        f"Run focused validation with `{_validation_command(test_files, impl_files, task)}` and inspect the retained trace before widening the rollout."
    )

    combined_text = " ".join([result.answer, *steps, *citations])
    file_recall, file_hits = _score_hits(citations, task.expected_files)
    keyword_coverage, keyword_hits = _score_keywords(combined_text, task.expected_keywords)
    return GroundedPlan(
        name=task.name,
        problem=task.problem,
        retained_answer=result.answer,
        steps=steps,
        citations=citations,
        unknowns=unknowns,
        file_recall=file_recall,
        file_hits=file_hits,
        keyword_coverage=keyword_coverage,
        keyword_hits=keyword_hits,
    )


def run_planning_suite(
    tasks: Sequence[PlanningTask],
    *,
    budget: int,
    max_depth: int,
    provider: str = "heuristic",
    model: str,
    base_url: str | None,
    api_key: str | None,
    use_openai_backend: bool | None,
    seed: int,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    if use_openai_backend is not None:
        provider = "openai_compatible" if use_openai_backend else provider
    config = RLMConfig(
        model=model,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        max_depth=max_depth,
        max_steps=256,
        memory_budget_tokens=budget,
        retention_policy="pairwise_tournament",
        seed=seed,
    )
    engine = RLM(config=config)

    output_path = Path(output_dir) if output_dir else None
    if output_path is not None:
        (output_path / "plans").mkdir(parents=True, exist_ok=True)
        (output_path / "traces").mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for task in tasks:
        result = engine.completion(task.problem, task.context)
        plan = synthesize_plan(task, result)
        row = {
            "name": task.name,
            "problem": task.problem,
            "citations": plan.citations,
            "unknowns": plan.unknowns,
            "steps": plan.steps,
            "retained_answer": plan.retained_answer,
            "file_recall": plan.file_recall,
            "file_hits": plan.file_hits,
            "keyword_coverage": plan.keyword_coverage,
            "keyword_hits": plan.keyword_hits,
            "retention_stats": result.retention_stats,
        }
        rows.append(row)
        if output_path is not None:
            (output_path / "plans" / f"{task.name}.md").write_text(plan.to_markdown())
            result.trace.write_jsonl(output_path / "traces" / f"{task.name}.jsonl")
            result.trace.write_tree(output_path / "traces" / f"{task.name}.tree.txt")

    summary = {
        "tasks": len(tasks),
        "avg_file_recall": round(sum(row["file_recall"] for row in rows) / len(rows), 3) if rows else 0.0,
        "avg_keyword_coverage": round(sum(row["keyword_coverage"] for row in rows) / len(rows), 3) if rows else 0.0,
        "missing_critical_file_rate": round(sum(1 for row in rows if row["file_recall"] < 1.0) / len(rows), 3) if rows else 0.0,
        "results": rows,
    }
    if output_path is not None:
        (output_path / "summary.json").write_text(json.dumps(summary, indent=2))
        with (output_path / "per_case.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    return summary


def _format_table(summary: dict[str, Any]) -> str:
    lines = ["| task | file recall | keyword | cited |", "| --- | ---: | ---: | ---: |"]
    for row in summary["results"]:
        lines.append(
            f"| {row['name']} | {row['file_recall']:.3f} | {row['keyword_coverage']:.3f} | {len(row['citations'])} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the grounded patch-planning showcase.")
    parser.add_argument("--repo-root", type=str, default="/tmp/nanorlm-verifiers")
    parser.add_argument("--tasks-path", type=str, default="")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--budget", type=int, default=120)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="showcases/outputs/planning")
    parser.add_argument("--provider", choices=CLI_PROVIDER_CHOICES, default="heuristic")
    parser.add_argument("--openai", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", type=str, default="demo/heuristic")
    parser.add_argument("--base-url", type=str, default="")
    parser.add_argument("--api-key", type=str, default="")
    args = parser.parse_args()

    provider = resolve_provider_choice(args.provider, args.openai)
    tasks = load_planning_tasks(args.repo_root, args.tasks_path or None)[: args.limit]
    summary = run_planning_suite(
        tasks,
        budget=args.budget,
        max_depth=args.depth,
        provider=provider,
        model=args.model,
        base_url=args.base_url or None,
        api_key=args.api_key or None,
        use_openai_backend=None,
        seed=0,
        output_dir=args.output_dir,
    )
    print(_format_table(summary))


if __name__ == "__main__":
    main()
