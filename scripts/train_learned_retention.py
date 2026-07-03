from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import BenchmarkExample, build_dataset, parse_csv_ints, parse_csv_strings  # noqa: E402
from learned_retention import retention_features, train_linear_retention_model  # noqa: E402
from nanorlm import ContextBlock, HeuristicBackend, MemoryItem, estimate_tokens, normalize_text  # noqa: E402


DEFAULT_DATASETS = "pairbench,dossierbench,ruler_synthetic,babilong_synthetic,external_jsonl"
NEGATIVE_SLOT_MARKERS = (
    "slot: distractor",
    "slot distractor",
    "belongs to another",
    "slot: duplicate",
    "slot duplicate",
    "duplicate:",
)


def _matches_expected_provenance(block: ContextBlock, expected_provenance: Sequence[str]) -> bool:
    block_name = block.name.lower()
    block_path = str(block.metadata.get("path", block.name)).lower()
    for expected in expected_provenance:
        expected_lower = expected.lower()
        expected_name = Path(expected).name.lower()
        if expected_lower in block_path or expected_lower in block_name:
            return True
        if expected_name and (expected_name in Path(block_name).name.lower() or expected_name in block_path):
            return True
    return False


def _contains_answer_fragment(block: ContextBlock, example: BenchmarkExample) -> bool:
    haystack = normalize_text(f"{block.name}\n{block.text}")
    fragments = list(example.must_contain)
    if example.answer:
        fragments.extend(part.strip() for part in example.answer.split("|") if part.strip())
    for fragment in fragments:
        normalized = normalize_text(fragment)
        if normalized and normalized in haystack:
            return True
    return False


def _is_explicit_negative_block(block: ContextBlock) -> bool:
    haystack = f"{block.name}\n{block.text}".lower()
    return any(marker in haystack for marker in NEGATIVE_SLOT_MARKERS)


def label_block(block: ContextBlock, example: BenchmarkExample) -> bool:
    if _matches_expected_provenance(block, example.expected_provenance):
        return True
    if _is_explicit_negative_block(block):
        return False
    return _contains_answer_fragment(block, example)


def memory_item_from_block(
    *,
    backend: HeuristicBackend,
    example: BenchmarkExample,
    block: ContextBlock,
    dataset: str,
    seed: int,
    index: int,
) -> MemoryItem:
    inspection = backend.inspect(example.query, [block], depth=1, branch=f"{dataset}.{example.name}.{index}")
    return MemoryItem(
        summary=inspection.summary,
        provenance=block.name,
        raw_pointer=f"{dataset}.{seed}.{example.name}.{index}",
        tokens=estimate_tokens(inspection.summary),
        depth=1,
        timestamp=float(index),
        answer_candidate=inspection.answer_candidate,
        confidence=inspection.confidence,
        metadata={
            "source_paths": [str(block.metadata.get("path", block.name))],
            "block_names": [block.name],
            "training_dataset": dataset,
            "training_seed": seed,
            "task_class": example.task_class,
        },
    )


def rows_from_examples(
    examples: Sequence[BenchmarkExample],
    *,
    dataset: str,
    seed: int,
    feature_budget: int,
) -> list[dict[str, Any]]:
    backend = HeuristicBackend(seed=seed)
    rows: list[dict[str, Any]] = []
    for example in examples:
        for index, block in enumerate(example.context):
            item = memory_item_from_block(
                backend=backend,
                example=example,
                block=block,
                dataset=dataset,
                seed=seed,
                index=index,
            )
            label = label_block(block, example)
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "case": example.name,
                    "task_class": example.task_class,
                    "query": example.query,
                    "provenance": item.provenance,
                    "summary": item.summary,
                    "label": label,
                    "features": retention_features(example.query, item, feature_budget),
                }
            )
    return rows


def build_training_rows(
    *,
    datasets: Sequence[str],
    seeds: Sequence[int],
    limit: int,
    repo_root: str,
    dataset_path: str | None,
    feature_budget: int,
    allow_missing_datasets: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    dataset_records: list[dict[str, Any]] = []
    for dataset in datasets:
        for seed in seeds:
            try:
                examples = build_dataset(
                    dataset,
                    limit=limit,
                    seed=seed,
                    repo_root=repo_root,
                    dataset_path=dataset_path if dataset == "external_jsonl" else None,
                )
            except (FileNotFoundError, ValueError) as exc:
                if not allow_missing_datasets:
                    raise
                dataset_records.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "status": "skipped",
                        "reason": str(exc),
                    }
                )
                continue
            dataset_rows = rows_from_examples(
                examples,
                dataset=dataset,
                seed=seed,
                feature_budget=feature_budget,
            )
            positives = sum(1 for row in dataset_rows if row["label"])
            rows.extend(dataset_rows)
            dataset_records.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "status": "included",
                    "examples": len(examples),
                    "rows": len(dataset_rows),
                    "positive_rows": positives,
                    "negative_rows": len(dataset_rows) - positives,
                }
            )
    return rows, dataset_records


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a small offline learned_retention model from benchmark traces.")
    parser.add_argument("--datasets", default=DEFAULT_DATASETS)
    parser.add_argument("--train-seeds", default="0,1")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--repo-root", default=str(ROOT / "tests" / "fixtures" / "verifiers-mini"))
    parser.add_argument("--dataset-path", default=str(ROOT / "tests" / "fixtures" / "external-benchmark-mini.jsonl"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "learned_retention"))
    parser.add_argument("--model-out", default="")
    parser.add_argument("--examples-out", default="")
    parser.add_argument("--feature-budget", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument("--l2", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-missing-datasets", action="store_true")
    return parser


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    model_path = Path(args.model_out) if args.model_out else output_dir / "learned_retention_model.json"
    examples_path = Path(args.examples_out) if args.examples_out else output_dir / "training_examples.jsonl"
    datasets = parse_csv_strings(args.datasets)
    train_seeds = parse_csv_ints(args.train_seeds)
    rows, dataset_records = build_training_rows(
        datasets=datasets,
        seeds=train_seeds,
        limit=args.limit,
        repo_root=args.repo_root,
        dataset_path=args.dataset_path or None,
        feature_budget=args.feature_budget,
        allow_missing_datasets=args.allow_missing_datasets,
    )
    model = train_linear_retention_model(
        rows,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        seed=args.seed,
    )
    model.metadata.update(
        {
            "datasets": datasets,
            "train_seeds": train_seeds,
            "limit": args.limit,
            "feature_budget": args.feature_budget,
            "repo_root": args.repo_root,
            "dataset_path": args.dataset_path,
        }
    )
    write_jsonl(examples_path, rows)
    model.save(model_path)
    manifest = {
        "status": "trained",
        "model_path": str(model_path),
        "training_examples_path": str(examples_path),
        "datasets": dataset_records,
        "training": model.metadata,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
