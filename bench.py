from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from nanorlm import ContextBlock, HeuristicBackend, OpenAIChatBackend, RLM, RLMConfig, load_text_blocks, normalize_text, write_trace


ROOT = Path(__file__).resolve().parent


@dataclass(slots=True)
class BenchmarkExample:
    name: str
    query: str
    context: list[ContextBlock]
    answer: str
    must_contain: list[str]


def extract_anchor_blocks(path: str | Path, anchors: Sequence[str], window: int = 6) -> list[ContextBlock]:
    file_path = Path(path)
    lines = file_path.read_text().splitlines()
    if not lines:
        return [ContextBlock(name=file_path.name, text="")]
    blocks: list[ContextBlock] = []
    seen_ranges: set[tuple[int, int]] = set()
    lower_lines = [line.lower() for line in lines]
    for anchor in anchors:
        anchor_lower = anchor.lower()
        for index, line in enumerate(lower_lines):
            if anchor_lower in line:
                start = max(0, index - window)
                end = min(len(lines), index + window + 1)
                key = (start, end)
                if key in seen_ranges:
                    break
                seen_ranges.add(key)
                text = "\n".join(lines[start:end])
                blocks.append(ContextBlock(name=f"{file_path.name}:{start + 1}-{end}", text=text, metadata={"path": str(file_path)}))
                break
    if not blocks:
        return load_text_blocks(file_path, chunk_size_lines=24)[:1]
    return blocks


def score_answer(answer: str, must_contain: Sequence[str]) -> float:
    normalized = normalize_text(answer)
    return 1.0 if all(normalize_text(fragment) in normalized for fragment in must_contain) else 0.0


def pair_words() -> list[str]:
    return [
        "amber",
        "comet",
        "frost",
        "lattice",
        "mango",
        "orbit",
        "quartz",
        "raven",
        "signal",
        "topaz",
        "vector",
        "willow",
        "yonder",
        "zephyr",
    ]


def build_pairbench(n: int = 100, seed: int = 0) -> list[BenchmarkExample]:
    rng = random.Random(seed)
    words = pair_words()
    examples: list[BenchmarkExample] = []
    distractor_space = max(128, n + 64)
    for index in range(n):
        pair_id = f"pair-{index:03d}"
        left_value = words[index % len(words)]
        right_value = words[(index * 3 + 5) % len(words)]
        docs: list[ContextBlock] = []
        for distractor in range(18):
            distractor_pair = f"pair-{(index + distractor + 7) % distractor_space:03d}"
            if distractor_pair == pair_id:
                distractor_pair = f"pair-{(index + distractor + 37) % distractor_space:03d}"
            distractor_kind = "left" if distractor % 2 == 0 else "right"
            distractor_value = words[(index + distractor * 2) % len(words)]
            docs.append(
                ContextBlock(
                    name=f"notes/{distractor_pair}-{distractor_kind}-{distractor}.md",
                    text=(
                        f"PAIR_ID: {distractor_pair}\n"
                        f"FACT_KIND: {distractor_kind}\n"
                        f"FACT_VALUE: {distractor_value}\n"
                        f"SLOT: memo\n"
                        "This scratch note may look relevant but belongs to another pair.\n"
                    ),
                )
            )
        docs.extend(
            [
                ContextBlock(
                    name=f"vault/{pair_id}-left.md",
                    text=(
                        f"PAIR_ID: {pair_id}\n"
                        "FACT_KIND: left\n"
                        f"FACT_VALUE: {left_value}\n"
                        "SLOT: durable\n"
                        "The left token must be combined with the right token.\n"
                    ),
                ),
                ContextBlock(
                    name=f"vault/{pair_id}-right.md",
                    text=(
                        f"PAIR_ID: {pair_id}\n"
                        "FACT_KIND: right\n"
                        f"FACT_VALUE: {right_value}\n"
                        "SLOT: durable\n"
                        "The right token must be combined with the left token.\n"
                    ),
                ),
                ContextBlock(
                    name=f"vault/{pair_id}-left-duplicate.md",
                    text=(
                        f"PAIR_ID: {pair_id}\n"
                        "FACT_KIND: left\n"
                        f"FACT_VALUE: {left_value}\n"
                        "SLOT: duplicate\n"
                        "This duplicate memo exists to tempt single-item ranking.\n"
                    ),
                ),
                ContextBlock(
                    name=f"vault/{pair_id}-left-archive.md",
                    text=(
                        f"PAIR_ID: {pair_id}\n"
                        "FACT_KIND: left\n"
                        f"FACT_VALUE: {left_value}\n"
                        "SLOT: archive\n"
                        "An older archive copy repeats the left token and competes for memory.\n"
                    ),
                ),
            ]
        )
        rng.shuffle(docs)
        examples.append(
            BenchmarkExample(
                name=pair_id,
                query=f"For {pair_id}, what is the full code? Combine the left token and the right token.",
                context=docs,
                answer=f"{left_value} {right_value}",
                must_contain=[left_value, right_value],
            )
        )
    return examples


def build_needlepairs(n: int = 50, seed: int = 0) -> list[BenchmarkExample]:
    rng = random.Random(seed)
    base = build_pairbench(n=n, seed=seed)
    examples: list[BenchmarkExample] = []
    filler = "Noise block. " * 50
    for example in base:
        padded: list[ContextBlock] = []
        for index in range(96):
            padded.append(ContextBlock(name=f"haystack/noise-{index:03d}.txt", text=filler + f" slot {index}"))
        padded.extend(example.context)
        rng.shuffle(padded)
        examples.append(
            BenchmarkExample(
                name=f"needle-{example.name}",
                query=example.query,
                context=padded,
                answer=example.answer,
                must_contain=example.must_contain,
            )
        )
    return examples


def load_verifiers_20(repo_root: str | Path, dataset_path: str | Path | None = None, distractors: int = 4, seed: int = 0) -> list[BenchmarkExample]:
    rng = random.Random(seed)
    repo_root = Path(repo_root)
    dataset_path = Path(dataset_path or ROOT / "examples" / "verifiers_20.json")
    rows = json.loads(dataset_path.read_text())
    pool = sorted(path for path in repo_root.rglob("*") if path.is_file() and ".git" not in path.parts)
    examples: list[BenchmarkExample] = []
    for row in rows:
        provenance_paths = [repo_root / path for path in row["provenance"]]
        context: list[ContextBlock] = []
        for path in provenance_paths:
            context.extend(extract_anchor_blocks(path, row["must_contain"], window=8))
        distractor_pool = [path for path in pool if path not in provenance_paths and path.suffix in {".md", ".toml", ".py"}]
        rng.shuffle(distractor_pool)
        for path in distractor_pool[:distractors]:
            context.extend(load_text_blocks(path, chunk_size_lines=24)[:1])
        examples.append(
            BenchmarkExample(
                name=row["name"],
                query=row["query"],
                context=context,
                answer=row["answer"],
                must_contain=list(row["must_contain"]),
            )
        )
    return examples


def run_dataset(
    examples: Sequence[BenchmarkExample],
    policy: str,
    *,
    budget: int = 120,
    max_depth: int = 2,
    model: str = "demo/heuristic",
    base_url: str = "http://localhost:11434/v1",
    api_key: str | None = None,
    output_dir: str | Path | None = None,
    use_openai_backend: bool = False,
) -> dict[str, object]:
    config = RLMConfig(
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_depth=max_depth,
        max_steps=256,
        memory_budget_tokens=budget,
        retention_policy=policy,
        seed=0,
    )
    backend = OpenAIChatBackend(config) if use_openai_backend else HeuristicBackend(seed=0)
    results: list[dict[str, object]] = []
    total = 0.0
    traces_dir: Path | None = None
    if output_dir is not None:
        traces_dir = Path(output_dir)
        traces_dir.mkdir(parents=True, exist_ok=True)
    for example in examples:
        engine = RLM(config=config, backend=backend)
        result = engine.completion(example.query, example.context)
        accuracy = score_answer(result.answer, example.must_contain)
        total += accuracy
        if traces_dir is not None:
            write_trace(result, traces_dir / f"{policy}-{example.name}.jsonl")
            (traces_dir / f"{policy}-{example.name}.tree.txt").write_text(result.trace.tree)
        results.append(
            {
                "name": example.name,
                "policy": policy,
                "answer": result.answer,
                "expected": example.answer,
                "accuracy": accuracy,
                "retained": [item.summary for item in result.kept_items],
                "usage": {
                    "prompt_tokens": result.usage.prompt_tokens,
                    "completion_tokens": result.usage.completion_tokens,
                    "calls": result.usage.calls,
                },
            }
        )
    return {
        "policy": policy,
        "examples": len(examples),
        "accuracy": round(total / max(1, len(examples)), 3),
        "results": results,
    }


def policy_sweep(
    examples: Sequence[BenchmarkExample],
    policies: Sequence[str],
    *,
    budget: int,
    max_depth: int,
    output_dir: str | Path | None = None,
) -> list[dict[str, object]]:
    summaries = []
    for policy in policies:
        summaries.append(run_dataset(examples, policy, budget=budget, max_depth=max_depth, output_dir=output_dir))
    return summaries


def format_table(rows: Iterable[dict[str, object]]) -> str:
    lines = ["| policy | examples | accuracy |", "| --- | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['policy']} | {row['examples']} | {row['accuracy']:.3f} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run nanoRLM synthetic or repo-backed benchmarks.")
    parser.add_argument("--dataset", choices=["pairbench", "needlepairs", "verifiers_20"], default="pairbench")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--budget", type=int, default=120)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--repo-root", type=str, default="/tmp/nanorlm-verifiers")
    parser.add_argument("--output-dir", type=str, default="")
    args = parser.parse_args()

    if args.dataset == "pairbench":
        examples = build_pairbench(n=args.limit, seed=0)
    elif args.dataset == "needlepairs":
        examples = build_needlepairs(n=args.limit, seed=0)
    else:
        examples = load_verifiers_20(args.repo_root)[: args.limit]

    summaries = policy_sweep(
        examples,
        ["keep_recent", "summary_only", "single_critic_topk", "pairwise_tournament"],
        budget=args.budget,
        max_depth=args.depth,
        output_dir=args.output_dir or None,
    )
    print(format_table(summaries))
    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / f"{args.dataset}-summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
