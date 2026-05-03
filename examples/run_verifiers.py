from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import (
    CLI_PROVIDER_CHOICES,
    format_table,
    generate_curves,
    load_verifiers_30,
    policy_sweep,
    resolve_provider_choice,
    run_dataset,
    write_report_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Verifiers-30 demo benchmark.")
    parser.add_argument("--repo-root", type=str, default="/tmp/nanorlm-verifiers")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--budget", type=int, default=140)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="examples/outputs/verifiers_30")
    parser.add_argument("--provider", choices=CLI_PROVIDER_CHOICES, default="heuristic")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--base-url", type=str, default="")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--openai", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    provider = resolve_provider_choice(args.provider, args.openai)
    examples = load_verifiers_30(args.repo_root)[: args.limit]
    output_dir = Path(args.output_dir)
    if provider != "heuristic":
        summary = run_dataset(
            examples,
            "pairwise_tournament",
            budget=args.budget,
            max_depth=args.depth,
            output_dir=output_dir,
            provider=provider,
            model=args.model,
            base_url=args.base_url or None,
            api_key=args.api_key or None,
            dataset_name="verifiers_30",
        )
        print(json.dumps(summary, indent=2))
        return

    summaries = policy_sweep(
        examples,
        ["direct_full_context", "keep_recent", "summary_only", "single_critic_topk", "pairwise_tournament"],
        budget=args.budget,
        max_depth=args.depth,
        output_dir=output_dir,
        dataset_name="verifiers_30",
    )
    curves = generate_curves(
        "verifiers_30",
        lambda seed: load_verifiers_30(args.repo_root, seed=seed)[: args.limit],
        policies=["direct_full_context", "summary_only", "pairwise_tournament"],
        budgets=[100, args.budget, 180],
        depths=[1, args.depth],
        seeds=[0],
        provider=provider,
        model=args.model,
        base_url=args.base_url or None,
        api_key=args.api_key or None,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_report_bundle(
        output_dir,
        dataset_name="verifiers_30",
        summaries=summaries,
        curves=curves,
        command=(
            "python examples/run_verifiers.py "
            f"--limit {args.limit} --budget {args.budget} --depth {args.depth} --provider {args.provider}"
        ),
    )
    print(format_table(summaries))


if __name__ == "__main__":
    main()
