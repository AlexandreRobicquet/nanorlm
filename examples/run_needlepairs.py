from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import build_needlepairs, format_table, generate_curves, policy_sweep, write_report_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NeedlePairs demo benchmark.")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="examples/outputs/needlepairs")
    args = parser.parse_args()

    examples = build_needlepairs(n=args.limit, seed=0)
    output_dir = Path(args.output_dir)
    summaries = policy_sweep(
        examples,
        ["keep_recent", "summary_only", "single_critic_topk", "pairwise_tournament"],
        budget=args.budget,
        max_depth=args.depth,
        output_dir=output_dir,
        dataset_name="needlepairs",
    )
    curves = generate_curves(
        "needlepairs",
        lambda seed: build_needlepairs(n=args.limit, seed=seed),
        policies=["keep_recent", "single_critic_topk", "pairwise_tournament"],
        budgets=[args.budget, 80],
        depths=[2, args.depth],
        seeds=[0, 1, 2],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_report_bundle(
        output_dir,
        dataset_name="needlepairs",
        summaries=summaries,
        curves=curves,
        command=f"python examples/run_needlepairs.py --limit {args.limit} --budget {args.budget} --depth {args.depth}",
    )
    print(format_table(summaries))
    (output_dir / "summary.pretty.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
