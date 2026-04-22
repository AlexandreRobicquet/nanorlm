from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench import build_needlepairs, policy_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NeedlePairs demo benchmark.")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--budget", type=int, default=120)
    parser.add_argument("--depth", type=int, default=2)
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
    )
    print("| policy | examples | accuracy |")
    print("| --- | ---: | ---: |")
    for row in summaries:
        print(f"| {row['policy']} | {row['examples']} | {row['accuracy']:.3f} |")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
