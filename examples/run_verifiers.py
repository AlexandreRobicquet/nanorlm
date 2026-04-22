from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench import load_verifiers_20, policy_sweep, run_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Verifiers-20 demo benchmark.")
    parser.add_argument("--repo-root", type=str, default="/tmp/nanorlm-verifiers")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--budget", type=int, default=120)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="examples/outputs/verifiers")
    parser.add_argument("--openai", action="store_true", help="Use the OpenAI-compatible backend instead of the deterministic heuristic backend.")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--base-url", type=str, default="https://api.openai.com/v1")
    args = parser.parse_args()

    examples = load_verifiers_20(args.repo_root)[: args.limit]
    output_dir = Path(args.output_dir)
    if args.openai:
        summary = run_dataset(
            examples,
            "pairwise_tournament",
            budget=args.budget,
            max_depth=args.depth,
            output_dir=output_dir,
            use_openai_backend=True,
            model=args.model,
            base_url=args.base_url,
        )
        print(json.dumps(summary, indent=2))
        return

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
