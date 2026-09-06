"""One command: a repository question, source evidence, and a reviewable answer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from repoqa import run_question


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('question',help='Question about code, configuration, behavior, or tests')
    parser.add_argument('--repo',help='Repository or source directory; Git repositories use tracked files')
    parser.add_argument('--output',required=True,help='New empty evidence bundle directory')
    parser.add_argument('--model',help='OpenAI-compatible model; omitted means local evidence only')
    parser.add_argument('--preview',action='store_true',help='Build evidence without any model request')
    parser.add_argument('--evidence',help='Reuse a checksummed evidence.json for the same question')
    parser.add_argument('--strategy',choices=['lexical','full','retention'],default='lexical')
    parser.add_argument('--context-budget',type=int,default=6000)
    parser.add_argument('--candidate-budget',type=int,default=16000)
    parser.add_argument('--retention-budget',type=int,default=512)
    parser.add_argument('--retention-policy',choices=['keep_recent','summary_only','single_critic_topk','pairwise_tournament','learned_retention'],default='pairwise_tournament')
    parser.add_argument('--learned-model',help='Optional learned retention weights; experimental')
    parser.add_argument('--max-cost',type=float,default=.25,help='USD estimate guard before each model request')
    parser.add_argument('--max-output-tokens',type=int,default=1600)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run_question(repository=args.repo,question=args.question,output=args.output,
            strategy=args.strategy,model=args.model,preview=args.preview,evidence_in=args.evidence,
            context_budget=args.context_budget,candidate_budget=args.candidate_budget,
            retention_budget=args.retention_budget,retention_policy=args.retention_policy,
            learned_model=args.learned_model,max_cost=args.max_cost,max_output_tokens=args.max_output_tokens)
    except (ValueError,RuntimeError,OSError) as exc:
        print(f'Failed: {exc}')
        return 1
    print(json.dumps({'status':result['status'],'estimated_usd':result['estimated_usd'],
                      'answer':str(Path(args.output).resolve()/'answer.md')},indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
