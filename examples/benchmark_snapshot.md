# Benchmark Snapshot

These are deterministic heuristic smoke runs after the Integrity Pass.

They are useful for checking that the recursive loop, retention policies, traces, and report bundle still work end to end. They are **not** evidence of general long-context performance, and they should not be read as headline benchmark claims.

## PairBench Smoke

Command:

```bash
uv run python bench.py --dataset pairbench --limit 4 --budget 60 --depth 2
```

| policy | examples | answer | prov | compact | avg toks |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct_full_context | 4 | 0.000 | 0.000 | 0.233 | 46.0 |
| keep_recent | 4 | 0.000 | 0.000 | 0.008 | 59.5 |
| summary_only | 4 | 0.000 | 0.000 | 0.291 | 42.5 |
| single_critic_topk | 4 | 0.000 | 0.000 | 0.100 | 54.0 |
| pairwise_tournament | 4 | 0.000 | 0.000 | 0.000 | 60.0 |

## NeedlePairs Smoke

Command:

```bash
uv run python bench.py --dataset needlepairs --limit 4 --budget 60 --depth 2
```

| policy | examples | answer | prov | compact | avg toks |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct_full_context | 4 | 0.000 | 0.000 | 0.233 | 46.0 |
| keep_recent | 4 | 0.000 | 0.000 | 0.013 | 59.2 |
| summary_only | 4 | 0.000 | 0.000 | 0.291 | 42.5 |
| single_critic_topk | 4 | 0.000 | 0.000 | 0.062 | 56.2 |
| pairwise_tournament | 4 | 0.000 | 0.000 | 0.000 | 60.0 |

## Verifiers Smoke Fixture

Command:

```bash
uv run python bench.py --dataset verifiers_smoke --limit 2 --budget 80 --depth 2 --repo-root tests/fixtures/verifiers-mini
```

| policy | examples | answer | prov | compact | avg toks |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct_full_context | 2 | 0.000 | 1.000 | 0.294 | 56.5 |
| keep_recent | 2 | 0.000 | 0.500 | 0.194 | 64.5 |
| summary_only | 2 | 0.000 | 1.000 | 0.338 | 53.0 |
| single_critic_topk | 2 | 0.000 | 0.500 | 0.475 | 42.0 |
| pairwise_tournament | 2 | 0.000 | 0.750 | 0.319 | 54.5 |
