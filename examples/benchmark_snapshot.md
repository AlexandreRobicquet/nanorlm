# Benchmark Snapshot

These are deterministic heuristic smoke runs after the Integrity Pass.

They are useful for checking that the recursive loop, retention policies, traces, and report bundle still work end to end. They are **not** evidence of general long-context performance, and they should not be read as headline benchmark claims.

`direct_full_context` is intentionally unbudgeted here: it answers from raw context instead of retained recursive memory, so its compactness and retained-token numbers are not directly comparable to budgeted policies.

## PairBench Smoke

Command:

```bash
uv run python bench.py --dataset pairbench --limit 4 --budget 60 --depth 2
```

| policy | examples | answer | prov | compact | avg toks |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct_full_context | 4 | 1.000 | 0.000 | 0.000 | 506.0 |
| keep_recent | 4 | 0.250 | 0.000 | 0.133 | 52.0 |
| summary_only | 4 | 1.000 | 0.000 | 0.267 | 44.0 |
| single_critic_topk | 4 | 0.250 | 0.000 | 0.133 | 52.0 |
| pairwise_tournament | 4 | 1.000 | 0.000 | 0.133 | 52.0 |

## NeedlePairs Smoke

Command:

```bash
uv run python bench.py --dataset needlepairs --limit 4 --budget 60 --depth 2
```

| policy | examples | answer | prov | compact | avg toks |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct_full_context | 4 | 1.000 | 0.000 | 0.000 | 13274.0 |
| keep_recent | 4 | 0.750 | 0.000 | 0.133 | 52.0 |
| summary_only | 4 | 1.000 | 0.000 | 0.267 | 44.0 |
| single_critic_topk | 4 | 0.750 | 0.000 | 0.133 | 52.0 |
| pairwise_tournament | 4 | 1.000 | 0.000 | 0.133 | 52.0 |

## Verifiers Smoke Fixture

Command:

```bash
uv run python bench.py --dataset verifiers_smoke --limit 2 --budget 80 --depth 2 --repo-root tests/fixtures/verifiers-mini
```

| policy | examples | answer | prov | compact | avg toks |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct_full_context | 2 | 1.000 | 1.000 | 0.000 | 255.5 |
| keep_recent | 2 | 0.000 | 0.500 | 0.188 | 65.0 |
| summary_only | 2 | 0.000 | 1.000 | 0.338 | 53.0 |
| single_critic_topk | 2 | 0.000 | 0.750 | 0.269 | 58.5 |
| pairwise_tournament | 2 | 0.000 | 0.750 | 0.269 | 58.5 |
