# Benchmark Snapshot

These numbers come from local smoke runs with the deterministic heuristic backend.

## PairBench

```text
uv run python bench.py --dataset pairbench --limit 10 --budget 60 --depth 2

| policy | examples | accuracy |
| --- | ---: | ---: |
| keep_recent | 10 | 0.400 |
| summary_only | 10 | 0.000 |
| single_critic_topk | 10 | 0.100 |
| pairwise_tournament | 10 | 0.700 |
```

## NeedlePairs

```text
uv run python bench.py --dataset needlepairs --limit 10 --budget 60 --depth 2

| policy | examples | accuracy |
| --- | ---: | ---: |
| keep_recent | 10 | 0.300 |
| summary_only | 10 | 0.000 |
| single_critic_topk | 10 | 0.300 |
| pairwise_tournament | 10 | 0.700 |
```

## Verifiers-20 Smoke Slice

```text
uv run python bench.py --dataset verifiers_20 --limit 5 --budget 120 --depth 2 --repo-root /tmp/nanorlm-verifiers

| policy | examples | accuracy |
| --- | ---: | ---: |
| keep_recent | 5 | 0.000 |
| summary_only | 5 | 0.000 |
| single_critic_topk | 5 | 0.200 |
| pairwise_tournament | 5 | 0.200 |
```
