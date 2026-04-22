# nanoRLM

`nanoRLM` is a minimal, inference-only reference implementation of Recursive Language Models with pluggable memory-retention policies.

The design goal is simple: you should be able to read the whole core in one sitting, understand the control flow, and then start swapping retention policies or benchmarks without wading through a framework.

## What Is In The Repo

- `nanorlm.py`: the core recursion loop, trace recorder, OpenAI-compatible transport, and deterministic offline backend
- `policies.py`: `keep_recent`, `summary_only`, `single_critic_topk`, and `pairwise_tournament`
- `bench.py`: synthetic ablations plus a curated `Verifiers-20` loader
- `examples/`: runnable scripts and the `verifiers_20.json` question set
- `tests/`: unit tests for recursion, budget enforcement, and policy behavior

## Why This Exists

There are already serious RLM implementations inside larger research stacks. What is still missing is the `nanoGPT`-style artifact: something small enough to study line by line, but real enough to produce interesting traces and retention-policy results.

This repo leans into that gap:

- one small OpenAI-compatible code path
- one deterministic offline backend for tests and smoke demos
- one novel retention angle: a pairwise-critic tournament that tries to preserve complementary evidence under a hard memory budget

## Quickstart

```bash
uv sync
uv run python -m unittest discover -s tests -v
uv run python bench.py --dataset pairbench --limit 10 --budget 60 --depth 2
uv run python bench.py --dataset needlepairs --limit 10 --budget 60 --depth 2
```

To run the curated repo-QA demo over Prime Intellect's `verifiers` repository:

```bash
git clone --depth 1 https://github.com/PrimeIntellect-ai/verifiers.git /tmp/nanorlm-verifiers
uv run python examples/run_verifiers.py --repo-root /tmp/nanorlm-verifiers
```

To swap in a real OpenAI-compatible model:

```bash
export OPENAI_API_KEY=...
uv run python examples/run_verifiers.py \
  --openai \
  --model gpt-4.1-mini \
  --base-url https://api.openai.com/v1 \
  --repo-root /tmp/nanorlm-verifiers
```

For a local OpenAI-compatible endpoint such as Ollama:

```bash
uv run python examples/run_verifiers.py \
  --openai \
  --model qwen3:14b \
  --base-url http://localhost:11434/v1 \
  --repo-root /tmp/nanorlm-verifiers
```

## Core API

```python
from nanorlm import ContextBlock, RLM, RLMConfig

context = [
    ContextBlock(name="left.txt", text="PAIR_ID: pair-000\nFACT_KIND: left\nFACT_VALUE: amber"),
    ContextBlock(name="right.txt", text="PAIR_ID: pair-000\nFACT_KIND: right\nFACT_VALUE: orbit"),
]

config = RLMConfig(
    model="demo/heuristic",
    base_url="http://localhost:11434/v1",
    max_depth=2,
    memory_budget_tokens=60,
    retention_policy="pairwise_tournament",
    seed=0,
)

result = RLM(config).completion(
    "For pair-000, what is the full code? Combine the left token and the right token.",
    context,
)

print(result.answer)
print(result.trace.tree)
```

`RLMConfig` exposes the public control surface:

- `model`
- `base_url`
- `api_key`
- `max_depth`
- `max_steps`
- `memory_budget_tokens`
- `retention_policy`
- `sandbox`
- `seed`

`RLM(config).completion(query, context)` returns an `RLMResult` with:

- `answer`
- `trace`
- `usage`
- `cost_estimate`
- `kept_items`

## Retention Policies

- `keep_recent`: blunt recency baseline
- `summary_only`: aggressively compresses memory and drops structured metadata
- `single_critic_topk`: scores each candidate independently
- `pairwise_tournament`: runs a lightweight tournament, then biases final selection toward complementary evidence

The memory item schema is intentionally simple:

- `summary`
- `provenance`
- `raw_pointer`
- `tokens`
- `depth`
- `timestamp`

The implementation also keeps optional `answer_candidate`, `confidence`, and lightweight metadata when a backend can extract it.

## Benchmarks

### PairBench

Synthetic long-context tasks where the answer depends on retaining the right pair of facts under a hard memory budget.

Local smoke run:

```text
uv run python bench.py --dataset pairbench --limit 10 --budget 60 --depth 2

| policy | examples | accuracy |
| --- | ---: | ---: |
| keep_recent | 10 | 0.400 |
| summary_only | 10 | 0.000 |
| single_critic_topk | 10 | 0.100 |
| pairwise_tournament | 10 | 0.700 |
```

### NeedlePairs

The same complementary-fact problem dropped into a much noisier haystack.

Local smoke run:

```text
uv run python bench.py --dataset needlepairs --limit 10 --budget 60 --depth 2

| policy | examples | accuracy |
| --- | ---: | ---: |
| keep_recent | 10 | 0.300 |
| summary_only | 10 | 0.000 |
| single_critic_topk | 10 | 0.300 |
| pairwise_tournament | 10 | 0.700 |
```

### Verifiers-20

`examples/verifiers_20.json` is a curated set of codebase-QA questions over `PrimeIntellect-ai/verifiers`. The deterministic heuristic backend is only a sanity path here; the real point of this benchmark is to run the same recursion engine against a real OpenAI-compatible model and inspect the retained trace.

Heuristic smoke run over `5` examples:

```text
uv run python bench.py --dataset verifiers_20 --limit 5 --budget 120 --depth 2 --repo-root /tmp/nanorlm-verifiers

| policy | examples | accuracy |
| --- | ---: | ---: |
| keep_recent | 5 | 0.000 |
| summary_only | 5 | 0.000 |
| single_critic_topk | 5 | 0.200 |
| pairwise_tournament | 5 | 0.200 |
```

## Example Trace

This tree came from a real local run of `pairwise_tournament` on `pair-000`:

```text
- [inspect] root query (query=For pair-000, what is the full code? Combine the left token and the right token., blocks=22)
- [split] root split (groups=2, blocks=22)
  - [recurse] root.0 (blocks=11, tokens=269)
  - [split] root.0 split (groups=2, blocks=11)
    - [recurse] root.0.0 (blocks=5, tokens=120)
    - [inspect] root.0.0 leaf (tokens=120, blocks=5)
  - [retain] root.0 policy=pairwise_tournament (before=1, after=1, budget=60)
    - [recurse] root.0.1 (blocks=6, tokens=149)
    - [inspect] root.0.1 leaf (tokens=149, blocks=6)
  - [retain] root.0 policy=pairwise_tournament (before=2, after=2, budget=60)
- [retain] root policy=pairwise_tournament (before=2, after=2, budget=60)
  - [recurse] root.1 (blocks=11, tokens=276)
  - [split] root.1 split (groups=2, blocks=11)
    - [recurse] root.1.0 (blocks=5, tokens=126)
    - [inspect] root.1.0 leaf (tokens=126, blocks=5)
  - [retain] root.1 policy=pairwise_tournament (before=1, after=1, budget=60)
    - [recurse] root.1.1 (blocks=6, tokens=150)
    - [inspect] root.1.1 leaf (tokens=150, blocks=6)
  - [retain] root.1 policy=pairwise_tournament (before=2, after=2, budget=60)
- [retain] root policy=pairwise_tournament (before=4, after=3, budget=60)
- [final_answer] compose answer (retained=3, answer_preview=amber orbit)
```

Each run also emits JSONL traces. The benchmark scripts write them to `outputs/<dataset>/`.

## Current Scope

This is a strong `v0.1`, not the entire July launch in one commit.

Implemented now:

- minimal recursive inference engine
- four retention policies
- deterministic offline backend for tests and smoke demos
- OpenAI-compatible backend for real runs
- synthetic ablations and a curated `verifiers` dataset
- JSONL traces plus a compact tree renderer

Deliberately not implemented yet:

- Docker sandbox execution
- polished plotting or paper-ready figures
- a full public `PairBench-100` artifact checked into the repo
- stronger model-backed `Verifiers-20` numbers in the README

## Testing

```bash
uv run python -m unittest discover -s tests -v
```

The test suite covers:

- recursive execution and trace generation
- memory-budget enforcement
- policy-specific behavior
- synthetic cases where pairwise retention beats a simpler baseline
