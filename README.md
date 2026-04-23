# nanoRLM

`nanoRLM` is a small, inference-only reference implementation of Recursive Language Models with pluggable retention policies and a query-conditioned `pairwise_tournament` memory selector.

The goal is not to be a framework. The goal is to be the repo you can read in one sitting and still get real recursive traces, real benchmark bundles, and real retention-policy ablations out of it.

## What We Are Building

![nanoRLM recursive memory loop](showcases/assets/dossierbench/architecture.svg)

The whole repo is this loop: start with a root query over too much context, recurse until each shard is small enough to inspect, turn leaf inspections into explicit `MemoryItem`s, keep only what survives the token budget, then answer from retained evidence instead of the full context.

If the retention policy drops a needed fact, the final answer loses it too. That is the central research surface in `nanoRLM`.

## Thesis

Modern long-context systems still fail in a very specific way: they look at everything, but they do not reliably keep the right intermediate evidence under pressure.

`nanoRLM` focuses on that exact gap:

- a tiny recursive inference loop with one small provider seam
- a deterministic offline backend for tests and smoke demos
- two tiny network backends: OpenAI-compatible and Anthropic Messages
- a minimal memory interface with swappable retention policies
- a novel `pairwise_tournament` policy that tries to preserve complementary evidence instead of just top-scoring snippets

## Quickstart With `uv`

This repo is meant to stay easy to run from a fresh machine with `uv`.

If you are learning the repo day to day, use this flow first:

```bash
uv sync
uv run python --version
uv run python -m unittest discover -s tests -v
uv run python bench.py --dataset pairbench --limit 4 --budget 60 --depth 2
```

The repo pins Python in [`.python-version`](.python-version), keeps project metadata in [`pyproject.toml`](pyproject.toml), and resolves the environment through [`uv.lock`](uv.lock).

For the repo-specific mental model, exact smoke commands, and a short cheat sheet, see [UV.md](UV.md).

## Tiny Example

```python
from nanorlm import ContextBlock, RLM, RLMConfig

context = [
    ContextBlock(name="left.txt", text="PAIR_ID: pair-000\nFACT_KIND: left\nFACT_VALUE: amber"),
    ContextBlock(name="right.txt", text="PAIR_ID: pair-000\nFACT_KIND: right\nFACT_VALUE: orbit"),
]

config = RLMConfig(
    model="demo/heuristic",
    provider="heuristic",
    max_depth=4,
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

`provider` selects `heuristic`, `openai_compatible`, `anthropic`, or `auto`. `base_url` is optional and defaults to the right endpoint for the chosen network provider.

`RLM(config).completion(query, context)` returns an `RLMResult` with:

- `answer`
- `trace`
- `usage`
- `cost_estimate`
- `kept_items`
- `retention_stats`
- `provenance_hits`
- `drop_reasons`
- `per_step_budget`

## Benchmarks

The repo now emits a stable report bundle:

- `summary.json`
- `per_case.jsonl`
- `curves.json`
- `trace_examples/`

Current local dossier run (`12` cases, budget `80`, depth `4`):

| policy | answer | prov | compact | avg toks |
| --- | ---: | ---: | ---: | ---: |
| direct_full_context | 0.667 | 1.000 | 0.552 | 35.8 |
| keep_recent | 0.000 | 0.222 | 0.144 | 68.5 |
| summary_only | 0.000 | 0.444 | 0.066 | 74.8 |
| single_critic_topk | 0.083 | 0.222 | 0.134 | 69.2 |
| pairwise_tournament | 0.333 | 0.583 | 0.162 | 67.1 |

On the dossier showcase, the important comparison is the controlled ablation: at budget `80` / depth `4`, `pairwise_tournament` beats `single_critic_topk`, and the same win persists across seeds in the test suite.

![Retained trace](showcases/assets/dossierbench/trace_card.svg)

## Showcases

### 1. Codebase QA

`examples/verifiers_30.json` is a curated `Verifiers-30` benchmark over `PrimeIntellect-ai/verifiers`, organized across:

- `defaults-flags`
- `config-resolution`
- `implementation-location`

Run it with:

```bash
git clone --depth 1 https://github.com/PrimeIntellect-ai/verifiers.git /tmp/nanorlm-verifiers
python examples/run_verifiers.py --repo-root /tmp/nanorlm-verifiers
```

The deterministic backend is only a smoke path here. The flagship use is to point the same engine at a real OpenAI-compatible model:

```bash
export OPENAI_API_KEY=...
python examples/run_verifiers.py \
  --provider openai-compatible \
  --model gpt-4.1-mini \
  --base-url https://api.openai.com/v1 \
  --repo-root /tmp/nanorlm-verifiers
```

For a local OpenAI-compatible endpoint such as Ollama:

```bash
python examples/run_verifiers.py \
  --provider openai-compatible \
  --model qwen3:14b \
  --base-url http://localhost:11434/v1 \
  --repo-root /tmp/nanorlm-verifiers
```

For native Claude through Anthropic:

```bash
export ANTHROPIC_API_KEY=...
python examples/run_verifiers.py \
  --provider anthropic \
  --model <your-claude-model> \
  --repo-root /tmp/nanorlm-verifiers
```

Portability limits:

- `any local LLM` here means any local model served behind an OpenAI-compatible `chat/completions` endpoint such as Ollama, `vLLM`, `llama.cpp` server, `LM Studio`, or `LocalAI`
- native Claude works through the Anthropic Messages API
- bespoke local runtime APIs are intentionally out of scope

### 2. Long-Horizon Dossiers

`examples/run_dossiers.py` is the “sexy” retention demo: noisy incident, migration, and release-blocker dossiers where the answer depends on keeping complementary clues across recursive branches.

```bash
python examples/run_dossiers.py --limit 12 --budget 80 --depth 4
```

This is the cleanest local story for the retention thesis.

### 3. Grounded Planning

`examples/run_planning.py` turns retained evidence into a read-only patch plan with ordered steps, citations, and explicit unknowns.

```bash
python examples/run_planning.py \
  --repo-root /tmp/nanorlm-verifiers \
  --limit 10 \
  --budget 140 \
  --depth 2
```

The planning suite writes markdown plans plus `summary.json` / `per_case.jsonl` under `showcases/outputs/planning/`.

### 4. PairBench And NeedlePairs

For the most minimal synthetic demonstrations:

```bash
python bench.py --dataset pairbench --limit 10 --budget 60 --depth 2
python examples/run_needlepairs.py --limit 10 --budget 60 --depth 3
```

These are useful for quick policy sanity checks and test-friendly regressions.

## Generate Assets

Run a benchmark, then turn its saved report bundle into launch-ready figures:

```bash
python showcases/generate_assets.py --report-dir outputs/dossierbench
```

This writes:

- `benchmark_snapshot.md`
- `architecture.svg`
- `policy_curve.svg`
- `trace_card.svg`

The showcase workflow is documented in [showcases/README.md](showcases/README.md).

## Repo Layout

- `nanorlm.py`: recursion loop, trace recorder, OpenAI-compatible backend, Anthropic backend, deterministic backend
- `policies.py`: `keep_recent`, `summary_only`, `single_critic_topk`, `pairwise_tournament`
- `bench.py`: datasets, evaluation harness, curve generation, report bundle writer
- `examples/`: minimal runnable demos
- `showcases/`: launch-facing demos, planning suite, asset generation
- `tests/`: recursion, policy, report-bundle, smoke-fixture, and planning tests

## Testing

If you want the repo-safe `uv` version of each command, prefix it as `uv run python ...`.

```bash
python -m unittest discover -s tests -v
python -m py_compile nanorlm.py policies.py bench.py
python bench.py --dataset pairbench --limit 4 --budget 60 --depth 2
python bench.py --dataset verifiers_smoke --limit 2 --budget 80 --depth 2 --repo-root tests/fixtures/verifiers-mini
```

A GitHub Actions smoke workflow runs the same core checks on pushes and pull requests.

## Current Scope

Implemented now:

- small recursive inference engine with stable public API
- four retention policies
- provider portability across heuristic, OpenAI-compatible, and Anthropic backends
- richer `RLMResult` metadata for retention analysis
- synthetic `PairBench`, `NeedlePairs`, and dossier benchmarks
- curated `Verifiers-30` repo-QA benchmark
- grounded planning showcase
- JSONL/tree traces and asset generation from saved reports

Still intentionally out of scope for this phase:

- training infrastructure
- framework-style agent abstractions
- Docker sandbox execution
- fully autonomous coding loops
