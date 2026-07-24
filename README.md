# nanoRLM

`nanoRLM` is a small, inference-only reference implementation for recursive long-context inspection with pluggable retention policies.

The goal is not to be a framework. The goal is to be the repo you can read in one sitting and still get real recursive traces, provider-portable runs, and reproducible report bundles out of it.

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
- a tournament-style `pairwise_tournament` policy that tries to preserve complementary evidence instead of just top-scoring snippets
- an offline-trained `learned_retention` policy for testing whether retention can be learned from benchmark traces under fixed budgets

## Quickstart With `uv`

This repo is meant to stay easy to run from a fresh machine with `uv`.

If you are learning the repo day to day, use this flow first:

```bash
uv sync
uv run python --version
uv run python -m unittest discover -s tests -v
uv run python bench.py --dataset verifiers_smoke --limit 2 --budget 80 --depth 2 --repo-root tests/fixtures/verifiers-mini
uv run python examples/run_dossiers.py --limit 4 --budget 80 --depth 4
uv run python scripts/run_benchmark_e2e.py --phases learned --learned-train-limit 4 --learned-eval-limit 4
```

The repo pins Python in [`.python-version`](.python-version), keeps project metadata in [`pyproject.toml`](pyproject.toml), and resolves the environment through [`uv.lock`](uv.lock).

For the repo-specific mental model, exact smoke commands, and a short cheat sheet, see [UV.md](UV.md).

## Tiny Example

```python
from nanorlm import ContextBlock, RLM, RLMConfig

context = [
    ContextBlock(name="incident-a.txt", text="The API gateway rollout is blocked by a stale endpoint registry cache."),
    ContextBlock(name="incident-b.txt", text="Reloading the registry and invalidating the cache unblocks the rollout."),
    ContextBlock(name="incident-c.txt", text="The infra team owns the fix and plans the patch after the next deploy window."),
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
    "What is blocking the rollout, and what change fixes it?",
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
- `drop_reasons`
- `per_step_budget`
- `retention_decisions`, with the complete candidate set, selected ranks, and budget for each retention step

Benchmark rows add scoring fields such as `answer_accuracy`, `provenance_score`, and `provenance_hits`. Those are harness-level checks against expected answers and expected provenance, not engine output.

## Evidence Status

The repo already emits a stable report bundle:

- `summary.json`
- `per_case.jsonl`
- `curves.json`
- `experiment_report.md`
- `trace_examples/`

That makes the current artifact useful for:

- provider-portable recursive runs over the same engine
- readable recursive traces and retained-memory inspection
- compact post-run analysis of policy deltas, task-level misses, and failure clusters
- `Verifiers-30` codebase-QA runs with real model backends
- dossier and planning showcases that demonstrate how retention changes what evidence survives
- learned-retention training rows, model JSON, and positive/negative reports against `pairwise_tournament`

What it does **not** claim yet:

- real-model headline results on established long-context benchmarks such as `RULER` or `BABILong`
- a paper-faithful model-directed RLM runtime
- public benchmark evidence that `pairwise_tournament` is generally superior beyond the repo's internal smoke and regression fixtures
- leaderboard evidence that `learned_retention` wins on real RULER or BABILong exports

`examples/benchmark_snapshot.md` is intentionally a deterministic smoke snapshot, not a public benchmark leaderboard.

![Retained trace](showcases/assets/dossierbench/trace_card.svg)

## Showcases

### 1. Codebase QA

`examples/verifiers_30.json` is a curated `Verifiers-30` benchmark over `PrimeIntellect-ai/verifiers`, organized across:

- `defaults-flags`
- `config-resolution`
- `implementation-location`

Run it with:

```bash
git init /tmp/nanorlm-verifiers
git -C /tmp/nanorlm-verifiers remote add origin https://github.com/PrimeIntellect-ai/verifiers.git
git -C /tmp/nanorlm-verifiers fetch --depth 1 origin 482e28ffa1f2613325867badaba4707b7c751d28
git -C /tmp/nanorlm-verifiers checkout --detach FETCH_HEAD

uv run python examples/run_verifiers.py \
  --repo-root /tmp/nanorlm-verifiers \
  --limit 30
```

This is the full 30-case benchmark. The CLI default remains a quick 10-case sample when `--limit` is omitted. The pinned revision is recorded alongside the actual checkout revision in generated `summary.json` metadata.
The compatibility source of truth is `examples/verifiers_compatibility.json`. To check either dataset against a checkout without running benchmark policies:

```bash
uv run python scripts/check_verifiers_compatibility.py --repo-root /tmp/nanorlm-verifiers
```

The deterministic backend is only a smoke path here. The flagship use is to point the same engine at a real OpenAI-compatible model:

```bash
export OPENAI_API_KEY=...
uv run python examples/run_verifiers.py \
  --provider openai-compatible \
  --model gpt-4.1-mini \
  --base-url https://api.openai.com/v1 \
  --max-estimated-cost 5 \
  --repo-root /tmp/nanorlm-verifiers \
  --limit 10
```

For a local OpenAI-compatible endpoint such as Ollama:

```bash
uv run python examples/run_verifiers.py \
  --provider openai-compatible \
  --model qwen3:14b \
  --base-url http://localhost:11434/v1 \
  --repo-root /tmp/nanorlm-verifiers \
  --limit 10
```

The Anthropic Messages backend is implemented, but the benchmark harness currently rejects Anthropic and unknown remote models because report bundles include cost estimates and there is no checked-in pricing table for those models.

Portability limits:

- `any local LLM` here means any local model served behind an OpenAI-compatible `chat/completions` endpoint such as Ollama, `vLLM`, `llama.cpp` server, `LM Studio`, or `LocalAI`
- native Claude works through the Anthropic Messages API at the backend-contract level, but it is not a priced benchmark-report path yet
- bespoke local runtime APIs are intentionally out of scope

### 2. Long-Horizon Dossiers

`examples/run_dossiers.py` is the main retention showcase: noisy incident, migration, and release-blocker dossiers where the answer depends on keeping complementary clues across recursive branches.

```bash
uv run python examples/run_dossiers.py --limit 12 --budget 80 --depth 4
```

Treat dossier results as an internal synthetic regression surface, not as headline evidence of general long-context performance.

### 3. Learned Retention

`learned_retention` treats memory retention as a small offline contextual-bandit-style scorer. The trainer runs a collection policy, records every candidate set seen at real retention steps, labels candidates from answer and provenance evidence, and optimizes a trajectory-reward-weighted pairwise ranking objective within each decision. The saved trajectory reward uses the same answer, provenance, compactness, latency, and cost contract as evaluation; offline heuristic collection has zero model cost and uses zero collection-latency penalty for deterministic training. The trainer writes both raw trajectory records and derived candidate rows as JSONL before saving the model.

Pairwise training requires `--training-source traces`, where candidates share an explicit retention decision. The legacy `--training-source blocks` ablation is only valid with `--objective pointwise`.

```bash
uv run python scripts/train_learned_retention.py \
  --datasets pairbench,dossierbench,ruler_synthetic,babilong_synthetic,external_jsonl \
  --train-seeds 0,1 \
  --limit 12 \
  --output-dir outputs/learned_retention

uv run python bench.py \
  --dataset ruler_synthetic \
  --seed 2 \
  --limit 12 \
  --budget 90 \
  --depth 4 \
  --policies direct_full_context,keep_recent,single_critic_topk,pairwise_tournament,learned_retention \
  --learned-retention-model outputs/learned_retention/learned_retention_model.json \
  --output-dir outputs/learned_retention/ruler_eval
```

For the full offline workflow, use:

```bash
uv run python scripts/run_benchmark_e2e.py --phases learned
```

That phase trains on offline slices, evaluates on held-out seeds, and writes `learned_retention_report.md`. The report is allowed to be negative. A win requires a reward delta of at least `0.01` with no answer or provenance regression. Only completed, equal-size DossierBench, Verifiers-30, or explicitly supplied external RULER/BABILong comparisons with at least eight examples are acceptance-eligible. If the learned policy does not beat `pairwise_tournament` on at least two eligible slices, the bundle should be read as evidence for where hand-coded retention is still enough.

To add distinct external RULER and BABILong exports to the same fixed-budget comparison, convert them to the external JSONL contract and pass both paths:

```bash
uv run python scripts/prepare_ruler_external_jsonl.py \
  --input /tmp/ruler-raw.jsonl \
  --output /tmp/nanorlm-ruler.jsonl

uv run python scripts/prepare_ruler_external_jsonl.py \
  --input /tmp/babilong-raw.jsonl \
  --output /tmp/nanorlm-babilong.jsonl \
  --benchmark BABILong \
  --task-prefix babilong

uv run python scripts/run_benchmark_e2e.py \
  --phases learned \
  --learned-verifiers-repo-root /tmp/nanorlm-verifiers \
  --learned-ruler-path /tmp/nanorlm-ruler.jsonl \
  --learned-babilong-path /tmp/nanorlm-babilong.jsonl
```

The learned report labels these as `ruler_external` and `babilong_external`. They remain local evaluation slices, not leaderboard submissions.

To include the full `Verifiers-30` curated slice in training or eval, first use the pinned shallow checkout above and pass it as `--repo-root`; for example add `verifiers_30` to `--datasets` when running `scripts/train_learned_retention.py`.

### 4. Grounded Planning

`examples/run_planning.py` turns retained evidence into a read-only patch plan with ordered steps, citations, and explicit unknowns.

```bash
uv run python examples/run_planning.py \
  --repo-root /tmp/nanorlm-verifiers \
  --limit 10 \
  --budget 140 \
  --depth 2
```

The planning suite writes markdown plans plus `summary.json` / `per_case.jsonl` under `showcases/outputs/planning/`.
It uses the same compatibility preflight and records the pinned and actual Verifiers revisions in `summary.json`.

### 5. PairBench, NeedlePairs, RULER Synthetic, And BABILong Synthetic

For the smallest synthetic sanity checks:

```bash
uv run python bench.py --dataset pairbench --limit 10 --budget 60 --depth 2
uv run python examples/run_needlepairs.py --limit 10 --budget 60 --depth 3
uv run python bench.py --dataset ruler_synthetic --limit 10 --budget 90 --depth 4
uv run python bench.py --dataset babilong_synthetic --limit 10 --budget 90 --depth 4
```

These are useful for quick smoke tests, trace demos, and test-friendly regressions. The RULER and BABILong variants are synthetic task-shape slices for multi-hop, aggregation, and distributed-fact retention; they are not official benchmark results.

### 6. External Benchmark JSONL

`external_jsonl` is an adapter for externally generated long-context benchmark exports, including RULER-style JSONL rows. It lets the same nanoRLM harness run over established benchmark data without vendoring benchmark datasets into this repo.

```bash
uv run python bench.py \
  --dataset external_jsonl \
  --dataset-path /tmp/ruler-or-other-long-context-export.jsonl \
  --limit 4 \
  --budget 80 \
  --depth 2
```

This is adapter support, not a published benchmark result. Any README metrics from external data should include the exact generation source, command, model, and output bundle.

For RULER-generated JSON or JSONL files, first normalize the export into the adapter shape:

```bash
uv run python scripts/prepare_ruler_external_jsonl.py \
  --input /tmp/ruler-generated.jsonl \
  --output /tmp/nanorlm-ruler-small.jsonl \
  --limit 12
```

For a bounded OpenAI-compatible real-model run, use a cache directory plus a cost cap:

```bash
uv run python bench.py \
  --dataset external_jsonl \
  --dataset-path /tmp/nanorlm-ruler-small.jsonl \
  --limit 12 \
  --policies direct_full_context,keep_recent,pairwise_tournament \
  --budget 120 \
  --depth 3 \
  --provider openai-compatible \
  --model gpt-5.4-mini \
  --base-url https://api.openai.com/v1 \
  --cache-dir outputs/cache/openai-gpt-5.4-mini \
  --max-estimated-cost 20 \
  --output-dir outputs/real-runs/openai-ruler-small
```

`direct_full_context` is a true direct-answer baseline: the answer step receives every raw context block and does not apply the retention budget. Recursive policies inspect shards into retained memory and then answer only from what survives the budget.

`--max-estimated-cost` is global for the whole policy sweep, not per policy. The harness checks the completed-case cumulative estimate before starting each case, so a single final case can move the total slightly past the cap before the next case or policy stops. Remote cost reporting is intentionally limited to the built-in priced OpenAI-compatible model table; unknown OpenAI-compatible models and Anthropic benchmark runs are rejected instead of reported as zero-cost.

Network-provider report bundles avoid hidden second-pass API calls: their `curves.json` is derived from the already completed summaries rather than re-running the sweep.

Small OpenAI-backed snapshots are tracked as mechanics and reproducibility artifacts, not headline benchmark claims:

- [`examples/real_runs/openai_ruler_small/benchmark_snapshot.md`](examples/real_runs/openai_ruler_small/benchmark_snapshot.md)
- [`examples/real_runs/openai_external_mini/benchmark_snapshot.md`](examples/real_runs/openai_external_mini/benchmark_snapshot.md)

## Generate Assets

Run a benchmark, then turn its saved report bundle into launch-ready figures:

```bash
uv run python showcases/generate_assets.py --report-dir outputs/dossierbench
```

This writes:

- `benchmark_snapshot.md`
- `architecture.svg`
- `policy_curve.svg`
- `trace_card.svg`

The showcase workflow is documented in [showcases/README.md](showcases/README.md).

## Benchmark E2E Workflow

Use the e2e runner when you want the repo checks, benchmark smoke paths, report bundles, and generated assets captured in one manifest:

```bash
uv run python scripts/run_benchmark_e2e.py
```

By default this runs local checks, smoke benchmarks, synthetic benchmarks, the checked-in external JSONL fixture, and asset generation under `outputs/e2e/<run-id>/`.

For repo-QA coverage against a local Verifiers checkout:

```bash
uv run python scripts/run_benchmark_e2e.py --phases offline --repo-root /tmp/nanorlm-verifiers
```

Use the pinned shallow Verifiers checkout from the Codebase QA section; a current-HEAD clone is intentionally not the reproducible compatibility target.

For a bounded hosted-model run, first generate or provide an external benchmark JSONL file, then run only the real-model phase with an explicit cache:

```bash
export OPENAI_API_KEY=...
uv run python scripts/run_benchmark_e2e.py \
  --phases real_model \
  --external-dataset-path /tmp/nanorlm-ruler-small.jsonl \
  --real-model gpt-4.1-mini \
  --real-cache-dir outputs/cache/openai-gpt-4.1-mini
```

Hosted OpenAI-compatible runs fail fast when the model has no cost table entry or no API key. The cost cap is enforced between benchmark cases, not before each individual model call.

## Repo Layout

- `nanorlm.py`: recursion loop, trace recorder, OpenAI-compatible backend, Anthropic backend, deterministic backend
- `policies.py`: `keep_recent`, `summary_only`, `single_critic_topk`, `pairwise_tournament`
- `learned_retention.py`: feature extraction, pairwise/pointwise offline training, and the learned policy
- `bench.py`: datasets, evaluation harness, curve generation, report bundle writer
- `scripts/train_learned_retention.py`: retention-trace collection and offline model training
- `scripts/run_benchmark_e2e.py`: e2e benchmark orchestration, manifests, and artifact checks
- `examples/`: minimal runnable demos
- `showcases/`: launch-facing demos, planning suite, asset generation
- `tests/`: recursion, policy, report-bundle, smoke-fixture, and planning tests

## Testing

Use [UV.md](UV.md#canonical-verification) as the canonical local verification path.

GitHub Actions keeps PR checks fast:

- `CI` runs `uv lock --check`, frozen sync, unit tests, and the compile check on Python 3.11 and 3.12.
- `smoke` uses the same `uv` setup on Python 3.11, then runs the same core checks plus the deterministic PairBench and Verifiers smoke fixtures.

CI intentionally does not run real-model jobs, networked benchmark jobs, or full benchmark sweeps.

## Current Scope

Implemented now:

- small recursive inference engine with stable public API
- five retention policies
- provider portability across heuristic, OpenAI-compatible, and Anthropic backends
- richer `RLMResult` metadata for retention analysis
- synthetic `PairBench`, `NeedlePairs`, and dossier fixtures for smoke and regression use
- curated `Verifiers-30` repo-QA benchmark
- external JSONL adapter for established benchmark exports
- grounded planning showcase
- JSONL/tree traces and asset generation from saved reports

Still intentionally out of scope for this phase:

- large-scale online RL and distributed training infrastructure
- framework-style agent abstractions
- Docker sandbox execution
- fully autonomous coding loops
