# nanoRLM

`nanoRLM` is a small, inference-only reference implementation for recursive long-context inspection with pluggable retention policies.

The goal is not to be a framework. The goal is a compact reference whose documented minimum
reading path explains the core end to end while still producing real recursive traces,
provider-portable runs, and reproducible report bundles.

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

`nanoRLM` is a clone-only reference repository. Run it from a source checkout; a
pip-installed library and an installed public API are not supported. The import examples below
work because the checkout root is the active working directory.

The repository is meant to stay easy to run from a fresh machine with `uv`.
Install `uv` with its
[official installation instructions](https://docs.astral.sh/uv/getting-started/installation/)
before running the first command.

If you are learning the repo day to day, use this flow first:

The initial `uv sync` may download the exact locked tools and therefore needs package-network
access on an empty machine; it needs no account credential and incurs no model API cost. Every test
and benchmark step after that sync is offline. The stdout-only smoke writes nothing; the dossier and
learned phases write small JSON, JSONL, Markdown, and trace bundles only under the named ignored
`outputs/` roots. The fixed limits make this a short local workflow, although exact runtime and
trace size vary by machine.

```bash
uv sync
uv run python --version
uv run python -m unittest discover -s tests -v
uv run python bench.py --dataset verifiers_smoke --limit 2 --budget 80 --depth 2 --repo-root tests/fixtures/verifiers-mini
uv run python examples/run_dossiers.py --limit 4 --budget 80 --depth 4 --output-dir outputs/quickstart/dossierbench
uv run python scripts/run_benchmark_e2e.py --phases learned --learned-train-limit 4 --learned-eval-limit 4 --output-root outputs/e2e --run-id quickstart-learned
```

The `verifiers_smoke` command intentionally omits `--output-dir`: it prints a policy table and
writes no report bundle. The dossier and e2e commands persist evidence at the paths named above.

The repo pins Python in [`.python-version`](.python-version), keeps project metadata in [`pyproject.toml`](pyproject.toml), and resolves the environment through [`uv.lock`](uv.lock).

For the repo-specific mental model, exact smoke commands, and a short cheat sheet, see [UV.md](UV.md).
To make a change, start with the one-page [contributor guide](CONTRIBUTING.md).

## Tiny Example

```python
from nanorlm import ContextBlock, RLM, RLMConfig

context = [
    ContextBlock(
        name="incident-a.txt",
        text="Deployment validation says the API gateway rollout is blocked by a "
        "stale endpoint registry cache from the previous release; the new gateway "
        "binary passed all of its health checks.",
    ),
    ContextBlock(
        name="incident-b.txt",
        text="The rollout can proceed by reloading the endpoint registry and "
        "invalidating the cache before the gateway reads route metadata again, then "
        "rerunning deployment validation against every refreshed endpoint.",
    ),
    ContextBlock(
        name="incident-c.txt",
        text="The observability team completed its dashboard migration and archived "
        "the old alert definitions after confirming that historical charts and "
        "service-level panels render correctly in every production region.",
    ),
    ContextBlock(
        name="incident-d.txt",
        text="A separate storage review recommends revisiting backup retention next "
        "quarter, after capacity forecasts, recovery drills, and vendor pricing "
        "have been updated by the infrastructure finance group.",
    ),
]

config = RLMConfig(
    model="demo/heuristic",
    provider="heuristic",
    max_depth=4,
    memory_budget_tokens=80,
    retention_policy="pairwise_tournament",
    seed=0,
)

result = RLM(config).completion(
    (
        "What blocks the API gateway rollout, and how should the endpoint "
        "registry and cache be refreshed to fix it?"
    ),
    context,
)

print(result.answer)
print(result.trace.tree)
print("retained:", sorted(item.provenance for item in result.kept_items))
print("dropped:", sorted(item["provenance"] for item in result.drop_reasons))
print("max memory depth:", result.retention_stats["max_memory_depth"])
```

Expected output (abridged):

```text
... stale endpoint registry cache ...
... reloading the endpoint registry and invalidating the cache ...
- [split] root split ...
  - [split] root.0 split ...
    - [inspect] root.0.0 leaf ...
retained: ['incident-a.txt', 'incident-b.txt']
dropped: ['incident-c.txt', 'incident-d.txt']
max memory depth: 2
```

The root context and both of its halves exceed the engine's 64-token leaf floor, so the run creates four depth-2 leaf memories; the 80-token budget then keeps the complementary blocker and fix while dropping both distractors.

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

A direct `bench.py` run writes this bundle only when `--output-dir` is supplied. Omitting the flag
selects intentional stdout-only smoke mode. For a saved run, open
`<output-dir>/experiment_report.md` first; use `<output-dir>/summary.json` as the machine-readable
entry point.

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

Operational boundary: the shallow Git fetch is networked but uses a public repository and needs no
model credential. The benchmark that follows is deterministic and has no API cost. The checkout's
disk use and fetch time depend on the upstream repository and connection. The run uses a 30-row
dataset across the documented policy and curve sweep, writes one report bundle under
`outputs/verifiers_30/heuristic/`, and scales with the policy/curve grid as well as checkout size.

Run it with:

```bash
git init /tmp/nanorlm-verifiers
git -C /tmp/nanorlm-verifiers remote add origin https://github.com/PrimeIntellect-ai/verifiers.git
git -C /tmp/nanorlm-verifiers fetch --depth 1 origin 482e28ffa1f2613325867badaba4707b7c751d28
git -C /tmp/nanorlm-verifiers checkout --detach FETCH_HEAD

uv run python examples/run_verifiers.py \
  --repo-root /tmp/nanorlm-verifiers \
  --limit 30 \
  --output-dir outputs/verifiers_30/heuristic
```

This is the full 30-case benchmark. The CLI default remains a quick 10-case sample when `--limit` is omitted. The pinned revision is recorded alongside the actual checkout revision in generated `summary.json` metadata.
The compatibility source of truth is `examples/verifiers_compatibility.json`. To check either dataset against a checkout without running benchmark policies:

```bash
uv run python scripts/check_verifiers_compatibility.py --repo-root /tmp/nanorlm-verifiers
```

The deterministic backend is only a smoke path here. The flagship use is to point the same engine at
a real OpenAI-compatible model.

Operational boundary: this command is networked, requires `OPENAI_API_KEY`, and writes both cached
responses and a report bundle below `outputs/verifiers_30/`. Runtime depends on provider latency and
rate limits. The `$5` estimate guard is enforced between cases, so the final completed case can move
the estimate slightly beyond the nominal cap.

```bash
export OPENAI_API_KEY=...
uv run python examples/run_verifiers.py \
  --provider openai-compatible \
  --model gpt-4.1-mini \
  --base-url https://api.openai.com/v1 \
  --cache-dir outputs/cache/openai-gpt-4.1-mini \
  --max-estimated-cost 5 \
  --repo-root /tmp/nanorlm-verifiers \
  --limit 10 \
  --output-dir outputs/verifiers_30/openai-gpt-4.1-mini
```

For a local OpenAI-compatible endpoint such as Ollama, start the server before running this command.
It uses loopback networking, needs no hosted API credential, and incurs no hosted-model charge; the
repository therefore applies no dollar guard. Output lands in
`outputs/verifiers_30/local-qwen3-14b/`, while runtime and external model storage depend on the
local server and hardware.

```bash
uv run python examples/run_verifiers.py \
  --provider openai-compatible \
  --model qwen3:14b \
  --base-url http://localhost:11434/v1 \
  --repo-root /tmp/nanorlm-verifiers \
  --limit 10 \
  --output-dir outputs/verifiers_30/local-qwen3-14b
```

The Anthropic Messages backend is implemented, but the benchmark harness currently rejects Anthropic and unknown remote models because report bundles include cost estimates and there is no checked-in pricing table for those models.

Portability limits:

- `any local LLM` here means any local model served behind an OpenAI-compatible `chat/completions` endpoint such as Ollama, `vLLM`, `llama.cpp` server, `LM Studio`, or `LocalAI`
- native Claude works through the Anthropic Messages API at the backend-contract level, but it is not a priced benchmark-report path yet
- bespoke local runtime APIs are intentionally out of scope

### 2. Long-Horizon Dossiers

`examples/run_dossiers.py` is the main retention showcase: noisy incident, migration, and release-blocker dossiers where the answer depends on keeping complementary clues across recursive branches.

This 12-case workflow is offline, deterministic, credential-free, and has no API cost. It is a
bounded local run that writes one report bundle and its traces under `outputs/dossierbench/`; exact
runtime and bundle size scale with the policy sweep, curve grid, and trace depth.

```bash
uv run python examples/run_dossiers.py \
  --limit 12 \
  --budget 80 \
  --depth 4 \
  --output-dir outputs/dossierbench
```

Treat dossier results as an internal synthetic regression surface, not as headline evidence of general long-context performance.

### 3. Learned Retention

`learned_retention` treats memory retention as a small offline contextual-bandit-style scorer. The trainer runs a collection policy, records every candidate set seen at real retention steps, labels candidates from answer and provenance evidence, and optimizes a trajectory-reward-weighted pairwise ranking objective within each decision. The saved trajectory reward uses the same answer, provenance, compactness, latency, and cost contract as evaluation; offline heuristic collection has zero model cost and uses zero collection-latency penalty for deterministic training. The trainer writes both raw trajectory records and derived candidate rows as JSONL before saving the model.

Pairwise training requires `--training-source traces`, where candidates share an explicit retention decision. The legacy `--training-source blocks` ablation is only valid with `--objective pointwise`.

These training and evaluation commands are offline, require no credentials, and incur no API cost.
They write raw training rows, traces, a small model JSON, and evaluation bundles below
`outputs/learned_retention/`. Treat them as a multi-step local workflow rather than a smoke test:
runtime and disk use scale with datasets, seeds, and trace counts.

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

For the full offline workflow, use the e2e form below. It needs no credentials or API budget and
writes multiple report bundles plus `learned_retention_report.md` under
`outputs/e2e/learned/`; runtime and disk use scale with the configured training and evaluation
slices.

```bash
uv run python scripts/run_benchmark_e2e.py \
  --phases learned \
  --output-root outputs/e2e \
  --run-id learned
```

That phase trains on offline slices and evaluates held-out seeds. A top-level e2e status of
`passed` means the commands and artifact checks completed; it is not a positive research verdict.
`learned_retention_report.md` may still report `negative_or_inconclusive`. A win requires a reward
delta of at least `0.01` with no answer or provenance regression. Only completed, equal-size
DossierBench, Verifiers-30, or explicitly supplied external RULER/BABILong comparisons with at
least eight examples are acceptance-eligible. If the learned policy does not beat
`pairwise_tournament` on at least two eligible slices, the bundle should be read as evidence for
where hand-coded retention is still enough.

To add distinct external RULER and BABILong exports to the same fixed-budget comparison, convert
them to the external JSONL contract and pass both paths. After the input files and pinned Verifiers
checkout exist, these commands are offline, credential-free, and have no API cost. Conversion
outputs go to `/tmp`; the e2e run writes multiple bundles under
`outputs/e2e/learned-external/`. Runtime and disk use depend on the supplied dataset sizes and
checkout.

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
  --learned-babilong-path /tmp/nanorlm-babilong.jsonl \
  --output-root outputs/e2e \
  --run-id learned-external
```

The learned report labels these as `ruler_external` and `babilong_external`. They remain local evaluation slices, not leaderboard submissions.

To include the full `Verifiers-30` curated slice in training or eval, first use the pinned shallow checkout above and pass it as `--repo-root`; for example add `verifiers_30` to `--datasets` when running `scripts/train_learned_retention.py`.

### 4. Grounded Planning

`examples/run_planning.py` turns retained evidence into a read-only patch plan with ordered steps, citations, and explicit unknowns.

Once the pinned Verifiers checkout exists, this 10-task workflow is offline, deterministic,
credential-free, and has no API cost. It writes Markdown plans, JSON/JSONL summaries, and traces
under `showcases/outputs/planning/`; runtime and disk use scale with checkout size and trace depth.

```bash
uv run python examples/run_planning.py \
  --repo-root /tmp/nanorlm-verifiers \
  --limit 10 \
  --budget 140 \
  --depth 2 \
  --output-dir showcases/outputs/planning
```

The planning suite writes markdown plans plus `summary.json` / `per_case.jsonl` under `showcases/outputs/planning/`.
It uses the same compatibility preflight and records the pinned and actual Verifiers revisions in `summary.json`.

### 5. PairBench, NeedlePairs, RULER Synthetic, And BABILong Synthetic

For the smallest synthetic sanity checks:

All four commands are offline, deterministic, credential-free, and have no API cost. They are
bounded smoke-class runs: three print tables only, while NeedlePairs writes its small report bundle
under `examples/outputs/needlepairs/`.

```bash
uv run python bench.py --dataset pairbench --limit 10 --budget 60 --depth 2
uv run python examples/run_needlepairs.py --limit 10 --budget 60 --depth 3 --output-dir examples/outputs/needlepairs
uv run python bench.py --dataset ruler_synthetic --limit 10 --budget 90 --depth 4
uv run python bench.py --dataset babilong_synthetic --limit 10 --budget 90 --depth 4
```

The three direct `bench.py` commands intentionally use stdout-only smoke mode. The NeedlePairs
wrapper is evidence-producing and writes to its named output directory. These runs are useful for
quick smoke tests, trace demos, and test-friendly regressions. The RULER and BABILong variants are
synthetic task-shape slices for multi-hop, aggregation, and distributed-fact retention; they are
not official benchmark results.

### 6. External Benchmark JSONL

`external_jsonl` is an adapter for externally generated long-context benchmark exports, including RULER-style JSONL rows. It lets the same nanoRLM harness run over established benchmark data without vendoring benchmark datasets into this repo.

The runnable smoke below uses the two-row tracked fixture. It is offline, credential-free,
API-cost-free, and stdout-only. Replace the fixture path and limit with your own normalized export
when doing external-data work; runtime then scales with its row and context sizes.

```bash
uv run python bench.py \
  --dataset external_jsonl \
  --dataset-path tests/fixtures/external-benchmark-mini.jsonl \
  --limit 2 \
  --budget 80 \
  --depth 2
```

This intentionally omits `--output-dir` and is a stdout-only adapter smoke run; it does not write a
report bundle. This is adapter support, not a published benchmark result. Any README metrics from
external data should include the exact generation source, command, model, and output bundle.

For RULER-generated JSON or JSONL files, first normalize the export into the adapter shape. This
conversion is offline, needs no credentials or API budget, writes the named `/tmp` JSONL file, and
scales linearly with the input size.

```bash
uv run python scripts/prepare_ruler_external_jsonl.py \
  --input /tmp/ruler-generated.jsonl \
  --output /tmp/nanorlm-ruler-small.jsonl \
  --limit 12
```

For a bounded OpenAI-compatible real-model run, use a cache directory plus a cost cap. This command
is networked and requires `OPENAI_API_KEY`. It writes cached responses under
`outputs/cache/openai-gpt-5.4-mini/` and a report bundle under
`outputs/real-runs/openai-ruler-small/`; runtime depends on model latency, context size, and rate
limits. The `$20` guard is enforced between cases, so the final completed case may move the estimate
slightly beyond the nominal cap.

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

Run a benchmark, then turn its saved report bundle into launch-ready figures. Asset generation is
offline, needs no credentials, and incurs no API cost. It reads an existing bundle and writes four
small Markdown/SVG artifacts below `outputs/dossierbench/assets/`; runtime and disk use scale with
the supplied traces and curves.

```bash
uv run python showcases/generate_assets.py \
  --report-dir outputs/dossierbench \
  --assets-dir outputs/dossierbench/assets
```

This writes:

- `benchmark_snapshot.md`
- `architecture.svg`
- `policy_curve.svg`
- `trace_card.svg`

The showcase workflow is documented in [showcases/README.md](showcases/README.md).

## Benchmark E2E Workflow

Use the e2e runner when you want the repo checks, benchmark smoke paths, report bundles, and generated assets captured in one manifest:

The default command is model/data-offline after the locked tools are synced, credential-free, and
has no API cost. Its internal check repeats `uv sync --frozen`, which may need package-network
access in an empty environment. It writes several report bundles, logs, a manifest, and generated
assets under `outputs/e2e/default/`; expect a multi-step local workflow whose runtime and disk use
scale with case counts and traces. A top-level `status: passed` establishes operational completion
only, not a positive benchmark or policy verdict.

```bash
uv run python scripts/run_benchmark_e2e.py \
  --output-root outputs/e2e \
  --run-id default
```

By default this runs local checks, smoke benchmarks, synthetic benchmarks, the checked-in external JSONL fixture, and asset generation under `outputs/e2e/<run-id>/`.

For repo-QA coverage against a local Verifiers checkout, use the pinned shallow checkout from the
Codebase QA section. After that networked fetch and the locked-tool sync, this e2e command is
model/data-offline, needs no credentials, and has no API cost. Its internal frozen sync may still
need the package network in an empty environment. It writes its manifest, logs, and bundles under
`outputs/e2e/offline/`; runtime and disk use scale with the external checkout and bounded case
counts.

```bash
uv run python scripts/run_benchmark_e2e.py \
  --phases offline \
  --repo-root /tmp/nanorlm-verifiers \
  --output-root outputs/e2e \
  --run-id offline
```

The command name `offline` refers to model/network behavior during the run; it does not create or
update the required checkout. A current-HEAD clone is intentionally not the reproducible
compatibility target.

For a bounded hosted-model run, first generate or provide an external benchmark JSONL file, then run only the real-model phase with an explicit cache:

This phase is networked, requires `OPENAI_API_KEY`, and writes cached responses plus a report bundle
under the named `outputs/` roots. Runtime depends on dataset size, model latency, and provider rate
limits. The explicit `$20` estimate guard is enforced between cases, so a final completed case can
move slightly past it.

```bash
export OPENAI_API_KEY=...
uv run python scripts/run_benchmark_e2e.py \
  --phases real_model \
  --external-dataset-path /tmp/nanorlm-ruler-small.jsonl \
  --real-model gpt-4.1-mini \
  --real-cache-dir outputs/cache/openai-gpt-4.1-mini \
  --real-max-estimated-cost 20 \
  --output-root outputs/e2e \
  --run-id real-model
```

Hosted OpenAI-compatible runs fail fast when the model has no cost table entry or no API key. The cost cap is enforced between benchmark cases, not before each individual model call.

## Minimum Reading Path

To understand the core without reading every workflow and receipt:

1. Read [`nanorlm.py`](nanorlm.py) for the recursive engine and result/trace contract.
2. Read [`policies.py`](policies.py) for side-by-side retention behavior.
3. Read `build_pairbench` in [`bench.py`](bench.py) as one concrete dataset builder.
4. Inspect the saved tree in [`examples/pairbench_trace.txt`](examples/pairbench_trace.txt).
5. Run the quickstart dossier command and open
   `outputs/quickstart/dossierbench/experiment_report.md`.

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

- `CI` runs the lock and Markdown-link checks, frozen sync, stdlib unittest and locked pytest
  suites, and compilation on Python 3.11 and 3.12.
- `smoke` uses the same locked `uv` setup on Python 3.11, then runs unittest, compilation, and the
  deterministic PairBench and Verifiers smoke fixtures.

CI intentionally does not run real-model jobs, networked benchmark jobs, or full benchmark sweeps.

## Current Scope

Implemented now:

- small recursive inference engine with a stable source-checkout interface
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
