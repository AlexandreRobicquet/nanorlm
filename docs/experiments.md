# Experiments and evidence

Start with the [project overview](../README.md) or the [engine example](engine.md).
Run these commands from the source-checkout root after following the [setup guide](development.md).

Choose a workflow:

- [Codebase QA](#1-codebase-qa) with the pinned Verifiers checkout.
- [Dossiers](#2-long-horizon-dossiers) and [synthetic smoke tests](#5-pairbench-needlepairs-ruler-synthetic-and-babilong-synthetic) for offline exploration.
- [Learned retention](#3-learned-retention) for training and held-out evaluation.
- [Matched retention](#matched-retention-experiments) for comparisons using the same captured inspections.
- [External benchmarks](#6-external-benchmark-jsonl) for supplied datasets and bounded model runs.
- [Grounded planning](#4-grounded-planning), [figures](#generate-assets), and [the e2e runner](#benchmark-e2e-workflow).

## Evidence Status

The repo already emits a stable report bundle:

- `summary.json`
- `per_case.jsonl`
- `curves.json`
- `experiment_report.md`
- `trace_examples/`
- `loom_traces/`, exported as standalone LOOM trace-contract v0.1 JSONL

A direct `python -m nanorlm.bench` run writes this bundle only when `--output-dir` is supplied. Omitting the flag
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

### Matched Retention Experiments

For policy comparisons, keep retention scoring local and capture each leaf inspection once. Later policies replay the same inspection outputs while preserving the captured token/call ledger:

```bash
uv run python -m nanorlm.bench \
  --dataset external_jsonl \
  --dataset-path /tmp/nanorlm-ruler.jsonl \
  --limit 8 \
  --budget 128 \
  --depth 3 \
  --policies keep_recent,summary_only,single_critic_topk,pairwise_tournament \
  --retention-judge heuristic \
  --inspection-replay-dir outputs/matched-retention/inspection-replay \
  --output-dir outputs/matched-retention/ruler
```

`--retention-judge heuristic` prevents the generation model from receiving extra score/compare calls for critic policies. `--inspection-replay-dir` stores integrity-checked inspection outputs keyed by model configuration, query, branch, and context hashes; it does not store raw input context. Replay preserves logical prompt/completion/call usage for matched accounting, while `wall_ms` records the actual faster replay path. Use `--inspection-replay-mode replay_only` to fail closed if any expected capture is missing.

Because replay preserves the captured usage ledger, `--max-estimated-cost` remains a conservative counterfactual allocation across policies; it is an upper bound on the API work actually issued by a replayed sweep, not a billing receipt.

For a protocol-bound sweep, use `scripts/run_matched_retention.py`. It runs the five frozen policies task-by-task, selects the smallest structurally eligible budget from the development grid, reruns a fixed task in replay-only mode, validates every exported trace with a clean LOOM checkout, and writes a manifest, privacy audit, and checksums. Accuracy is deliberately excluded from budget selection.

Matched bundle schema 0.2 binds the bounded-inspection v2 runtime contract. Task IDs are assigned before execution and checked against every exported trace; duplicate policy rows and incomplete traversal fail the evidence gate. Reservation formula v3 accounts for inspections created by splitting oversized source blocks. These changes require fresh same-commit training, offline and preflight bundles. The historical 96/128/192 development grid is retained. Training evidence freezes the three-family suite, seeds 0/1, twelve tasks per family/seed, pairwise collection, and the documented default optimizer settings. Offline evidence also freezes all three families, four tasks per family from index 0, seed 0, depth 3, output cap 512 and the heuristic provider. In-memory path metadata is made portable before results and traces are generated. A failed eligibility gate remains a failed gate.

The default protocol sweep uses `dossierbench`, `ruler_synthetic`, and `babilong_synthetic`. PairBench remains an engineering fixture and is not included in the protocol model or budget-selection evidence.

The release gate requires a hash-bound learned-retention model and portable training manifest, clean nanoRLM and LOOM commits, and exact 40-character commit bindings. Hosted preflight and execution additionally require the offline bundle's nanoRLM commit to equal the current runner commit. A convenient offline sequence is:

```bash
uv run python scripts/train_learned_retention.py \
  --datasets dossierbench,ruler_synthetic,babilong_synthetic \
  --train-seeds 0,1 \
  --limit 12 \
  --repo-root tests/fixtures/verifiers-mini \
  --dataset-path tests/fixtures/external-benchmark-mini.jsonl \
  --output-dir outputs/protocol-model

uv run python scripts/run_matched_retention.py \
  --phase offline \
  --budgets 96,128,192 \
  --learned-retention-model outputs/protocol-model/learned_retention_model.json \
  --learned-retention-training-manifest outputs/protocol-model/manifest.json \
  --loom-root ../loom \
  --expected-nanorlm-commit "$(git rev-parse HEAD)" \
  --expected-loom-commit "$(git -C ../loom rev-parse HEAD)" \
  --output-dir outputs/matched-retention-offline
```

Before a hosted-model pilot, build a no-network preflight bundle from the passed offline manifest and the exact external JSONL exports. The pilot configuration is frozen to two eight-task families, the 96-token budget, `gpt-5.4-mini`, depth 3, a 512-token output cap, seed 0, and a USD 5 conservative reservation cap:

```bash
uv run python scripts/run_matched_retention.py \
  --phase pilot \
  --preflight-only \
  --dataset-spec ruler:external_jsonl:/path/to/ruler.jsonl \
  --dataset-spec babilong:external_jsonl:/path/to/babilong.jsonl \
  --expected-dataset-sha256 ruler=<64-character-source-sha256> \
  --expected-dataset-sha256 babilong=<64-character-source-sha256> \
  --limit 8 \
  --budgets 96 \
  --provider openai_compatible \
  --model gpt-5.4-mini \
  --max-estimated-cost 5 \
  --learned-retention-model outputs/protocol-model/learned_retention_model.json \
  --learned-retention-training-manifest outputs/protocol-model/manifest.json \
  --offline-manifest outputs/matched-retention-offline/manifest.json \
  --expected-offline-sha256 <64-character-offline-manifest-sha256> \
  --cache-dir outputs/cache/matched-retention-pilot \
  --loom-root ../loom \
  --expected-nanorlm-commit "$(git rev-parse HEAD)" \
  --expected-loom-commit "$(git -C ../loom rev-parse HEAD)" \
  --output-dir outputs/matched-retention-pilot-preflight
```

Preflight validates the source hashes, RULER/BABILong family metadata, task conversion, frozen model and training bundle, expected offline-manifest hash and complete offline-bundle checksum coverage, exact clean commits, persistent-cache mode, and conservative cost reservation without issuing model requests. It embeds canonical JSONL copies, replacing only local path metadata with portable placeholders, and records both source and embedded hashes.

The paid command must use the same arguments without `--preflight-only`, a new empty output directory, and both `--preflight-manifest <passed-preflight>/manifest.json` and `--expected-preflight-sha256 <64-character-manifest-sha256>`. Use a dedicated external `--cache-dir` for that frozen execution. Execution creates an exact binding over the nanoRLM/LOOM commits, task manifest, offline evidence, preflight evidence, and model configuration before network access. If a provider quota interrupts the run, keep that cache unchanged and rerun the identical command with another empty output directory; successful responses are reused without duplicate requests while their logical token and call usage remains in the matched ledger. The completed release copies the validated cache into `artifacts/response_cache` and includes every record in its inventory and checksum index.

Execution fails before network access unless the passed zero-network bundle has the same nanoRLM/LOOM commits, task manifest, dataset artifacts, model/training hashes, configuration, offline evidence, and a checksum index that exactly covers the manifest inventory and release bundle. A real-model bundle also fails the gate unless every policy row records exactly one model identifier returned by the provider; the configured alias alone is not accepted as runtime evidence. Do not edit, merge, or reuse a bound cache for a different execution.

The command exits nonzero whenever a release gate fails. Inspect `manifest.json` and each `budget-*/validation.json`; do not treat `smallest_eligible_budget_candidate` as frozen unless `selected_budget` is populated and the manifest status is `passed`.

`direct_full_context` remains an unmatched descriptive reference because it skips recursive inspection and retention. Do not include it in a matched-policy significance claim.

The exporter has no runtime dependency on LOOM. When both repos are available, validate generated traces with LOOM itself:

```bash
uv run loom-validate-trace /path/to/output/loom_traces/pairwise_tournament/example.jsonl
```

![Retained trace](../showcases/assets/dossierbench/trace_card.svg)

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

`learned_retention` treats memory retention as a small offline contextual-bandit-style scorer. The trainer runs a collection policy, records every candidate set seen at real retention steps, labels candidates from answer and provenance evidence, and optimizes a trajectory-reward-weighted pairwise ranking objective within each decision. The saved trajectory reward uses the same answer, provenance, compactness, and cost contract as evaluation; offline heuristic collection has zero model cost and records latency separately from quality reward. The trainer writes both raw trajectory records and derived candidate rows as JSONL before saving the model.

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

uv run python -m nanorlm.bench \
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
uv run python -m nanorlm.bench --dataset pairbench --limit 10 --budget 60 --depth 2
uv run python examples/run_needlepairs.py --limit 10 --budget 60 --depth 3 --output-dir examples/outputs/needlepairs
uv run python -m nanorlm.bench --dataset ruler_synthetic --limit 10 --budget 90 --depth 4
uv run python -m nanorlm.bench --dataset babilong_synthetic --limit 10 --budget 90 --depth 4
```

The three direct `python -m nanorlm.bench` commands intentionally use stdout-only smoke mode. The NeedlePairs
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
uv run python -m nanorlm.bench \
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
uv run python -m nanorlm.bench \
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

- [`examples/real_runs/openai_ruler_small/benchmark_snapshot.md`](../examples/real_runs/openai_ruler_small/benchmark_snapshot.md)
- [`examples/real_runs/openai_external_mini/benchmark_snapshot.md`](../examples/real_runs/openai_external_mini/benchmark_snapshot.md)

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

The showcase workflow is documented in [showcases/README.md](../showcases/README.md).

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
