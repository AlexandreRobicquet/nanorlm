# Showcases

`showcases/` holds the launch-facing demos and artifact generators for `nanoRLM`.
Run them from the clone-only source-checkout root through `uv`; package installation is not
supported. Install `uv` from its
[official instructions](https://docs.astral.sh/uv/getting-started/installation/) and complete the
root [quickstart](../README.md#quickstart-with-uv) first.

## Recommended Runs

Full Verifiers-30 repo QA:

Prerequisite and boundary: create the pinned shallow Verifiers checkout documented in the root
[Codebase QA section](../README.md#1-codebase-qa). That fetch is networked and its time/disk use
depend on the public upstream repository, but it needs no model credential. The command below is
then offline, deterministic, and API-cost-free. It uses a 30-row dataset across the configured
policy/curve sweep, writes one report bundle under `outputs/verifiers_30/`, and scales with that
sweep as well as checkout size.

```bash
uv run python bench.py \
  --dataset verifiers_30 \
  --limit 30 \
  --budget 140 \
  --depth 2 \
  --repo-root /tmp/nanorlm-verifiers \
  --output-dir outputs/verifiers_30
```

Use `--limit 10` for a quick sample. The checkout must use the verified revision documented in the root README.

Long-horizon dossier benchmark:

This curve sweep is offline, deterministic, credential-free, and has no API cost. It writes one
bundle plus traces under `outputs/dossierbench/`; runtime and disk use scale with the three budgets,
two depths, three seeds, and saved traces, so treat it as a benchmark workflow rather than a
one-command smoke.

```bash
uv run python bench.py \
  --dataset dossierbench \
  --limit 12 \
  --budget 80 \
  --depth 4 \
  --curve-budgets 60,80,100 \
  --curve-depths 3,4 \
  --curve-seeds 0,1,2 \
  --output-dir outputs/dossierbench
```

External benchmark JSONL adapter:

The runnable smoke below uses the two-row tracked fixture. It is offline, needs no credentials or
API budget, and intentionally writes no files because it omits `--output-dir`. Substitute a
normalized export path and matching limit for external-data work; runtime then scales with its row
and context sizes.

```bash
uv run python bench.py \
  --dataset external_jsonl \
  --dataset-path tests/fixtures/external-benchmark-mini.jsonl \
  --limit 2 \
  --budget 80 \
  --depth 2
```

This runs externally generated benchmark rows through nanoRLM. It does not vendor benchmark data or turn adapter smoke output into headline evidence.
It intentionally omits `--output-dir` and is stdout-only; no report bundle is written.

Grounded planning showcase:

Prerequisite and boundary: use the same pinned Verifiers checkout as Codebase QA. After that public
network fetch, this 10-task command is offline, deterministic, credential-free, and API-cost-free.
It writes plans, JSON/JSONL summaries, and traces under `showcases/outputs/planning/`; runtime and
disk use scale with checkout size and trace depth.

```bash
uv run python examples/run_planning.py \
  --repo-root /tmp/nanorlm-verifiers \
  --limit 10 \
  --budget 140 \
  --depth 2 \
  --output-dir showcases/outputs/planning
```

Render launch assets from a saved report bundle:

This command requires the dossier bundle above. It is offline, credential-free, and API-cost-free,
and writes four small Markdown/SVG artifacts below `outputs/dossierbench/assets/`; duration and
disk use scale with the saved curves and traces.

```bash
uv run python showcases/generate_assets.py \
  --report-dir outputs/dossierbench \
  --assets-dir outputs/dossierbench/assets
```

## Output Contract

Direct `bench.py` runs and benchmark wrappers such as Verifiers and DossierBench produce the
following bundle when given `--output-dir`. Start with `experiment_report.md`; `summary.json` is
the machine-readable entry point.

- `summary.json`
- `per_case.jsonl`
- `curves.json`
- `experiment_report.md`
- `trace_examples/`

The planning workflow has its own contract under `showcases/outputs/planning/`:
`summary.json`, `per_case.jsonl`, `plans/`, and `traces/`.

Asset generation produces:

- `benchmark_snapshot.md`
- `architecture.svg`
- `policy_curve.svg`
- `trace_card.svg` when a saved tree trace is available
