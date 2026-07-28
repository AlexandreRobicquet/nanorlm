# Contributing to nanoRLM

`nanoRLM` is a clone-only reference repository, not a pip-installable library. Make changes and run
commands from the source-checkout root. Keep contributions small, inspectable, and independent of
any one benchmark.

## Prerequisites and first setup

- Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) using its official
  instructions.
- Use Python 3.11 or 3.12. CI tests both; [`.python-version`](.python-version) selects 3.11 by
  default.
- Clone the repository and work inside that checkout. There is no supported package-install path.

Set up a fresh checkout, confirm the interpreter, and run the fastest deterministic benchmark
smoke:

```bash
uv sync --frozen
uv run python --version
uv run python bench.py --dataset pairbench --limit 4 --budget 60 --depth 2
```

The smoke run is offline, uses the deterministic backend, prints a table, and intentionally writes
no report bundle.

## Edit, test, and debug

During an edit, run the narrowest relevant test module. For example:

```bash
uv run python -m unittest tests/test_nanorlm.py -v
```

Then run both complete test entrypoints:

```bash
uv run python -m unittest discover -s tests -v
uv run --frozen pytest
```

Before every PR, run the entire [canonical verification block](UV.md#canonical-verification).
That block is the source of truth for the lock check, frozen sync, compile check, deterministic
smoke commands, and smoke/learned e2e commands; do not copy a shortened version into a PR.

Use this ladder to choose fast feedback while editing. The complete canonical block remains the
pre-PR requirement for every row.

| Change | Fast feedback before the full block |
| --- | --- |
| Documentation | Check rendered links and commands you changed; run `tests/test_readme.py` for README examples. |
| Tests | Run the changed test module directly. |
| Engine, provider, trace, or result contract | Run `tests/test_nanorlm.py` and, when applicable, `tests/test_backends.py`. |
| Retention policy or learned scorer | Run `tests/test_policies.py` and the relevant deterministic benchmark. |
| Benchmark, example, showcase, or report | Run its smallest offline command and inspect any saved report bundle. |
| Workflow, Python version, or dependency metadata | Run the lock check and frozen sync first; let CI confirm both supported Python minors. |

CI is intentionally offline and lightweight. Networked benchmarks and real-model runs are evidence
workflows, not routine PR gates.

## Code ownership map

- [`nanorlm.py`](nanorlm.py): recursive engine, providers, and trace/result contract.
- [`policies.py`](policies.py): retention baselines and their budget behavior.
- [`learned_retention.py`](learned_retention.py): dependency-free learned scorer and model format.
- [`bench.py`](bench.py): datasets, scoring, policy sweeps, and report bundles.
- [`examples/`](examples/): small runnable demos and deliberately reviewed benchmark receipts.
- [`showcases/`](showcases/): planning and launch-facing asset workflows.
- [`scripts/`](scripts/): training, conversion, e2e orchestration, and maintenance utilities.
- [`tests/`](tests/): deterministic contracts for all of the above.

Keep core behavior benchmark-agnostic. A dataset-specific rule belongs in the harness, not in the
engine or a general retention policy.

## Generated files and durable receipts

Local environments and caches such as `.venv/`, `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`,
and `.local/` are ignored. Run outputs belong under the ignored `outputs/`, `examples/outputs/`, or
`showcases/outputs/` roots. Raw `*.jsonl` and `*.log` files are also ignored except for deliberate
test fixtures.

Start benchmark work in an ignored output root. Promote only small, reviewed, reproducible receipts
to [`examples/`](examples/) or [`showcases/assets/`](showcases/assets/), together with the exact
source and command that produced them. Never commit `.env` files, API keys, provider caches, raw
model-response caches, or unreviewed large outputs.

For a saved run, read `experiment_report.md` first and keep the complete ignored bundle available
for review: `summary.json`, `per_case.jsonl`, `curves.json`, `experiment_report.md`, and
`trace_examples/`. Do not commit that bundle wholesale.

## Research evidence

- Synthetic datasets and checked-in fixtures are regression evidence, not leaderboard evidence.
- External-data or real-model claims must name the exact dataset source or revision, model and
  provider, full command, cost cap, and report-bundle path.
- Keep hosted-model runs cached and bounded. Normal validation must not spend API budget.
- Report negative or inconclusive results plainly in the receipt and the PR's **Risks** section.
  Do not hide them, select only favorable seeds, or turn an operationally successful run into a
  positive research claim.

## Project conventions and PRs

The repository currently enforces lockfile consistency, unittest and pytest suites, compilation,
and deterministic smoke checks. It deliberately has no separate lint, autoformat, static-typing,
or coverage threshold gate. Do not claim or invent those gates in a contribution.

Preserve source-checkout behavior, offline determinism, and conservative benchmark language.
Explain user-visible or research behavior changes, add a focused regression test, and keep unrelated
refactors out of the same PR. Finish with the
[pull-request checklist](.github/pull_request_template.md),
including weak, negative, or follow-up results under **Risks**.
