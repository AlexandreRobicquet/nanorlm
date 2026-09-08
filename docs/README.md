# Documentation

The [project README](../README.md) explains the motivation and the first run.
Use these guides when you want to go further. All commands run from the repository
root.

| Guide | Purpose |
| --- | --- |
| [Repository questions](repository-questions.md) | Ask about code, inspect citations, reuse evidence, and control costs. |
| [Recursive engine](engine.md) | Run the small Python example and understand memory, budgets, and traces. |
| [Experiments](experiments.md) | Compare policies, train retention, use external datasets, and reproduce evidence bundles. |
| [Results](results.md) | Read the frozen repository-answer comparison, costs, failures, and limitations. |
| [Next steps](roadmap.md) | See the work justified by the current results and its acceptance criteria. |
| [Development](development.md) | Set up `uv` and run the canonical verification commands. |

For code changes, start with [Contributing](../CONTRIBUTING.md). The
[evaluation protocol](../evaluations/README.md) records how the published scores
were produced; [showcases](../showcases/README.md) covers planning and figure
generation.

## Repository layout

- [`nanorlm/`](../nanorlm/): runtime, question CLI, policies, and benchmark harness.
- [`scripts/`](../scripts/): evaluation, training, conversion, and maintenance commands.
- [`examples/`](../examples/): runnable demos, dataset definitions, and historical snapshots.
- [`evaluations/`](../evaluations/): frozen questions, scoring protocol, and audited results.
- [`showcases/`](../showcases/): planning demos and generated figures.
- [`tests/`](../tests/): offline tests and fixtures.
- [`docs/`](./): guides, optional templates, and archived project records.

## Commands after the layout cleanup

The Python files now live together in `nanorlm/`. Run repository questions with
`uv run python -m nanorlm` and benchmarks with `uv run python -m nanorlm.bench`.
These replace the former `ask.py` and `bench.py` commands. The engine import stays
`from nanorlm import ContextBlock, RLM, RLMConfig`; supporting imports now use
`nanorlm.policies`, `nanorlm.bench`, and the other package modules. The project
still runs from a source checkout.

## Templates and history

[Agent instructions](templates/AGENTS.example.md),
[Claude reminders](templates/CLAUDE.example.md), and the
[local roadmap template](templates/ROADMAP.example.md) are optional starting
points. Copy them to `AGENTS.md`, `CLAUDE.md`, or `ROADMAP.md` at the repository root
for private use; those local files remain ignored. The current public priorities
are in [Next steps](roadmap.md).

The completed [onboarding audit](archive/onboarding-audit.md) and
[July 2026 acceptance report](archive/onboarding-acceptance-2026-07-28.md) preserve
historical commands and observations. They describe their recorded revisions.
