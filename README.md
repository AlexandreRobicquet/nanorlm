# nanoRLM

**What should a model keep when it cannot keep everything?**

nanoRLM is a small Python research project for studying how language models retain
evidence under a fixed memory budget. It includes a practical way to ask questions
about a codebase, inspect the cited sources, and compare simple retrieval, full
context, and recursive retention.

## Why this exists

A question about a codebase rarely has its whole answer in one place. A default
lives in one file, an override in another, and the test that explains an exception
somewhere else. A useful answer has to connect those pieces.

Recursive inspection breaks a large context into smaller pieces. But each time
those pieces are compressed into memory, evidence can disappear. Keep the default
and lose the override, and the final answer can sound right while being wrong.

nanoRLM makes that tradeoff explicit: **what was inspected, what survived, what was
dropped, and what it cost.** You can change the retention policy, follow the trace,
and check whether the answer improves against simpler baselines. The implementation
stays small enough to read, and additional complexity has to earn its place in the
results.

## How it works

The recursive engine splits context into bounded pieces, inspects each piece into
memory items with source provenance, and applies a retention policy as the branches
merge. The final answer uses the evidence that survives the budget.

```text
Question + context → split → inspect → retain within budget → answer
                               └──── sources, decisions, omissions, usage ────┘
```

Five policies expose different choices: keep recent evidence, summarize it, rank
items individually, compare pairs, or use an offline-trained scorer. The
[engine guide](docs/engine.md) contains a runnable example that keeps a deployment
blocker and its fix while dropping two distractors.

Repository questions provide a concrete test of the idea. They return original
source excerpts and citations, plus reusable evidence and a cost receipt. The
default uses lexical retrieval; recursive retention is available for comparison.

## Try it

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/), clone this
repository, and run from its root. Python 3.11 and 3.12 are supported. This project
is used from a source checkout; package installation is not supported.

Start with a local evidence preview:

```bash
uv sync --frozen
uv run python -m nanorlm \
  'Where is the HTTP retry limit set, what overrides the delay, and which tests cover it?' \
  --repo . --output outputs/retry-evidence
```

Open `outputs/retry-evidence/sources.md` to inspect the selected source spans.
Without `--model`, this creates evidence only and makes no model requests. The
initial sync may download the locked development tools.

To generate an answer from that same evidence, set `OPENAI_API_KEY` in your
environment and run:

```bash
uv run python -m nanorlm \
  'Where is the HTTP retry limit set, what overrides the delay, and which tests cover it?' \
  --evidence outputs/retry-evidence/evidence.json \
  --model gpt-4.1-mini-2025-04-14 --max-cost 0.25 \
  --output outputs/retry-answer
```

This sends the question and selected source spans to OpenAI. Open
`outputs/retry-answer/answer.md` for the cited answer and `run.json` in the same
directory for usage and estimated cost. Use a fresh output directory for each run.
The [repository-question guide](docs/repository-questions.md) explains strategies,
budgets, evidence reuse, and source coverage.

To explore retention without an API key, run the deterministic dossier demo:

```bash
uv run python examples/run_dossiers.py \
  --limit 4 --budget 80 --depth 4 --output-dir outputs/quickstart/dossierbench
```

Open `outputs/quickstart/dossierbench/experiment_report.md`, then follow a case into
`trace_examples/` to see which evidence each policy kept. This synthetic demo tests
the mechanics; it does not measure real-model performance.

## What the evidence says

**Simple retrieval is the current default because it performed best in the
checked-in comparison.** On 30 frozen questions across backoff, python-dotenv, and
cachetools, using `gpt-4.1-mini-2025-04-14`:

| Approach | Complete answers | Supported claims | Model calls |
| --- | ---: | ---: | ---: |
| Lexical retrieval | **16/30** | **77.5%** | **30** |
| Full eligible context | 9/30 | 72.4% | 30 |
| Recursive retention | 3/30 | 69.2% | 514 |

Retrieval also had the lowest inference cost. This is a small, assistant-authored
and audited evaluation on three Python repositories, with one answer per
question and strategy. “Complete” means satisfying the frozen fact checklist with
no material error; citation support is measured separately. Even the best approach
requires checking the cited source.

The [results and limitations](docs/results.md) and
[frozen protocol](evaluations/README.md) document the failures, costs, and grading.
These results test the pairwise retention policy; they do not evaluate learned
retention. The project does not yet establish a general retention advantage or
headline results on official RULER or BABILong benchmarks.

The [next steps](docs/roadmap.md) follow that evidence: improve citation reliability
and retrieval, then evaluate on fresh repositories before adding complexity.

## Read the code

Start with the [engine and its example](docs/engine.md), then follow the path you
care about:

| File | What to look for |
| --- | --- |
| [`nanorlm/__init__.py`](nanorlm/__init__.py) | The recursive loop, memory items, providers, and trace contract. |
| [`nanorlm/policies.py`](nanorlm/policies.py) | Retention policies, side by side. |
| [`nanorlm/repoqa.py`](nanorlm/repoqa.py) | Source retrieval, answer validation, and evidence receipts. |
| [`nanorlm/bench.py`](nanorlm/bench.py) | Datasets, scoring, and reproducible policy comparisons. |

The runtime uses the Python standard library. It includes a deterministic backend
for offline work and OpenAI-compatible and Anthropic backends for model calls.
The engine uses fixed recursive splits; model-directed recursion and autonomous
coding are outside its current scope.

[Documentation](docs/README.md) · [Experiment recipes](docs/experiments.md) ·
[Contributing](CONTRIBUTING.md) · [MIT license](LICENSE)
