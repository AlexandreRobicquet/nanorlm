## Why

Briefly explain the user-facing, research, or maintenance reason for this change.

## What Changed

- 

## Validation

- [ ] `uv lock --check`
- [ ] `uv sync --frozen`
- [ ] `uv run python -m unittest discover -s tests -v`
- [ ] `uv run python -m py_compile nanorlm.py policies.py bench.py scripts/prepare_ruler_external_jsonl.py scripts/run_benchmark_e2e.py examples/run_verifiers.py examples/run_needlepairs.py examples/run_dossiers.py examples/run_planning.py showcases/planning.py showcases/generate_assets.py`
- [ ] `uv run python bench.py --dataset pairbench --limit 4 --budget 60 --depth 2`
- [ ] `uv run python bench.py --dataset verifiers_smoke --limit 2 --budget 80 --depth 2 --repo-root tests/fixtures/verifiers-mini`
- [ ] `uv run python scripts/run_benchmark_e2e.py --phases smoke --smoke-limit 1`

## Repo-Specific Evidence

- [ ] If benchmark numbers or README metrics changed, I included the exact command and the updated numbers.
- [ ] If recursion, retention, or trace behavior changed, I added a test or included a trace snippet.

## Risks

Call out any behavior changes, weak results, or follow-up work plainly.
