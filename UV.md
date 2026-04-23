# Using `uv` In nanoRLM

This is the smallest `uv` guide you need to work confidently in this repo.

## Mental Model

- [`pyproject.toml`](pyproject.toml) is the project definition.
- [`.python-version`](.python-version) pins the expected interpreter to `3.11`.
- [`uv.lock`](uv.lock) is the resolved lockfile.
- [`.venv/`](.venv/) is the local environment `uv` manages for this repo.
- `[tool.uv] package = false` means this repo uses `uv` as an environment-and-runner workflow, not as a package publishing workflow.

## Three Commands That Matter Most

- `uv run ...`
  Run a command inside the repo's managed environment.
- `uv sync`
  Make `.venv` match `pyproject.toml` and `uv.lock`.
- `uv lock --check`
  Verify the lockfile still matches the project metadata.

## First Learning Pass

Run these in order once:

```bash
uv run python --version
uv run python -m unittest discover -s tests -v
uv run python -m py_compile nanorlm.py policies.py bench.py examples/run_verifiers.py examples/run_needlepairs.py examples/run_dossiers.py examples/run_planning.py showcases/planning.py showcases/generate_assets.py
uv run python bench.py --dataset pairbench --limit 4 --budget 60 --depth 2
uv run python bench.py --dataset verifiers_smoke --limit 2 --budget 80 --depth 2 --repo-root tests/fixtures/verifiers-mini
```

You should see:

- `uv run python --version` use a `3.11.x` interpreter
- the test suite pass
- the compile check pass
- both smoke runs complete cleanly

## Rules Of Thumb

- Default to `uv run python ...` instead of raw `python ...`
- Use `uv sync` after dependency or Python-version changes
- Use `uv lock --check` before assuming your environment metadata is current
- Avoid `uv add` until you actually need a new dependency
- Do not worry about activating `.venv` manually unless you have a shell or editor reason

## Common Scenarios

Fresh clone:

```bash
uv sync
uv run python --version
uv run python -m unittest discover -s tests -v
```

README says `python`, but you want the `uv`-safe version:

```bash
uv run python ...
```

You pulled changes or touched project metadata:

```bash
uv lock --check
uv sync
```

You want confidence before editing code:

```bash
uv run python -m unittest discover -s tests -v
uv run python bench.py --dataset pairbench --limit 4 --budget 60 --depth 2
```

You want to confirm which Python `uv` is actually using:

```bash
uv run python --version
cat .python-version
```

## Cheat Sheet

- Use the repo: `uv run python ...`
- Refresh env: `uv sync`
- Check lockfile: `uv lock --check`
- Which Python is active: `uv run python --version`
- Something feels off: rerun `uv sync`

## Next Layer, Later

Once this flow feels natural, learn:

- `uv sync --frozen` for stricter reproducibility
- `uv add ...` when the repo actually needs a new dependency
- `uv python ...` if you want `uv` to manage Python installs directly
