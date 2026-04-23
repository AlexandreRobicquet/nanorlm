## Why

Briefly explain the user-facing, research, or maintenance reason for this change.

## What Changed

- 

## Validation

- [ ] `uv run python -m unittest discover -s tests -v`
- [ ] `uv run python -m py_compile nanorlm.py policies.py bench.py examples/run_needlepairs.py examples/run_verifiers.py`

## Repo-Specific Evidence

- [ ] If benchmark numbers or README metrics changed, I included the exact command and the updated numbers.
- [ ] If recursion, retention, or trace behavior changed, I added a test or included a trace snippet.

## Risks

Call out any behavior changes, weak results, or follow-up work plainly.
