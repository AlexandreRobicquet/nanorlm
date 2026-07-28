# Newcomer Audit Remediation

This document turns the 2026-07-17 full newcomer audit into a durable, reviewable backlog.
It is the source of truth for onboarding fixes; `ROADMAP.example.md` keeps the higher-level
research sequence and links here for implementation detail.

Last reconciled: 2026-07-27 against NR-ONB-04 implementation commit `37d038d`, based on
`master` at `1556960`.

## Status Legend

- `Open`: no complete fix is present at the reconciled repository revision.
- `In progress`: related changes are present, but the acceptance checks have not yet established completion.
- `Done`: the documented acceptance checks passed from a clean, isolated checkout.
- `Decision required`: maintainers must choose a supported product direction before implementation.

Do not mark an item `Done` because code exists or a narrow unit test passes. Preserve the literal
newcomer journey and rerun the item-specific acceptance checks.

## Priority Order

| Priority | Item | Status | Roadmap mapping |
| --- | --- | --- | --- |
| P0 | Restore pinned Verifiers compatibility | Done | Integrity Pass, Honest Baseline, Real Benchmarks |
| P0 | Ship the declared MIT license | Done | Integrity Pass, Release Bar |
| P1 | Make the Tiny Example exercise recursion and retention | Done | Honest Baseline |
| P1 | Make report bundles part of the golden path | Done | Honest Baseline |
| P1 | Resolve clone-only versus installable packaging | Done | Integrity Pass, Release Bar |
| P2 | Add a concise contributor entrypoint | Done | Integrity Pass |
| P2 | Make the pytest verification dependency reproducible | Done | Integrity Pass |
| P3 | Clean up command, prerequisite, link, and scope language | Open | Honest Baseline, Release Bar |

## Recommended Execution Order

The seven completed fixes establish a trustworthy baseline. Finish the remaining work in one
reviewable slice:

1. **Run the documentation consistency pass.** Apply the settled contracts everywhere, add
   prerequisite/cost/runtime notes, fix links and command forms, then rerun all Markdown and
   newcomer-path checks.

Keep the slice independently mergeable. Do not combine this onboarding fix with benchmark
result changes or new research claims.

## P0 — Restore Pinned Verifiers Compatibility

Status: **Done**

The documented current-head Verifiers checkout had drifted away from the paths required by
`examples/verifiers_30.json` and `showcases/planning_tasks.json`. Codebase QA failed on
`docs/evaluation.md`; grounded planning failed on `tests/test_eval_cli.py`.

Merged in [PR #33](https://github.com/AlexandreRobicquet/nanorlm/pull/33) with:

- a compatibility manifest pinned to `482e28ffa1f2613325867badaba4707b7c751d28`;
- aggregated path preflight validation;
- source-revision metadata in report bundles;
- explicit 30-case versus 10-case documentation;
- a scheduled current-upstream compatibility workflow;
- deterministic compatibility fixtures and tests.

### Completion evidence

- [x] Reviewed and merged the compatibility implementation.
- [x] Passed the full 77-test suite from an isolated checkout of `262cc4e`.
- [x] Created a fresh checkout of Verifiers revision
      `482e28ffa1f2613325867badaba4707b7c751d28`.
- [x] Passed the compatibility preflight over all 25 required paths.
- [x] Ran all 30 Codebase QA cases and all 10 grounded-planning tasks.
- [x] Verified missing checkouts fail before policy execution with one diagnostic and no traceback.
- [x] Verified both report summaries record the pinned and actual revision as matching.
- [x] Made current-upstream drift a failing scheduled maintenance check.

### Acceptance

```bash
git init /tmp/nanorlm-verifiers
git -C /tmp/nanorlm-verifiers remote add origin https://github.com/PrimeIntellect-ai/verifiers.git
git -C /tmp/nanorlm-verifiers fetch --depth 1 origin 482e28ffa1f2613325867badaba4707b7c751d28
git -C /tmp/nanorlm-verifiers checkout --detach FETCH_HEAD

uv run python scripts/check_verifiers_compatibility.py \
  --repo-root /tmp/nanorlm-verifiers

uv run python examples/run_verifiers.py \
  --repo-root /tmp/nanorlm-verifiers \
  --limit 30

uv run python examples/run_planning.py \
  --repo-root /tmp/nanorlm-verifiers \
  --limit 10 \
  --budget 140 \
  --depth 2
```

All three commands must pass from an isolated checkout. A deliberately incompatible checkout must
exit before policy execution, list all missing required files, and point to the verified revision.

## P0 — Ship the Declared MIT License

Status: **Done**

The audit found that `pyproject.toml` declared MIT without shipping the license text. That gap was
closed in [PR #30](https://github.com/AlexandreRobicquet/nanorlm/pull/30).

### Completion evidence

- [x] Added the canonical MIT text as `LICENSE`.
- [x] Replaced the deprecated license table with `license = "MIT"`.
- [x] Declared `license-files = ["LICENSE"]`.
- [x] Added repository-metadata regression tests.
- [x] Passed clean Python 3.11, Python 3.12, and smoke CI.
- [x] Verified GitHub reports the repository license as MIT on 2026-07-24.

### Acceptance

- `LICENSE` is tracked at the repository root.
- Build metadata emits no license deprecation warning.
- GitHub reports MIT for the public repository.

## P1 — Make the Tiny Example Prove the Thesis

Status: **Done**

The audited README example ran as a single leaf, produced zero retention decisions, and omitted
part of the answer. [PR #31](https://github.com/AlexandreRobicquet/nanorlm/pull/31) replaced it with
an executable recursive example.

### Completion evidence

- [x] The root and both halves split into four depth-2 leaf memories.
- [x] The 80-token budget makes a real retention decision.
- [x] The retained answer includes both the stale cache blocker and the registry/cache fix.
- [x] The README shows an abridged expected answer, tree, retained set, and dropped set.
- [x] `tests/test_readme.py` executes the fenced example and asserts its answer and trace contract.
- [x] The full 77-test suite passed from an isolated checkout of `262cc4e`.

### Acceptance

The checked-in example must satisfy equivalent assertions:

```python
assert "stale endpoint registry cache" in result.answer
assert "invalidating the cache" in result.answer
assert result.retention_stats["total_retention_steps"] >= 1
assert result.retention_decisions
assert result.retention_stats["max_memory_depth"] >= 1
```

## P1 — Make Report Bundles Part of the Golden Path

Status: **Done**

Several documented `bench.py` commands print a table but omit `--output-dir`, so they do not create
the report bundle described by the README.

### Work

- [x] Inventory every benchmark command in `README.md`, `UV.md`, and `showcases/README.md`, and
      classify it as stdout-only smoke or evidence-producing run.
- [x] Add explicit, deterministic `--output-dir` arguments to every evidence-producing command.
- [x] State in `bench.py --help` and the docs that omitting `--output-dir` is intentional
      stdout-only mode.
- [x] Print the normalized report location after `bench.py` writes a bundle.
- [x] Point newcomers first to `experiment_report.md`, then to machine-readable `summary.json`.
- [x] Extend `tests/test_nanorlm.py` or add a documentation test that runs the golden command shape
      and checks the complete bundle.

### Required output contract

- `summary.json`
- `per_case.jsonl`
- `curves.json`
- `experiment_report.md`
- `trace_examples/`

### Acceptance

Every command described as producing evidence names its output directory, and a fresh run creates
the complete contract at that location.

### Completion evidence

- [x] Classified 32 benchmark-runner command occurrences across the three documentation files:
      11 intentional stdout-only smoke runs and 21 evidence-producing runs.
- [x] Named deterministic destinations under ignored `outputs/`, `examples/outputs/`, or
      `showcases/outputs/` roots; e2e commands name both `--output-root` and `--run-id`.
- [x] Normalized the direct CLI output path once, used it for traces and report writing, and printed
      one stable location line only after a successful write.
- [x] Documented stdout-only omission in CLI help and pointed saved-run readers to
      `experiment_report.md` before `summary.json`.
- [x] Added a real CLI-level test covering all five bundle components, normalized stdout, and the
      no-claim stdout-only path.
- [x] From a clean archive of `80022c4`, passed `uv lock --check`, `uv sync --frozen`, all 35
      targeted nanoRLM tests, and all 79 tests on Python 3.11.15 and Python 3.12.13.
- [x] Passed the 79-test temporary pytest run, compile check, all five direct offline smoke
      benchmarks, and the explicitly rooted smoke and learned e2e phases.
- [x] Ran the four-case PairBench golden command and verified `summary.json`, `per_case.jsonl`,
      `curves.json`, `experiment_report.md`, `trace_examples/`, and the printed normalized path.

## P1 — Resolve Packaging Intent

Status: **Done — Option B**

`UV.md` says the repository is not a package-publishing workflow, but the build metadata can emit a
wheel. A fresh build from `262cc4e` on 2026-07-24 contained `showcases` and distribution metadata
but did not expose the documented `nanorlm` API.

### Option A — Support installation

- [ ] Record the installable-library decision in `pyproject.toml`, `README.md`, and `UV.md`.
- [ ] Explicitly package `nanorlm.py`, `policies.py`, and `learned_retention.py`.
- [ ] Decide whether benchmark, example, script, and showcase modules belong in the artifact.
- [ ] Add source-distribution and wheel-content checks.
- [ ] Install the wheel outside the checkout and smoke-test the public imports.

Acceptance:

```bash
python -c "from nanorlm import RLM, RLMConfig"
```

must pass using only the installed artifact.

### Option B — Stay clone-only

- [x] Record clone-only as the supported product boundary in `README.md` and `UV.md`.
- [x] Remove or minimize the accidental distribution surface.
- [x] State prominently that nanoRLM is run from a checkout.
- [x] Ensure ordinary contributor commands do not create a misleading publishable artifact.

Acceptance: the repository no longer presents a wheel as a supported installation path.

Option B was selected because the repository already set `[tool.uv] package = false`, presented a
checkout workflow, and contained no explicit maintainer commitment to an installed public API.

### Completion evidence

- [x] Added explicit empty setuptools package and module lists so automatic discovery cannot expose
      `showcases` or a partial `nanorlm` API.
- [x] Added a repository-metadata regression test for the uv and setuptools boundary.
- [x] Documented the checkout-only contract before the first setup command and in `UV.md`.
- [x] From a clean archive of `673d00f`, passed `uv lock --check`, `uv sync --frozen`, all three
      metadata tests, and all 78 tests on Python 3.11.15 and Python 3.12.13.
- [x] Rebuilt the sdist and wheel; the wheel contained only `.dist-info` entries with an empty
      `top_level.txt`, while neither artifact exposed runtime modules or `showcases`.
- [x] Verified the embedded MIT license matched `LICENSE` and an empty environment outside the
      checkout could not discover `nanorlm`, `policies`, `learned_retention`, or `showcases`.
- [x] Removed validation-only build metadata and confirmed the implementation worktree was clean.

## P2 — Add a Concise Contributor Entrypoint

Status: **Done**

`UV.md`, CI, and the pull-request template contain useful pieces of the workflow, but a first-time
contributor must assemble the edit-test-debug loop themselves.

### Work

- [x] Add a short `CONTRIBUTING.md` linked from the README and pull-request template.
- [x] Document uv installation and supported Python versions.
- [x] Identify the fastest local feedback command and the complete pre-PR command.
- [x] Explain the boundaries of `nanorlm.py`, `policies.py`, `bench.py`, examples, and showcases.
- [x] Document where generated outputs belong and that they are ignored by default.
- [x] State the evidence rules for synthetic, external, and real-model benchmark changes.
- [x] State whether linting, formatting, typing, and coverage are required or deliberately absent.
- [x] Explain how negative or inconclusive research results should be reported.
- [x] Test every copied command from a clean checkout on each supported Python minor used by CI.

### Acceptance

A newcomer can identify setup, one-file-change validation, full validation, generated-artifact
handling, and PR expectations from one page.

### Completion evidence

- [x] Added one concise `CONTRIBUTING.md` covering prerequisites, clone-only setup, the
      edit-test-debug ladder, ownership boundaries, generated files, evidence rules, and project
      conventions.
- [x] Linked the guide from `README.md` and the pull-request template while keeping the full
      canonical command block in `UV.md`.
- [x] Removed the pull-request template's shortened command copy and aligned UV's compile targets
      with CI so contributors have one authoritative verification path.
- [x] Documented ignored output roots, reviewed-receipt locations, and the prohibition on
      committing `.env` files, API keys, raw caches, or unreviewed large outputs.
- [x] Stated that synthetic and fixture results are regression evidence; external and real-model
      claims require reproducible source, model, command, cost-cap, and bundle metadata; and
      negative or inconclusive results remain reportable evidence.
- [x] Recorded that linting, formatting, static typing, and coverage are deliberately not enforced;
      the existing lock, test, compile, and smoke gates remain authoritative.
- [x] From separate clean archives of `37d038d`, executed every copied command and the complete
      canonical verification block on Python 3.11.15 and Python 3.12.13, with 35 targeted tests
      and 80 tests per complete runner passing on both minors.
- [x] Verified all 29 local Markdown targets and the canonical UV anchor from both clean archives.
- [x] Passed the GitHub Python 3.11, Python 3.12, and smoke checks for
      [PR #39](https://github.com/AlexandreRobicquet/nanorlm/pull/39).

## P2 — Make Pytest Verification Reproducible

Status: **Done**

The canonical `uv run --with pytest pytest` command resolves the current pytest release at execution
time, leaving part of the verification path outside the lockfile.

### Work

- [x] Add a locked development dependency group for pytest; keep runtime dependencies empty.
- [x] Update `UV.md`, CI where appropriate, and the pull-request template to use the same versioned path.
- [x] Keep the stdlib `unittest` path available as the dependency-free core check.
- [x] Add `uv lock --check` coverage that fails when the declared development dependency and lock
      drift apart.
- [x] Confirm the locked pytest path and stdlib discovery path collect the same tests.

### Acceptance

The lockfile or the documented command fixes the pytest version, and both unittest and pytest runs
execute the same complete suite without dependency drift.

### Completion evidence

- [x] Declared `pytest>=9.1.1,<9.2` in the standardized default `dev` dependency group while
      retaining an empty runtime dependency set.
- [x] Locked pytest 9.1.1 with iniconfig 2.3.0, packaging 26.2, pluggy 1.6.0, Pygments 2.20.0,
      and the Windows-only colorama 0.4.6 dependency.
- [x] Replaced execution-time pytest resolution in `UV.md` and the benchmark e2e check phase,
      aligned the pull-request checklist, and made both locked and stdlib suites explicit in CI.
- [x] Added an explicit CI matrix interpreter override and version log so `.python-version` cannot
      silently collapse the Python 3.12 job onto Python 3.11.
- [x] From clean archives of `8fba876`, passed `uv lock --check`, frozen sync, all 80 unittest
      cases, and all 80 locked pytest items on Python 3.11.15 and Python 3.12.13.
- [x] Compared every public pytest collection ID with recursively discovered unittest IDs and
      confirmed the same complete 80-test set on both Python minors.
- [x] Passed offline frozen pytest runs, proving no execution-time resolution, and separately
      passed all 80 unittest cases from a frozen `--no-dev` environment with no importable pytest.
- [x] Passed the GitHub 3.11, 3.12, and smoke checks for
      [PR #38](https://github.com/AlexandreRobicquet/nanorlm/pull/38).

## P3 — Documentation Consistency and Scope Polish

Status: **Open**

### Work

- [ ] Link official uv installation instructions before the first uv command.
- [ ] Use `uv run python ...` consistently in showcase documentation.
- [ ] Render `.venv/` as generated-path code rather than a link to an untracked directory.
- [ ] State expected duration, disk output, credentials, and cost before heavier workflows.
- [ ] Explain that an e2e phase can pass operationally while its research verdict is negative.
- [ ] Replace the broad “read the repo in one sitting” promise with a minimum reading path if the
      full repository no longer supports that expectation.
- [ ] Apply the chosen clone-only/installable language consistently across `README.md`, `UV.md`,
      `pyproject.toml`, and the contributor guide.
- [ ] Run a local-link checker and execute every newcomer command from a clean checkout.

Suggested minimum reading path:

1. `nanorlm.py` for the recursive engine and result contract.
2. `policies.py` for side-by-side retention behavior.
3. One dataset builder in `bench.py`.
4. One saved tree trace.
5. One `experiment_report.md`.

### Acceptance

- All local Markdown links resolve.
- All setup and showcase commands use the canonical environment workflow.
- A newcomer sees prerequisites and operational boundaries before running expensive or external paths.

## Shared Completion Gate

After all P0 and P1 items are complete, rerun the literal newcomer journey in a clean archive or
temporary checkout:

```bash
uv sync
uv run python --version
uv run python -m unittest discover -s tests -v
uv run python bench.py --dataset verifiers_smoke --limit 2 --budget 80 --depth 2 --repo-root tests/fixtures/verifiers-mini
uv run python examples/run_dossiers.py --limit 4 --budget 80 --depth 4
uv run python scripts/run_benchmark_e2e.py --phases learned --learned-train-limit 4 --learned-eval-limit 4
```

Then run the README Tiny Example and the pinned external Codebase QA and planning commands. Record:

- time to first meaningful output;
- exact output directories;
- semantic answer checks;
- recursive and retention trace checks;
- expected versus actual external revisions;
- all generated files;
- any diagnostic workaround separately from the documented result.

The onboarding gate is complete only when the documented path reaches the promised outcomes without
hidden commands, and the public evidence remains conservative about synthetic and local results.
