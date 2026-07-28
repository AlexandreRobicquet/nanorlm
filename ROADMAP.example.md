# ROADMAP.md

This file is a tracked starting point for a local `ROADMAP.md`.

The live `ROADMAP.md` is gitignored on purpose so contributors can keep private planning notes, sequencing ideas, and half-formed experiments out of the repo history.

## North Star

`nanoRLM` should become a small, readable, benchmark-credible reference implementation of recursive inference with memory retention under hard budgets.

## Current Milestones

### Gate A. Newcomer Readiness

Status: **Closed — 2026-07-28**

The detailed, prioritized backlog is tracked in
[`ONBOARDING_AUDIT_TODOS.md`](ONBOARDING_AUDIT_TODOS.md).

Completed on `master`:

- [x] restore pinned external Codebase QA and grounded-planning paths,
- [x] ship the declared MIT license and metadata guard,
- [x] make the smallest example visibly exercise recursion and retention,
- [x] enforce the clone-only source-checkout packaging boundary,
- [x] make report bundles explicit in the golden path,
- [x] lock the pytest verification dependency,
- [x] add a single contributor entrypoint,
- [x] complete the command, prerequisite, link, and scope-language pass.

Completion evidence:

- [x] run and record the literal clean-checkout newcomer gate, including pinned external QA and
      planning paths, in
      [`ONBOARDING_ACCEPTANCE_REPORT.md`](ONBOARDING_ACCEPTANCE_REPORT.md).

The acceptance run tested `master` at `4467fc2` from a genuinely fresh checkout. It recorded the
full newcomer and pinned external paths without hidden setup or any documented-path workaround. No
model provider was configured, and realized API spend was $0. Later roadmap gates and release
claims remain independent.

### 0. Integrity Pass

- remove benchmark-specific shortcuts from backends and policies,
- stop using synthetic wins as headline README evidence,
- replace oracle-dependent tests with invariant-based tests.

### 1. Honest Baseline

- center the repo-QA story around a real inspectable demo,
- improve traces and benchmark snapshots,
- make one-command reproduction easy.

### 2. Retention Upgrade

- keep the simple baselines,
- add one stronger benchmark-agnostic policy,
- test ranking stability, diversity, and budget behavior.

### 3. Real Benchmarks

- add at least one established long-context benchmark,
- keep benchmark dependencies optional,
- publish exact reproduction commands with every number.

### 4. Real-Model Runs

- support one hosted model path and one local-model path,
- cache responses so reruns are cheap,
- publish one honest sweep before expanding scope.

### 5. Optional Engine Expansion

- add a paper-faithful model-directed engine only if it stays small,
- keep the current fixed-split engine as the pedagogical default.

## Non-Goals

- no agent framework,
- no vector database,
- no training stack,
- no sprawling CLI,
- no benchmark zoo just to look busy.

## Release Bar

The repo is ready for a bigger `v1.0` push when:

- the literal newcomer path passes from a clean checkout,
- the public license and packaging intent are unambiguous,
- the pinned external benchmark and planning paths pass their compatibility preflight,
- the headline results are honest,
- at least one headline result comes from an established benchmark,
- the retention story generalizes beyond synthetic markers,
- the documented minimum reading path still explains the core end to end.
