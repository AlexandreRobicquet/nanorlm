# ROADMAP.md

This file is a tracked starting point for a local `ROADMAP.md`.

The live `ROADMAP.md` is gitignored on purpose so contributors can keep private planning notes, sequencing ideas, and half-formed experiments out of the repo history.

## North Star

`nanoRLM` should become a small, readable, benchmark-credible reference implementation of recursive inference with memory retention under hard budgets.

## Current Milestones

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

- the headline results are honest,
- at least one headline result comes from an established benchmark,
- the retention story generalizes beyond synthetic markers,
- the core is still readable in one sitting.
