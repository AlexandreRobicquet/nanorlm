# AGENTS.md

This repository is meant to become a small, legible, high-signal research artifact in the spirit of the best Karpathy repos: one sitting to understand, but still sharp enough to teach, benchmark, and extend.

## Project Thesis

- `nanoRLM` is the pedagogical reference implementation of recursive inference with explicit memory retention.
- The moat is clarity, not framework breadth.
- Every non-trivial line should either:
  - explain recursive inference clearly,
  - support the retention-policy research story,
  - or make the demos and benchmarks more convincing.

## Product Standard

- Prefer a tiny codebase with excellent taste over a broad codebase with shallow features.
- Bias toward plain Python, explicit control flow, and traceable behavior.
- Keep the core mental model small:
  - split context
  - recurse
  - inspect leaves
  - retain under budget
  - answer from memory
- Avoid turning this repo into an agent framework, orchestration layer, trainer, or config-heavy platform.

## Architecture Guardrails

- `nanorlm.py` should remain the canonical place to understand the full recursive loop.
- `policies.py` should hold retention strategies and be easy to compare side by side.
- `bench.py` should stay runnable and readable, not evolve into a generalized eval framework.
- `examples/` should contain memorable, narrative demos.
- `tests/` should prove the invariants that matter:
  - recursion works
  - memory budgets are enforced
  - pairwise retention beats simpler baselines on the right tasks

## What To Optimize For

- First: pedagogical clarity
- Second: compelling traces and examples
- Third: a real empirical edge for `pairwise_tournament`
- Fourth: interoperability with OpenAI-compatible backends

Do not trade clarity away for premature abstraction.

## Coding Style

- Favor small functions and explicit data structures.
- Prefer dataclasses and simple protocols over deep class hierarchies.
- Keep dependencies minimal.
- Avoid hidden global state.
- Keep names literal and research-legible.
- Add comments sparingly and only when they reduce true conceptual load.

## Research Bar

- New benchmark or policy additions must strengthen the central story:
  - recursive decomposition
  - memory bottlenecks
  - complementary evidence retention
- If a result is weak, say so plainly in the docs.
- Do not overclaim synthetic results as real-world capability gains.
- Any README metric should be reproducible with a documented command.

## Demo Bar

- Every headline demo should answer one fast question:
  - why recursion?
  - why retention?
  - why pairwise tournament?
- Prefer examples with strong visual or cognitive contrast:
  - tight budget vs naive retention
  - haystack search with complementary facts
  - codebase QA with compact retained evidence

## Before Making Changes

- Read `README.md`, this file, and the relevant module end to end.
- Preserve the small-repo aesthetic.
- If adding a feature, ask whether it belongs in:
  - core engine
  - policy layer
  - benchmark layer
  - or should stay out entirely

## Before Merging or Publishing

- Run:
  - `python3 -m unittest discover -s tests -v`
  - `python3 -m py_compile nanorlm.py policies.py bench.py examples/run_needlepairs.py examples/run_verifiers.py`
- If benchmark numbers in the README changed, regenerate them and update the snapshot honestly.
- Keep the repo publishable from a fresh machine using `uv`.

## Current Known Gaps

- The OpenAI-compatible backend exists, but headline model-backed numbers are not yet baked into the README.
- `Verifiers-20` is currently a strong demo scaffold, not yet a polished benchmark artifact.
- Docker sandboxing is still unimplemented.
- The July-quality public release still needs better figures, stronger examples, and a tighter narrative.

## Anti-Goals

- No LangGraph rewrite
- No plugin ecosystem
- No training stack in this repo
- No sprawling CLI
- No giant dependency surface unless it unlocks a major, concrete benefit

## Repo Personality

This project should feel:

- minimal
- exact
- tasteful
- empirically honest
- easy to fork and understand

If a proposed change makes the repo feel heavier, noisier, or less legible, it is probably the wrong change.
