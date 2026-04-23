# AGENTS.md

This file is a tracked starting point for a local `AGENTS.md`.

The live `AGENTS.md` is gitignored on purpose so each contributor can keep repo-specific agent instructions, notes, and working preferences without shipping them in commits.

## Project Thesis

- `nanoRLM` should teach recursive inference and memory retention in one sitting.
- The moat is clarity, not framework breadth.
- The strongest version of this repo is small, exact, and empirically honest.

## What To Optimize For

- First: pedagogical clarity.
- Second: honest benchmark and trace story.
- Third: benchmark-agnostic retention quality under hard budgets.
- Fourth: interoperability with OpenAI-compatible backends.

## Guardrails

- Keep `nanorlm.py` as the canonical place to understand the full recursive loop until a split clearly improves readability.
- Keep `policies.py` easy to compare side by side.
- Keep `bench.py` runnable and readable, not a framework.
- Treat synthetic datasets as tests or smoke demos, not headline evidence.
- Do not add benchmark-specific hacks that leak answer structure into backends or policies.

## Repo Tone

This repo should feel:

- minimal,
- legible,
- tasteful,
- empirically honest,
- easy to fork and understand.

If a change makes the repo feel heavier or noisier, it is probably the wrong change.

## Before Changing Code

- Read `README.md` and the relevant module end to end.
- Preserve explicit control flow and plain Python.
- Prefer dataclasses and small functions over abstraction layers.
- Add comments only when they reduce real conceptual load.
- Keep optional dependencies out of the core path.

## Before Publishing

- Run `python3 -m unittest discover -s tests -v` if code changed.
- Run `python3 -m py_compile nanorlm.py policies.py bench.py examples/run_needlepairs.py examples/run_verifiers.py` if Python source changed.
- If benchmark numbers in the README changed, regenerate them honestly and keep the reproduction command close to the claim.
- Keep the repo publishable from a fresh machine with `uv`.

## Current Direction

- Integrity pass first.
- Honest real-context baseline second.
- Principled retention upgrade third.
- Established benchmarks before bigger architecture moves.
- Only add a second, paper-faithful engine if it clearly earns the complexity.
