# Showcases

`showcases/` holds the launch-facing demos and artifact generators for `nanoRLM`.

## Recommended Runs

Flagship repo QA:

```bash
python bench.py \
  --dataset verifiers_30 \
  --limit 10 \
  --budget 140 \
  --depth 2 \
  --repo-root /tmp/nanorlm-verifiers \
  --output-dir outputs/verifiers_30
```

Long-horizon dossier benchmark:

```bash
python bench.py \
  --dataset dossierbench \
  --limit 12 \
  --budget 80 \
  --depth 4 \
  --curve-budgets 60,80,100 \
  --curve-depths 3,4 \
  --curve-seeds 0,1,2 \
  --output-dir outputs/dossierbench
```

Grounded planning showcase:

```bash
python examples/run_planning.py \
  --repo-root /tmp/nanorlm-verifiers \
  --limit 10 \
  --budget 140 \
  --depth 2 \
  --output-dir showcases/outputs/planning
```

Render launch assets from a saved report bundle:

```bash
python showcases/generate_assets.py --report-dir outputs/dossierbench
```

## Output Contract

Benchmark runs produce:

- `summary.json`
- `per_case.jsonl`
- `curves.json`
- `trace_examples/`

Asset generation produces:

- `benchmark_snapshot.md`
- `architecture.svg`
- `policy_curve.svg`
- `trace_card.svg` when a saved tree trace is available
