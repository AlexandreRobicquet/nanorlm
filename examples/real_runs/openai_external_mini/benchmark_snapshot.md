# OpenAI External JSONL Mini Snapshot

Date: 2026-06-13

This is a tiny real-model smoke artifact for the external JSONL adapter and report-bundle path. It is not a headline benchmark result or leaderboard claim.

## Model And Backend

- Model: `gpt-4.1-mini`
- Provider: OpenAI-compatible Chat Completions
- Base URL: `https://api.openai.com/v1`
- Cache directory: `outputs/cache/openai-gpt-4.1-mini`
- Cost guard: `--max-estimated-cost 0.50`
- Cache secret check: `cache_files_containing_key=0`

## Source Data

- Dataset adapter: `external_jsonl`
- Fixture: `tests/fixtures/external-benchmark-mini.jsonl`
- Rows: 2
- Task classes: `ruler/niah`, `ruler/variable_tracking`

## Command

```bash
uv run python bench.py \
  --dataset external_jsonl \
  --dataset-path tests/fixtures/external-benchmark-mini.jsonl \
  --limit 2 \
  --policies direct_full_context,pairwise_tournament \
  --budget 80 \
  --depth 2 \
  --provider openai-compatible \
  --model gpt-4.1-mini \
  --base-url https://api.openai.com/v1 \
  --cache-dir outputs/cache/openai-gpt-4.1-mini \
  --max-output-tokens 256 \
  --max-estimated-cost 0.50 \
  --output-dir outputs/real-runs/openai-external-mini
```

## Completed Metrics

| Dataset | Policy | Examples | Answer Accuracy | Avg Retained Tokens | Estimated Cost |
| --- | --- | ---: | ---: | ---: | ---: |
| External JSONL mini | `direct_full_context` | 2 | 1.000 | 22.0 | `$0.000439` |
| External JSONL mini | `pairwise_tournament` | 2 | 1.000 | 22.0 | `$0.000439` |

Raw output bundle, not committed:

- `outputs/real-runs/openai-external-mini/summary.json`
- `outputs/real-runs/openai-external-mini/per_case.jsonl`
- `outputs/real-runs/openai-external-mini/curves.json`
- `outputs/real-runs/openai-external-mini/experiment_report.md`

Checksums:

```text
be24be2e65e9efd59e2905e40755911f5b6e3f0c57654bada1b439ab22718bf7  summary.json
a00c5f2d9b20270e8b7a28fe38c065ec57fdef89ca8c4c8b27cbc61e4f7a3f5f  per_case.jsonl
9edae4f77c27fa2428b2a75e4e436c57d36e64315474623168e3f1bfd3356fa1  curves.json
4bee0cdcbfd3a7f930e103549e9598ec0120619d08c4894041bd9337d11e2b80  experiment_report.md
```

## Limitations

- This uses a two-row fixture, not a full RULER export.
- Provenance score is 0.0 because the fixture rows do not carry expected source-file provenance.
- The result validates adapter, cache, cost-estimation, and `experiment_report.md` mechanics on a real model; it does not establish broad long-context performance.
