# OpenAI RULER Small Snapshot

Date: 2026-05-21

This is a small real-model artifact for the nanoRLM harness. It is not a headline RULER result, not a leaderboard submission, and not evidence that one retention policy is generally better than another.

## Model And Backend

- Model: `gpt-5.4-mini`
- Provider: OpenAI-compatible Chat Completions
- Base URL: `https://api.openai.com/v1`
- Cache directory: `outputs/cache/openai-gpt-5.4-mini`
- Cost guard: `--max-estimated-cost 20`
- Cache secret check: `cache_files_containing_key=0`

## Source Data

- Upstream RULER checkout: `/tmp/nanorlm-ruler`, commit `ab17b78`
- Generated source rows: RULER `niah_single_1` and `vt`
- Intended source slice: 3 examples per task at 4K and 8K, 12 rows total
- Completed committed snapshot slice: 4 rows from `/tmp/nanorlm-ruler-small.jsonl`
- Context block conversion: `scripts/prepare_ruler_external_jsonl.py`, 96-line chunks
- Verifiers checkout for secondary slice attempt: `/tmp/nanorlm-verifiers`, commit `482e28f`

## Commands

RULER data generation used upstream RULER:

```bash
uv run --with pyyaml --with nltk --with numpy --with tqdm --with wonderwords --with tiktoken --with tenacity \
  python prepare.py \
  --save_dir /tmp/nanorlm-ruler-generated/4096 \
  --benchmark synthetic \
  --task niah_single_1 \
  --tokenizer_path cl100k_base \
  --tokenizer_type openai \
  --max_seq_length 4096 \
  --model_template_type base \
  --num_samples 3 \
  --random_seed 42
```

The same command shape was used for `vt` at 4096 tokens and for `niah_single_1` / `vt` at 8192 tokens with seed `43`.

The completed OpenAI run:

```bash
uv run python bench.py \
  --dataset external_jsonl \
  --dataset-path /tmp/nanorlm-ruler-small.jsonl \
  --limit 4 \
  --policies direct_full_context \
  --budget 120 \
  --depth 3 \
  --provider openai-compatible \
  --model gpt-5.4-mini \
  --base-url https://api.openai.com/v1 \
  --cache-dir outputs/cache/openai-gpt-5.4-mini \
  --max-estimated-cost 20 \
  --output-dir outputs/real-runs/openai-ruler-small-direct
```

## Completed Metrics

| Dataset | Policy | Examples | Answer Accuracy | Avg Retained Tokens | Estimated Cost |
| --- | --- | ---: | ---: | ---: | ---: |
| RULER-derived external JSONL | `direct_full_context` | 4 | 1.000 | 18.25 | `$0.014798` |

Completed cases:

| Case | Task | Expected fragment count | Answer accuracy | Estimated cost |
| --- | --- | ---: | ---: | ---: |
| `niah_single_1-4093-1179` | `ruler/niah_single_1` | 1 | 1.000 | `$0.003683` |
| `niah_single_1-4093-8824` | `ruler/niah_single_1` | 1 | 1.000 | `$0.003589` |
| `niah_single_1-3496-1717` | `ruler/niah_single_1` | 1 | 1.000 | `$0.003278` |
| `vt-4082-0` | `ruler/vt` | 5 | 1.000 | `$0.004248` |

Raw output bundle, not committed:

- `outputs/real-runs/openai-ruler-small-direct/summary.json`
- `outputs/real-runs/openai-ruler-small-direct/per_case.jsonl`
- `outputs/real-runs/openai-ruler-small-direct/curves.json`

Checksums:

```text
a8bf89e6feaf2ab3260cd93c6024633db57b145c2dbc8befdf6405fb0ce89c97  summary.json
99a8b5f6bdae3ea34345987e999603359c8eaa821d035f9ca45031079a0af95d  per_case.jsonl
dbaa6c174fff6d1c5acff04079ecd0e67ed7d51eda44d9785f02723d587e6bd3  curves.json
```

## Limitations

- The available OpenAI project was rate-limited to 3 requests per minute for `gpt-5.4-mini`.
- A 12-example, three-policy RULER run was attempted but did not complete in a practical time window under that RPM cap.
- A one-case Verifiers-30 run was also attempted and blocked before the first trace by the same rate limit.
- The committed snapshot therefore reports the completed RULER direct-context slice only.
- Provenance score is 0.0 because these generated RULER rows do not carry expected source-file provenance.
- This artifact validates adapter, cache, cost-estimation, and report-bundle mechanics; it does not establish benchmark performance.
