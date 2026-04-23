# Evaluation

Evaluations use OpenAI-compatible endpoints. The default endpoint registry lives at `./configs/endpoints.toml`.

You can configure an evaluation by setting `endpoint_id` in TOML configs. `endpoint_id` is only supported with TOML endpoint registries.

Results are written under `./outputs/evals/...` and a valid saved run contains:

- `results.jsonl`
- `metadata.json`
