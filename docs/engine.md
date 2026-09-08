# Recursive engine

Read the full loop in [`nanorlm/__init__.py`](../nanorlm/__init__.py) and compare the retention policies in
[`nanorlm/policies.py`](../nanorlm/policies.py). This guide keeps the runnable example and result contract
together. Imports below work from the source-checkout root.

![Recursive memory loop](../showcases/assets/dossierbench/architecture.svg)

## Tiny Example

```python
from nanorlm import ContextBlock, RLM, RLMConfig

context = [
    ContextBlock(
        name="incident-a.txt",
        text="Deployment validation says the API gateway rollout is blocked by a "
        "stale endpoint registry cache from the previous release; the new gateway "
        "binary passed all of its health checks.",
    ),
    ContextBlock(
        name="incident-b.txt",
        text="The rollout can proceed by reloading the endpoint registry and "
        "invalidating the cache before the gateway reads route metadata again, then "
        "rerunning deployment validation against every refreshed endpoint.",
    ),
    ContextBlock(
        name="incident-c.txt",
        text="The observability team completed its dashboard migration and archived "
        "the old alert definitions after confirming that historical charts and "
        "service-level panels render correctly in every production region.",
    ),
    ContextBlock(
        name="incident-d.txt",
        text="A separate storage review recommends revisiting backup retention next "
        "quarter, after capacity forecasts, recovery drills, and vendor pricing "
        "have been updated by the infrastructure finance group.",
    ),
]

config = RLMConfig(
    model="demo/heuristic",
    provider="heuristic",
    max_depth=4,
    memory_budget_tokens=120,
    retention_policy="pairwise_tournament",
    seed=0,
)

result = RLM(config).completion(
    (
        "What blocks the API gateway rollout, and how should the endpoint "
        "registry and cache be refreshed to fix it?"
    ),
    context,
)

print(result.answer)
print(result.trace.tree)
print("retained:", sorted(item.provenance for item in result.kept_items))
print("dropped:", sorted(item["provenance"] for item in result.drop_reasons))
print("max memory depth:", result.retention_stats["max_memory_depth"])
```

Expected output (abridged):

```text
... stale endpoint registry cache ...
... reloading the endpoint registry and invalidating the cache ...
- [split] root split ...
  - [split] root.0 split ...
    - [inspect] root.0.0 leaf ...
retained: ['incident-a.txt', 'incident-b.txt']
dropped: ['incident-c.txt', 'incident-d.txt']
max memory depth: 2
```

The root context and both of its halves exceed the engine's 64-token leaf floor, so the run creates four depth-2 leaf memories; the 120-token budget then keeps the complementary blocker and fix while dropping both distractors.

## Result and budget contract

`provider` selects `heuristic`, `openai_compatible`, `anthropic`, or `auto`. `base_url` is optional and defaults to the right endpoint for the chosen network provider.

`RLM(config).completion(query, context)` returns an `RLMResult` with:

- `answer`
- `trace`
- `usage`
- `cost_estimate`
- `kept_items`
- `retention_stats`
- `drop_reasons`
- `per_step_budget`
- `retention_decisions`, with the complete candidate set, selected ranks, and budget for each retention step
- `completed` and `stop_reasons`, plus omitted source spans when traversal limits prevent full inspection
- `stage_budgets`, with prompt tokens, completion tokens, calls, and wall time for inspection and final-answer stages

Memory budgets apply to estimated summary tokens at every leaf and parent exit. The v2 estimate
uses the larger of the word estimate and UTF-8 bytes / 4; it is not a provider tokenizer. Oversized
individual inputs are split losslessly with source coordinates (`max_leaf_tokens`, default 2048).
If the depth or step limit prevents inspection, the result explicitly reports incomplete coverage.
Remote requests also enforce `max_input_tokens` (default 32768) using a conservative UTF-8 byte
bound with prompt headroom. `max_output_tokens` is a separate provider-enforced response cap.

Report filenames are opaque content-bound IDs; use each row's `artifact_stem` to locate its trace.
Use a fresh output directory for each saved run. Quality rewards exclude observed wall time;
`latency_ms` reports actual execution including cache/replay speedups. These contract changes
invalidate comparison with older receipts unless those runs are regenerated.

Benchmark rows add scoring fields such as `answer_accuracy`, `provenance_score`, and `provenance_hits`. Those are harness-level checks against expected answers and expected provenance, not engine output.

For a saved trace, read [`examples/pairbench_trace.txt`](../examples/pairbench_trace.txt).
For the smallest dataset construction, find `build_pairbench` in
[`nanorlm/bench.py`](../nanorlm/bench.py). The [experiment guide](experiments.md)
explains how to generate and compare full report bundles.
