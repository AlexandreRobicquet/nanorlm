# Repository QA usefulness evaluation

The [v0.2 results](../docs/results.md) select lexical retrieval. This directory contains
the frozen 30-question dataset, upstream licenses, machine-readable results and
source adjudication. The release evidence archive contains the original execution
and grading receipts. Neither the questions nor the grading are an independent
human benchmark.

## Frozen answer comparison

`repoqa_v1.json` contains 30 questions, 90 expected facts, reference excerpts and
exact commits for [backoff](https://github.com/litl/backoff),
[python-dotenv](https://github.com/theskumar/python-dotenv) and
[cachetools](https://github.com/tkem/cachetools). The implementation assistant wrote
the questions after the initial implementation freeze and before generating any
evaluation answers. Two source-lineage/rendering fixes landed before execution.
The dataset records that implementation commit. No answer or retrieval prompt was
tuned using the evaluation responses. Public sources may occur in pretraining.

All three strategies use `gpt-4.1-mini-2025-04-14`, the same final-answer prompt and
1,600-token output limit. Lexical retrieval and recursive retention receive the
same 6,000-token candidate pool. Retention uses `pairwise_tournament`, a 512-token
memory budget, and retained **original source spans** for the final answer. Full
context receives every span admitted by the same scanner and refuses truncation.
Expected facts and reference answers never enter the answer model's context.

Each case binds the dataset, implementation, source commit/snapshot, evidence,
answer and usage receipts. All attempts and failures remain recorded. The answer
experiment has a USD 5 estimated inference cap. Cost reservations precede network
calls; unknown billing stops that execution rather than silently retrying it.

The original synchronous runner at `532a3b8` completed four cases before the
account exhausted its 50-request daily model quota during the fifth. That pilot is
excluded from the comparison. The complete experiment uses the batch runner at
`c6708b4`, with unchanged questions, model, prompts, budgets and source snapshots.
The original all-request batch was rejected by the account's queued-input limit;
the completed execution submits at most 140,000 tokenizer-counted input tokens
per round. This scheduling amendment changes transport, not answer selection.

Fourteen completed batches contain 574 billed requests and all 90 cases. The
engine generates inspection, repair and final-answer requests in dependency order.
Only provider responses bound to their exact request bodies become evidence;
planning placeholders are discarded. No failed final answer is retried or fuzzily
repaired. Later scanner fixes for non-UTF-8 filenames and special files were
verified not to change these three eligible source inventories.

To reproduce original answer generation, use the pinned checkout and clone the
source commits into a parent directory named `backoff`, `dotenv`, and `cachetools`:

```bash
# At nanoRLM commit c6708b4, with OPENAI_API_KEY configured:
uv run --with tiktoken python scripts/batch_repoqa.py \
  --repositories /path/to/pinned-checkouts --output outputs/repoqa-v1-batch --submit
```

Run the identical command to collect a finished round and submit its dependents;
omit `--submit` to inspect without submitting. Each batch may take up to 24 hours.
Use a fresh experiment for changed implementations. Resuming an interrupted request
requires reconciliation, never an unrecorded retry.

## Grading and adjudication

A complete answer expresses all three expected facts and contains no materially
incorrect extra claim. Each factual claim is separately judged against **its own
cited excerpts**; their union must establish every material assertion. Valid source
IDs and hashes prove identity, not semantic support. A failed answer receives zero
completeness and contributes no accepted claims to the citation denominator.

Final grading is pinned at `05ae795` and uses
`scripts/grade_grouped_repoqa.py`: GPT-5.4 mini, medium reasoning, strict JSON schema
counts, and each claim's original cited text attached directly. Seed-0 grouping
places at most two different questions in each call. Strategy names, costs and
competing answers to one question are hidden. The 81 usable answers took 41 calls;
the nine failed answers needed no grading request.

The complete USD 3.346866 conservative reservation fit a separate USD 6 grading
cap. Group receipts retain exact requests, responses, model identifiers, timing
and usage. Per-case grading expense is an explicitly equal allocation of actual
group cost, not separately observed token usage. Actual final grading cost was
USD 0.870716, reconciled against every group and case receipt.

The final synthetic calibration correctly classified coverage and citation support
but missed a conditional error label. It is preserved as a failure. Therefore
**model judgments alone cannot publish the report**. The implementation assistant
reviewed all 270 fact judgments and 205/276 claims against their actual citations,
including all flagged claims, all claims in provisionally complete answers, and 16
predefined source-audit cases. Remaining claims retain model judgments. This audit
could see strategy labels and is not independent human adjudication.

```bash
# At nanoRLM commit 05ae795:
uv run --with tiktoken python scripts/grade_grouped_repoqa.py \
  --experiment outputs/repoqa-v1-batch --output outputs/repoqa-v1-grouped-grades
# Inspect the plan, then repeat the identical command with --execute.
```

An audit JSON contains a `method` and `cases` keyed by case directory. Every case
binds its original `receipt_sha256`, records `facts_reviewed: true`, and lists the
`source_reviewed_claims`. Boolean amendments identify `kind` (`facts` or `claims`),
1-based `index`, `values`, and an evidence-specific `reason`. Raw model receipts
remain unchanged. Malformed grading requires a complete explained
`replacement_grade`; missing judgments never become automatic passes or zeroes.
The report verifies audit coverage for both original and amended judgments, so
promoting an answer to complete requires checking all its claims.

```bash
uv run python scripts/report_repoqa.py \
  --experiment /path/to/release-evidence/answers \
  --grades /path/to/release-evidence/grading-grouped-v4/grades \
  --audit /path/to/release-evidence/repoqa-audit.json \
  --output outputs/reproduced-report
```

The [checked-in audit](results/repoqa-v1/audit.json) and
[summary](results/repoqa-v1/summary.json) bind the full source evidence archive.
Helper names and equivalent wording may be paraphrased, but composite checklist
facts require material behavior, distinct modes and requested test coverage.
Some checklist details go beyond the literal question; completeness is not a
human usefulness score. Acknowledged missing evidence is not itself treated as a
positive false behavioral claim.

## Preserved grading failures and expense

| Attempt | Outcome | Estimated billed-price expense |
|---|---|---:|
| GPT-4.1 batch, V1 | 61 malformed receipts plus semantic contradictions; entirely excluded | $0.316245 |
| GPT-5.4 mini batch, V2 | One malformed receipt and repeated source/coverage mistakes; entirely excluded | $0.153634 |
| GPT-5.4 batch, V3 | No responses before cancellation was requested; excluded | See terminal or pending provider status in release accounting |
| GPT-5.4 mini grouped, V4 | Complete strict-schema output, then mandatory source adjudication | $0.870716 |

Calibration receipts, discarded responses and billing limitations are preserved in
the archive. One early calibration did not retain usage and has only its USD 0.05
conservative bound. Any unfinished provider cancellation is explicitly bounded,
not called a zero-cost completion. These expenses are separate from user-facing
answer inference. Grading amendments never regenerate or tune the frozen answers.

Price tables use official [GPT-4.1](https://developers.openai.com/api/docs/models/gpt-4.1),
[GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini),
[GPT-5.4](https://developers.openai.com/api/docs/models/gpt-5.4) and
[GPT-5.4 mini](https://developers.openai.com/api/docs/models/gpt-5.4-mini) prices
checked on 2026-09-06. These are estimates from reported tokens, not invoices;
server-side prompt-cache discounts are not deducted. GPT-5.4 requests are bounded
below its long-context price tier.

## Latency and release decision

**The batch comparison does not establish interactive latency.** Interactive
latency fields are null. Batch availability includes shared queue waiting; local
replay assembly times are not API response times. The interrupted synchronous pilot
and single release-candidate smoke do not substitute for a 30-question timing study.

The frozen selection rule prefers the cheapest approach within one complete answer
of the best and five percentage points of its citation support. Retention needs
at least three additional complete answers over lexical to justify its overhead.
Only lexical qualifies on these data. Learned retention remains experimental and
was not evaluated here. Publish the raw counts and these limitations; use a fresh,
independently authored set for the [next version](../docs/roadmap.md).
