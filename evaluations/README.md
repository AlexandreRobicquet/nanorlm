# Repository QA usefulness evaluation

`repoqa_v1.json` freezes 30 questions, 90 expected facts, source excerpts and exact
repository commits before the first evaluation call. The three public Python
repositories are [backoff](https://github.com/litl/backoff),
[python-dotenv](https://github.com/theskumar/python-dotenv) and
[cachetools](https://github.com/tkem/cachetools). Their licenses are preserved in
`licenses/`. The reference excerpts retain the respective upstream licenses.

Questions were written by the implementation assistant using source inspection,
after the initial implementation freeze. Two review-driven fixes to source
lineage and Markdown rendering landed before execution. The final implementation
commit is pinned in the dataset. No evaluation answers were used for retrieval or
prompt tuning. This is a small engineering evaluation, not an independent public
benchmark; public source code may occur in the model's pretraining data.

All strategies use `gpt-4.1-mini-2025-04-14`, the same answer prompt and output
limit. Lexical retrieval and retention receive the same 6,000-token candidate
pool. Retention compresses that pool with a 512-token memory budget and then
answers from its retained **original source spans**. Full context receives every
span admitted by the same scanner, including tests and documentation. It refuses
truncation. Scanner exclusions apply to all strategies. The model never receives
expected facts or reference answers.

The runner rotates strategy order, executes sequentially, disables local response
caching and records every attempted case, including failures. Each case binds the
dataset, implementation, question, strategy, source snapshot, evidence and usage
receipts. The total estimated inference cap is USD 5. Unknown remote billing stops
the experiment; reported prices are list-price estimates, not invoice totals.

Clone the three repositories into a parent directory with the names `backoff`,
`dotenv`, and `cachetools`, checking out the exact commits in the dataset. From a
clean nanoRLM checkout:

```bash
uv run python scripts/evaluate_repoqa.py \
  --repositories /path/to/pinned-checkouts --output outputs/repoqa-v1
```

The original synchronous runner is pinned at `532a3b8`; use that checkout to
reproduce it. It deliberately refuses later implementation changes. The quota-aware
batch execution is pinned at `c6708b4`. New experiments record their own checkout
and script hashes; do not overwrite an existing experiment after changing code.

Set `OPENAI_API_KEY` first. `--resume` accepts only the exact same experiment and
checksummed completed case directories; it neither overwrites nor retries failed
answers. An interrupted partial case requires investigation, not silent retry.

## Scoring and selection

Score each of the three expected facts as present and correct or missing/wrong.
A fully correct question needs all three facts and no materially incorrect
additional claim. Score semantic support for every factual claim against its
actual cited excerpts, separately from valid source IDs and checksums. Report
abstentions and failures. Do not substitute substring matching for correctness.
Report factual coverage, fully correct questions, citation support, median and
tail latency, and total estimated cost separately, by strategy and repository.

`scripts/grade_repoqa.py` applies the fixed checklist using
`gpt-4.1-2025-04-14`, with strategy names, costs and competing answers hidden from
the grader. It receives each claim's actual cited excerpts and separate reference
excerpts, and records per-fact and per-claim explanations. Grading starts after all
answer runs finish. Its separate USD 5 cap and full usage receipts distinguish
evaluation expense from user-facing inference expense. The local price table uses the published [GPT-4.1 prices](https://developers.openai.com/api/docs/models/gpt-4.1) and [GPT-4.1 mini prices](https://developers.openai.com/api/docs/models/gpt-4.1-mini), checked on 2026-09-06. A subsequent assistant
audit of flagged cases and a spread of passing cases is recorded separately;
this is **not human adjudication**. Both models share a family, so correlated
grading errors remain a limitation. The grader and its prompt are frozen before
grading, and raw judgments are retained when an audit changes a score.

```bash
uv run python scripts/grade_repoqa.py \
  --experiment outputs/repoqa-v1 --output outputs/repoqa-v1-grades
```

## Batch transport amendment, 2026-09-06

The synchronous run completed four cases before the API account exhausted its
50-request daily model quota during the fifth. Those partial results are retained
separately and cannot select a default. The Anthropic connection was unavailable.
The comparison therefore has a separate batch execution protocol, with the same
questions, model, prompts, budgets, source snapshots and failure scoring. A later
review-driven fix omits undecodable filenames; it does not change the eligible
source inventory in these three repositories.

The [Batch API](https://developers.openai.com/api/docs/guides/batch) uses separate
limits. The first round submits independent inspection leaves and baseline
answers. Subsequent rounds submit required schema repairs and retention answers
after their actual inspections are available. The original engine constructs and
validates all prompts. Local planning placeholders are discarded; only complete
responses bound to their exact request bodies can become experiment artifacts.

```bash
uv run --with tiktoken python scripts/batch_repoqa.py --repositories /path/to/pinned-checkouts \
  --output outputs/repoqa-v1-batch --submit
```

Run the same command to collect a completed round and submit dependent requests.
Omit `--submit` to prepare or inspect without submitting another round. Batches
can take up to 24 hours per round. Each submission records its conservative cost
reservation. No failed/expired batch is silently retried.

The initial all-request submission was rejected before executing any requests
because this account also has a 200,000 queued-input-token limit. A fresh batch
experiment therefore sends at most 140,000 tokenizer-counted input tokens at a
time. The rejected submission is retained separately. Each submission records the
tokenizer version; unsubmitted requests wait until earlier batches complete.
This scheduling change does not modify prompts or select results by quality.

## Audit and report

Use `scripts/report_repoqa.py --experiment ... --grades ... --output ... --audit
audit.json` to aggregate completed cases. The audit JSON contains a `method`
description and a `cases` object keyed by case directory name. Each audited case
must include `receipt_sha256` matching the original grading receipt's `sha256`.
An optional `changes` list identifies `kind` (`facts` or `claims`), a 1-based
`index`, the changed boolean `values`, and an evidence-specific `reason`.

A malformed grader response remains an error, with its raw response and paid
receipt preserved. To adjudicate it, provide a complete `replacement_grade`
containing all fact and claim judgments in the grader schema, plus a nonempty
case-level `reason`. Missing judgments cannot become automatic zeroes or passes.
The original receipt is never overwritten, and the report records the audit hash.

**Batch results cannot establish interactive latency.** Receipts set interactive
latency to null and distinguish local replay assembly time from provider batch
turnaround. Normal list-price estimates and the 50%-discounted batch estimates
are reported separately. The earlier interactive observations remain a small,
quota-interrupted pilot, not a 30-question latency comparison. A release decision
must disclose this limitation.

Prefer the lowest-cost strategy within one fully correct question of the best and
within five percentage points of its citation-support precision. Retention needs
at least three additional fully correct questions over lexical retrieval to
justify its overhead. Publish raw counts and limitations, even if no strategy
meets the rule. Learned retention remains experimental and is not evaluated here.
