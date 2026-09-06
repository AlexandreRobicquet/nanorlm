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
evaluation expense from user-facing inference expense. A subsequent assistant
audit of flagged cases and a spread of passing cases is recorded separately;
this is **not human adjudication**. Both models share a family, so correlated
grading errors remain a limitation. The grader and its prompt are frozen before
grading, and raw judgments are retained when an audit changes a score.

```bash
uv run python scripts/grade_repoqa.py \
  --experiment outputs/repoqa-v1 --output outputs/repoqa-v1-grades
```

Prefer the lowest-cost strategy within one fully correct question of the best and
within five percentage points of its citation-support precision. Retention needs
at least three additional fully correct questions over lexical retrieval to
justify its overhead. Publish raw counts and limitations, even if no strategy
meets the rule. Learned retention remains experimental and is not evaluated here.
