# Repository answers: v0.2 evaluation

**Keep lexical retrieval as the default.** On 30 frozen questions across backoff,
python-dotenv and cachetools, it produced more complete answers and cost less than
full context or recursive retention. This supports a simple, inspectable starting
point; **77.5% citation support is not sufficient for unattended use**. Read the
cited source before relying on an answer.

| Approach | Complete, no material error | Correct checklist facts | Supported claims | Failed answers | API calls |
|---|---:|---:|---:|---:|---:|
| Lexical retrieval | **16/30** | 64/90 | **86/111 (77.5%)** | 1/30 | 30 |
| Full eligible context | 9/30 | 54/90 | 63/87 (72.4%) | 7/30 | 30 |
| Recursive retention | 3/30 | 37/90 | 54/78 (69.2%) | 1/30 | 514 |

A complete answer must express all three frozen reference facts and contain no
materially incorrect additional claim. Citation support is a **separate** measure:
each claim's own cited excerpts must support every material assertion. Correct
answers can still cite the wrong source. Failed answers receive zero completeness;
the supported-claim denominator includes only claims in accepted answer artifacts.
All accepted citations passed source identity and hash validation, which does not
establish semantic support.

| Approach | Normal-price inference estimate, all 30 | Batch inference estimate, all 30 | Median batch availability |
|---|---:|---:|---:|
| Lexical retrieval | **$0.088019** | **$0.044009** | 75 s |
| Full eligible context | $0.541172 | $0.270586 | 76 s |
| Recursive retention | $0.308321 | $0.154161 | 251.5 s |

Costs include all 574 submitted answer/inspection requests, including inspection
responses unused after a failure. They use provider-reported tokens and published
list prices, with the Batch API's 50% discount shown separately; they are not
invoice totals. The complete answer experiment cost **$0.468756 at batch prices**.
Lexical's normal-price mean was $0.002934 per question. Retention cost 3.50 times
as much, and full context 6.15 times as much, on this set.

**Interactive latency is unavailable for the 30-question comparison.** The account's
synchronous quota interrupted a separate four-case pilot, so the complete experiment
used batch transport. Availability measures first submission to completion of the
last required batch; it includes shared queue waiting. It is not response latency,
and local replay timings are not model latency. A separate release-candidate CLI
smoke on nanoRLM itself answered the retry-limit question in 2.69 seconds for
$0.005714 using GPT-5.4 mini; that single observation uses a different model and is
not comparative latency evidence.

## What the results show

| Repository | Lexical complete | Full complete | Retention complete |
|---|---:|---:|---:|
| backoff | 5/10 | 2/10 | 1/10 |
| python-dotenv | 5/10 | 4/10 | 1/10 |
| cachetools | 6/10 | 3/10 | 1/10 |

The frozen decision rule chooses the cheapest approach within one complete answer
of the best and five percentage points of its citation support. Retention also
needs at least three more complete answers than lexical to earn its extra calls.
**Only lexical qualifies.** The tested retention policy is `pairwise_tournament`;
these results do not evaluate or disprove learned retention.

Seven of nine answer failures were mistyped long citation IDs; two were invalid
JSON. Those failures remain in the results with the original responses and billed
usage. No fuzzy citation repair, answer retry, or post-result retrieval tuning was
used to improve the comparison.

Source review found concrete semantic misses: callable retry limits described as
being evaluated on every attempt, finite-generator exceptions incorrectly described
as suppressed, TTL `len` described as leaving expired entries, and missing-file
behavior confused with dotenv search. Several accurate descriptions cited an
unrelated test or an adjacent source window missing the actual implementation.
These observations motivate the ordered [next steps](roadmap.md): short citation
aliases with strict output validation, complete-definition and test retrieval, then
an independently authored evaluation with synchronous timing.

## Method and limits

All answer runs use `gpt-4.1-mini-2025-04-14`, the same answer prompt and 1,600-token
output limit. Lexical and retention share a 6,000-token candidate pool; retention
uses a 512-token memory budget and answers from retained original source spans.
Full context receives every span admitted by the same bounded scanner. Source
commits, questions and reference excerpts were frozen before answer generation.
The [protocol](../evaluations/README.md) records implementation and transport changes.

Final grading used 41 synchronous GPT-5.4 mini calls, with strict schema counts,
medium reasoning and each claim's cited text attached directly. Strategy names,
costs and competing answers to the same question were hidden from that grader.
The implementation assistant then reviewed **all 270 fact judgments and 205 of 276
claims against their sources**, including every flagged claim, every claim in any
provisionally complete answer, and 16 predefined source-audit cases. The remaining
71 claim judgments retain model grading. The audit changed 39 fact judgments and
34 claim records; raw responses remain unchanged. A mandatory report gate checks
audit coverage and binds amendments to the original grading receipt hashes.

The questions and audit were written by the implementation assistant, **not an
independent human evaluator**. The audit could see strategy labels. Composite
reference facts sometimes require details beyond the literal question, such as
defaults, exception branches or a test's exact assertions. These scores measure
checklist completeness under documented judgments, not a human usefulness rating.
There is one answer per question/strategy, only three Python repositories, and
public source may appear in pretraining. Model grading made substantial errors;
assistant adjudication reduces known errors but is not an independent ground truth.
The final grader also missed an error label in synthetic calibration, which is why
its output alone was insufficient to publish scores.

Final grading cost **$0.870716**, separate from inference. Two discarded grading
passes cost $0.316245 and $0.153634 at batch prices; calibration and the cancellation-requested
third grading attempt are itemized separately in the release evidence, including
unknown billing and conservative bounds. No discarded pass contributes a score.
The frozen answer costs above are fully reconciled regardless of those separate
protocol-development expenses.

Machine-readable [summary](../evaluations/results/repoqa-v1/summary.json) and
[adjudication](../evaluations/results/repoqa-v1/audit.json) are checked in. The v0.2.0
release evidence archive includes the 90 original answer bundles, batch requests
and responses, final raw grading, audit seeds, fact audit, failure audit, discarded
grading passes, licenses and checksums. See its README to reproduce aggregation.
