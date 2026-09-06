# Next steps after the repository-answer release

The useful product boundary is now concrete: ask a question about a source
checkout, inspect the cited answer, and keep a reusable evidence and cost receipt.
The next investment should make that answer dependable on unfamiliar repositories.

## 1. Eliminate avoidable output failures

Seven of the nine failed answers in the frozen 90-case experiment mistyped a
long source identifier; two returned invalid JSON. Keep those failures in the
published results. For the next version, expose short numbered citation aliases
to the model while retaining content hashes in the evidence bundle, and constrain
the answer schema at the provider boundary. Reject invented aliases. Preserve
original responses and charge any repair calls explicitly.

Done means zero malformed answers or invalid citation aliases on a **fresh**
30-question set, plus tests that reject fabricated aliases and tampered evidence.
Do not silently repair the historical experiment or reuse it as held-out evidence.

## 2. Retrieve implementation and relevant tests together

The evaluation includes plausible explanations that miss exceptional control
flow, cite documentation too broadly, or name tests that do not establish the
claimed behavior. Improve deterministic retrieval around complete definitions and
associated tests, within the same context and cost budgets. Start with concrete
misses: finite retry-generator exhaustion, TTL expiration during `len`, callable
configuration timing, and path-versus-stream precedence.

Compare the change with the released baseline on new questions. Keep it only if
it improves complete answers and citation support without hidden extra calls.
Embeddings, an index service, and new retention policies need evidence of added
value before becoming dependencies.

## 3. Validate everyday usefulness independently

Collect questions from an engineer who did not implement the tool, across three
new repositories including a language other than Python. Freeze the questions,
reference facts and source commits before running. Use human review for ambiguous
behavior and test-coverage judgments; model grading alone proved insufficient.

Run the alternatives through the synchronous API after quota is available, with
three runs per question. Report complete answers, supported claims, failures,
median/p95 response time and total normal-price cost separately. Suggested release
targets are at least 24/30 complete answers, at least 95% supported claims, no
output-format failures, mean inference cost below USD 0.01 per question and p95
interactive latency below 10 seconds. These are proposed acceptance targets,
not measurements from the batch experiment.

## 4. Keep the research evidence reproducible

The repaired contract PRs #26 and #28 are merged. The rebuilt matched offline
bundle establishes execution, budget, task identity and trace integrity; it does
not establish a hosted-model quality win. Keep learned retention optional until
an independent comparison demonstrates enough additional correct answers to
justify its total inspection and generation cost.

The pinned Verifiers compatibility target passes. Its current-upstream canary
has failed after upstream paths changed. Refresh that separate dataset and its
source references against an explicit new upstream commit, then restore the
canary with a new compatibility receipt. Preserve the existing supported pin
and do not turn off the drift signal to obtain a green badge.
