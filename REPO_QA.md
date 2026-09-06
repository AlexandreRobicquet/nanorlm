# Ask a repository question

Use nanoRLM to locate a behavior, explain its configuration, and find the tests that actually cover it. Start with a specific question:

```bash
uv run python ask.py \
  'Where is the HTTP retry limit set, what overrides the delay, and which tests cover it?' \
  --repo /path/to/repository \
  --model gpt-4.1-mini-2025-04-14 \
  --output outputs/retry-answer
```

The model option sends the question and selected source spans to OpenAI and reads `OPENAI_API_KEY` from the environment. Without `--model`, the command creates local evidence only. Add `--preview` to inspect evidence without making model requests even when a model is configured. No server, vector database, LOOM installation, or training job is required.

Open `answer.md` first. Its citations link to `sources.md`, which contains the original snippets with file names, line ranges and file hashes. The bundle also contains:

- `evidence.json`: the question, repository commit and working-tree snapshot, original source spans, hashes, retrieval settings, and omitted files/spans.
- `answer.json`: factual claims with source IDs and explicit uncertainties.
- `run.json`: completion status, configured and returned model identifiers, code hashes, per-call token usage, latency, and estimated API cost, including inspection and repair calls.
- `checksums.json`: a checksum for every generated artifact. A failed run preserves its completed call ledger and available candidate evidence.

Use a fresh output directory for every run. Evidence is reusable for the same question:

```bash
uv run python ask.py 'Where is the retry limit set?' \
  --repo /path/to/repository --output outputs/retry-evidence

uv run python ask.py 'Where is the retry limit set?' \
  --evidence outputs/retry-evidence/evidence.json \
  --model gpt-4.1-mini-2025-04-14 --output outputs/retry-answer
```

Evidence reuse verifies its checksum, source identifiers and span hashes. It uses the captured snapshot; it does not assume that the live checkout remains unchanged. Retrieval depends only on the repository and question, never expected answers.

## Strategies and budgets

`--strategy lexical` is the default selected by the [audited comparison](RESULTS.md): BM25 over source text and path terms, with neighboring windows to preserve function boundaries, followed by one answer call. The normal evidence budget is 6,000 estimated tokens including source headers.

`--strategy retention` runs recursive inspection and the selected retention policy over the lexical candidate pool, then answers from the retained original source spans. It adds model calls; use it only when an evaluation justifies that cost. `--retention-policy` accepts the existing policies. `--learned-model` is optional and experimental. Defaults are 16,000 candidate tokens and 512 summary-memory tokens. Incomplete inspection is reported in the run receipt. A retention `--preview` shows the entire candidate pool that inspections would receive, even when it exceeds the smaller final-answer context budget; its evidence stage is `candidates`.

`--strategy full` supplies every scanned source span. Increase `--context-budget` as needed; the command refuses to silently truncate this baseline. “Full” means the declared eligible source inventory, subject to the same file exclusions as other strategies.

`--max-cost` defaults to USD 0.25. Each request reserves a conservative input-byte bound and maximum output cost before network access. Usage receipts use provider-reported tokens and the local list-price table; they are estimates, not invoices. Server-side cache discounts are not deducted. Failed requests whose billing cannot be observed are explicitly marked. The default response cap is 1,600 tokens. A malformed or uncited final answer fails with its raw response and receipt retained.

## Source boundaries

Git repositories use tracked files from the current working tree, including local edits. Other directories use a bounded recursive scan. Generated/vendor directories, lockfiles, binaries, non-UTF-8 files, symlinks, common credential filenames and recognized key patterns are excluded. Files are limited to 1 MB and the scanned source inventory to 20 MB. Exclusions and size-limit omissions appear in the evidence bundle. Pattern filtering is not a comprehensive secret detector; inspect the preview for repositories containing sensitive source material.

Coverage counts show how much source was selected, not whether an answer is correct. Source IDs, coordinates and content hashes establish where a citation points. They do not automatically prove that the cited code supports the claim. Questions about runtime deployment state, external services or untracked files need evidence outside this command's source inventory.
