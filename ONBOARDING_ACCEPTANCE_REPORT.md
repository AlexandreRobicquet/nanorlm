# Newcomer Acceptance Report — 2026-07-28

## Gate decision

**Gate A: Passed and closed.**

The literal clone-only newcomer journey, canonical contributor checks, Task 06 report-bundle and
E2E paths, and pinned external showcases all completed from a fresh checkout. No documented command
needed an undisclosed setup step or a product/runtime workaround.

| Receipt fact | Recorded value |
| --- | --- |
| nanoRLM revision under test | [`4467fc25d020567ddf04624249414d71e1c1e116`](https://github.com/AlexandreRobicquet/nanorlm/commit/4467fc25d020567ddf04624249414d71e1c1e116) |
| nanoRLM branch under test | `master` |
| Verifiers revision under test | [`482e28ffa1f2613325867badaba4707b7c751d28`](https://github.com/PrimeIntellect-ai/verifiers/commit/482e28ffa1f2613325867badaba4707b7c751d28) |
| Packaging contract | Option B: run from a source checkout; no installable runtime API |
| Gate interval | 2026-07-28 00:12:56–00:20:19 PDT (07:12:56–07:20:19 UTC) |
| Gate wall time | 7 minutes 23 seconds |
| Hosted-model phases | Not run |
| Model-provider configuration | None; heuristic/deterministic paths only |
| API spend | **$0** |
| Documented-path workarounds | **None** |
| Observer-side diagnostic incidents | Three, recorded below |

This report was added after the tested revision. Its eventual documentation-only commit is not the
system-under-test revision above.

## Scope and evidence standard

This run exercised the final documentation produced by the newcomer-remediation sequence recorded
in [`ONBOARDING_AUDIT_TODOS.md`](ONBOARDING_AUDIT_TODOS.md). It did not edit runtime, benchmark,
provider, model, or research-result behavior. A genuine product or documentation-path failure would
have left Gate A open and been routed to a separate fix.

Operational completion and research evidence are deliberately separate in this receipt:

- `status: passed` means a command completed and its declared artifacts passed structural checks.
- `negative_or_inconclusive` remains the learned-retention research verdict where the evidence does
  not clear the repository's improvement threshold.
- Deterministic fixtures, synthetic slices, and local pinned-repository runs are regression and
  mechanics evidence. They are not claims of general benchmark or real-model performance.

## Freshness, environment, and network boundary

### Fresh resources

| Resource | Fresh location | Pre-run state |
| --- | --- | --- |
| nanoRLM network clone | `/tmp/nanorlm-onboarding-gate.rS1pjn/repo` | Newly cloned at the tested `master`; clean |
| uv cache | `/tmp/nanorlm-onboarding-uv-cache.5JKF72` | Newly created and empty |
| Verifiers checkout | `/tmp/nanorlm-verifiers` | Absent before the run; initialized and shallow-fetched at the exact pin |
| Packaging inspection | `/tmp/nanorlm-onboarding-package.AGzbaB` | Newly created outside the checkout |

Before setup, the nanoRLM clone had no `.venv/`, `outputs/`, `examples/outputs/`, or
`showcases/outputs/`. The gate did not reuse another worktree, environment, generated output,
package archive, uv cache, or Verifiers checkout. Both repositories were clean at the end.

### Environment

| Fact | Value |
| --- | --- |
| Operating system | macOS 26.6, build 25G5028f |
| Kernel | Darwin 25.6.0 |
| Architecture | arm64 |
| uv | 0.11.21 (Homebrew 2026-06-11, aarch64-apple-darwin) |
| Repository Python selector | `.python-version` = `3.11` |
| Selected interpreter | Python 3.11.15 |

Every nanoRLM validation subprocess ran with the two supported hosted-provider key variables
explicitly removed:

```bash
env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY \
  UV_CACHE_DIR=/tmp/nanorlm-onboarding-uv-cache.5JKF72 <command>
```

Network access was limited to the public nanoRLM clone, the locked Python/tool downloads needed by
the first empty-cache sync and build isolation, and the public shallow fetch of the exact Verifiers
revision. No hosted or local model provider was configured, and no model phase was executed.

### Timing

The 7-minute-23-second wall interval includes evidence inspection and observer diagnostics between
commands. Separately recorded command timings were:

| Milestone | Duration |
| --- | ---: |
| Fresh nanoRLM network clone | 0.70 s |
| Empty-cache `uv sync` | 0.27 s |
| Clone + sync + Python-version active command time | 0.97 s |
| First meaningful environment output (`Python 3.11.15`) | 58 s wall-clock from gate start |
| First benchmark result table | 2.47 s cumulative active command time, after the full unittest run |
| Full gate interval | 443 s |

“Active command time” is subprocess runtime and excludes the evidence inspection between commands.
Wall time is reported separately so the faster active figure is not presented as end-to-end elapsed
time.

## Literal newcomer and contributor commands

The commands below are shown after shell-variable expansion. Except for public Git operations,
each was run under the key-removal and fresh-cache prefix above.

### Clone and README quickstart

```bash
git clone https://github.com/AlexandreRobicquet/nanorlm.git \
  /tmp/nanorlm-onboarding-gate.rS1pjn/repo
cd /tmp/nanorlm-onboarding-gate.rS1pjn/repo

uv sync
uv run python --version
uv run python -m unittest discover -s tests -v
uv run python bench.py --dataset verifiers_smoke --limit 2 --budget 80 --depth 2 --repo-root tests/fixtures/verifiers-mini
uv run python examples/run_dossiers.py --limit 4 --budget 80 --depth 4 --output-dir outputs/quickstart/dossierbench
uv run python scripts/run_benchmark_e2e.py --phases learned --learned-train-limit 4 --learned-eval-limit 4 --output-root outputs/e2e --run-id quickstart-learned
```

| Command/result | Status | Separately recorded duration | Evidence |
| --- | --- | ---: | --- |
| Python selection | Passed | included above | `Python 3.11.15` |
| Complete unittest discovery | Passed | 1.50 s tool time; 1.332 s test time | 92 tests |
| `verifiers_smoke` quickstart | Passed | included in 2.47 s first-table active time | 2 cases; stdout-only; no bundle, as documented |
| Four-case DossierBench quickstart | Passed | 0.817 s | Complete bundle at `outputs/quickstart/dossierbench` |
| Learned quickstart E2E | Passed operationally | 1.351 s | Manifest passed at `outputs/e2e/quickstart-learned`; verdict `negative_or_inconclusive` |

### Contributor fast loop and canonical verification

The copied contributor fast loop passed:

```bash
uv sync --frozen
uv run python --version
uv run python bench.py --dataset pairbench --limit 4 --budget 60 --depth 2
```

The canonical verification block also passed:

```bash
uv lock --check
uv sync --frozen
uv run python scripts/check_markdown_links.py
uv run python -m unittest discover -s tests -v
uv run --frozen pytest
uv run python -m py_compile learned_retention.py nanorlm.py policies.py bench.py scripts/check_markdown_links.py scripts/check_verifiers_compatibility.py scripts/prepare_ruler_external_jsonl.py scripts/train_learned_retention.py scripts/run_benchmark_e2e.py examples/run_verifiers.py examples/run_needlepairs.py examples/run_dossiers.py examples/run_planning.py showcases/planning.py showcases/generate_assets.py
uv run python bench.py --dataset pairbench --limit 4 --budget 60 --depth 2
uv run python bench.py --dataset ruler_synthetic --limit 4 --budget 90 --depth 4 --policies pairwise_tournament,learned_retention
uv run python bench.py --dataset babilong_synthetic --limit 4 --budget 90 --depth 4 --policies pairwise_tournament,learned_retention
uv run python bench.py --dataset verifiers_smoke --limit 2 --budget 80 --depth 2 --repo-root tests/fixtures/verifiers-mini
uv run python bench.py --dataset external_jsonl --dataset-path tests/fixtures/external-benchmark-mini.jsonl --limit 2 --budget 80 --depth 2
uv run python scripts/run_benchmark_e2e.py --phases smoke --smoke-limit 1 --output-root outputs/e2e --run-id verify-smoke
uv run python scripts/run_benchmark_e2e.py --phases learned --learned-train-limit 2 --learned-eval-limit 2 --output-root outputs/e2e --run-id verify-learned
```

Exact duplicates already executed in the quickstart or contributor loop were cross-referenced
above; they were not silently omitted. The targeted command

```bash
uv run python -m unittest tests/test_nanorlm.py -v
```

passed 35 tests. The complete locked pytest run passed all 92 items in 1.896 seconds. Lock
consistency, frozen sync, compilation, PairBench, both four-case synthetic long-context slices, the
external fixture adapter, smoke E2E, and learned E2E all passed. `verify-learned` remained
`negative_or_inconclusive`.

## Complete-suite parity

An observer diagnostic normalized recursively discovered unittest IDs and public pytest collection
IDs, then compared their sets. Pytest IDs of the form
`tests/test_x.py::ClassName::test_name` were normalized to
`test_x.ClassName.test_name`, matching `unittest.TestCase.id()`. The following reproducible
comparator passed:

```bash
uv run python - <<'PY'
from pathlib import Path
import subprocess
import sys
import unittest

def iter_tests(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from iter_tests(item)
        else:
            yield item

unittest_ids = {
    test.id()
    for test in iter_tests(unittest.defaultTestLoader.discover("tests"))
}
collected = subprocess.run(
    [sys.executable, "-m", "pytest", "--collect-only", "-q"],
    check=True,
    capture_output=True,
    text=True,
).stdout.splitlines()
pytest_ids = set()
for nodeid in collected:
    if "::" not in nodeid:
        continue
    path, *parts = nodeid.split("::")
    pytest_ids.add(".".join([Path(path).stem, *parts]))

assert unittest_ids == pytest_ids, (
    sorted(unittest_ids - pytest_ids),
    sorted(pytest_ids - unittest_ids),
)
print(f"unittest: {len(unittest_ids)} unique test IDs")
print(f"pytest:   {len(pytest_ids)} unique test IDs")
print(f"Equivalent complete collection: {len(unittest_ids)} tests")
PY
```

Both runners collected 92 unique tests and the sets were exactly equal:

```text
unittest: 92 unique test IDs
pytest:   92 unique test IDs
Equivalent complete collection: 92 tests
```

This confirms that the two green totals represent the same complete suite, rather than coincidental
counts over different tests.

## Tiny Example semantic acceptance

The checked-in `tests/test_readme.py` helper was invoked through unittest and extracted and executed
the exact fenced Python source under the README's “Tiny Example” heading:

```bash
uv run python -m unittest tests/test_readme.py -v
```

It passed in 0.024 seconds. The receipt confirmed all promised semantics:

- the answer contains the blocker, “stale endpoint registry cache”;
- the answer contains the fix: reloading the endpoint registry and invalidating the cache;
- root and child splits, leaf inspection, and retention events appear in the trace;
- retention decisions are nonempty and at least one candidate is dropped;
- retained tokens stay within the configured memory budget;
- `incident-a.txt` and `incident-b.txt` are retained;
- `incident-c.txt` and `incident-d.txt` are dropped; and
- maximum memory depth is exactly 2.

## Explicit report bundles

The minimum evidence-producing commands from the gate task passed:

```bash
uv run python bench.py \
  --dataset verifiers_smoke \
  --limit 2 \
  --budget 80 \
  --depth 2 \
  --repo-root tests/fixtures/verifiers-mini \
  --output-dir outputs/onboarding/verifiers-smoke

uv run python examples/run_dossiers.py \
  --limit 4 \
  --budget 80 \
  --depth 4 \
  --output-dir outputs/onboarding/dossiers

uv run python scripts/run_benchmark_e2e.py \
  --phases learned \
  --learned-train-limit 4 \
  --learned-eval-limit 4
```

| Output root | Structural result | Row/trace evidence |
| --- | --- | --- |
| `outputs/onboarding/verifiers-smoke` | All five required components; normalized output path printed | 6 policies × 2 cases = 12 rows; 24 trace files |
| `outputs/onboarding/dossiers` | All five required components plus `summary.pretty.json` | 5 policies × 4 cases = 20 rows; 40 trace files |
| `outputs/e2e/e2e-20260728-071505` | Manifest `status: passed` | Learned verdict `negative_or_inconclusive` |

For both direct bundles, the five-part contract was present:

1. `summary.json`
2. `per_case.jsonl`
3. `curves.json`
4. `experiment_report.md`
5. `trace_examples/`

Every row completed and all recorded costs were zero.

## Offline E2E results

| Run root | Operational status | Research verdict |
| --- | --- | --- |
| `outputs/e2e/quickstart-learned` | Passed | `negative_or_inconclusive` |
| `outputs/e2e/verify-smoke` | Passed | Not a learned-policy claim |
| `outputs/e2e/verify-learned` | Passed | `negative_or_inconclusive` |
| `outputs/e2e/e2e-20260728-071505` | Passed | `negative_or_inconclusive` |
| `outputs/e2e/offline` | Passed in 14.786 s | Operational pinned-repository acceptance, not a real-model claim |

The offline command was:

```bash
uv run python scripts/run_benchmark_e2e.py \
  --phases offline \
  --repo-root /tmp/nanorlm-verifiers \
  --output-root outputs/e2e \
  --run-id offline
```

No learned run was promoted to a positive result merely because its manifest passed.

## Pinned Verifiers acceptance

The external repository was initialized from an absent path and shallow-fetched at the exact
documented revision:

```bash
git init /tmp/nanorlm-verifiers
git -C /tmp/nanorlm-verifiers remote add origin https://github.com/PrimeIntellect-ai/verifiers.git
git -C /tmp/nanorlm-verifiers fetch --depth 1 origin 482e28ffa1f2613325867badaba4707b7c751d28
git -C /tmp/nanorlm-verifiers checkout --detach FETCH_HEAD

uv run python scripts/check_verifiers_compatibility.py \
  --repo-root /tmp/nanorlm-verifiers

uv run python examples/run_verifiers.py \
  --repo-root /tmp/nanorlm-verifiers \
  --limit 30

uv run python examples/run_planning.py \
  --repo-root /tmp/nanorlm-verifiers \
  --limit 10 \
  --budget 140 \
  --depth 2
```

The shallow checkout contained one commit and remained clean. Preflight found all 25 required paths.
Both expected and actual revisions were
`482e28ffa1f2613325867badaba4707b7c751d28`, with `matches: true`.

| Acceptance path | Result |
| --- | --- |
| Default 30-case Codebase QA audit path | 30/30 cases executed for each of 5 policies; 150 completed rows; 300 trace files; answer accuracy 0.200–0.767; operational pass only |
| Default 10-task grounded-planning audit path | 10/10 tasks executed; plans and JSON/JSONL/tree traces complete; average file recall 0.834, keyword coverage 0.900, missing-critical-file rate 0.500; operational pass only |
| README Verifiers wrapper with `--output-dir outputs/verifiers_30/heuristic` | 30 completed cases; complete revision metadata |
| Direct six-policy `verifiers_30` showcase bundle | Complete bundle at `outputs/verifiers_30`; normalized path printed |
| Explicit planning output path | 10 completed tasks at `showcases/outputs/planning`; complete revision metadata |

The explicit documented forms were also executed:

```bash
uv run python examples/run_verifiers.py \
  --repo-root /tmp/nanorlm-verifiers \
  --limit 30 \
  --output-dir outputs/verifiers_30/heuristic

uv run python examples/run_planning.py \
  --repo-root /tmp/nanorlm-verifiers \
  --limit 10 \
  --budget 140 \
  --depth 2 \
  --output-dir showcases/outputs/planning

uv run python bench.py \
  --dataset verifiers_30 \
  --limit 30 \
  --budget 140 \
  --depth 2 \
  --repo-root /tmp/nanorlm-verifiers \
  --output-dir outputs/verifiers_30
```

A recursive post-run scan counted 2,592 per-case `cost_estimate` values and 427 summary
`total_cost_estimate` values across generated JSON and JSONL. All 3,019 realized execution-cost
observations were zero. Five `real_max_estimated_cost: 20.0` configuration guards were excluded
because they are caps, not incurred cost:

```bash
uv run python - <<'PY'
from collections import Counter
import json
from pathlib import Path

counts = Counter()
nonzero = []
caps = []
roots = [Path("outputs"), Path("examples/outputs"), Path("showcases/outputs")]

def inspect(value, source):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"cost_estimate", "total_cost_estimate"} and isinstance(child, (int, float)):
                counts[key] += 1
                if child != 0:
                    nonzero.append((str(source), key, child))
            elif key == "real_max_estimated_cost" and isinstance(child, (int, float)):
                caps.append((str(source), child))
            inspect(child, source)
    elif isinstance(value, list):
        for child in value:
            inspect(child, source)

for root in roots:
    for source in sorted(root.rglob("*.json")):
        inspect(json.loads(source.read_text()), source)
    for source in sorted(root.rglob("*.jsonl")):
        for line in source.read_text().splitlines():
            if line.strip():
                inspect(json.loads(line), source)

assert counts == {"cost_estimate": 2592, "total_cost_estimate": 427}
assert not nonzero
assert len(caps) == 5 and {value for _, value in caps} == {20.0}
print(dict(counts), "total", sum(counts.values()), "all zero")
print("excluded configured caps:", len(caps), "at", caps[0][1])
PY
```

## Clone-only packaging acceptance

The repository-metadata test module passed all four tests, including the Option B boundary:

```bash
uv run python tests/test_repository_metadata.py
uv build --out-dir /tmp/nanorlm-onboarding-package.AGzbaB
```

The build produced:

- `nanorlm-0.1.0-py3-none-any.whl`
  (`sha256:691ceb5155d973f3207c064dab4c44ec0be23544cdb7cc8e440ccec4a8bf7e84`)
- `nanorlm-0.1.0.tar.gz`
  (`sha256:3f38bba3646d75c5c959123f886e005a1c8e02f85ec5801c85f17ef3733503f4`)

The archive assertion was:

```bash
uv run python - <<'PY'
from email.parser import Parser
from pathlib import Path
import tarfile
import zipfile

from packaging.specifiers import SpecifierSet

artifacts = Path("/tmp/nanorlm-onboarding-package.AGzbaB")
wheel = artifacts / "nanorlm-0.1.0-py3-none-any.whl"
sdist = artifacts / "nanorlm-0.1.0.tar.gz"
license_bytes = Path("LICENSE").read_bytes()
forbidden = {"nanorlm.py", "policies.py", "learned_retention.py", "bench.py"}

with zipfile.ZipFile(wheel) as archive:
    names = archive.namelist()
    assert len(names) == 5 and all(".dist-info/" in name for name in names)
    metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
    metadata = Parser().parsestr(archive.read(metadata_name).decode())
    assert SpecifierSet(metadata["Requires-Python"]) == SpecifierSet(">=3.11,<3.13")
    assert not metadata.get_all("Requires-Dist")
    top_level = next(name for name in names if name.endswith("top_level.txt"))
    assert not archive.read(top_level).strip()
    embedded_license = next(name for name in names if name.endswith("/licenses/LICENSE"))
    assert archive.read(embedded_license) == license_bytes
    assert forbidden.isdisjoint({Path(name).name for name in names})

with tarfile.open(sdist, "r:gz") as archive:
    members = [member for member in archive.getmembers() if member.isfile()]
    relative = [Path(*Path(member.name).parts[1:]).as_posix() for member in members]
    assert forbidden.isdisjoint({Path(name).name for name in relative})
    assert not any(name == "showcases" or name.startswith("showcases/") for name in relative)
    license_member = next(member for member in members if Path(member.name).name == "LICENSE")
    extracted = archive.extractfile(license_member)
    assert extracted is not None and extracted.read() == license_bytes

print("wheel: 5 dist-info files; no runtime dependencies or import surface")
print("sdist: no runtime modules or showcases")
print("wheel and sdist licenses match LICENSE")
PY
```

The metadata-only wheel was then installed into a new external environment and inspected from
outside the checkout:

```bash
uv venv --python 3.11 /tmp/nanorlm-onboarding-package.AGzbaB/venv
uv pip install \
  --python /tmp/nanorlm-onboarding-package.AGzbaB/venv/bin/python \
  --no-deps \
  /tmp/nanorlm-onboarding-package.AGzbaB/nanorlm-0.1.0-py3-none-any.whl
cd /tmp/nanorlm-onboarding-package.AGzbaB
./venv/bin/python - <<'PY'
from importlib.util import find_spec

modules = ("nanorlm", "policies", "learned_retention", "showcases")
found = {module: find_spec(module) for module in modules}
assert all(spec is None for spec in found.values()), found
print("No runtime import surface:", ", ".join(modules))
PY
```

These checks established:

- `[tool.uv] package = false`;
- setuptools has explicit empty `packages` and `py-modules` lists;
- runtime dependencies are empty;
- `Requires-Python` is semantically `>=3.11,<3.13`;
- the wheel contains only `.dist-info` entries and has an empty `top_level.txt`;
- neither archive exposes `nanorlm.py`, `policies.py`, `learned_retention.py`, `bench.py`, or
  `showcases`;
- the embedded MIT license exactly matches [`LICENSE`](LICENSE); and
- after installing the metadata-only wheel in an isolated environment outside the checkout,
  `nanorlm`, `policies`, `learned_retention`, and `showcases` had no import specifications.

The build backend created `nanorlm.egg-info/` in the source checkout. After inspection, the exact
cleanup command was:

```bash
mv nanorlm.egg-info /tmp/nanorlm-onboarding-package.AGzbaB/build-egg-info
```

This moved the generated directory intact and restored the fresh clone to a clean state. It is
packaging-output cleanup, not a command needed to make a documented newcomer result pass.

## Documentation and link integrity

The final baseline command passed:

```bash
uv run python scripts/check_markdown_links.py
```

On the tested system revision, it checked 15 tracked Markdown files and 34 local targets, and
skipped 11 external URLs without fetching them. Setup, contributor, showcase, clone-only,
output-location, credential, cost, and operational-versus-research language were therefore
exercised in the same fresh checkout as the commands they describe.

After this report and the two Gate A closeout edits were staged, the checker passed again over 16
tracked Markdown files, 42 local targets, and 13 skipped external URLs. The report-only closeout is
also subject to the Python 3.11/3.12 CI matrix before merge.

## Diagnostic incidents and workarounds

No documented-path workaround was used. Three observer-side incidents were corrected and retained
in the receipt:

1. A freshness preflight used `path` as a zsh variable name. Because `path` is tied to `PATH` in
   zsh, the observer command then emitted `find: command not found`. It was rerun with
   `candidate_path` and `/usr/bin/find`, confirming the fresh cache was empty. No system-under-test
   or documented command was affected.
2. The first archive assertion compared the literal text
   `Requires-Python: >=3.11,<3.13`. The wheel emitted the semantically identical normalized order
   `Requires-Python: <3.13,>=3.11`. A set-based specifier assertion passed. The archive did not
   change.
3. `uv build` created `nanorlm.egg-info/` in the checkout. It was inspected and moved to the
   external packaging directory so final cleanliness could be verified. No runtime or tracked file
   changed.

None of these incidents converted a failed newcomer promise into a pass.

## Output footprint

At the end of validation:

| Location | Size |
| --- | ---: |
| Fresh nanoRLM `.venv/` | 9.4 MB |
| `outputs/` | 119 MB |
| `examples/outputs/` | 5.8 MB |
| `showcases/outputs/` | 164 KB |
| Fresh uv cache | 12 MB |
| External package inspection directory | 232 KB |
| Fresh shallow Verifiers checkout | 34 MB |

The three nanoRLM output roots contained 2,776 generated files. These stayed under ignored
generated-output locations and were not added to the repository.

## Limitations

- This one-shot clean-checkout receipt selected Python 3.11.15 from `.python-version`. The same
  repository checks also run on Python 3.11 and 3.12 in CI; the report-only closeout must pass that
  matrix before merge.
- The external result is intentionally pinned acceptance at the documented Verifiers revision, not
  a statement about current upstream HEAD.
- No hosted or local real model was exercised. This proves the credential-free path and mechanics,
  not model quality.
- Synthetic, fixture, and deterministic results remain non-headline evidence.
- Generated output bundles are ephemeral receipts whose checked structures and counts are recorded
  here; they are not committed benchmark snapshots.

## Final requirement matrix

| Requirement | Result |
| --- | --- |
| Fresh nanoRLM checkout, environment, cache, and outputs | Passed |
| Literal prerequisite and README quickstart | Passed |
| Contributor fast loop and canonical verification | Passed |
| unittest/pytest complete-suite equivalence | Passed: 92 = 92, exact ID set |
| Tiny Example recursion, retention, blocker, and fix | Passed |
| Explicit five-part report bundles | Passed |
| Smoke and learned E2E, with separate research verdicts | Passed operationally; learned verdicts remain `negative_or_inconclusive` |
| Fresh pinned Verifiers checkout and 25-path preflight | Passed |
| 30-case Codebase QA path | Operational pass: 30/30 cases for each of 5 policies; 150 completed rows; accuracy 0.200–0.767 |
| 10-task grounded-planning path | Operational pass: 10/10 tasks; average file recall 0.834; missing-critical-file rate 0.500 |
| Expected/actual external revision metadata | Passed: exact match |
| Option B clone-only archive and import boundary | Passed |
| Local Markdown links | Passed |
| Hidden documented-path workaround | None |
| Hosted-provider keys/model configuration/API cost | Removed / none / $0 realized execution cost |

On this evidence, the shared completion gate in
[`ONBOARDING_AUDIT_TODOS.md`](ONBOARDING_AUDIT_TODOS.md) is complete and Gate A in
[`ROADMAP.example.md`](ROADMAP.example.md) is closed.
