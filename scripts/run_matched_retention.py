from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import (  # noqa: E402
    DATASET_CHOICES,
    BenchmarkExample,
    build_dataset,
    curves_from_summaries,
    run_dataset,
    write_report_bundle,
)
from nanorlm import (  # noqa: E402
    REMOTE_MODEL_PRICES,
    estimate_tokens,
    is_local_base_url,
    normalize_provider_name,
    slugify,
)


MATCHED_POLICIES = [
    "keep_recent",
    "summary_only",
    "single_critic_topk",
    "pairwise_tournament",
    "learned_retention",
]
SIMPLE_POLICIES = ["keep_recent", "summary_only", "single_critic_topk"]
DEFAULT_BUDGETS = [96, 128, 192]
PHASES = ("offline", "pilot", "confirmation")
SCHEMA_VERSION = "0.1"
FULL_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
CACHE_RECORD_RE = re.compile(r"[0-9a-f]{64}\.json")
CACHE_BINDING_NAME = "binding.json"
HOSTED_FAMILY_METADATA = {"ruler": "RULER", "babilong": "BABILong"}


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    label: str
    dataset: str
    path: Path | None = None


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def response_cache_namespace(binding: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(binding).encode("utf-8"))


def validate_response_cache_record(
    path: Path,
    *,
    expected_namespace: str,
    expected_model: str = "",
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid response-cache record: {path.name}") from exc
    request = payload.get("request") if isinstance(payload, dict) else None
    response = payload.get("response") if isinstance(payload, dict) else None
    usage = response.get("usage") if isinstance(response, dict) else None
    messages = request.get("messages") if isinstance(request, dict) else None
    if (
        payload.get("provider") != "openai_compatible"
        or payload.get("cache_key") != path.stem
        or (expected_model and payload.get("model") != expected_model)
        or payload.get("cache_namespace") != expected_namespace
        or not isinstance(messages, list)
        or not messages
        or not isinstance(response.get("content"), str)
        or not str(response.get("model", "")).strip()
        or not isinstance(usage, dict)
    ):
        raise ValueError(f"response-cache record has invalid structure: {path.name}")
    for key in ("prompt_tokens", "completion_tokens", "calls"):
        value = usage.get(key)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"response-cache record has invalid usage: {path.name}")
    if usage["calls"] < 1:
        raise ValueError(f"response-cache record has no logical model call: {path.name}")
    serialized = canonical_json(payload).lower()
    if "authorization" in serialized or "api_key" in serialized or "api-key" in serialized:
        raise ValueError(f"response-cache record contains credential material: {path.name}")
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "response_model": str(response["model"]),
        "prompt_tokens": int(usage["prompt_tokens"]),
        "completion_tokens": int(usage["completion_tokens"]),
        "calls": int(usage["calls"]),
    }


def prepare_response_cache(
    cache_root: Path,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    if cache_root.is_symlink():
        raise ValueError("response-cache directory must not be a symlink")
    cache_root.mkdir(parents=True, exist_ok=True)
    if not cache_root.is_dir():
        raise ValueError("response-cache path is not a directory")
    marker = cache_root / CACHE_BINDING_NAME
    namespace = response_cache_namespace(binding)
    configuration = binding.get("configuration")
    expected_model = (
        str(configuration.get("model", "")) if isinstance(configuration, dict) else ""
    )
    if marker.exists():
        if marker.is_symlink() or not marker.is_file():
            raise ValueError("response-cache binding is not a regular file")
        try:
            observed_binding = json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("response-cache binding is not valid JSON") from exc
        if observed_binding != binding:
            raise ValueError("response-cache binding does not match this frozen execution")
    else:
        write_json(marker, binding)

    records = []
    for item in sorted(cache_root.iterdir()):
        if item.name == CACHE_BINDING_NAME:
            continue
        if item.is_symlink() or not item.is_file() or CACHE_RECORD_RE.fullmatch(item.name) is None:
            raise ValueError(f"unexpected response-cache entry: {item.name}")
        records.append(
            validate_response_cache_record(
                item,
                expected_namespace=namespace,
                expected_model=expected_model,
            )
        )
    return {
        "path": "artifacts/response_cache",
        "namespace": namespace,
        "binding_sha256": sha256_file(marker),
        "record_count": len(records),
        "logical_calls": sum(record["calls"] for record in records),
        "prompt_tokens": sum(record["prompt_tokens"] for record in records),
        "completion_tokens": sum(record["completion_tokens"] for record in records),
        "response_models": sorted({record["response_model"] for record in records}),
        "records_sha256": sha256_bytes(canonical_json(records).encode("utf-8")),
    }


def snapshot_response_cache(
    cache_root: Path,
    output_root: Path,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    validation = prepare_response_cache(cache_root, binding)
    destination = output_root / validation["path"]
    if destination.exists():
        raise ValueError("response-cache snapshot destination already exists")
    destination.mkdir(parents=True)
    for item in sorted(cache_root.iterdir()):
        shutil.copyfile(item, destination / item.name)
    copied = prepare_response_cache(destination, binding)
    if copied != validation:
        raise ValueError("response-cache snapshot changed during publication copy")
    return copied


def parse_csv_ints(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values or any(item <= 0 for item in values):
        raise ValueError("budgets must be positive integers")
    return values


def parse_dataset_spec(value: str) -> DatasetSpec:
    parts = value.split(":", 2)
    if len(parts) < 2:
        raise ValueError("dataset spec must be LABEL:DATASET[:PATH]")
    label, dataset = parts[0].strip(), parts[1].strip()
    if not label or slugify(label) != label:
        raise ValueError("dataset label must already be a lowercase filesystem-safe slug")
    if dataset not in DATASET_CHOICES:
        raise ValueError(f"unknown dataset in spec {value}: {dataset}")
    path = Path(parts[2]).expanduser().resolve() if len(parts) == 3 and parts[2].strip() else None
    if dataset == "external_jsonl" and path is None:
        raise ValueError(f"external_jsonl spec requires a path: {value}")
    if dataset != "external_jsonl" and path is not None:
        raise ValueError(f"only external_jsonl specs accept a path: {value}")
    return DatasetSpec(label=label, dataset=dataset, path=path)


def parse_expected_dataset_hashes(values: Sequence[str]) -> dict[str, str]:
    expected: dict[str, str] = {}
    for value in values:
        label, separator, digest = value.partition("=")
        if not separator or not label or SHA256_RE.fullmatch(digest) is None:
            raise ValueError("expected dataset hash must be LABEL=64_HEX_SHA256")
        if label in expected:
            raise ValueError(f"duplicate expected dataset hash: {label}")
        expected[label] = digest
    return expected


def validate_dataset_hashes(
    specs: Sequence[DatasetSpec],
    expected: Mapping[str, str],
    *,
    required: bool,
) -> dict[str, str]:
    observed: dict[str, str] = {}
    external_labels = {spec.label for spec in specs if spec.path is not None}
    unknown_labels = set(expected) - external_labels
    if unknown_labels:
        raise ValueError(f"expected dataset hash has no external dataset: {sorted(unknown_labels)[0]}")
    for spec in specs:
        if spec.path is None:
            continue
        if not spec.path.is_file():
            raise ValueError(f"external dataset does not exist: {spec.path}")
        digest = sha256_file(spec.path)
        observed[spec.label] = digest
        expected_digest = expected.get(spec.label)
        if required and expected_digest is None:
            raise ValueError(f"external dataset requires an expected SHA-256: {spec.label}")
        if expected_digest is not None and digest != expected_digest:
            raise ValueError(f"external dataset SHA-256 mismatch: {spec.label}")
    return observed


def git_snapshot(path: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=path,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    status = run("status", "--porcelain")
    commit = run("rev-parse", "HEAD")
    return {
        "is_repository": bool(commit),
        "commit": commit,
        "branch": run("branch", "--show-current") or "detached",
        "clean": not bool(status),
        "status_entries": len(status.splitlines()) if status else 0,
    }


def commit_binding(snapshot: Mapping[str, Any] | None, expected: str) -> dict[str, Any]:
    expected = expected.strip().lower()
    actual = str(snapshot.get("commit", "")).lower() if snapshot else ""
    if not expected:
        reason = "expected_commit_missing"
    elif FULL_GIT_SHA_RE.fullmatch(expected) is None:
        reason = "expected_commit_not_full_sha"
    elif not snapshot or not snapshot.get("is_repository"):
        reason = "repository_unavailable"
    elif actual != expected:
        reason = "commit_mismatch"
    else:
        reason = None
    return {"ok": reason is None, "expected": expected or None, "actual": actual or None, "reason": reason}


def example_record(spec: DatasetSpec, index: int, example: BenchmarkExample) -> dict[str, Any]:
    context = [
        {
            "index": block_index,
            "name": block.name,
            "text_sha256": sha256_bytes(block.text.encode("utf-8")),
            "estimated_tokens": block.tokens,
        }
        for block_index, block in enumerate(example.context)
    ]
    payload = {
        "family": spec.label,
        "position": index,
        "name": example.name,
        "task_class": example.task_class,
        "query_sha256": sha256_bytes(example.query.encode("utf-8")),
        "answer_sha256": sha256_bytes(example.answer.encode("utf-8")),
        "must_contain_sha256": sha256_bytes(canonical_json(example.must_contain).encode("utf-8")),
        "context": context,
    }
    return {"task_id": f"task_{sha256_bytes(canonical_json(payload).encode('utf-8'))[:20]}", **payload}


def conversion_audit(
    tasks: Sequence[tuple[DatasetSpec, BenchmarkExample]],
) -> dict[str, Any]:
    violations = []
    for spec, example in tasks:
        context = "\n".join(block.text for block in example.context).lower()
        reasons = []
        if not example.query.strip():
            reasons.append("empty_query")
        if not example.answer.strip() or not example.must_contain:
            reasons.append("empty_answer_rule")
        if any(fragment.lower() not in context for fragment in example.must_contain):
            reasons.append("answer_fragment_absent_from_context")
        if reasons:
            violations.append(
                {
                    "family": spec.label,
                    "name": example.name,
                    "reasons": reasons,
                }
            )
    return {
        "ok": not violations,
        "tasks_checked": len(tasks),
        "violations": violations,
    }


def hosted_family_audit(
    tasks: Sequence[tuple[DatasetSpec, BenchmarkExample]],
) -> dict[str, Any]:
    violations = []
    observed: dict[str, set[str]] = {label: set() for label in HOSTED_FAMILY_METADATA}
    for spec, example in tasks:
        benchmark = str(example.metadata.get("benchmark", ""))
        observed.setdefault(spec.label, set()).add(benchmark)
        expected = HOSTED_FAMILY_METADATA.get(spec.label)
        if expected is None or benchmark != expected:
            violations.append(
                {
                    "family": spec.label,
                    "name": example.name,
                    "expected_benchmark": expected,
                    "observed_benchmark": benchmark or None,
                }
            )
    return {
        "ok": not violations,
        "tasks_checked": len(tasks),
        "expected": HOSTED_FAMILY_METADATA,
        "observed": {label: sorted(values) for label, values in sorted(observed.items())},
        "violations": violations,
    }


def load_spec_examples(
    specs: Sequence[DatasetSpec],
    *,
    limit: int,
    start_index: int,
    seed: int,
    repo_root: str,
) -> dict[str, list[BenchmarkExample]]:
    loaded: dict[str, list[BenchmarkExample]] = {}
    for spec in specs:
        examples = build_dataset(
            spec.dataset,
            limit=limit,
            start_index=start_index,
            seed=seed,
            repo_root=repo_root,
            dataset_path=spec.path,
        )
        names = [example.name for example in examples]
        if len(names) != len(set(names)):
            raise ValueError(f"dataset {spec.label} contains duplicate example names")
        loaded[spec.label] = examples
    return loaded


def round_robin_tasks(
    specs: Sequence[DatasetSpec],
    examples: Mapping[str, Sequence[BenchmarkExample]],
) -> list[tuple[DatasetSpec, BenchmarkExample]]:
    ordered: list[tuple[DatasetSpec, BenchmarkExample]] = []
    max_length = max((len(examples[spec.label]) for spec in specs), default=0)
    for index in range(max_length):
        for spec in specs:
            family = examples[spec.label]
            if index < len(family):
                ordered.append((spec, family[index]))
    return ordered


def conservative_cost_upper_bound(
    tasks: Sequence[tuple[DatasetSpec, BenchmarkExample]],
    *,
    provider: str,
    model: str,
    base_url: str | None,
    budget: int,
    depth: int,
    max_output_tokens: int,
) -> dict[str, Any]:
    normalized_provider = normalize_provider_name(provider)
    if normalized_provider == "heuristic" or is_local_base_url(base_url):
        return {
            "formula_version": 1,
            "logical_policy_upper_bound_usd": 0.0,
            "prompt_safety_factor": 4,
            "tasks": len(tasks),
        }
    price_key = (normalized_provider, model)
    if price_key not in REMOTE_MODEL_PRICES:
        raise ValueError(f"no cost table entry for pilot model: {provider}/{model}")
    prompt_price, completion_price = REMOTE_MODEL_PRICES[price_key]
    prompt_tokens_upper = 0
    completion_tokens_upper = 0
    for _, example in tasks:
        leaf_calls = min(max(1, len(example.context)), 2**depth)
        context_tokens = sum(block.tokens for block in example.context)
        query_tokens = estimate_tokens(example.query)
        inspection_prompt = context_tokens + leaf_calls * (query_tokens + 512)
        final_prompt = budget + query_tokens + 512
        prompt_tokens_upper += 4 * (inspection_prompt + final_prompt) * len(MATCHED_POLICIES)
        completion_tokens_upper += (
            (leaf_calls + 1) * max_output_tokens * len(MATCHED_POLICIES)
        )
    cost = prompt_tokens_upper * prompt_price + completion_tokens_upper * completion_price
    return {
        "formula_version": 1,
        "logical_policy_upper_bound_usd": round(cost, 6),
        "prompt_tokens_upper_bound": prompt_tokens_upper,
        "completion_tokens_upper_bound": completion_tokens_upper,
        "prompt_safety_factor": 4,
        "tasks": len(tasks),
    }


def combine_policy_summaries(
    parts: Sequence[dict[str, Any]],
    *,
    policy: str,
    dataset: str,
    requested_examples: int,
    max_estimated_cost: float | None,
) -> dict[str, Any]:
    results = [row for part in parts for row in part.get("results", [])]

    def mean(key: str) -> float:
        return round(statistics.fmean(float(row.get(key, 0.0)) for row in results), 3) if results else 0.0

    replay_rows = [
        row["retention_stats"]["inspection_replay"]
        for row in results
        if isinstance(row.get("retention_stats", {}).get("inspection_replay"), dict)
    ]
    total_cost = round(sum(float(row.get("cost_estimate", 0.0)) for row in results), 6)
    completed = len(results) == requested_examples and all(part.get("completed", False) for part in parts)
    return {
        "dataset": dataset,
        "policy": policy,
        "retention_judge": "heuristic",
        "inspection_replay": {
            "mode": "capture_or_replay",
            "captured": sum(int(row.get("captured", 0)) for row in replay_rows),
            "replayed": sum(int(row.get("replayed", 0)) for row in replay_rows),
            "stores": len(replay_rows),
            "store_sha256": sorted(
                str(row["store_sha256"]) for row in replay_rows if row.get("store_sha256")
            ),
        },
        "examples": len(results),
        "requested_examples": requested_examples,
        "accuracy": mean("answer_accuracy"),
        "answer_accuracy": mean("answer_accuracy"),
        "provenance_score": mean("provenance_score"),
        "compactness": mean("compactness"),
        "reward_score": mean("reward_score"),
        "avg_retained_tokens": mean("retained_tokens"),
        "avg_latency_ms": mean("latency_ms"),
        "avg_cost_estimate": round(total_cost / len(results), 6) if results else 0.0,
        "total_cost_estimate": total_cost,
        "initial_cost_estimate": 0.0,
        "final_cumulative_cost_estimate": total_cost,
        "max_estimated_cost": max_estimated_cost,
        "completed": completed,
        "stop_reason": None if completed else "incomplete_task_blocks",
        "last_completed_case": results[-1]["name"] if results else None,
        "results": results,
    }


def candidate_identity(candidate: Mapping[str, Any]) -> tuple[str, str]:
    raw_input = candidate.get("input_item")
    source = raw_input if isinstance(raw_input, Mapping) else candidate
    return str(source.get("raw_pointer", "")), str(source.get("provenance", ""))


def decision_signature(row: Mapping[str, Any]) -> list[tuple[tuple[str, str], ...]]:
    signatures: list[tuple[tuple[str, str], ...]] = []
    for decision in row.get("retention_decisions", []):
        selected = sorted(
            candidate_identity(candidate)
            for candidate in decision.get("candidates", [])
            if isinstance(candidate, Mapping) and candidate.get("selected")
        )
        signatures.append(tuple(selected))
    return signatures


def normalized_row(row: Mapping[str, Any]) -> dict[str, Any]:
    usage = row.get("usage", {})
    return {
        "answer": row.get("answer"),
        "retained_tokens": row.get("retained_tokens"),
        "retained_summaries": row.get("retained_summaries"),
        "retained_provenance": row.get("retained_provenance"),
        "decision_signature": decision_signature(row),
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "calls": usage.get("calls"),
        },
    }


def budget_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    *,
    budget: int,
    expected_tasks: int,
    require_response_model_identifier: bool = False,
    configured_model_alias: str = "",
) -> dict[str, Any]:
    expected_rows = expected_tasks * len(MATCHED_POLICIES)
    pressures = []
    budget_violations = []
    judge_call_violations = []
    final_call_violations = []
    model_identifier_violations = []
    observed_model_identifiers: set[str] = set()
    for row in rows:
        decisions = list(row.get("retention_decisions", []))
        pressures.append(
            max((float(decision.get("before_tokens", 0)) / budget for decision in decisions), default=0.0)
        )
        if int(row.get("retained_tokens", 0)) > budget:
            budget_violations.append(f"{row.get('dataset')}:{row.get('name')}:{row.get('policy')}")
        for decision_index, decision in enumerate(decisions):
            if (
                int(decision.get("budget", -1)) != budget
                or int(decision.get("after_tokens", budget + 1)) > budget
            ):
                budget_violations.append(
                    f"{row.get('dataset')}:{row.get('name')}:{row.get('policy')}:decision-{decision_index}"
                )
        judge_calls = sum(
            int(decision.get("budget_used", {}).get("calls", 0))
            for decision in decisions
        )
        if judge_calls:
            judge_call_violations.append(f"{row.get('dataset')}:{row.get('name')}:{row.get('policy')}")
        final_calls = int(row.get("stage_budgets", {}).get("final_answer", {}).get("calls", 0))
        if final_calls != 1:
            final_call_violations.append(f"{row.get('dataset')}:{row.get('name')}:{row.get('policy')}")
        identifiers = row.get("retention_stats", {}).get("response_model_identifiers", [])
        normalized_identifiers = sorted(str(item) for item in identifiers) if isinstance(identifiers, list) else []
        observed_model_identifiers.update(normalized_identifiers)
        alias_only = (
            len(normalized_identifiers) == 1
            and bool(configured_model_alias)
            and normalized_identifiers[0] == configured_model_alias
        )
        if require_response_model_identifier and (
            len(normalized_identifiers) != 1 or alias_only
        ):
            model_identifier_violations.append(
                f"{row.get('dataset')}:{row.get('name')}:{row.get('policy')}"
            )

    by_task: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        by_task.setdefault((str(row.get("dataset")), str(row.get("name"))), {})[
            str(row.get("policy"))
        ] = row

    matched_ledger_violations = []
    replay_hash_violations = []
    difference_counts = {policy: [0, 0] for policy in SIMPLE_POLICIES}
    for task_key, policies in sorted(by_task.items()):
        if set(policies) != set(MATCHED_POLICIES):
            matched_ledger_violations.append(f"{task_key[0]}:{task_key[1]}:missing-policy")
            continue
        inspect_ledgers = {
            (
                int(row.get("stage_budgets", {}).get("inspect", {}).get("prompt_tokens", 0)),
                int(row.get("stage_budgets", {}).get("inspect", {}).get("completion_tokens", 0)),
                int(row.get("stage_budgets", {}).get("inspect", {}).get("calls", 0)),
            )
            for row in policies.values()
        }
        if len(inspect_ledgers) != 1:
            matched_ledger_violations.append(f"{task_key[0]}:{task_key[1]}:inspect-ledger")
        replay_hashes = {
            str(row.get("retention_stats", {}).get("inspection_replay", {}).get("store_sha256", ""))
            for row in policies.values()
        }
        if len(replay_hashes) != 1 or "" in replay_hashes:
            replay_hash_violations.append(f"{task_key[0]}:{task_key[1]}")

        pairwise_signature = decision_signature(policies["pairwise_tournament"])
        for simple_policy in SIMPLE_POLICIES:
            simple_signature = decision_signature(policies[simple_policy])
            count = min(len(pairwise_signature), len(simple_signature))
            difference_counts[simple_policy][0] += abs(len(pairwise_signature) - len(simple_signature)) + sum(
                pairwise_signature[index] != simple_signature[index] for index in range(count)
            )
            difference_counts[simple_policy][1] += max(len(pairwise_signature), len(simple_signature))

    difference_rates = {
        policy: round(different / total, 6) if total else 0.0
        for policy, (different, total) in difference_counts.items()
    }
    nonempty_rate = (
        sum(int(row.get("retained_items", 0)) > 0 for row in rows) / len(rows)
        if rows
        else 0.0
    )
    median_pressure = statistics.median(pressures) if pressures else 0.0
    structurally_complete = len(rows) == expected_rows and len(by_task) == expected_tasks
    distinct = max(difference_rates.values(), default=0.0) >= 0.20
    eligible = (
        structurally_complete
        and nonempty_rate >= 0.95
        and median_pressure >= 1.5
        and not budget_violations
        and not judge_call_violations
        and not final_call_violations
        and not model_identifier_violations
        and not matched_ledger_violations
        and not replay_hash_violations
        and distinct
    )
    return {
        "budget": budget,
        "eligible": eligible,
        "expected_rows": expected_rows,
        "observed_rows": len(rows),
        "expected_tasks": expected_tasks,
        "observed_tasks": len(by_task),
        "nonempty_rate": round(nonempty_rate, 6),
        "median_max_pre_retention_pressure": round(median_pressure, 6),
        "budget_violations": budget_violations,
        "remote_retention_judge_call_violations": judge_call_violations,
        "final_answer_call_violations": final_call_violations,
        "response_model_identifier_violations": model_identifier_violations,
        "observed_response_model_identifiers": sorted(observed_model_identifiers),
        "configured_model_alias": configured_model_alias or None,
        "matched_inspection_ledger_violations": matched_ledger_violations,
        "replay_hash_violations": replay_hash_violations,
        "pairwise_difference_rates": difference_rates,
        "distinctness_pass": distinct,
    }


def validate_loom_traces(
    loom_root: Path | None,
    trace_paths: Sequence[Path],
    *,
    expected_count: int,
) -> dict[str, Any]:
    if loom_root is None:
        return {
            "available": False,
            "all_valid": False,
            "expected_traces": expected_count,
            "observed_traces": len(trace_paths),
            "traces": [],
        }
    code = (
        "import json,sys\n"
        "from loom.trace import load_trace_events,validate_trace\n"
        "bad=0\n"
        "for raw in sys.argv[1:]:\n"
        " events=load_trace_events(raw); report=validate_trace(events)\n"
        " issues=[str(issue) for issue in report.issues]\n"
        " print(json.dumps({'path':raw,'ok':report.ok,'events':len(events),'issues':issues}))\n"
        " bad += 0 if report.ok else 1\n"
        "raise SystemExit(1 if bad else 0)\n"
    )
    command = ["uv", "run", "--project", str(loom_root), "python", "-c", code, *map(str, trace_paths)]
    result = subprocess.run(command, cwd=loom_root, text=True, capture_output=True, check=False)
    records = [json.loads(line) for line in result.stdout.splitlines() if line.strip().startswith("{")]
    by_path = {str(record["path"]): record for record in records}
    sanitized = []
    for path in trace_paths:
        record = by_path.get(str(path), {"ok": False, "events": 0, "issues": ["validator returned no row"]})
        parts = path.parts
        display_path = (
            Path(*parts[parts.index("reports") :]).as_posix()
            if "reports" in parts
            else path.name
        )
        sanitized.append(
            {
                "path": display_path,
                "ok": bool(record.get("ok")),
                "events": int(record.get("events", 0)),
                "issues": list(record.get("issues", [])),
            }
        )
    stderr = result.stderr.strip()
    return {
        "available": True,
        "all_valid": (
            result.returncode == 0
            and len(trace_paths) == expected_count
            and len(records) == len(trace_paths)
            and all(record["ok"] for record in sanitized)
        ),
        "expected_traces": expected_count,
        "observed_traces": len(trace_paths),
        "traces": sanitized,
        "validator_returncode": result.returncode,
        "validator_stderr_lines": len(stderr.splitlines()) if stderr else 0,
        "validator_stderr_sha256": sha256_bytes(stderr.encode("utf-8")) if stderr else None,
    }


AUDIT_PATTERNS = {
    "mac_user_path": re.compile(r"/Users/[^\s\"']+"),
    "linux_user_path": re.compile(r"/home/[^\s\"']+"),
    "temporary_path": re.compile(r"/tmp/[^\s\"']+"),
    "windows_user_path": re.compile(r"[A-Za-z]:\\\\Users\\\\[^\s\"']+"),
    "openai_style_secret": re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    "bearer_secret": re.compile(r"Authorization[^\n]{0,40}Bearer\s+[A-Za-z0-9._-]{12,}", re.IGNORECASE),
}


def release_audit(root: Path, *, excluded: Sequence[str] = ()) -> dict[str, Any]:
    excluded_set = set(excluded)
    findings = []
    scanned = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded_set:
            continue
        data = path.read_bytes()
        if b"\x00" in data:
            continue
        text = data.decode("utf-8", errors="replace")
        scanned += 1
        for code, pattern in AUDIT_PATTERNS.items():
            match = pattern.search(text)
            if match:
                findings.append({"file": relative, "code": code, "match_sha256": sha256_bytes(match.group(0).encode())})
    return {"ok": not findings, "scanned_text_files": scanned, "findings": findings}


def artifact_inventory(root: Path, *, excluded: Sequence[str]) -> list[dict[str, Any]]:
    excluded_set = set(excluded)
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path.relative_to(root).as_posix() not in excluded_set
    ]


def write_checksums(root: Path, path: Path) -> None:
    files = [item for item in root.rglob("*") if item.is_file() and item != path]
    lines = [f"{sha256_file(item)}  {item.relative_to(root).as_posix()}" for item in sorted(files)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize_release_manifest(output_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    write_json(output_root / "manifest.json", manifest)
    audit = release_audit(output_root, excluded=["release_audit.json", "checksums.txt"])
    manifest["gate_checks"]["release_audit"] = audit["ok"]
    manifest["release_audit"] = audit
    manifest["status"] = "passed" if all(manifest["gate_checks"].values()) else "failed"
    write_json(output_root / "manifest.json", manifest)
    final_audit = release_audit(output_root, excluded=["release_audit.json", "checksums.txt"])
    write_json(output_root / "release_audit.json", final_audit)
    if final_audit["ok"] != audit["ok"]:
        manifest["release_audit"] = final_audit
        manifest["status"] = "failed"
        write_json(output_root / "manifest.json", manifest)
    write_checksums(output_root, output_root / "checksums.txt")
    return manifest


def run_budget(
    *,
    phase: str,
    specs: Sequence[DatasetSpec],
    examples: Mapping[str, Sequence[BenchmarkExample]],
    budget: int,
    budget_root: Path,
    provider: str,
    model: str,
    base_url: str | None,
    learned_model: Path | None,
    seed: int,
    depth: int,
    max_output_tokens: int,
    max_estimated_cost: float | None,
    response_cache_dir: Path | None,
    response_cache_namespace_value: str,
) -> dict[str, Any]:
    parts: dict[str, dict[str, list[dict[str, Any]]]] = {
        spec.label: {policy: [] for policy in MATCHED_POLICIES} for spec in specs
    }
    ordered_tasks = round_robin_tasks(specs, examples)
    execution_order = []
    cumulative_cost = 0.0
    cost_cap_exceeded = False
    for task_index, (spec, example) in enumerate(ordered_tasks):
        if max_estimated_cost is not None and cumulative_cost >= max_estimated_cost:
            break
        block = {"task_index": task_index, "family": spec.label, "name": example.name, "policies": []}
        for policy in MATCHED_POLICIES:
            summary = run_dataset(
                [example],
                policy,
                budget=budget,
                max_depth=depth,
                provider=provider,
                model=model,
                base_url=base_url,
                cache_dir=response_cache_dir,
                cache_preserve_usage=response_cache_dir is not None,
                cache_namespace=response_cache_namespace_value,
                max_output_tokens=max_output_tokens,
                learned_retention_model=learned_model,
                output_dir=budget_root / "reports" / spec.label,
                seed=seed,
                dataset_name=spec.label,
                retention_judge="heuristic",
                inspection_replay_dir=budget_root / "inspection_replay",
            )
            if len(summary.get("results", [])) != 1:
                raise RuntimeError(f"incomplete policy result for {spec.label}:{example.name}:{policy}")
            row = summary["results"][0]
            cumulative_cost = round(cumulative_cost + float(row.get("cost_estimate", 0.0)), 6)
            row["cumulative_cost_estimate"] = cumulative_cost
            parts[spec.label][policy].append(summary)
            block["policies"].append(policy)
        block["cumulative_cost_estimate"] = cumulative_cost
        execution_order.append(block)
        if max_estimated_cost is not None and cumulative_cost > max_estimated_cost:
            cost_cap_exceeded = True
            break

    all_rows = []
    report_records = []
    for spec in specs:
        summaries = [
            combine_policy_summaries(
                parts[spec.label][policy],
                policy=policy,
                dataset=spec.label,
                requested_examples=len(examples[spec.label]),
                max_estimated_cost=max_estimated_cost,
            )
            for policy in MATCHED_POLICIES
        ]
        all_rows.extend(row for summary in summaries for row in summary["results"])
        report_root = budget_root / "reports" / spec.label
        curves = curves_from_summaries(spec.label, summaries, budget=budget, depth=depth, seed=seed)
        write_report_bundle(
            report_root,
            dataset_name=spec.label,
            summaries=summaries,
            curves=curves,
            command=(
                "python scripts/run_matched_retention.py "
                f"--phase {phase} --dataset-spec {spec.label}:{spec.dataset}"
                f"{':<embedded-dataset-path>' if spec.path else ''} "
                f"--budgets {budget} --output-dir <bundle>"
            ),
        )
        report_records.append(
            {
                "family": spec.label,
                "path": (Path("reports") / spec.label).as_posix(),
                "rows": sum(len(summary["results"]) for summary in summaries),
            }
        )

    diagnostics = budget_diagnostics(
        all_rows,
        budget=budget,
        expected_tasks=len(ordered_tasks),
        require_response_model_identifier=phase != "offline" and provider != "heuristic",
        configured_model_alias=model,
    )
    diagnostics["cost_cap_exceeded"] = cost_cap_exceeded
    diagnostics["total_estimated_cost"] = cumulative_cost
    diagnostics["eligible"] = diagnostics["eligible"] and not cost_cap_exceeded
    return {
        "budget": budget,
        "root": budget_root,
        "rows": all_rows,
        "diagnostics": diagnostics,
        "execution_order": execution_order,
        "reports": report_records,
    }


def determinism_check(
    budget_result: Mapping[str, Any],
    *,
    first_spec: DatasetSpec,
    first_example: BenchmarkExample,
    provider: str,
    model: str,
    base_url: str | None,
    learned_model: Path | None,
    seed: int,
    depth: int,
    max_output_tokens: int,
) -> dict[str, Any]:
    budget = int(budget_result["budget"])
    original_rows = {
        str(row["policy"]): row
        for row in budget_result["rows"]
        if row["dataset"] == first_spec.label and row["name"] == first_example.name
    }
    mismatches = []
    for policy in MATCHED_POLICIES:
        rerun = run_dataset(
            [first_example],
            policy,
            budget=budget,
            max_depth=depth,
            provider=provider,
            model=model,
            base_url=base_url,
            cache_dir=None,
            max_output_tokens=max_output_tokens,
            learned_retention_model=learned_model,
            output_dir=None,
            seed=seed,
            dataset_name=first_spec.label,
            retention_judge="heuristic",
            inspection_replay_dir=Path(budget_result["root"]) / "inspection_replay",
            inspection_replay_mode="replay_only",
        )
        candidate = rerun["results"][0]
        if normalized_row(original_rows[policy]) != normalized_row(candidate):
            mismatches.append(policy)
    return {"ok": not mismatches, "task": f"{first_spec.label}:{first_example.name}", "mismatches": mismatches}


def copy_dataset_sources(output_root: Path, specs: Sequence[DatasetSpec]) -> list[dict[str, Any]]:
    def portable_path(value: str) -> str | None:
        posix_path = PurePosixPath(value)
        windows_path = PureWindowsPath(value)
        if posix_path.is_absolute():
            return f"<portable-source>/{posix_path.name or 'source'}"
        if windows_path.is_absolute():
            return f"<portable-source>/{windows_path.name or 'source'}"
        return None

    def portable_value(value: Any, key: str = "") -> Any:
        if isinstance(value, dict):
            return {name: portable_value(item, str(name)) for name, item in value.items()}
        if isinstance(value, list):
            return [portable_value(item, key) for item in value]
        if isinstance(value, str) and key in {"path", "source_path", "repo_root", "dataset_path"}:
            scrubbed = portable_path(value)
            if scrubbed is not None:
                return scrubbed
        return value

    records = []
    for spec in specs:
        if spec.path is None:
            records.append({"label": spec.label, "dataset": spec.dataset, "embedded": False})
            continue
        destination = output_root / "datasets" / f"{spec.label}.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        portable_rows = []
        for line_number, raw_line in enumerate(spec.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"external dataset line {line_number} is not valid JSON: {spec.label}"
                ) from exc
            portable_rows.append(canonical_json(portable_value(row)))
        destination.write_text("\n".join(portable_rows) + "\n", encoding="utf-8")
        records.append(
            {
                "label": spec.label,
                "dataset": spec.dataset,
                "embedded": True,
                "path": destination.relative_to(output_root).as_posix(),
                "source_sha256": sha256_file(spec.path),
                "sha256": sha256_file(destination),
                "normalization": "canonical_json_and_portable_local_path_metadata_v1",
            }
        )
    return records


def reproduction_argv_template(
    args: argparse.Namespace,
    specs: Sequence[DatasetSpec],
    budgets: Sequence[int],
    dataset_hashes: Mapping[str, str],
) -> list[str]:
    argv = ["uv", "run", "python", "scripts/run_matched_retention.py", "--phase", args.phase]
    if args.preflight_only:
        argv.append("--preflight-only")
    for spec in specs:
        value = f"{spec.label}:{spec.dataset}"
        if spec.path is not None:
            value += f":<source-dataset>/{spec.label}.jsonl"
        argv.extend(["--dataset-spec", value])
    for label, digest in sorted(dataset_hashes.items()):
        argv.extend(["--expected-dataset-sha256", f"{label}={digest}"])
    argv.extend(
        [
            "--limit",
            str(args.limit),
            "--start-index",
            str(args.start_index),
            "--seed",
            str(args.seed),
            "--budgets",
            ",".join(map(str, budgets)),
            "--depth",
            str(args.depth),
            "--max-output-tokens",
            str(args.max_output_tokens),
            "--provider",
            args.provider,
            "--model",
            args.model,
        ]
    )
    if args.base_url:
        endpoint_hash = sha256_bytes(args.base_url.encode("utf-8"))[:16]
        argv.extend(["--base-url", f"<endpoint-sha256-{endpoint_hash}>"])
    if args.learned_retention_model:
        argv.extend(
            [
                "--learned-retention-model",
                "<bundle>/artifacts/learned_retention_training/learned_retention_model.json",
            ]
        )
    if args.learned_retention_training_manifest:
        argv.extend(
            [
                "--learned-retention-training-manifest",
                "<bundle>/artifacts/learned_retention_training/manifest.json",
            ]
        )
    if args.offline_manifest:
        argv.extend(["--offline-manifest", "<passed-offline-bundle>/manifest.json"])
    if args.expected_offline_sha256:
        argv.extend(["--expected-offline-sha256", args.expected_offline_sha256])
    if args.preflight_manifest:
        argv.extend(["--preflight-manifest", f"<passed-{args.phase}-preflight-bundle>/manifest.json"])
    if args.expected_preflight_sha256:
        argv.extend(["--expected-preflight-sha256", args.expected_preflight_sha256])
    if args.cache_dir:
        argv.extend(["--cache-dir", "<bound-response-cache-dir>"])
    if args.max_estimated_cost is not None:
        argv.extend(["--max-estimated-cost", str(args.max_estimated_cost)])
    if any(spec.dataset == "verifiers_smoke" for spec in specs):
        argv.extend(["--repo-root", "<verifiers-checkout>"])
    argv.extend(
        [
            "--loom-root",
            "<loom-checkout>",
            "--expected-nanorlm-commit",
            args.expected_nanorlm_commit or "<full-nanorlm-commit>",
            "--expected-loom-commit",
            args.expected_loom_commit or "<full-loom-commit>",
            "--output-dir",
            "<empty-output-dir>",
        ]
    )
    return argv


def copy_learned_training_bundle(
    source_manifest: Path,
    supplied_model: Path,
    output_root: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "trained":
        raise ValueError("learned-retention training manifest must have status=trained")
    training = payload.get("training")
    if not isinstance(training, dict):
        raise ValueError("learned-retention training manifest is missing training metadata")
    if training.get("source") != "offline_trace_training":
        raise ValueError("learned-retention model must come from offline trace training")
    if training.get("training_source") != "traces" or training.get("objective") != "pairwise":
        raise ValueError("learned-retention model must use pairwise training over decision traces")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not isinstance(artifacts.get("model"), dict):
        raise ValueError("learned-retention training manifest is missing artifact hashes")
    repository = payload.get("repository")
    if (
        not isinstance(repository, dict)
        or FULL_GIT_SHA_RE.fullmatch(str(repository.get("commit", ""))) is None
        or repository.get("clean") is not True
    ):
        raise ValueError("learned-retention training manifest must bind a clean full git commit")

    source_root = source_manifest.parent.resolve()
    destination_root = output_root / "artifacts" / "learned_retention_training"
    copied = []
    model_destination: Path | None = None
    supplied_hash = sha256_file(supplied_model)
    for name, record in artifacts.items():
        if record is None:
            continue
        if not isinstance(record, dict) or record.get("external"):
            raise ValueError(f"learned-retention artifact must be bundle-local: {name}")
        relative = Path(str(record.get("path", "")))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"learned-retention artifact has unsafe path: {name}")
        source = (source_root / relative).resolve()
        try:
            source.relative_to(source_root)
        except ValueError as exc:
            raise ValueError(f"learned-retention artifact escapes manifest root: {name}") from exc
        if not source.is_file():
            raise ValueError(f"learned-retention artifact is missing: {name}")
        actual_hash = sha256_file(source)
        if actual_hash != record.get("sha256"):
            raise ValueError(f"learned-retention artifact hash mismatch: {name}")
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied.append({"name": name, "path": relative.as_posix(), "sha256": actual_hash})
        if name == "model":
            model_destination = destination
            if actual_hash != supplied_hash:
                raise ValueError("supplied learned-retention model does not match its training manifest")
    if model_destination is None:
        raise ValueError("learned-retention training manifest did not resolve a model artifact")
    manifest_destination = destination_root / "manifest.json"
    shutil.copyfile(source_manifest, manifest_destination)
    return model_destination, manifest_destination, {
        "ok": True,
        "manifest_sha256": sha256_file(manifest_destination),
        "training_repository_commit": repository["commit"],
        "artifacts": copied,
    }


def copy_and_validate_offline_manifest(
    source_manifest: Path,
    output_root: Path,
    *,
    expected_manifest_sha256: str,
    expected_budget: int,
    expected_nanorlm_commit: str,
    expected_loom_commit: str,
    learned_model_sha256: str,
    learned_training_manifest_sha256: str,
) -> dict[str, Any]:
    if SHA256_RE.fullmatch(expected_manifest_sha256) is None:
        raise ValueError("expected offline manifest SHA-256 must be 64 lowercase hex characters")
    source_hash = sha256_file(source_manifest)
    if source_hash != expected_manifest_sha256:
        raise ValueError("offline manifest SHA-256 mismatch")
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("phase") != "offline" or payload.get("status") != "passed":
        raise ValueError("prior offline manifest must be a passed offline bundle")
    if int(payload.get("selected_budget", -1)) != expected_budget:
        raise ValueError("prior offline manifest selected a different memory budget")
    gate_checks = payload.get("gate_checks")
    if not isinstance(gate_checks, dict) or not gate_checks or not all(gate_checks.values()):
        raise ValueError("prior offline manifest has an incomplete gate")
    release_audit = payload.get("release_audit")
    if not isinstance(release_audit, dict) or release_audit.get("ok") is not True:
        raise ValueError("prior offline manifest did not pass its release audit")
    repositories = payload.get("repositories")
    if not isinstance(repositories, dict):
        raise ValueError("prior offline manifest is missing repository bindings")
    offline_loom = repositories.get("loom")
    offline_nanorlm = repositories.get("nanorlm")
    if not isinstance(offline_loom, dict) or offline_loom.get("commit") != expected_loom_commit:
        raise ValueError("prior offline manifest used a different LOOM commit")
    if (
        not isinstance(offline_nanorlm, dict)
        or offline_nanorlm.get("commit") != expected_nanorlm_commit
    ):
        raise ValueError("prior offline manifest used a different nanoRLM commit")
    configuration = payload.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("prior offline manifest is missing configuration bindings")
    if configuration.get("learned_model_sha256") != learned_model_sha256:
        raise ValueError("prior offline manifest used a different learned model")
    training_record = configuration.get("learned_training_manifest")
    if (
        not isinstance(training_record, dict)
        or training_record.get("sha256") != learned_training_manifest_sha256
    ):
        raise ValueError("prior offline manifest used a different learned training manifest")
    training_validation = training_record.get("validation")
    if (
        not isinstance(training_validation, dict)
        or training_validation.get("training_repository_commit") != offline_nanorlm.get("commit")
    ):
        raise ValueError("prior offline manifest did not bind training code to its runtime")
    task_manifest = payload.get("task_manifest")
    if not isinstance(task_manifest, dict) or SHA256_RE.fullmatch(
        str(task_manifest.get("sha256", ""))
    ) is None:
        raise ValueError("prior offline manifest is missing its task-manifest hash")
    checksum_name = str(payload.get("checksums", ""))
    checksum_relative = Path(checksum_name)
    if (
        not checksum_relative.parts
        or checksum_relative.is_absolute()
        or ".." in checksum_relative.parts
    ):
        raise ValueError("prior offline manifest has an unsafe checksum index path")
    checksum_validation = verify_checksum_index(
        source_manifest.parent.resolve(),
        (source_manifest.parent / checksum_relative).resolve(),
        artifact_inventory=payload.get("artifact_inventory"),
    )

    destination = output_root / "prior_evidence" / "offline_manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_manifest, destination)
    return {
        "ok": True,
        "path": destination.relative_to(output_root).as_posix(),
        "sha256": sha256_file(destination),
        "offline_nanorlm_commit": offline_nanorlm["commit"],
        "offline_loom_commit": offline_loom["commit"],
        "offline_task_manifest_sha256": task_manifest["sha256"],
        "training_repository_commit": training_validation["training_repository_commit"],
        "selected_budget": expected_budget,
        "checksum_index": checksum_validation,
    }


def verify_checksum_index(
    root: Path,
    index_path: Path,
    *,
    artifact_inventory: Any,
) -> dict[str, Any]:
    resolved_root = root.resolve()
    resolved_index = index_path.resolve()
    try:
        resolved_index.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("checksum index escapes its bundle") from exc
    if not resolved_index.is_file():
        raise ValueError("release bundle is missing its checksum index")
    if not isinstance(artifact_inventory, list):
        raise ValueError("release manifest is missing its artifact inventory")

    inventory_hashes: dict[str, str] = {}
    reserved_paths = {
        resolved_index.relative_to(resolved_root).as_posix(),
        "manifest.json",
        "release_audit.json",
    }
    for position, record in enumerate(artifact_inventory, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"invalid artifact inventory entry at position {position}")
        relative_value = str(record.get("path", ""))
        digest = str(record.get("sha256", ""))
        relative = Path(relative_value)
        if (
            not relative.parts
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() in reserved_paths
            or SHA256_RE.fullmatch(digest) is None
        ):
            raise ValueError(f"invalid artifact inventory entry at position {position}")
        relative_name = relative.as_posix()
        if relative_name in inventory_hashes:
            raise ValueError(f"duplicate artifact inventory path: {relative_name}")
        inventory_hashes[relative_name] = digest

    verified: dict[str, str] = {}
    for line_number, line in enumerate(resolved_index.read_text(encoding="utf-8").splitlines(), start=1):
        digest, separator, relative_value = line.partition("  ")
        if not separator or SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"invalid checksum entry at line {line_number}")
        relative = Path(relative_value)
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe checksum path at line {line_number}")
        artifact = (resolved_root / relative).resolve()
        try:
            artifact.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(f"checksum path escapes bundle at line {line_number}") from exc
        if not artifact.is_file() or sha256_file(artifact) != digest:
            raise ValueError(f"checksum mismatch: {relative.as_posix()}")
        relative_name = relative.as_posix()
        if relative_name in verified:
            raise ValueError(f"duplicate checksum path: {relative_name}")
        verified[relative_name] = digest

    actual_paths = {
        item.relative_to(resolved_root).as_posix()
        for item in resolved_root.rglob("*")
        if item.is_file() and item.resolve() != resolved_index
    }
    indexed_paths = set(verified)
    if indexed_paths != actual_paths:
        missing = sorted(actual_paths - indexed_paths)
        unexpected = sorted(indexed_paths - actual_paths)
        raise ValueError(
            "checksum index does not cover the release bundle exactly: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    expected_paths = set(inventory_hashes) | {"manifest.json", "release_audit.json"}
    if indexed_paths != expected_paths:
        missing = sorted(expected_paths - indexed_paths)
        unexpected = sorted(indexed_paths - expected_paths)
        raise ValueError(
            "checksum index does not match the manifest artifact inventory: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    for relative_name, expected_digest in inventory_hashes.items():
        if verified[relative_name] != expected_digest:
            raise ValueError(f"artifact inventory hash mismatch: {relative_name}")
    return {
        "ok": True,
        "path": resolved_index.relative_to(resolved_root).as_posix(),
        "sha256": sha256_file(resolved_index),
        "verified_files": len(verified),
        "inventory_files": len(inventory_hashes),
        "complete_coverage": True,
    }


def copy_and_validate_preflight_manifest(
    source_manifest: Path,
    output_root: Path,
    *,
    expected_manifest_sha256: str,
    phase: str,
    expected_nanorlm_commit: str,
    expected_loom_commit: str,
    expected_budget: int,
    expected_task_manifest_sha256: str,
    expected_configuration: Mapping[str, Any],
    expected_datasets: Sequence[Mapping[str, Any]],
    expected_offline_manifest_sha256: str,
) -> dict[str, Any]:
    if SHA256_RE.fullmatch(expected_manifest_sha256) is None:
        raise ValueError("expected preflight manifest SHA-256 must be 64 lowercase hex characters")
    source_hash = sha256_file(source_manifest)
    if source_hash != expected_manifest_sha256:
        raise ValueError("preflight manifest SHA-256 mismatch")
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("phase") != f"{phase}_preflight"
        or payload.get("requested_phase") != phase
        or payload.get("preflight_only") is not True
        or payload.get("network_calls_issued") != 0
        or payload.get("status") != "passed"
    ):
        raise ValueError(f"{phase} requires its passed zero-network preflight manifest")
    gate_checks = payload.get("gate_checks")
    if not isinstance(gate_checks, dict) or not gate_checks or not all(gate_checks.values()):
        raise ValueError("preflight manifest has an incomplete gate")
    release_audit = payload.get("release_audit")
    if not isinstance(release_audit, dict) or release_audit.get("ok") is not True:
        raise ValueError("preflight manifest did not pass its release audit")
    repositories = payload.get("repositories")
    if not isinstance(repositories, dict):
        raise ValueError("preflight manifest is missing repository bindings")
    nanorlm = repositories.get("nanorlm")
    loom = repositories.get("loom")
    if not isinstance(nanorlm, dict) or nanorlm.get("commit") != expected_nanorlm_commit:
        raise ValueError("preflight manifest used a different nanoRLM commit")
    if not isinstance(loom, dict) or loom.get("commit") != expected_loom_commit:
        raise ValueError("preflight manifest used a different LOOM commit")
    if int(payload.get("selected_budget", -1)) != expected_budget:
        raise ValueError("preflight manifest selected a different memory budget")
    task_manifest = payload.get("task_manifest")
    if (
        not isinstance(task_manifest, dict)
        or task_manifest.get("sha256") != expected_task_manifest_sha256
    ):
        raise ValueError("preflight manifest used a different task manifest")
    configuration = payload.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("preflight manifest is missing configuration bindings")
    for key, expected_value in expected_configuration.items():
        if configuration.get(key) != expected_value:
            raise ValueError(f"preflight manifest configuration mismatch: {key}")
    if payload.get("datasets") != list(expected_datasets):
        raise ValueError("preflight manifest used different dataset artifacts")
    prior_offline = payload.get("prior_offline_evidence")
    if (
        not isinstance(prior_offline, dict)
        or prior_offline.get("ok") is not True
        or prior_offline.get("sha256") != expected_offline_manifest_sha256
    ):
        raise ValueError("preflight manifest used different offline evidence")
    checksum_name = str(payload.get("checksums", ""))
    checksum_relative = Path(checksum_name)
    if (
        not checksum_relative.parts
        or checksum_relative.is_absolute()
        or ".." in checksum_relative.parts
    ):
        raise ValueError("preflight manifest has an unsafe checksum index path")
    checksum_validation = verify_checksum_index(
        source_manifest.parent.resolve(),
        (source_manifest.parent / checksum_relative).resolve(),
        artifact_inventory=payload.get("artifact_inventory"),
    )

    destination = output_root / "prior_evidence" / f"{phase}_preflight_manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_manifest, destination)
    return {
        "ok": True,
        "path": destination.relative_to(output_root).as_posix(),
        "sha256": sha256_file(destination),
        "phase": payload["phase"],
        "nanorlm_commit": nanorlm["commit"],
        "loom_commit": loom["commit"],
        "task_manifest_sha256": task_manifest["sha256"],
        "offline_manifest_sha256": prior_offline["sha256"],
        "checksum_index": checksum_validation,
        "network_calls_issued": 0,
    }


def validate_phase_configuration(
    args: argparse.Namespace,
    specs: Sequence[DatasetSpec],
    budgets: Sequence[int],
) -> None:
    if args.phase == "offline":
        if args.cache_dir:
            raise ValueError("offline phase does not accept a response cache")
        return
    expected_limit = 8 if args.phase == "pilot" else 25
    expected_cap = 5.0 if args.phase == "pilot" else 20.0
    if budgets != [96]:
        raise ValueError(f"{args.phase} must use the frozen 96-token budget")
    if args.provider != "openai_compatible" or args.model != "gpt-5.4-mini":
        raise ValueError(f"{args.phase} must use openai_compatible/gpt-5.4-mini")
    if args.base_url.rstrip("/") not in {"", "https://api.openai.com/v1"}:
        raise ValueError(f"{args.phase} must use the frozen OpenAI API endpoint")
    if args.depth != 3 or args.max_output_tokens != 512 or args.seed != 0:
        raise ValueError(f"{args.phase} must use depth=3, output cap=512, and seed=0")
    if args.start_index != 0 or args.limit != expected_limit:
        raise ValueError(f"{args.phase} must use start-index=0 and limit={expected_limit}")
    if (
        [spec.label for spec in specs] != ["ruler", "babilong"]
        or any(spec.dataset != "external_jsonl" for spec in specs)
    ):
        raise ValueError(
            f"{args.phase} requires ordered ruler and babilong external_jsonl families"
        )
    if args.max_estimated_cost != expected_cap:
        raise ValueError(f"{args.phase} must use the frozen USD {expected_cap:g} cost cap")
    if not args.cache_dir:
        raise ValueError(f"{args.phase} requires a bound persistent response cache")


def execute(args: argparse.Namespace) -> dict[str, Any]:
    specs = [parse_dataset_spec(value) for value in args.dataset_spec]
    if len({spec.label for spec in specs}) != len(specs):
        raise ValueError("dataset labels must be unique")
    expected_dataset_hashes = parse_expected_dataset_hashes(args.expected_dataset_sha256)
    observed_dataset_hashes = validate_dataset_hashes(
        specs,
        expected_dataset_hashes,
        required=args.phase != "offline",
    )
    budgets = parse_csv_ints(args.budgets)
    validate_phase_configuration(args, specs, budgets)
    if args.preflight_only and args.phase == "offline":
        raise ValueError("preflight-only is for pilot or confirmation phases")
    if args.phase != "offline" and len(budgets) != 1:
        raise ValueError("pilot and confirmation phases require exactly one frozen budget")
    if args.phase == "offline" and args.provider != "heuristic":
        raise ValueError("offline phase must use the heuristic provider")
    if args.phase != "offline" and (
        not args.learned_retention_model or not args.learned_retention_training_manifest
    ):
        raise ValueError(
            "pilot and confirmation require a frozen learned-retention model and training manifest"
        )
    if args.phase != "offline" and (
        not args.offline_manifest or not args.expected_offline_sha256
    ):
        raise ValueError(
            "pilot and confirmation require the passed offline manifest and its SHA-256"
        )
    if args.phase != "offline" and args.preflight_only and (
        args.preflight_manifest or args.expected_preflight_sha256
    ):
        raise ValueError("preflight-only creates evidence and does not accept prior preflight evidence")
    if args.phase != "offline" and not args.preflight_only and (
        not args.preflight_manifest or not args.expected_preflight_sha256
    ):
        raise ValueError(
            "pilot and confirmation execution require a passed preflight manifest and its SHA-256"
        )
    if args.learned_retention_training_manifest and not args.learned_retention_model:
        raise ValueError("a learned-retention training manifest requires its model")

    output_root = Path(args.output_dir).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"output directory must be empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    response_cache_root: Path | None = None
    if args.cache_dir:
        raw_cache_root = Path(args.cache_dir).expanduser()
        if raw_cache_root.is_symlink():
            raise ValueError("response-cache directory must not be a symlink")
        response_cache_root = raw_cache_root.resolve()
        try:
            response_cache_root.relative_to(output_root)
        except ValueError:
            pass
        else:
            raise ValueError("response-cache directory must be outside the output bundle")
        try:
            output_root.relative_to(response_cache_root)
        except ValueError:
            pass
        else:
            raise ValueError("output bundle must not be inside the response-cache directory")
    loom_root = Path(args.loom_root).expanduser().resolve() if args.loom_root else None
    source_learned_model = (
        Path(args.learned_retention_model).expanduser().resolve()
        if args.learned_retention_model
        else None
    )
    source_training_manifest = (
        Path(args.learned_retention_training_manifest).expanduser().resolve()
        if args.learned_retention_training_manifest
        else None
    )
    source_offline_manifest = (
        Path(args.offline_manifest).expanduser().resolve() if args.offline_manifest else None
    )
    source_preflight_manifest = (
        Path(args.preflight_manifest).expanduser().resolve() if args.preflight_manifest else None
    )
    learned_model: Path | None = None
    training_manifest: Path | None = None
    training_bundle_validation: dict[str, Any] | None = None
    if source_learned_model is not None:
        if not source_learned_model.is_file():
            raise ValueError(f"learned-retention model does not exist: {source_learned_model}")
        if source_training_manifest is not None:
            if not source_training_manifest.is_file():
                raise ValueError(
                    f"learned-retention training manifest does not exist: {source_training_manifest}"
                )
            learned_model, training_manifest, training_bundle_validation = copy_learned_training_bundle(
                source_training_manifest,
                source_learned_model,
                output_root,
            )
        else:
            learned_model = output_root / "artifacts" / "learned_retention_model.json"
            learned_model.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_learned_model, learned_model)
    examples = load_spec_examples(
        specs,
        limit=args.limit,
        start_index=args.start_index,
        seed=args.seed,
        repo_root=args.repo_root,
    )
    ordered_tasks = round_robin_tasks(specs, examples)
    if not ordered_tasks:
        raise ValueError("dataset specs produced no tasks")
    conversion_audit_result = conversion_audit(ordered_tasks)
    if args.phase != "offline" and not conversion_audit_result["ok"]:
        raise ValueError("external task conversion audit failed before execution")
    hosted_family_audit_result = (
        hosted_family_audit(ordered_tasks)
        if args.phase != "offline"
        else {"ok": True, "not_required": "offline development phase"}
    )
    if args.phase != "offline" and not hosted_family_audit_result["ok"]:
        raise ValueError("hosted task family metadata does not match RULER and BABILong")
    if args.phase != "offline":
        expected_per_family = 8 if args.phase == "pilot" else 25
        if any(len(examples[spec.label]) != expected_per_family for spec in specs):
            raise ValueError(
                f"{args.phase} requires exactly {expected_per_family} tasks per family"
            )
    cost_preflight = conservative_cost_upper_bound(
        ordered_tasks,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url or None,
        budget=budgets[0],
        depth=args.depth,
        max_output_tokens=args.max_output_tokens,
    )
    if (
        args.max_estimated_cost is not None
        and cost_preflight["logical_policy_upper_bound_usd"] > args.max_estimated_cost
    ):
        raise ValueError(
            "conservative task-block cost reservation exceeds the frozen cost cap before execution"
        )

    preflight_configuration = {
        "provider": args.provider,
        "model": args.model,
        "base_url_sha256": sha256_bytes((args.base_url or "").encode("utf-8")),
        "budget": budgets[0],
        "max_depth": args.depth,
        "max_output_tokens": args.max_output_tokens,
        "max_estimated_cost": args.max_estimated_cost,
        "cost_preflight": cost_preflight,
        "seed": args.seed,
        "expected_dataset_sha256": expected_dataset_hashes,
        "observed_dataset_sha256": observed_dataset_hashes,
        "learned_model_sha256": sha256_file(learned_model) if learned_model else None,
        "learned_training_manifest_sha256": (
            sha256_file(training_manifest) if training_manifest else None
        ),
        "response_cache": {
            "required": args.phase != "offline",
            "mode": "exact_binding_read_write_v1" if args.phase != "offline" else "disabled",
            "preserve_logical_usage_on_hit": args.phase != "offline",
            "publish_snapshot": args.phase != "offline",
        },
    }

    code_snapshot = git_snapshot(ROOT)
    loom_snapshot = git_snapshot(loom_root) if loom_root else None
    prior_offline_evidence: dict[str, Any] | None = None
    if source_offline_manifest is not None:
        if not source_offline_manifest.is_file():
            raise ValueError(f"prior offline manifest does not exist: {source_offline_manifest}")
        if learned_model is None or training_manifest is None or loom_snapshot is None:
            raise ValueError("prior offline binding requires learned and LOOM artifacts")
        prior_offline_evidence = copy_and_validate_offline_manifest(
            source_offline_manifest,
            output_root,
            expected_manifest_sha256=args.expected_offline_sha256,
            expected_budget=budgets[0],
            expected_nanorlm_commit=str(code_snapshot["commit"]),
            expected_loom_commit=str(loom_snapshot["commit"]),
            learned_model_sha256=sha256_file(learned_model),
            learned_training_manifest_sha256=sha256_file(training_manifest),
        )
    dataset_records = copy_dataset_sources(output_root, specs)
    task_records = [
        example_record(spec, index, example)
        for index, (spec, example) in enumerate(ordered_tasks)
    ]
    task_manifest = {
        "schema_version": SCHEMA_VERSION,
        "seed": args.seed,
        "start_index": args.start_index,
        "limit_per_family": args.limit,
        "ordering": "round_robin_family_then_source_index",
        "answer_evaluator": {
            "name": "normalized_required_substring_all",
            "implementation": "bench.score_answer",
            "case_sensitive": False,
        },
        "conversion_audit": conversion_audit_result,
        "hosted_family_audit": hosted_family_audit_result,
        "tasks": task_records,
    }
    write_json(output_root / "task_manifest.json", task_manifest)

    prior_preflight_evidence: dict[str, Any] | None = None
    response_cache_binding_payload: dict[str, Any] | None = None
    response_cache_initial_state: dict[str, Any] | None = None
    if source_preflight_manifest is not None:
        if not source_preflight_manifest.is_file():
            raise ValueError(f"prior preflight manifest does not exist: {source_preflight_manifest}")
        if prior_offline_evidence is None or loom_snapshot is None:
            raise ValueError("preflight binding requires passed offline and LOOM evidence")
        prior_preflight_evidence = copy_and_validate_preflight_manifest(
            source_preflight_manifest,
            output_root,
            expected_manifest_sha256=args.expected_preflight_sha256,
            phase=args.phase,
            expected_nanorlm_commit=str(code_snapshot["commit"]),
            expected_loom_commit=str(loom_snapshot["commit"]),
            expected_budget=budgets[0],
            expected_task_manifest_sha256=sha256_file(output_root / "task_manifest.json"),
            expected_configuration=preflight_configuration,
            expected_datasets=dataset_records,
            expected_offline_manifest_sha256=str(prior_offline_evidence["sha256"]),
        )
        pre_execution_bindings = {
            "nanorlm": commit_binding(code_snapshot, args.expected_nanorlm_commit),
            "loom": commit_binding(loom_snapshot, args.expected_loom_commit),
        }
        if (
            not code_snapshot["is_repository"]
            or not code_snapshot["clean"]
            or not loom_snapshot["is_repository"]
            or not loom_snapshot["clean"]
            or not all(binding["ok"] for binding in pre_execution_bindings.values())
        ):
            raise ValueError(
                "hosted execution requires clean repositories and exact commit bindings before network access"
            )
        if response_cache_root is None:
            raise ValueError("hosted execution requires a persistent response cache")
        response_cache_binding_payload = {
            "schema_version": "nanorlm-response-cache-binding-v1",
            "phase": args.phase,
            "nanorlm_commit": str(code_snapshot["commit"]),
            "loom_commit": str(loom_snapshot["commit"]),
            "task_manifest_sha256": sha256_file(output_root / "task_manifest.json"),
            "offline_manifest_sha256": str(prior_offline_evidence["sha256"]),
            "preflight_manifest_sha256": str(prior_preflight_evidence["sha256"]),
            "configuration": preflight_configuration,
        }
        response_cache_initial_state = prepare_response_cache(
            response_cache_root,
            response_cache_binding_payload,
        )

    if args.preflight_only:
        commit_bindings = {
            "nanorlm": commit_binding(code_snapshot, args.expected_nanorlm_commit),
            "loom": commit_binding(loom_snapshot, args.expected_loom_commit),
        }
        frozen_training_bundle = bool(
            learned_model is not None
            and training_manifest is not None
            and training_bundle_validation
            and training_bundle_validation["ok"]
        )
        gate_checks = {
            "nanorlm_repository": bool(code_snapshot["is_repository"]),
            "nanorlm_clean": bool(code_snapshot["clean"]),
            "loom_repository": bool(loom_snapshot and loom_snapshot["is_repository"]),
            "loom_clean": bool(loom_snapshot and loom_snapshot["clean"]),
            "commit_bindings": all(binding["ok"] for binding in commit_bindings.values()),
            "frozen_learned_training_bundle": frozen_training_bundle,
            "prior_offline_evidence": bool(
                prior_offline_evidence and prior_offline_evidence["ok"]
            ),
            "dataset_hashes": all(
                expected_dataset_hashes.get(label) == digest
                for label, digest in observed_dataset_hashes.items()
            ),
            "task_count": len(task_records) == (16 if args.phase == "pilot" else 50),
            "conversion_audit": conversion_audit_result["ok"],
            "hosted_family_audit": hosted_family_audit_result["ok"],
            "cost_cap_reservation": (
                args.max_estimated_cost is not None
                and cost_preflight["logical_policy_upper_bound_usd"] <= args.max_estimated_cost
            ),
            "bound_response_cache_configuration": bool(args.cache_dir),
            "zero_network_calls": True,
        }
        inventory_exclusions = ["manifest.json", "release_audit.json", "checksums.txt"]
        preflight_manifest = {
            "schema_version": SCHEMA_VERSION,
            "phase": f"{args.phase}_preflight",
            "requested_phase": args.phase,
            "preflight_only": True,
            "network_calls_issued": 0,
            "status": "pending_release_audit",
            "repositories": {"nanorlm": code_snapshot, "loom": loom_snapshot},
            "commit_bindings": commit_bindings,
            "prior_offline_evidence": prior_offline_evidence,
            "environment": {
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "uv_lock_sha256": sha256_file(ROOT / "uv.lock"),
            },
            "reproduction": {
                "argv_template": reproduction_argv_template(
                    args,
                    specs,
                    budgets,
                    {
                        str(record["label"]): str(record["source_sha256"])
                        for record in dataset_records
                        if record.get("embedded")
                    },
                ),
                "note": (
                    "Source-dataset placeholders require the exact original exports matching "
                    "source_sha256; bundle placeholders refer to copied frozen artifacts."
                ),
            },
            "configuration": preflight_configuration,
            "datasets": dataset_records,
            "task_manifest": {
                "path": "task_manifest.json",
                "sha256": sha256_file(output_root / "task_manifest.json"),
                "tasks": len(task_records),
            },
            "selected_budget": budgets[0],
            "gate_checks": gate_checks,
            "artifact_inventory": artifact_inventory(
                output_root,
                excluded=inventory_exclusions,
            ),
            "release_audit": {"pending": True},
            "checksums": "checksums.txt",
        }
        return finalize_release_manifest(output_root, preflight_manifest)

    budget_results = []
    for budget in budgets:
        budget_root = output_root / f"budget-{budget:03d}"
        budget_root.mkdir(parents=True, exist_ok=True)
        result = run_budget(
            phase=args.phase,
            specs=specs,
            examples=examples,
            budget=budget,
            budget_root=budget_root,
            provider=args.provider,
            model=args.model,
            base_url=args.base_url or None,
            learned_model=learned_model,
            seed=args.seed,
            depth=args.depth,
            max_output_tokens=args.max_output_tokens,
            max_estimated_cost=args.max_estimated_cost,
            response_cache_dir=response_cache_root,
            response_cache_namespace_value=(
                response_cache_namespace(response_cache_binding_payload)
                if response_cache_binding_payload is not None
                else ""
            ),
        )
        result["determinism"] = (
            determinism_check(
                result,
                first_spec=ordered_tasks[0][0],
                first_example=ordered_tasks[0][1],
                provider=args.provider,
                model=args.model,
                base_url=args.base_url or None,
                learned_model=learned_model,
                seed=args.seed,
                depth=args.depth,
                max_output_tokens=args.max_output_tokens,
            )
            if args.phase == "offline"
            else {"ok": True, "not_run": "real-model phase uses frozen offline determinism evidence"}
        )
        trace_paths = sorted(budget_root.glob("reports/*/loom_traces/*/*.jsonl"))
        result["loom_validation"] = validate_loom_traces(
            loom_root,
            trace_paths,
            expected_count=len(result["rows"]),
        )
        write_json(
            budget_root / "validation.json",
            {
                "diagnostics": result["diagnostics"],
                "determinism": result["determinism"],
                "loom_validation": result["loom_validation"],
            },
        )
        budget_results.append(result)

    eligible_budgets = sorted(
        int(result["budget"])
        for result in budget_results
        if result["diagnostics"]["eligible"]
        and result["determinism"]["ok"]
        and result["loom_validation"]["all_valid"]
    )
    candidate_budget = eligible_budgets[0] if eligible_budgets else None
    frozen_learned_artifact = bool(
        learned_model is not None
        and training_manifest is not None
        and training_bundle_validation
        and training_bundle_validation["ok"]
        and (
            args.phase != "offline"
            or training_bundle_validation["training_repository_commit"] == code_snapshot["commit"]
        )
    )
    selected_budget = candidate_budget if frozen_learned_artifact else None
    commit_bindings = {
        "nanorlm": commit_binding(code_snapshot, args.expected_nanorlm_commit),
        "loom": commit_binding(loom_snapshot, args.expected_loom_commit),
    }
    gate_checks = {
        "nanorlm_repository": bool(code_snapshot["is_repository"]),
        "nanorlm_clean": bool(code_snapshot["clean"]),
        "loom_repository": bool(loom_snapshot and loom_snapshot["is_repository"]),
        "loom_clean": bool(loom_snapshot and loom_snapshot["clean"]),
        "commit_bindings": all(binding["ok"] for binding in commit_bindings.values()),
        "frozen_learned_model": learned_model is not None,
        "frozen_learned_training_manifest": frozen_learned_artifact,
        "training_code_binding": bool(
            (
                args.phase == "offline"
                and training_bundle_validation
                and training_bundle_validation.get("training_repository_commit") == code_snapshot["commit"]
            )
            or (args.phase != "offline" and prior_offline_evidence and prior_offline_evidence["ok"])
        ),
        "prior_offline_evidence": args.phase == "offline" or bool(
            prior_offline_evidence and prior_offline_evidence["ok"]
        ),
        "prior_preflight_evidence": args.phase == "offline" or bool(
            prior_preflight_evidence and prior_preflight_evidence["ok"]
        ),
        "eligible_budget": selected_budget is not None,
        "all_determinism_checks": all(result["determinism"]["ok"] for result in budget_results),
        "all_loom_traces_valid": all(result["loom_validation"]["all_valid"] for result in budget_results),
        "all_phase_diagnostics": (
            args.phase == "offline"
            or all(result["diagnostics"]["eligible"] for result in budget_results)
        ),
        "conversion_audit": conversion_audit_result["ok"],
        "hosted_family_audit": hosted_family_audit_result["ok"],
        "cost_cap_reservation": (
            args.max_estimated_cost is None
            or cost_preflight["logical_policy_upper_bound_usd"] <= args.max_estimated_cost
        ),
    }
    response_cache_snapshot: dict[str, Any] | None = None
    if response_cache_root is not None and response_cache_binding_payload is not None:
        response_cache_snapshot = snapshot_response_cache(
            response_cache_root,
            output_root,
            response_cache_binding_payload,
        )
    gate_checks["bound_response_cache"] = (
        args.phase == "offline"
        or bool(
            response_cache_snapshot
            and response_cache_snapshot["record_count"] > 0
            and response_cache_snapshot["namespace"]
            == response_cache_namespace(response_cache_binding_payload or {})
        )
    )
    inventory_exclusions = ["manifest.json", "release_audit.json", "checksums.txt"]
    inventory = artifact_inventory(output_root, excluded=inventory_exclusions)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "phase": args.phase,
        "status": "pending_release_audit",
        "repositories": {"nanorlm": code_snapshot, "loom": loom_snapshot},
        "commit_bindings": commit_bindings,
        "prior_offline_evidence": prior_offline_evidence,
        "prior_preflight_evidence": prior_preflight_evidence,
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "uv_lock_sha256": sha256_file(ROOT / "uv.lock"),
        },
        "reproduction": {
            "argv_template": reproduction_argv_template(
                args,
                specs,
                budgets,
                {
                    str(record["label"]): str(record["source_sha256"])
                    for record in dataset_records
                    if record.get("embedded")
                },
            ),
            "note": (
                "Source-dataset placeholders require the exact original exports matching "
                "source_sha256; replace other placeholders with artifacts or the endpoint "
                "matching the recorded hashes."
            ),
        },
        "configuration": {
            "policies": MATCHED_POLICIES,
            "retention_judge": "heuristic",
            "provider": args.provider,
            "model": args.model,
            "base_url_sha256": sha256_bytes((args.base_url or "").encode("utf-8")),
            "max_depth": args.depth,
            "max_steps": 256,
            "max_output_tokens": args.max_output_tokens,
            "budgets": budgets,
            "max_estimated_cost": args.max_estimated_cost,
            "cost_preflight": cost_preflight,
            "seed": args.seed,
            "expected_dataset_sha256": expected_dataset_hashes,
            "observed_dataset_sha256": observed_dataset_hashes,
            "learned_model_sha256": sha256_file(learned_model) if learned_model else None,
            "learned_model_source": "external_frozen_artifact" if learned_model else "built_in_development_default",
            "learned_training_manifest": (
                {
                    "path": training_manifest.relative_to(output_root).as_posix(),
                    "sha256": sha256_file(training_manifest),
                    "validation": training_bundle_validation,
                }
                if training_manifest
                else None
            ),
            "response_cache": (
                {
                    "binding": response_cache_binding_payload,
                    "initial_record_count": int(
                        response_cache_initial_state["record_count"]
                        if response_cache_initial_state
                        else 0
                    ),
                    "snapshot": response_cache_snapshot,
                }
                if response_cache_snapshot is not None
                else None
            ),
        },
        "datasets": dataset_records,
        "task_manifest": {
            "path": "task_manifest.json",
            "sha256": sha256_file(output_root / "task_manifest.json"),
            "tasks": len(task_records),
        },
        "budget_results": [
            {
                "budget": result["budget"],
                "diagnostics": result["diagnostics"],
                "determinism": result["determinism"],
                "loom_validation": result["loom_validation"],
                "reports": result["reports"],
                "execution_order": result["execution_order"],
            }
            for result in budget_results
        ],
        "eligible_budgets": eligible_budgets,
        "smallest_eligible_budget_candidate": candidate_budget,
        "selected_budget": selected_budget,
        "gate_checks": gate_checks,
        "artifact_inventory": inventory,
        "release_audit": {"pending": True},
        "checksums": "checksums.txt",
    }
    return finalize_release_manifest(output_root, manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the preregistered matched-retention bundle workflow.")
    parser.add_argument("--phase", choices=PHASES, default="offline")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--dataset-spec",
        action="append",
        default=[],
        help="Repeat LABEL:DATASET[:PATH]; defaults to the three protocol development families.",
    )
    parser.add_argument("--expected-dataset-sha256", action="append", default=[])
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--provider", choices=["heuristic", "openai_compatible"], default="heuristic")
    parser.add_argument("--model", default="demo/heuristic")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--learned-retention-model", default="")
    parser.add_argument("--learned-retention-training-manifest", default="")
    parser.add_argument("--offline-manifest", default="")
    parser.add_argument("--expected-offline-sha256", default="")
    parser.add_argument("--preflight-manifest", default="")
    parser.add_argument("--expected-preflight-sha256", default="")
    parser.add_argument(
        "--cache-dir",
        default="",
        help="External persistent response cache; required and exact-bound for hosted phases.",
    )
    parser.add_argument("--max-estimated-cost", type=float, default=None)
    parser.add_argument("--repo-root", default="/tmp/nanorlm-verifiers")
    parser.add_argument("--loom-root", default="")
    parser.add_argument("--expected-nanorlm-commit", default="")
    parser.add_argument("--expected-loom-commit", default="")
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.dataset_spec:
        args.dataset_spec = [
            "dossierbench:dossierbench",
            "ruler-synthetic:ruler_synthetic",
            "babilong-synthetic:babilong_synthetic",
        ]
    try:
        manifest = execute(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": manifest["status"], "selected_budget": manifest["selected_budget"]}, indent=2))
    return 0 if manifest["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
