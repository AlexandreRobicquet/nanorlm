from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import (  # noqa: E402
    DEFAULT_POLICIES,
    build_dataset,
    curves_from_summaries,
    generate_curves,
    parse_csv_strings,
    policy_sweep,
    resolve_provider_choice,
    write_report_bundle,
)
from learned_retention import FEATURE_NAMES, TRAINING_OBJECTIVES, LearnedRetentionModel  # noqa: E402
from nanorlm import is_local_base_url, supports_cost_estimate  # noqa: E402
from showcases.generate_assets import (  # noqa: E402
    load_payload,
    render_architecture_svg,
    render_curve_svg,
    render_trace_svg,
    summary_table,
)
from scripts import train_learned_retention  # noqa: E402


PHASE_ORDER = ["check", "smoke", "synthetic", "learned", "repo_qa", "external", "real_model", "assets"]
DEFAULT_PHASES = ["check", "smoke", "synthetic", "external", "assets"]
OFFLINE_PHASES = ["check", "smoke", "synthetic", "learned", "repo_qa", "external", "assets"]
COMPILE_TARGETS = [
    "learned_retention.py",
    "nanorlm.py",
    "policies.py",
    "bench.py",
    "scripts/prepare_ruler_external_jsonl.py",
    "scripts/train_learned_retention.py",
    "scripts/run_benchmark_e2e.py",
    "examples/run_verifiers.py",
    "examples/run_needlepairs.py",
    "examples/run_dossiers.py",
    "examples/run_planning.py",
    "showcases/planning.py",
    "showcases/generate_assets.py",
]
LEARNED_ACCEPTANCE_DATASETS = {"dossierbench", "verifiers_30", "ruler_external", "babilong_external"}
LEARNED_MIN_REWARD_DELTA = 0.01
LEARNED_MIN_ELIGIBLE_EXAMPLES = 8
LEARNED_REQUIRED_WINS = 2


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    name: str
    dataset: str
    limit: int
    budget: int
    depth: int
    policies: list[str]
    curve_policies: list[str]
    curve_budgets: list[int]
    curve_depths: list[int]
    curve_seeds: list[int]
    repo_root: str
    dataset_path: str | None = None
    provider: str = "heuristic"
    model: str = "demo/heuristic"
    base_url: str | None = None
    api_key: str | None = None
    cache_dir: str | None = None
    max_output_tokens: int = 1024
    max_estimated_cost: float | None = None
    seed: int = 0
    start_index: int = 0
    learned_retention_model: str | None = None
    dataset_label: str | None = None


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def shell_join(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_id() -> str:
    return time.strftime("e2e-%Y%m%d-%H%M%S", time.gmtime())


def repo_value(command: Sequence[str], fallback: str = "") -> str:
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    except OSError:
        return fallback
    if result.returncode != 0:
        return fallback
    return result.stdout.strip() or fallback


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_phases(value: str) -> list[str]:
    normalized = value.strip().lower()
    if normalized == "default":
        return DEFAULT_PHASES
    if normalized == "offline":
        return OFFLINE_PHASES
    if normalized == "all":
        return PHASE_ORDER
    phases = parse_csv_strings(normalized)
    unknown = [phase for phase in phases if phase not in PHASE_ORDER]
    if unknown:
        raise ValueError(f"unknown phase(s): {', '.join(unknown)}")
    return [phase for phase in PHASE_ORDER if phase in phases]


def validate_report_bundle(path: Path) -> dict[str, Any]:
    required = ["summary.json", "per_case.jsonl", "curves.json", "experiment_report.md"]
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise RuntimeError(f"{path} is missing report file(s): {', '.join(missing)}")
    summary = json.loads((path / "summary.json").read_text())
    curves = json.loads((path / "curves.json").read_text())
    with (path / "per_case.jsonl").open(encoding="utf-8") as handle:
        per_case_rows = sum(1 for line in handle if line.strip())
    return {
        "path": str(path),
        "dataset": summary.get("dataset"),
        "policies": summary.get("policies", []),
        "per_case_rows": per_case_rows,
        "curve_points": len(curves.get("points", [])),
        "curve_aggregates": len(curves.get("aggregates", [])),
        "trace_examples": str(path / "trace_examples"),
    }


def run_command(command: Sequence[str], *, phase_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    elapsed = round(time.perf_counter() - started, 3)
    phase_dir.mkdir(parents=True, exist_ok=True)
    stem = "_".join("".join(char if char.isalnum() else "_" for char in part) for part in command[:5])[:96]
    stdout_path = phase_dir / f"{stem}.stdout.txt"
    stderr_path = phase_dir / f"{stem}.stderr.txt"
    stdout_path.write_text(result.stdout)
    stderr_path.write_text(result.stderr)
    record = {
        "command": shell_join(command),
        "elapsed_seconds": elapsed,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {shell_join(command)}")
    return record


def run_check_phase(run_root: Path) -> dict[str, Any]:
    phase_dir = run_root / "check"
    commands = [
        ["uv", "lock", "--check"],
        ["uv", "sync", "--frozen"],
        ["uv", "run", "python", "-m", "unittest", "discover", "-s", "tests", "-v"],
        ["uv", "run", "--with", "pytest", "pytest"],
        ["uv", "run", "python", "-m", "py_compile", *COMPILE_TARGETS],
    ]
    return {"commands": [run_command(command, phase_dir=phase_dir) for command in commands]}


def benchmark_command(spec: BenchmarkSpec) -> str:
    parts = [
        "python",
        "bench.py",
        "--dataset",
        spec.dataset,
        "--limit",
        str(spec.limit),
        "--start-index",
        str(spec.start_index),
        "--seed",
        str(spec.seed),
        "--budget",
        str(spec.budget),
        "--depth",
        str(spec.depth),
        "--provider",
        spec.provider.replace("_", "-"),
        "--model",
        spec.model,
        "--policies",
        ",".join(spec.policies),
        "--max-output-tokens",
        str(spec.max_output_tokens),
    ]
    if spec.dataset_path:
        parts.extend(["--dataset-path", spec.dataset_path])
    if spec.repo_root:
        parts.extend(["--repo-root", spec.repo_root])
    if spec.base_url:
        parts.extend(["--base-url", spec.base_url])
    if spec.cache_dir:
        parts.extend(["--cache-dir", spec.cache_dir])
    if spec.max_estimated_cost is not None:
        parts.extend(["--max-estimated-cost", str(spec.max_estimated_cost)])
    if spec.learned_retention_model:
        parts.extend(["--learned-retention-model", spec.learned_retention_model])
    return shell_join(parts)


def run_benchmark_spec(run_root: Path, spec: BenchmarkSpec) -> dict[str, Any]:
    output_dir = run_root / spec.name
    dataset_label = spec.dataset_label or spec.dataset
    examples = build_dataset(
        spec.dataset,
        limit=spec.limit,
        seed=spec.seed,
        repo_root=spec.repo_root,
        dataset_path=spec.dataset_path,
        start_index=spec.start_index,
    )
    summaries = policy_sweep(
        examples,
        spec.policies,
        budget=spec.budget,
        max_depth=spec.depth,
        output_dir=output_dir,
        provider=spec.provider,
        model=spec.model,
        base_url=spec.base_url,
        api_key=spec.api_key,
        cache_dir=spec.cache_dir,
        max_output_tokens=spec.max_output_tokens,
        max_estimated_cost=spec.max_estimated_cost,
        learned_retention_model=spec.learned_retention_model,
        dataset_name=dataset_label,
        seed=spec.seed,
    )
    if spec.provider == "heuristic":
        curves = generate_curves(
            dataset_label,
            lambda seed: build_dataset(
                spec.dataset,
                limit=spec.limit,
                seed=seed,
                repo_root=spec.repo_root,
                dataset_path=spec.dataset_path,
                start_index=spec.start_index,
            ),
            policies=spec.curve_policies,
            budgets=spec.curve_budgets,
            depths=spec.curve_depths,
            seeds=spec.curve_seeds,
            provider=spec.provider,
            model=spec.model,
            base_url=spec.base_url,
            api_key=spec.api_key,
            cache_dir=spec.cache_dir,
            max_output_tokens=spec.max_output_tokens,
            learned_retention_model=spec.learned_retention_model,
        )
    else:
        curves = curves_from_summaries(dataset_label, summaries, budget=spec.budget, depth=spec.depth)
    write_report_bundle(
        output_dir,
        dataset_name=dataset_label,
        summaries=summaries,
        curves=curves,
        command=benchmark_command(spec),
    )
    report = validate_report_bundle(output_dir)
    report.update(
        {
            "name": spec.name,
            "limit": spec.limit,
            "budget": spec.budget,
            "depth": spec.depth,
            "start_index": spec.start_index,
            "provider": spec.provider,
            "model": spec.model,
            "base_url": spec.base_url,
            "cache_dir": spec.cache_dir,
            "completed": all(summary.get("completed", False) for summary in summaries),
            "total_cost_estimate": round(sum(float(summary.get("total_cost_estimate", 0.0)) for summary in summaries), 6),
        }
    )
    return report


def run_specs(run_root: Path, specs: Sequence[BenchmarkSpec]) -> dict[str, Any]:
    return {"reports": [run_benchmark_spec(run_root, spec) for spec in specs]}


def smoke_specs(args: argparse.Namespace) -> list[BenchmarkSpec]:
    policies = list(DEFAULT_POLICIES)
    return [
        BenchmarkSpec(
            name="smoke_pairbench",
            dataset="pairbench",
            limit=args.smoke_limit,
            budget=60,
            depth=2,
            policies=policies,
            curve_policies=policies,
            curve_budgets=[60],
            curve_depths=[2],
            curve_seeds=[0],
            repo_root=args.repo_root,
        ),
        BenchmarkSpec(
            name="smoke_verifiers",
            dataset="verifiers_smoke",
            limit=min(2, args.smoke_limit),
            budget=80,
            depth=2,
            policies=policies,
            curve_policies=policies,
            curve_budgets=[80],
            curve_depths=[2],
            curve_seeds=[0],
            repo_root=args.smoke_repo_root,
        ),
        BenchmarkSpec(
            name="smoke_external_jsonl",
            dataset="external_jsonl",
            limit=min(2, args.smoke_limit),
            budget=80,
            depth=2,
            policies=policies,
            curve_policies=policies,
            curve_budgets=[80],
            curve_depths=[2],
            curve_seeds=[0],
            repo_root=args.repo_root,
            dataset_path=args.fixture_external_dataset_path,
        ),
    ]


def synthetic_specs(args: argparse.Namespace) -> list[BenchmarkSpec]:
    policies = list(DEFAULT_POLICIES)
    return [
        BenchmarkSpec(
            name="synthetic_pairbench",
            dataset="pairbench",
            limit=args.synthetic_limit,
            budget=60,
            depth=2,
            policies=policies,
            curve_policies=policies,
            curve_budgets=[60],
            curve_depths=[2],
            curve_seeds=[0],
            repo_root=args.repo_root,
        ),
        BenchmarkSpec(
            name="synthetic_needlepairs",
            dataset="needlepairs",
            limit=args.synthetic_limit,
            budget=60,
            depth=3,
            policies=["keep_recent", "summary_only", "single_critic_topk", "pairwise_tournament"],
            curve_policies=["keep_recent", "single_critic_topk", "pairwise_tournament"],
            curve_budgets=[60, 80],
            curve_depths=[2, 3],
            curve_seeds=[0, 1, 2],
            repo_root=args.repo_root,
        ),
        BenchmarkSpec(
            name="synthetic_dossierbench",
            dataset="dossierbench",
            limit=args.dossier_limit,
            budget=80,
            depth=4,
            policies=policies,
            curve_policies=["single_critic_topk", "pairwise_tournament"],
            curve_budgets=[60, 80, 100],
            curve_depths=[3, 4],
            curve_seeds=[0, 1, 2],
            repo_root=args.repo_root,
        ),
    ]


def repo_qa_specs(args: argparse.Namespace) -> list[BenchmarkSpec]:
    return [
        BenchmarkSpec(
            name="repo_qa_verifiers_30",
            dataset="verifiers_30",
            limit=args.repo_qa_limit,
            budget=140,
            depth=2,
            policies=list(DEFAULT_POLICIES),
            curve_policies=["direct_full_context", "summary_only", "pairwise_tournament"],
            curve_budgets=[100, 140, 180],
            curve_depths=[1, 2],
            curve_seeds=[0],
            repo_root=args.repo_root,
        )
    ]


def external_specs(args: argparse.Namespace) -> list[BenchmarkSpec]:
    return [
        BenchmarkSpec(
            name="external_jsonl",
            dataset="external_jsonl",
            limit=args.external_limit,
            budget=120,
            depth=3,
            policies=["direct_full_context", "keep_recent", "pairwise_tournament"],
            curve_policies=["direct_full_context", "keep_recent", "pairwise_tournament"],
            curve_budgets=[120],
            curve_depths=[3],
            curve_seeds=[0],
            repo_root=args.repo_root,
            dataset_path=args.external_dataset_path,
        )
    ]


def learned_eval_specs(args: argparse.Namespace, model_path: str) -> list[BenchmarkSpec]:
    policies = ["direct_full_context", "keep_recent", "single_critic_topk", "pairwise_tournament", "learned_retention"]
    seed = args.learned_eval_seed
    heldout_start = args.learned_eval_start_index
    if heldout_start < 0:
        heldout_start = args.learned_train_limit
    specs = [
        BenchmarkSpec(
            name="learned_pairbench",
            dataset="pairbench",
            limit=args.learned_eval_limit,
            budget=60,
            depth=2,
            policies=policies,
            curve_policies=policies,
            curve_budgets=[60],
            curve_depths=[2],
            curve_seeds=[seed],
            repo_root=args.repo_root,
            seed=seed,
            start_index=heldout_start,
            learned_retention_model=model_path,
        ),
        BenchmarkSpec(
            name="learned_dossierbench",
            dataset="dossierbench",
            limit=args.learned_eval_limit,
            budget=80,
            depth=4,
            policies=policies,
            curve_policies=policies,
            curve_budgets=[80],
            curve_depths=[4],
            curve_seeds=[seed],
            repo_root=args.repo_root,
            seed=seed,
            start_index=heldout_start,
            learned_retention_model=model_path,
        ),
        BenchmarkSpec(
            name="learned_ruler_synthetic",
            dataset="ruler_synthetic",
            limit=args.learned_eval_limit,
            budget=90,
            depth=4,
            policies=policies,
            curve_policies=policies,
            curve_budgets=[90],
            curve_depths=[4],
            curve_seeds=[seed],
            repo_root=args.repo_root,
            seed=seed,
            start_index=heldout_start,
            learned_retention_model=model_path,
        ),
        BenchmarkSpec(
            name="learned_babilong_synthetic",
            dataset="babilong_synthetic",
            limit=args.learned_eval_limit,
            budget=90,
            depth=4,
            policies=policies,
            curve_policies=policies,
            curve_budgets=[90],
            curve_depths=[4],
            curve_seeds=[seed],
            repo_root=args.repo_root,
            seed=seed,
            start_index=heldout_start,
            learned_retention_model=model_path,
        ),
        BenchmarkSpec(
            name="learned_external_jsonl",
            dataset="external_jsonl",
            limit=min(args.learned_eval_limit, args.external_limit),
            budget=120,
            depth=3,
            policies=policies,
            curve_policies=policies,
            curve_budgets=[120],
            curve_depths=[3],
            curve_seeds=[seed],
            repo_root=args.repo_root,
            dataset_path=args.external_dataset_path,
            seed=seed,
            start_index=0,
            learned_retention_model=model_path,
        ),
    ]
    for name, label, dataset_path in (
        ("learned_ruler_external", "ruler_external", args.learned_ruler_path),
        ("learned_babilong_external", "babilong_external", args.learned_babilong_path),
    ):
        if not dataset_path:
            continue
        specs.append(
            BenchmarkSpec(
                name=name,
                dataset="external_jsonl",
                dataset_label=label,
                limit=args.external_limit,
                budget=120,
                depth=3,
                policies=policies,
                curve_policies=policies,
                curve_budgets=[120],
                curve_depths=[3],
                curve_seeds=[seed],
                repo_root=args.repo_root,
                dataset_path=dataset_path,
                seed=seed,
                start_index=0,
                learned_retention_model=model_path,
            )
        )
    if args.learned_verifiers_repo_root:
        specs.append(
            BenchmarkSpec(
                name="learned_verifiers_30",
                dataset="verifiers_30",
                limit=args.repo_qa_limit,
                budget=140,
                depth=2,
                policies=policies,
                curve_policies=policies,
                curve_budgets=[140],
                curve_depths=[2],
                curve_seeds=[seed],
                repo_root=args.learned_verifiers_repo_root,
                seed=seed,
                start_index=args.learned_train_limit,
                learned_retention_model=model_path,
            )
        )
    return specs


def _summary_by_policy(summary_path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(summary_path.read_text())
    return {summary["policy"]: summary for summary in payload.get("summaries", [])}


def _learned_acceptance_deltas(
    learned: dict[str, Any],
    pairwise: dict[str, Any],
) -> tuple[float, float, float] | None:
    def rows_by_name(summary: dict[str, Any]) -> dict[str, dict[str, Any]] | None:
        rows = summary.get("results")
        if not isinstance(rows, list) or not rows:
            return None
        indexed: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                return None
            name = str(row.get("name", ""))
            if not name or name in indexed:
                return None
            indexed[name] = row
        return indexed

    learned_rows = rows_by_name(learned)
    pairwise_rows = rows_by_name(pairwise)
    if learned_rows is None or pairwise_rows is None or learned_rows.keys() != pairwise_rows.keys():
        return None

    def mean(rows: dict[str, dict[str, Any]], key: str) -> float:
        return sum(float(row.get(key, 0.0)) for row in rows.values()) / len(rows)

    return (
        mean(learned_rows, "reward_score") - mean(pairwise_rows, "reward_score"),
        mean(learned_rows, "answer_accuracy") - mean(pairwise_rows, "answer_accuracy"),
        mean(learned_rows, "provenance_score") - mean(pairwise_rows, "provenance_score"),
    )


def _per_case_by_policy(report_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    per_case_path = report_path / "per_case.jsonl"
    if not per_case_path.exists():
        return rows
    with per_case_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[(str(row.get("policy", "")), str(row.get("name", "")))] = row
    return rows


def _evidence_delta(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    right_values = {str(value) for value in right.get("retained_provenance", [])}
    return [str(value) for value in left.get("retained_provenance", []) if str(value) not in right_values]


def _dropped_expected_provenance(row: dict[str, Any]) -> list[str]:
    expected = [str(value).lower() for value in row.get("expected_provenance", [])]
    matches: list[str] = []
    for dropped in row.get("drop_reasons", []):
        provenance = str(dropped.get("provenance", ""))
        lower = provenance.lower()
        if any(value in lower or Path(value).name in lower for value in expected):
            matches.append(provenance)
    return matches


def _compact_evidence(values: Sequence[str], limit: int = 3) -> str:
    if not values:
        return "none"
    compact = [value if len(value) <= 100 else value[:97] + "..." for value in values[:limit]]
    if len(values) > limit:
        compact.append(f"+{len(values) - limit} more")
    return "; ".join(compact)


def _underperforming_cases(report_path: Path, limit: int = 3) -> list[dict[str, Any]]:
    rows = _per_case_by_policy(report_path)
    case_names = sorted({name for policy, name in rows if policy in {"learned_retention", "pairwise_tournament"}})
    failures: list[dict[str, Any]] = []
    for name in case_names:
        learned = rows.get(("learned_retention", name))
        pairwise = rows.get(("pairwise_tournament", name))
        if not learned or not pairwise:
            continue
        learned_answer = float(learned.get("answer_accuracy", 0.0))
        pairwise_answer = float(pairwise.get("answer_accuracy", 0.0))
        learned_prov = float(learned.get("provenance_score", 0.0))
        pairwise_prov = float(pairwise.get("provenance_score", 0.0))
        if learned_answer >= pairwise_answer and learned_prov >= pairwise_prov:
            continue
        failures.append(
            {
                "name": name,
                "learned_answer": learned_answer,
                "pairwise_answer": pairwise_answer,
                "learned_provenance": learned_prov,
                "pairwise_provenance": pairwise_prov,
                "learned_only_provenance": _evidence_delta(learned, pairwise),
                "pairwise_only_provenance": _evidence_delta(pairwise, learned),
                "learned_dropped_expected_provenance": _dropped_expected_provenance(learned),
                "pairwise_dropped_expected_provenance": _dropped_expected_provenance(pairwise),
                "learned_trace": str(report_path / "trace_examples" / "learned_retention" / f"{name}.tree.txt"),
                "pairwise_trace": str(report_path / "trace_examples" / "pairwise_tournament" / f"{name}.tree.txt"),
            }
        )
        if len(failures) >= limit:
            break
    return failures


def _is_learned_acceptance_win(
    *,
    acceptance_eligible: bool,
    reward_delta: float,
    answer_delta: float,
    provenance_delta: float,
) -> bool:
    return (
        acceptance_eligible
        and reward_delta >= LEARNED_MIN_REWARD_DELTA
        and answer_delta >= 0.0
        and provenance_delta >= 0.0
    )


def _round_report_metric(value: float) -> float:
    rounded = round(value, 3)
    return 0.0 if rounded == 0.0 else rounded


def write_learned_report(run_root: Path, training_manifest: dict[str, Any], reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    non_toy_wins: list[dict[str, Any]] = []
    failure_evidence: list[dict[str, Any]] = []
    for report in reports:
        report_path = Path(str(report["path"]))
        by_policy = _summary_by_policy(report_path / "summary.json")
        learned = by_policy.get("learned_retention")
        pairwise = by_policy.get("pairwise_tournament")
        if not learned or not pairwise:
            continue
        exact_deltas = _learned_acceptance_deltas(learned, pairwise)
        if exact_deltas is None:
            reward_delta = float(learned.get("reward_score", 0.0)) - float(pairwise.get("reward_score", 0.0))
            answer_delta = float(learned.get("answer_accuracy", 0.0)) - float(pairwise.get("answer_accuracy", 0.0))
            provenance_delta = float(learned.get("provenance_score", 0.0)) - float(pairwise.get("provenance_score", 0.0))
        else:
            reward_delta, answer_delta, provenance_delta = exact_deltas
        learned_examples = int(learned.get("examples", 0))
        pairwise_examples = int(pairwise.get("examples", 0))
        acceptance_eligible = (
            exact_deltas is not None
            and len(learned["results"]) == learned_examples
            and len(pairwise["results"]) == pairwise_examples
            and report.get("dataset") in LEARNED_ACCEPTANCE_DATASETS
            and learned_examples >= LEARNED_MIN_ELIGIBLE_EXAMPLES
            and learned_examples == pairwise_examples
            and bool(learned.get("completed"))
            and bool(pairwise.get("completed"))
        )
        row = {
            "dataset": report.get("dataset"),
            "report": str(report_path),
            "examples": learned_examples,
            "budget": report.get("budget"),
            "depth": report.get("depth"),
            "start_index": report.get("start_index", 0),
            "learned_reward": learned.get("reward_score", 0.0),
            "pairwise_reward": pairwise.get("reward_score", 0.0),
            "reward_delta": _round_report_metric(reward_delta),
            "answer_delta": _round_report_metric(answer_delta),
            "provenance_delta": _round_report_metric(provenance_delta),
            "learned_answer": learned.get("answer_accuracy", 0.0),
            "pairwise_answer": pairwise.get("answer_accuracy", 0.0),
            "learned_provenance": learned.get("provenance_score", 0.0),
            "pairwise_provenance": pairwise.get("provenance_score", 0.0),
            "learned_compactness": learned.get("compactness", 0.0),
            "pairwise_compactness": pairwise.get("compactness", 0.0),
            "acceptance_eligible": acceptance_eligible,
            "gate_metrics_source": "per_case" if exact_deltas is not None else "unavailable",
        }
        rows.append(row)
        if _is_learned_acceptance_win(
            acceptance_eligible=acceptance_eligible,
            reward_delta=reward_delta,
            answer_delta=answer_delta,
            provenance_delta=provenance_delta,
        ):
            non_toy_wins.append(row)
        for failure in _underperforming_cases(report_path):
            failure_evidence.append({"dataset": report.get("dataset"), **failure})
    verdict = "positive" if len(non_toy_wins) >= LEARNED_REQUIRED_WINS else "negative_or_inconclusive"
    training = training_manifest.get("training", {})
    model = LearnedRetentionModel.load(training_manifest["model_path"])
    ranked_weights = sorted(
        ((name, float(model.weights.get(name, 0.0))) for name in FEATURE_NAMES),
        key=lambda item: (-abs(item[1]), item[0]),
    )
    external_long_context = any(
        row["dataset"] in {"ruler_external", "babilong_external"}
        for row in rows
    )
    evidence_note = (
        "This is a budgeted local evaluation bundle. Synthetic task-shape rows and explicitly supplied external "
        "RULER/BABILong slices are reported separately; neither is a leaderboard submission."
        if external_long_context
        else "This is a budgeted internal evaluation bundle. The RULER and BABILong rows here are synthetic "
        "task-shape slices, not public leaderboard claims."
    )
    report_path = run_root / "learned_retention_report.md"
    lines = [
        "# Learned Retention Report",
        "",
        f"- Verdict: `{verdict}`",
        f"- Acceptance check: learned_retention beats pairwise_tournament on {len(non_toy_wins)} eligible non-toy slice(s).",
        f"- Win rule: reward delta >= {LEARNED_MIN_REWARD_DELTA:.2f}, with no answer or provenance regression.",
        f"- Eligible slice rule: at least {LEARNED_MIN_ELIGIBLE_EXAMPLES} completed examples for both policies on DossierBench, Verifiers-30, or an explicit external RULER/BABILong slice.",
        f"- Promotion rule: at least {LEARNED_REQUIRED_WINS} eligible slice wins.",
        f"- Model: `{training_manifest['model_path']}`",
        f"- Training source: `{training.get('training_source', 'unknown')}` via `{training.get('collection_policy', 'unknown')}`",
        f"- Objective: `{training.get('objective', 'unknown')}`",
        f"- Training rows / decision pairs: {training.get('training_rows', 0)} / {training.get('training_pairs', 0)}",
        f"- Reward-weighted pairs / mean weight: {training.get('reward_weighted_pairs', 0)} / {training.get('mean_pair_reward_weight')}",
        f"- Trace trajectories: {training.get('trace_trajectories', 0)}",
        f"- Pairwise training accuracy: {training.get('pairwise_accuracy_before')} -> {training.get('pairwise_accuracy_after')}",
        "",
        evidence_note,
        "",
        "## Pairwise Comparison",
        "",
        "| dataset | eligible | examples | start | budget | learned reward | pairwise reward | reward delta | answer delta | provenance delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['dataset']}` | {'yes' if row['acceptance_eligible'] else 'no'} | {row['examples']} | {row['start_index']} | {row['budget']} | {row['learned_reward']:.3f} | "
            f"{row['pairwise_reward']:.3f} | {row['reward_delta']:+.3f} | {row['answer_delta']:+.3f} | "
            f"{row['provenance_delta']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Model Diagnostics",
            "",
            "| feature | weight |",
            "| --- | ---: |",
        ]
    )
    for name, weight in ranked_weights[:8]:
        lines.append(f"| `{name}` | {weight:+.4f} |")
    if verdict == "positive":
        lines.extend(
            [
                "",
                "## Conservative Read",
                "",
                "The learned controller cleared the local acceptance check on at least two non-toy slices. Treat this as a reproducible internal result until the same model is evaluated against real exported RULER or BABILong data with a hosted-model bundle.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Conservative Read",
                "",
                "The learned controller did not clear the local acceptance check. The saved `per_case.jsonl` rows and `trace_examples/` directories are the evidence for where hand-coded retention remains sufficient or where the learned scorer failed to preserve the right facts.",
            ]
        )
    if failure_evidence:
        lines.extend(
            [
                "",
                "## Trace Evidence For Negative Cases",
            ]
        )
        for row in failure_evidence[:10]:
            lines.extend(
                [
                    "",
                    f"### {row['dataset']} / {row['name']}",
                    "",
                    f"- Answer: learned {row['learned_answer']:.3f}, pairwise {row['pairwise_answer']:.3f}",
                    f"- Provenance: learned {row['learned_provenance']:.3f}, pairwise {row['pairwise_provenance']:.3f}",
                    f"- Learned-only retained provenance: {_compact_evidence(row['learned_only_provenance'])}",
                    f"- Pairwise-only retained provenance: {_compact_evidence(row['pairwise_only_provenance'])}",
                    f"- Expected provenance dropped by learned: {_compact_evidence(row['learned_dropped_expected_provenance'])}",
                    f"- Traces: `{row['learned_trace']}` and `{row['pairwise_trace']}`",
                ]
            )
    report_path.write_text("\n".join(lines) + "\n")
    return {
        "report_path": str(report_path),
        "verdict": verdict,
        "non_toy_wins": len(non_toy_wins),
        "acceptance_rule": {
            "eligible_datasets": sorted(LEARNED_ACCEPTANCE_DATASETS),
            "minimum_examples": LEARNED_MIN_ELIGIBLE_EXAMPLES,
            "minimum_reward_delta": LEARNED_MIN_REWARD_DELTA,
            "required_wins": LEARNED_REQUIRED_WINS,
            "requires_answer_non_regression": True,
            "requires_provenance_non_regression": True,
            "requires_completed_equal_size_runs": True,
            "gate_metrics_source": "per_case",
        },
        "comparisons": rows,
        "failure_evidence": failure_evidence,
    }


def run_learned_phase(run_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    phase_dir = run_root / "learned_retention"
    training_dir = phase_dir / "training"
    training_datasets = parse_csv_strings(args.learned_train_datasets)
    training_repo_root = args.smoke_repo_root
    if args.learned_verifiers_repo_root:
        training_datasets = [dataset for dataset in training_datasets if dataset != "verifiers_smoke"]
        if "verifiers_30" not in training_datasets:
            training_datasets.append("verifiers_30")
        training_repo_root = args.learned_verifiers_repo_root
    training_manifest = train_learned_retention.run(
        [
            "--datasets",
            ",".join(training_datasets),
            "--train-seeds",
            args.learned_train_seeds,
            "--limit",
            str(args.learned_train_limit),
            "--repo-root",
            training_repo_root,
            "--dataset-path",
            args.fixture_external_dataset_path,
            "--output-dir",
            str(training_dir),
            "--epochs",
            str(args.learned_epochs),
            "--learning-rate",
            str(args.learned_learning_rate),
            "--l2",
            str(args.learned_l2),
            "--seed",
            str(args.learned_train_seed),
            "--training-source",
            args.learned_training_source,
            "--objective",
            args.learned_objective,
            "--collection-policy",
            args.learned_collection_policy,
        ]
    )
    reports = run_specs(run_root, learned_eval_specs(args, training_manifest["model_path"]))["reports"]
    learned_report = write_learned_report(run_root, training_manifest, reports)
    return {
        "training": training_manifest,
        "reports": reports,
        "learned_report": learned_report,
    }


def real_model_spec(args: argparse.Namespace, run_root: Path) -> BenchmarkSpec:
    provider = resolve_provider_choice(args.real_provider, False)
    base_url = args.real_base_url or None
    if provider != "openai_compatible":
        raise RuntimeError("real_model phase currently requires --real-provider openai-compatible")
    hosted = not is_local_base_url(base_url)
    if hosted and not supports_cost_estimate(provider, args.real_model, base_url):
        raise RuntimeError(f"real_model phase has no cost table entry for hosted model: {args.real_model}")
    if hosted and not args.real_api_key:
        raise RuntimeError("real_model phase requires OPENAI_API_KEY or --real-api-key for hosted OpenAI-compatible runs")
    cache_dir = args.real_cache_dir or str(run_root / "cache" / args.real_model)
    return BenchmarkSpec(
        name="real_model_external_jsonl",
        dataset="external_jsonl",
        limit=args.real_model_limit,
        budget=120,
        depth=3,
        policies=["direct_full_context", "keep_recent", "pairwise_tournament"],
        curve_policies=["direct_full_context", "keep_recent", "pairwise_tournament"],
        curve_budgets=[120],
        curve_depths=[3],
        curve_seeds=[0],
        repo_root=args.repo_root,
        dataset_path=args.external_dataset_path,
        provider=provider,
        model=args.real_model,
        base_url=base_url,
        api_key=args.real_api_key or None,
        cache_dir=cache_dir,
        max_output_tokens=args.real_max_output_tokens,
        max_estimated_cost=args.real_max_estimated_cost,
    )


def find_report(run_root: Path, preferred_name: str) -> Path:
    preferred = run_root / preferred_name
    if (preferred / "summary.json").exists():
        return preferred
    candidates = sorted(path for path in run_root.iterdir() if path.is_dir() and (path / "summary.json").exists())
    if not candidates:
        raise RuntimeError("assets phase could not find a benchmark report bundle")
    return candidates[0]


def run_assets_phase(run_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    report_dir = find_report(run_root, args.assets_source)
    assets_dir = run_root / "artifacts"
    assets_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = load_payload(report_dir / "summary.json")
    curves_payload = load_payload(report_dir / "curves.json")
    (assets_dir / "benchmark_snapshot.md").write_text(summary_table(summary_payload))
    (assets_dir / "architecture.svg").write_text(render_architecture_svg())
    (assets_dir / "policy_curve.svg").write_text(render_curve_svg(curves_payload, metric=args.assets_metric))
    trace_dir = report_dir / "trace_examples" / args.assets_trace_policy
    tree_files = sorted(trace_dir.glob("*.tree.txt"))
    if tree_files:
        (assets_dir / "trace_card.svg").write_text(render_trace_svg(tree_files[0].read_text()))
    manifest = {
        "report_dir": str(report_dir),
        "assets_dir": str(assets_dir),
        "files": sorted(path.name for path in assets_dir.iterdir() if path.is_file()),
    }
    write_json(assets_dir / "manifest.json", manifest)
    return manifest


def phase_runner(args: argparse.Namespace, run_root: Path) -> dict[str, Callable[[], dict[str, Any]]]:
    return {
        "check": lambda: run_check_phase(run_root),
        "smoke": lambda: run_specs(run_root, smoke_specs(args)),
        "synthetic": lambda: run_specs(run_root, synthetic_specs(args)),
        "learned": lambda: run_learned_phase(run_root, args),
        "repo_qa": lambda: run_specs(run_root, repo_qa_specs(args)),
        "external": lambda: run_specs(run_root, external_specs(args)),
        "real_model": lambda: run_specs(run_root, [real_model_spec(args, run_root)]),
        "assets": lambda: run_assets_phase(run_root, args),
    }


def initial_manifest(args: argparse.Namespace, phases: Sequence[str], run_root: Path) -> dict[str, Any]:
    return {
        "generated_by": "scripts/run_benchmark_e2e.py",
        "status": "running",
        "started_at": utc_timestamp(),
        "ended_at": None,
        "run_root": str(run_root),
        "phases_requested": list(phases),
        "git": {
            "head": repo_value(["git", "rev-parse", "HEAD"]),
            "branch": repo_value(["git", "branch", "--show-current"], "detached"),
            "status_short": repo_value(["git", "status", "--short"]),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "config": {
            "repo_root": args.repo_root,
            "smoke_repo_root": args.smoke_repo_root,
            "external_dataset_path": args.external_dataset_path,
            "fixture_external_dataset_path": args.fixture_external_dataset_path,
            "real_provider": args.real_provider,
            "real_model": args.real_model,
            "real_base_url": args.real_base_url,
            "real_cache_dir": args.real_cache_dir,
            "real_max_estimated_cost": args.real_max_estimated_cost,
            "learned_train_datasets": args.learned_train_datasets,
            "learned_train_seeds": args.learned_train_seeds,
            "learned_eval_seed": args.learned_eval_seed,
            "learned_eval_start_index": args.learned_eval_start_index,
            "learned_training_source": args.learned_training_source,
            "learned_objective": args.learned_objective,
            "learned_collection_policy": args.learned_collection_policy,
            "learned_ruler_path": args.learned_ruler_path,
            "learned_babilong_path": args.learned_babilong_path,
            "learned_verifiers_repo_root": args.learned_verifiers_repo_root,
            "cost_cap_note": "max_estimated_cost is enforced between benchmark cases, not before each model call.",
        },
        "phases": [],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the nanoRLM benchmark workflow end to end.")
    parser.add_argument("--phases", default="default", help="default, offline, all, or comma-separated phase names")
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "e2e"))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--repo-root", default="/tmp/nanorlm-verifiers")
    parser.add_argument("--smoke-repo-root", default=str(ROOT / "tests" / "fixtures" / "verifiers-mini"))
    parser.add_argument(
        "--fixture-external-dataset-path",
        default=str(ROOT / "tests" / "fixtures" / "external-benchmark-mini.jsonl"),
    )
    parser.add_argument("--external-dataset-path", default=str(ROOT / "tests" / "fixtures" / "external-benchmark-mini.jsonl"))
    parser.add_argument("--smoke-limit", type=int, default=4)
    parser.add_argument("--synthetic-limit", type=int, default=10)
    parser.add_argument("--dossier-limit", type=int, default=12)
    parser.add_argument("--repo-qa-limit", type=int, default=10)
    parser.add_argument("--external-limit", type=int, default=12)
    parser.add_argument(
        "--learned-train-datasets",
        default="pairbench,dossierbench,ruler_synthetic,babilong_synthetic,external_jsonl,verifiers_smoke",
    )
    parser.add_argument("--learned-train-seeds", default="0,1")
    parser.add_argument("--learned-train-limit", type=int, default=8)
    parser.add_argument("--learned-train-seed", type=int, default=0)
    parser.add_argument("--learned-training-source", choices=["traces", "blocks"], default="traces")
    parser.add_argument("--learned-objective", choices=TRAINING_OBJECTIVES, default="pairwise")
    parser.add_argument(
        "--learned-collection-policy",
        choices=["keep_recent", "single_critic_topk", "pairwise_tournament"],
        default="pairwise_tournament",
    )
    parser.add_argument("--learned-eval-limit", type=int, default=10)
    parser.add_argument("--learned-eval-seed", type=int, default=2)
    parser.add_argument(
        "--learned-eval-start-index",
        type=int,
        default=-1,
        help="Held-out start index for synthetic learned eval; -1 means learned-train-limit.",
    )
    parser.add_argument("--learned-epochs", type=int, default=20)
    parser.add_argument("--learned-learning-rate", type=float, default=0.15)
    parser.add_argument("--learned-l2", type=float, default=0.0005)
    parser.add_argument("--learned-ruler-path", default="")
    parser.add_argument("--learned-babilong-path", default="")
    parser.add_argument("--learned-verifiers-repo-root", default="")
    parser.add_argument("--real-provider", choices=["openai-compatible", "anthropic"], default="openai-compatible")
    parser.add_argument("--real-model", default="gpt-4.1-mini")
    parser.add_argument("--real-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--real-api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--real-cache-dir", default="")
    parser.add_argument("--real-model-limit", type=int, default=12)
    parser.add_argument("--real-max-output-tokens", type=int, default=1024)
    parser.add_argument("--real-max-estimated-cost", type=float, default=20.0)
    parser.add_argument("--assets-source", default="synthetic_dossierbench")
    parser.add_argument("--assets-metric", default="answer_accuracy")
    parser.add_argument("--assets-trace-policy", default="pairwise_tournament")
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        phases = parse_phases(args.phases)
    except ValueError as exc:
        parser.error(str(exc))

    current_run_id = args.run_id or run_id()
    run_root = Path(args.output_root) / current_run_id
    run_root.mkdir(parents=True, exist_ok=True)
    manifest = initial_manifest(args, phases, run_root)
    write_json(run_root / "manifest.json", manifest)
    runners = phase_runner(args, run_root)
    started = time.perf_counter()
    try:
        for phase in phases:
            started = time.perf_counter()
            phase_record: dict[str, Any] = {"name": phase, "status": "running", "started_at": utc_timestamp()}
            manifest["phases"].append(phase_record)
            write_json(run_root / "manifest.json", manifest)
            result = runners[phase]()
            phase_record.update(
                {
                    "status": "passed",
                    "ended_at": utc_timestamp(),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "result": result,
                }
            )
            write_json(run_root / "manifest.json", manifest)
        manifest["status"] = "passed"
        return_code = 0
    except Exception as exc:
        manifest["status"] = "failed"
        if manifest["phases"]:
            manifest["phases"][-1].update(
                {
                    "status": "failed",
                    "ended_at": utc_timestamp(),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "error": str(exc),
                }
            )
        else:
            manifest["error"] = str(exc)
        print(f"benchmark e2e failed: {exc}", file=sys.stderr)
        return_code = 1
    finally:
        manifest["ended_at"] = utc_timestamp()
        write_json(run_root / "manifest.json", manifest)
        print(json.dumps({"status": manifest["status"], "run_root": str(run_root)}, indent=2))
    return return_code


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
