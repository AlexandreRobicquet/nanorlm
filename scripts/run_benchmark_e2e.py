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
from nanorlm import is_local_base_url, supports_cost_estimate  # noqa: E402
from showcases.generate_assets import (  # noqa: E402
    load_payload,
    render_architecture_svg,
    render_curve_svg,
    render_trace_svg,
    summary_table,
)


PHASE_ORDER = ["check", "smoke", "synthetic", "repo_qa", "external", "real_model", "assets"]
DEFAULT_PHASES = ["check", "smoke", "synthetic", "external", "assets"]
OFFLINE_PHASES = ["check", "smoke", "synthetic", "repo_qa", "external", "assets"]
COMPILE_TARGETS = [
    "nanorlm.py",
    "policies.py",
    "bench.py",
    "scripts/prepare_ruler_external_jsonl.py",
    "scripts/run_benchmark_e2e.py",
    "examples/run_verifiers.py",
    "examples/run_needlepairs.py",
    "examples/run_dossiers.py",
    "examples/run_planning.py",
    "showcases/planning.py",
    "showcases/generate_assets.py",
]


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
    return shell_join(parts)


def run_benchmark_spec(run_root: Path, spec: BenchmarkSpec) -> dict[str, Any]:
    output_dir = run_root / spec.name
    examples = build_dataset(
        spec.dataset,
        limit=spec.limit,
        seed=0,
        repo_root=spec.repo_root,
        dataset_path=spec.dataset_path,
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
        dataset_name=spec.dataset,
    )
    if spec.provider == "heuristic":
        curves = generate_curves(
            spec.dataset,
            lambda seed: build_dataset(
                spec.dataset,
                limit=spec.limit,
                seed=seed,
                repo_root=spec.repo_root,
                dataset_path=spec.dataset_path,
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
        )
    else:
        curves = curves_from_summaries(spec.dataset, summaries, budget=spec.budget, depth=spec.depth)
    write_report_bundle(
        output_dir,
        dataset_name=spec.dataset,
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
