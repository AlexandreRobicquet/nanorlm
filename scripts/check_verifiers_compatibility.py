from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import (  # noqa: E402
    DatasetCompatibilityError,
    dataset_required_paths,
    load_verifiers_compatibility,
    validate_repository_paths,
    verifiers_report_metadata,
)


def required_paths() -> list[str]:
    sources = [
        (ROOT / "examples" / "verifiers_30.json", "provenance"),
        (ROOT / "showcases" / "planning_tasks.json", "evidence_files"),
    ]
    paths: set[str] = set()
    for dataset_path, path_field in sources:
        rows = json.loads(dataset_path.read_text())
        paths.update(dataset_required_paths(rows, path_field))
    return sorted(paths)


def main(argv: list[str] | None = None) -> int:
    compatibility = load_verifiers_compatibility()
    parser = argparse.ArgumentParser(
        description="Check nanoRLM Verifiers datasets against a repository checkout."
    )
    parser.add_argument("--repo-root", default=compatibility["default_checkout"])
    args = parser.parse_args(argv)

    paths = required_paths()
    try:
        validate_repository_paths(
            args.repo_root,
            paths,
            dataset_name="Verifiers datasets",
            compatibility=compatibility,
        )
    except DatasetCompatibilityError as exc:
        parser.exit(2, f"error: {exc}\n")

    print(
        json.dumps(
            {
                "datasets": ["verifiers_30", "grounded_planning"],
                "required_paths": len(paths),
                "metadata": verifiers_report_metadata(args.repo_root),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
