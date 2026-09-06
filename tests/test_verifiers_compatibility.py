from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bench import (
    DatasetCompatibilityError,
    dataset_required_paths,
    load_verifiers_30,
    load_verifiers_compatibility,
    verifiers_report_metadata,
)
from examples.run_verifiers import main as run_verifiers_main
from showcases.planning import load_planning_tasks, main as run_planning_main


ROOT = Path(__file__).resolve().parents[1]
TREE_FIXTURE = ROOT / "tests" / "fixtures" / "verifiers_482e28f_tree.json"


def load_tree_fixture() -> dict[str, object]:
    return json.loads(TREE_FIXTURE.read_text())


def populate_tree(root: Path, paths: list[str]) -> None:
    for relative_path in paths:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"Fixture for {relative_path}\n")


class VerifiersCompatibilityTests(unittest.TestCase):
    def test_pinned_tree_fixture_contains_every_dataset_path(self) -> None:
        fixture = load_tree_fixture()
        compatibility = load_verifiers_compatibility()
        required: set[str] = set()
        for dataset_path, path_field in [
            (ROOT / "examples" / "verifiers_30.json", "provenance"),
            (ROOT / "showcases" / "planning_tasks.json", "evidence_files"),
        ]:
            rows = json.loads(dataset_path.read_text())
            required.update(dataset_required_paths(rows, path_field))

        self.assertEqual(fixture["revision"], compatibility["revision"])
        self.assertEqual(required, set(fixture["required_paths"]))

    def test_full_datasets_load_from_pinned_tree_fixture(self) -> None:
        paths = list(load_tree_fixture()["required_paths"])
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            populate_tree(repo_root, paths)
            self.assertEqual(len(load_verifiers_30(repo_root)), 30)
            self.assertEqual(len(load_planning_tasks(repo_root)), 10)

    def test_preflight_aggregates_missing_paths_in_stable_order(self) -> None:
        paths = list(load_tree_fixture()["required_paths"])
        missing = ["docs/evaluation.md", "tests/test_eval_cli.py", "verifiers/scripts/eval.py"]
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            populate_tree(repo_root, [path for path in paths if path not in missing])
            with self.assertRaises(DatasetCompatibilityError) as caught:
                load_planning_tasks(repo_root)

        message = str(caught.exception)
        self.assertIn("Missing 3 required file(s):", message)
        self.assertLess(message.index(missing[0]), message.index(missing[1]))
        self.assertLess(message.index(missing[1]), message.index(missing[2]))
        self.assertIn("482e28ffa1f2613325867badaba4707b7c751d28", message)
        self.assertIn("git -C /tmp/nanorlm-verifiers fetch --depth 1 origin", message)

    def test_entrypoints_exit_without_traceback_or_outputs(self) -> None:
        entrypoints = [
            (
                run_verifiers_main,
                [
                    "run_verifiers.py",
                    "--repo-root",
                    "/missing/verifiers",
                    "--output-dir",
                    "verifiers-output",
                ],
            ),
            (
                run_planning_main,
                [
                    "run_planning.py",
                    "--repo-root",
                    "/missing/verifiers",
                    "--output-dir",
                    "planning-output",
                ],
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            for entrypoint, argv in entrypoints:
                stderr = io.StringIO()
                output_dir = Path(tmpdir) / argv[-1]
                argv[-1] = str(output_dir)
                with (
                    self.subTest(entrypoint=argv[0]),
                    patch.object(sys, "argv", argv),
                    contextlib.redirect_stderr(stderr),
                    self.assertRaises(SystemExit) as caught,
                ):
                    entrypoint()
                self.assertEqual(caught.exception.code, 2)
                self.assertIn("compatibility preflight failed", stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())
                self.assertFalse(output_dir.exists())

    def test_report_metadata_handles_matching_mismatch_and_non_git_roots(self) -> None:
        compatibility = load_verifiers_compatibility()
        with patch("bench._git_revision", return_value=compatibility["revision"]):
            matching = verifiers_report_metadata("/tmp/pinned")["source_repository"]
        self.assertEqual(matching["revision"], compatibility["revision"])
        self.assertTrue(matching["matches_compatibility_revision"])

        with patch("bench._git_revision", return_value="a" * 40):
            mismatched = verifiers_report_metadata("/tmp/current")["source_repository"]
        self.assertFalse(mismatched["matches_compatibility_revision"])

        with tempfile.TemporaryDirectory() as tmpdir:
            non_git = verifiers_report_metadata(tmpdir)["source_repository"]
        self.assertIsNone(non_git["revision"])
        self.assertIsNone(non_git["matches_compatibility_revision"])


if __name__ == "__main__":
    unittest.main()
