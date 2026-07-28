from __future__ import annotations

import tomllib
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class RepositoryMetadataTests(unittest.TestCase):
    def test_canonical_mit_license_is_present(self) -> None:
        license_path = REPO_ROOT / "LICENSE"

        self.assertTrue(license_path.is_file(), "repository root must contain LICENSE")
        license_text = license_path.read_text(encoding="utf-8")
        self.assertTrue(license_text.startswith("MIT License\n"))
        self.assertIn("Copyright (c) 2026 Alexandre Robicquet", license_text)
        self.assertIn("Permission is hereby granted, free of charge", license_text)
        self.assertIn('THE SOFTWARE IS PROVIDED "AS IS"', license_text)

    def test_pyproject_links_spdx_license_to_file(self) -> None:
        with (REPO_ROOT / "pyproject.toml").open("rb") as pyproject_file:
            project = tomllib.load(pyproject_file)["project"]

        self.assertEqual(project["license"], "MIT")
        self.assertEqual(project["license-files"], ["LICENSE"])

    def test_pyproject_enforces_clone_only_distribution_boundary(self) -> None:
        with (REPO_ROOT / "pyproject.toml").open("rb") as pyproject_file:
            pyproject = tomllib.load(pyproject_file)

        self.assertFalse(pyproject["tool"]["uv"]["package"])
        setuptools = pyproject["tool"]["setuptools"]
        self.assertEqual(setuptools["packages"], [])
        self.assertEqual(setuptools["py-modules"], [])

    def test_pytest_is_locked_as_a_development_dependency_only(self) -> None:
        with (REPO_ROOT / "pyproject.toml").open("rb") as pyproject_file:
            pyproject = tomllib.load(pyproject_file)

        self.assertEqual(pyproject["project"].get("dependencies", []), [])
        self.assertEqual(
            pyproject["dependency-groups"]["dev"],
            ["pytest>=9.1.1,<9.2"],
        )


if __name__ == "__main__":
    unittest.main()
