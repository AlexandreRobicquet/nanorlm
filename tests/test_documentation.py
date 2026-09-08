from __future__ import annotations

import re
import unittest
from pathlib import Path
from typing import Iterator


ROOT = Path(__file__).resolve().parents[1]
INSTALL_URL = "https://docs.astral.sh/uv/getting-started/installation/"
SETUP_DOCS = [
    ROOT / "README.md",
    ROOT / "docs" / "development.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "showcases" / "README.md",
]
COMMAND_DOCS = [
    *SETUP_DOCS,
    ROOT / "docs" / "engine.md",
    ROOT / "docs" / "experiments.md",
    ROOT / "docs" / "repository-questions.md",
    ROOT / "examples" / "benchmark_snapshot.md",
    ROOT / "examples" / "real_runs" / "openai_external_mini" / "benchmark_snapshot.md",
    ROOT / "examples" / "real_runs" / "openai_ruler_small" / "benchmark_snapshot.md",
]
BASH_BLOCK_RE = re.compile(
    r"^```(?:bash|sh)\n(?P<body>.*?)^```$",
    flags=re.MULTILINE | re.DOTALL,
)
UV_COMMAND_RE = re.compile(r"\buv (?:sync|run|lock|python|add|build)\b")
PYTHON_COMMAND_RE = re.compile(r"(?:^|\s)python(?:\d+(?:\.\d+)*)?\s")


def logical_shell_commands(document: Path) -> Iterator[str]:
    text = document.read_text(encoding="utf-8")
    for block in BASH_BLOCK_RE.finditer(text):
        command_parts: list[str] = []
        for raw_line in block.group("body").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            continued = line.endswith("\\")
            command_parts.append(line[:-1].rstrip() if continued else line)
            if continued:
                continue
            yield " ".join(command_parts)
            command_parts = []
        if command_parts:
            yield " ".join(command_parts)


class DocumentationConsistencyTests(unittest.TestCase):
    def test_official_uv_installation_precedes_commands(self) -> None:
        for document in SETUP_DOCS:
            with self.subTest(document=document.relative_to(ROOT)):
                text = document.read_text(encoding="utf-8")
                first_command = UV_COMMAND_RE.search(text)
                self.assertIsNotNone(first_command)
                self.assertIn(INSTALL_URL, text)
                self.assertLess(text.index(INSTALL_URL), first_command.start())

    def test_repository_python_commands_run_through_uv(self) -> None:
        failures: list[str] = []
        for document in COMMAND_DOCS:
            for command in logical_shell_commands(document):
                if PYTHON_COMMAND_RE.search(command) and not command.startswith("uv run "):
                    failures.append(f"{document.relative_to(ROOT)}: {command}")

        self.assertEqual(failures, [])

    def test_generated_environment_is_not_a_markdown_link(self) -> None:
        uv_guide = (ROOT / "docs" / "development.md").read_text(encoding="utf-8")
        self.assertNotIn("](.venv/)", uv_guide)
        self.assertIn("`.venv/` is the generated local environment", uv_guide)

    def test_reading_path_and_experiment_limits_remain_discoverable(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        experiments = (ROOT / "docs" / "experiments.md").read_text(encoding="utf-8")

        self.assertIn("## Read the code", readme)
        for expected in [
            "nanorlm/__init__.py",
            "nanorlm/policies.py",
            "docs/engine.md",
            "docs/experiments.md",
            "docs/results.md",
            "outputs/quickstart/dossierbench/experiment_report.md",
        ]:
            self.assertIn(expected, readme)
        self.assertIn("negative_or_inconclusive", experiments)
        self.assertRegex(experiments, r"operational completion\s+only")
        self.assertIn("--real-max-estimated-cost 20", experiments)


if __name__ == "__main__":
    unittest.main()
