from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import check_markdown_links  # noqa: E402


class MarkdownLinkCheckerTests(unittest.TestCase):
    def test_accepts_repo_local_targets_and_skips_non_links(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "docs").mkdir()
            (root / "images").mkdir()
            (root / "docs" / "guide.md").write_text(
                "# Guide\n\n## Details\n",
                encoding="utf-8",
            )
            (root / "images" / "trace.svg").write_text("<svg/>\n", encoding="utf-8")
            (root / ".python-version").write_text("3.11\n", encoding="utf-8")
            (root / "space name.md").write_text("# Encoded\n", encoding="utf-8")
            (root / "a(b).md").write_text("# Parenthesized\n", encoding="utf-8")
            readme = root / "README.md"
            readme.write_text(
                "\n".join(
                    [
                        "# Intro",
                        "[guide](docs/guide.md#details)",
                        "[directory](docs/)",
                        "[dotfile](.python-version)",
                        "![trace](images/trace.svg)",
                        "[encoded](space%20name.md)",
                        "[parenthesized](a(b).md)",
                        "[same document](#intro)",
                        "[external](https://example.com/docs)",
                        "[email](mailto:maintainer@example.com)",
                        "`[inline code](missing-inline.md)`",
                        "```text",
                        "[fenced code](missing-fenced.md)",
                        "```",
                        "[guide reference][guide]",
                        '[guide]: <docs/guide.md#details> "Guide title"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = check_markdown_links.check_markdown_links(root, [readme])

            self.assertEqual(result.issues, ())
            self.assertEqual(result.documents, 1)
            self.assertEqual(result.local_checked, 8)
            self.assertEqual(result.external_skipped, 2)

    def test_reports_missing_same_document_and_cross_document_anchors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target = root / "target.md"
            target.write_text("# Real Heading\n", encoding="utf-8")
            source = root / "README.md"
            source.write_text(
                "# Intro\n\n[same](#missing)\n[other](target.md#also-missing)\n",
                encoding="utf-8",
            )

            result = check_markdown_links.check_markdown_links(
                root,
                [source],
            )

            self.assertEqual(len(result.issues), 2)
            self.assertEqual(
                [issue.reason for issue in result.issues],
                ["missing anchor", "missing anchor"],
            )
            self.assertEqual(
                [issue.resolved for issue in result.issues],
                ["README.md#missing", "target.md#also-missing"],
            )

    def test_aggregates_missing_absolute_and_outside_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "repo"
            root.mkdir()
            source = root / "README.md"
            source.write_text(
                "\n".join(
                    [
                        "[first](missing-one.md)",
                        "![second](images/missing-two.svg)",
                        "[outside](../outside.md)",
                        "[absolute](/tmp/absolute.md)",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = check_markdown_links.check_markdown_links(root, [source])

            self.assertEqual(len(result.issues), 4)
            self.assertEqual(
                [issue.link.line for issue in result.issues],
                [1, 2, 3, 4],
            )
            self.assertEqual(
                [issue.reason for issue in result.issues],
                ["missing", "missing", "outside repository", "absolute local path"],
            )

    def test_tracked_mode_rejects_generated_target_that_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            generated = root / ".venv"
            generated.mkdir()
            (generated / "README.md").write_text("# Generated\n", encoding="utf-8")
            source = root / "README.md"
            source.write_text("[generated](.venv/)\n", encoding="utf-8")

            result = check_markdown_links.check_markdown_links(
                root,
                [source],
                tracked_files=[source],
            )

            self.assertEqual(len(result.issues), 1)
            self.assertEqual(result.issues[0].reason, "untracked or generated")

    def test_repository_links_resolve(self) -> None:
        tracked_files = check_markdown_links.tracked_repository_files(REPO_ROOT)
        documents = check_markdown_links.tracked_markdown_files(
            REPO_ROOT,
            tracked_files,
        )
        result = check_markdown_links.check_markdown_links(
            REPO_ROOT,
            documents,
            tracked_files,
        )

        self.assertGreater(result.documents, 0)
        self.assertGreater(result.local_checked, 0)
        self.assertEqual(result.issues, ())

    def test_main_reports_success(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = check_markdown_links.main()

        self.assertEqual(exit_code, 0)
        self.assertIn("Markdown link check passed:", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
