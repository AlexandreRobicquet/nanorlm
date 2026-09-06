from __future__ import annotations

import contextlib
import io
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def run_tiny_example() -> tuple[dict[str, object], str]:
    readme = README.read_text(encoding="utf-8")
    heading = "## Tiny Example"
    if heading not in readme:
        raise AssertionError(f"{README.name} is missing {heading!r}")

    section = readme.split(heading, 1)[1].split("\n## ", 1)[0]
    match = re.search(r"^```python\n(?P<source>.*?)^```$", section, flags=re.MULTILINE | re.DOTALL)
    if match is None:
        raise AssertionError(f"{heading!r} is missing its Python code fence")

    namespace: dict[str, object] = {"__name__": "__main__"}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(match.group("source"), "README.md#tiny-example", "exec"), namespace)
    return namespace, stdout.getvalue()


class ReadmeTinyExampleTests(unittest.TestCase):
    def test_tiny_example_recurses_and_retains_the_complete_answer(self) -> None:
        namespace, stdout = run_tiny_example()
        self.assertIn("result", namespace)
        self.assertIn("config", namespace)
        result = namespace["result"]
        config = namespace["config"]

        answer = result.answer.lower()
        self.assertIn("stale endpoint registry cache", answer)
        self.assertIn("reloading the endpoint registry", answer)
        self.assertIn("invalidating the cache", answer)

        self.assertIn("- [split] root split", result.trace.tree)
        self.assertIn("- [split] root.0 split", result.trace.tree)
        self.assertIn("- [inspect] root.0.0 leaf", result.trace.tree)
        self.assertEqual(result.retention_stats["max_memory_depth"], 2)
        self.assertTrue(result.retention_decisions)
        self.assertTrue(
            any(
                not candidate["selected"]
                for decision in result.retention_decisions
                for candidate in decision["candidates"]
            )
        )
        self.assertLessEqual(sum(item.tokens for item in result.kept_items), config.memory_budget_tokens)

        retained = {item.provenance for item in result.kept_items}
        dropped = {item["provenance"] for item in result.drop_reasons}
        self.assertEqual(retained, {"incident-a.txt", "incident-b.txt"})
        self.assertEqual(dropped, {"incident-c.txt", "incident-d.txt"})

        for marker in ("[split]", "retained:", "dropped:", "max memory depth:"):
            self.assertIn(marker, stdout)


if __name__ == "__main__":
    unittest.main()
