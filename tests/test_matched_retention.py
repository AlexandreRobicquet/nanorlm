from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_matched_retention import (
    MATCHED_POLICIES,
    budget_diagnostics,
    commit_binding,
    copy_learned_training_bundle,
    parse_dataset_spec,
    release_audit,
    sha256_file,
    validate_loom_traces,
)


def result_row(policy: str, *, selected_pointer: str, retained_tokens: int = 48) -> dict:
    return {
        "dataset": "fixture",
        "name": "case-1",
        "policy": policy,
        "retained_items": 1,
        "retained_tokens": retained_tokens,
        "stage_budgets": {
            "inspect": {"prompt_tokens": 100, "completion_tokens": 20, "calls": 2, "wall_ms": 1},
            "final_answer": {"prompt_tokens": 40, "completion_tokens": 5, "calls": 1, "wall_ms": 1},
        },
        "retention_stats": {
            "inspection_replay": {"store_sha256": "a" * 64},
        },
        "retention_decisions": [
            {
                "after_tokens": retained_tokens,
                "before_tokens": 192,
                "budget": 96,
                "budget_used": {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "wall_ms": 1},
                "candidates": [
                    {
                        "raw_pointer": "root.0",
                        "provenance": "a.txt",
                        "input_item": {"raw_pointer": "root.0", "provenance": "a.txt"},
                        "selected": selected_pointer == "root.0",
                    },
                    {
                        "raw_pointer": "root.1",
                        "provenance": "b.txt",
                        "input_item": {"raw_pointer": "root.1", "provenance": "b.txt"},
                        "selected": selected_pointer == "root.1",
                    },
                ],
            }
        ],
    }


class MatchedRetentionTests(unittest.TestCase):
    def test_parse_dataset_spec_requires_external_path(self) -> None:
        self.assertEqual(parse_dataset_spec("pairbench:pairbench").dataset, "pairbench")
        with self.assertRaisesRegex(ValueError, "requires a path"):
            parse_dataset_spec("ruler:external_jsonl")
        with self.assertRaisesRegex(ValueError, "filesystem-safe slug"):
            parse_dataset_spec("RULER:pairbench")

    def test_budget_gate_uses_non_accuracy_invariants(self) -> None:
        rows = [
            result_row(
                policy,
                selected_pointer="root.1" if policy == "pairwise_tournament" else "root.0",
            )
            for policy in MATCHED_POLICIES
        ]

        diagnostics = budget_diagnostics(rows, budget=96, expected_tasks=1)

        self.assertTrue(diagnostics["eligible"])
        self.assertEqual(diagnostics["nonempty_rate"], 1.0)
        self.assertEqual(diagnostics["median_max_pre_retention_pressure"], 2.0)
        self.assertEqual(diagnostics["pairwise_difference_rates"]["keep_recent"], 1.0)

    def test_budget_gate_fails_on_oversized_retained_state(self) -> None:
        rows = [
            result_row(policy, selected_pointer="root.0", retained_tokens=97 if policy == "keep_recent" else 48)
            for policy in MATCHED_POLICIES
        ]

        diagnostics = budget_diagnostics(rows, budget=96, expected_tasks=1)

        self.assertFalse(diagnostics["eligible"])
        self.assertEqual(len(diagnostics["budget_violations"]), 2)

    def test_budget_gate_checks_every_retention_decision(self) -> None:
        rows = [result_row(policy, selected_pointer="root.0") for policy in MATCHED_POLICIES]
        rows[0]["retained_tokens"] = 48
        rows[0]["retention_decisions"][0]["after_tokens"] = 97

        diagnostics = budget_diagnostics(rows, budget=96, expected_tasks=1)

        self.assertFalse(diagnostics["eligible"])
        self.assertEqual(diagnostics["budget_violations"], ["fixture:case-1:keep_recent:decision-0"])

    def test_commit_binding_requires_exact_full_sha(self) -> None:
        commit = "a" * 40
        snapshot = {"is_repository": True, "commit": commit}

        self.assertTrue(commit_binding(snapshot, commit)["ok"])
        self.assertEqual(commit_binding(snapshot, commit[:12])["reason"], "expected_commit_not_full_sha")
        self.assertEqual(commit_binding(snapshot, "b" * 40)["reason"], "commit_mismatch")

    def test_loom_validator_records_portable_paths_and_requires_trace_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trace = root / "budget-096" / "reports" / "pairbench" / "loom_traces" / "keep_recent" / "case.jsonl"
            trace.parent.mkdir(parents=True)
            trace.write_text("{}\n")
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps({"path": str(trace), "ok": True, "events": 4, "issues": []}) + "\n",
                stderr="built package at /Users/example/private/loom",
            )
            with patch("scripts.run_matched_retention.subprocess.run", return_value=completed):
                result = validate_loom_traces(root, [trace], expected_count=1)
                missing = validate_loom_traces(root, [trace], expected_count=2)

        self.assertTrue(result["all_valid"])
        self.assertEqual(
            result["traces"][0]["path"],
            "reports/pairbench/loom_traces/keep_recent/case.jsonl",
        )
        self.assertNotIn("/Users/", json.dumps(result))
        self.assertFalse(missing["all_valid"])

    def test_training_bundle_is_hash_bound_and_copied_portably(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            source.mkdir()
            model = source / "learned_retention_model.json"
            examples = source / "training_examples.jsonl"
            traces = source / "training_traces.jsonl"
            model.write_text('{"model": 1}\n')
            examples.write_text('{"example": 1}\n')
            traces.write_text('{"trace": 1}\n')
            artifact_paths = {
                "model": model,
                "training_examples": examples,
                "training_traces": traces,
            }
            manifest_payload = {
                "status": "trained",
                "repository": {"commit": "a" * 40, "clean": True, "status_entries": 0},
                "training": {
                    "source": "offline_trace_training",
                    "training_source": "traces",
                    "objective": "pairwise",
                },
                "artifacts": {
                    name: {
                        "path": path.name,
                        "external": False,
                        "sha256": sha256_file(path),
                    }
                    for name, path in artifact_paths.items()
                },
            }
            manifest = source / "manifest.json"
            manifest.write_text(json.dumps(manifest_payload))

            copied_model, copied_manifest, validation = copy_learned_training_bundle(
                manifest,
                model,
                root / "release",
            )

            self.assertTrue(validation["ok"])
            self.assertEqual(copied_model.read_bytes(), model.read_bytes())
            self.assertTrue(copied_manifest.is_file())
            self.assertNotIn(tmpdir, copied_manifest.read_text())

            model.write_text('{"model": 2}\n')
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                copy_learned_training_bundle(manifest, model, root / "rejected")

    def test_release_audit_hashes_path_or_secret_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "clean.json").write_text('{"value": "portable"}')
            self.assertTrue(release_audit(root)["ok"])

            (root / "bad.txt").write_text("/Users/alexandre/private sk-abcdefghijklmnop")
            audit = release_audit(root)

        self.assertFalse(audit["ok"])
        self.assertEqual({finding["code"] for finding in audit["findings"]}, {"mac_user_path", "openai_style_secret"})
        self.assertTrue(all("alexandre" not in str(finding) for finding in audit["findings"]))


if __name__ == "__main__":
    unittest.main()
