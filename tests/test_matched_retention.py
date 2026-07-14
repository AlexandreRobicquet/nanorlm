from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bench import BenchmarkExample
from nanorlm import ContextBlock
from scripts.run_matched_retention import (
    DatasetSpec,
    MATCHED_POLICIES,
    build_parser,
    budget_diagnostics,
    commit_binding,
    conservative_cost_upper_bound,
    copy_and_validate_offline_manifest,
    copy_and_validate_preflight_manifest,
    copy_dataset_sources,
    copy_learned_training_bundle,
    execute,
    hosted_family_audit,
    parse_dataset_spec,
    parse_expected_dataset_hashes,
    prepare_response_cache,
    release_audit,
    reproduction_argv_template,
    response_cache_namespace,
    run_budget,
    sha256_file,
    snapshot_response_cache,
    validate_dataset_hashes,
    validate_phase_configuration,
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

    def test_real_model_dataset_hashes_are_required_and_verified(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Path(tmpdir) / "pilot.jsonl"
            dataset.write_text('{"name":"case"}\n')
            spec = parse_dataset_spec(f"ruler:external_jsonl:{dataset}")
            digest = sha256_file(dataset)

            expected = parse_expected_dataset_hashes([f"ruler={digest}"])
            self.assertEqual(
                validate_dataset_hashes([spec], expected, required=True),
                {"ruler": digest},
            )
            with self.assertRaisesRegex(ValueError, "requires an expected SHA-256"):
                validate_dataset_hashes([spec], {}, required=True)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                validate_dataset_hashes([spec], {"ruler": "0" * 64}, required=True)

    def test_reproduction_template_preserves_original_dataset_hashes(self) -> None:
        source_hash = "a" * 64
        args = SimpleNamespace(
            phase="pilot",
            preflight_only=True,
            limit=8,
            start_index=0,
            seed=0,
            depth=3,
            max_output_tokens=512,
            provider="openai_compatible",
            model="gpt-5.4-mini",
            base_url="",
            learned_retention_model="model.json",
            learned_retention_training_manifest="training.json",
            offline_manifest="offline.json",
            expected_offline_sha256="b" * 64,
            preflight_manifest="",
            expected_preflight_sha256="",
            cache_dir="/tmp/bound-cache",
            max_estimated_cost=5.0,
            expected_nanorlm_commit="c" * 40,
            expected_loom_commit="d" * 40,
        )
        argv = reproduction_argv_template(
            args,
            [DatasetSpec("ruler", "external_jsonl", Path("/tmp/source-ruler.jsonl"))],
            [96],
            {"ruler": source_hash},
        )

        self.assertIn("ruler:external_jsonl:<source-dataset>/ruler.jsonl", argv)
        self.assertIn(f"ruler={source_hash}", argv)
        self.assertIn("<bound-response-cache-dir>", argv)
        self.assertFalse(any("<bundle>/datasets/" in value for value in argv))

    def test_response_cache_is_exact_bound_and_snapshotted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache = root / "cache"
            binding = {
                "schema_version": "nanorlm-response-cache-binding-v1",
                "nanorlm_commit": "a" * 40,
                "task_manifest_sha256": "b" * 64,
            }
            empty = prepare_response_cache(cache, binding)
            self.assertEqual(empty["record_count"], 0)

            namespace = response_cache_namespace(binding)
            record = {
                "cache_key": "c" * 64,
                "provider": "openai_compatible",
                "model": "gpt-5.4-mini",
                "cache_namespace": namespace,
                "created_at": 1.0,
                "request": {
                    "messages": [{"role": "user", "content": "hello"}],
                    "temperature": 0.0,
                    "max_completion_tokens": 512,
                },
                "response": {
                    "content": "answer",
                    "model": "gpt-5.4-mini-2026-07-01",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "calls": 1},
                },
            }
            (cache / f"{'c' * 64}.json").write_text(json.dumps(record))

            populated = prepare_response_cache(cache, binding)
            self.assertEqual(populated["record_count"], 1)
            self.assertEqual(populated["logical_calls"], 1)
            self.assertEqual(populated["response_models"], ["gpt-5.4-mini-2026-07-01"])

            copied = snapshot_response_cache(cache, root / "bundle", binding)
            self.assertEqual(copied, populated)
            self.assertTrue((root / "bundle" / "artifacts" / "response_cache" / "binding.json").is_file())
            with self.assertRaisesRegex(ValueError, "does not match"):
                prepare_response_cache(cache, {**binding, "nanorlm_commit": "d" * 40})

    def test_response_cache_rejects_unexpected_or_credential_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir) / "cache"
            binding = {"schema_version": "nanorlm-response-cache-binding-v1"}
            prepare_response_cache(cache, binding)
            (cache / "notes.txt").write_text("not a cache record")
            with self.assertRaisesRegex(ValueError, "unexpected response-cache entry"):
                prepare_response_cache(cache, binding)

            (cache / "notes.txt").unlink()
            namespace = response_cache_namespace(binding)
            record = {
                "cache_key": "e" * 64,
                "provider": "openai_compatible",
                "cache_namespace": namespace,
                "request": {"messages": [{"role": "user", "content": "api_key"}]},
                "response": {
                    "content": "answer",
                    "model": "dated-model",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "calls": 1},
                },
            }
            (cache / f"{'e' * 64}.json").write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "credential material"):
                prepare_response_cache(cache, binding)

    def test_response_cache_rejects_dangling_binding_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache = root / "cache"
            cache.mkdir()
            (cache / "binding.json").symlink_to(root / "missing-binding.json")

            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                prepare_response_cache(
                    cache,
                    {"schema_version": "nanorlm-response-cache-binding-v1"},
                )

    def test_hosted_budget_preserves_usage_from_bound_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            spec = DatasetSpec("ruler", "external_jsonl", root / "ruler.jsonl")
            example = BenchmarkExample(
                name="case-1",
                query="What?",
                context=[ContextBlock(name="context.txt", text="alpha")],
                answer="alpha",
                must_contain=["alpha"],
            )
            observed_kwargs = []

            def fake_run_dataset(_examples, policy, **kwargs):
                observed_kwargs.append(kwargs)
                row = result_row(
                    policy,
                    selected_pointer="root.1" if policy == "pairwise_tournament" else "root.0",
                )
                row["dataset"] = "ruler"
                row["cost_estimate"] = 0.001
                row["retention_stats"]["response_model_identifiers"] = [
                    "gpt-5.4-mini-2026-07-01"
                ]
                return {"completed": True, "results": [row]}

            with (
                patch("scripts.run_matched_retention.run_dataset", side_effect=fake_run_dataset),
                patch("scripts.run_matched_retention.curves_from_summaries", return_value={}),
                patch("scripts.run_matched_retention.write_report_bundle"),
            ):
                result = run_budget(
                    phase="pilot",
                    specs=[spec],
                    examples={"ruler": [example]},
                    budget=96,
                    budget_root=root / "budget-096",
                    provider="openai_compatible",
                    model="gpt-5.4-mini",
                    base_url=None,
                    learned_model=root / "model.json",
                    seed=0,
                    depth=3,
                    max_output_tokens=512,
                    max_estimated_cost=5.0,
                    response_cache_dir=root / "cache",
                    response_cache_namespace_value="f" * 64,
                )

            self.assertTrue(result["diagnostics"]["eligible"])
            self.assertEqual(len(observed_kwargs), len(MATCHED_POLICIES))
            self.assertTrue(all(item["cache_preserve_usage"] for item in observed_kwargs))
            self.assertTrue(all(item["cache_namespace"] == "f" * 64 for item in observed_kwargs))

    def test_hosted_execution_rejects_missing_preflight_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ruler = root / "ruler.jsonl"
            babilong = root / "babilong.jsonl"
            ruler.write_text("")
            babilong.write_text("")
            output = root / "output"
            args = build_parser().parse_args(
                [
                    "--phase",
                    "pilot",
                    "--dataset-spec",
                    f"ruler:external_jsonl:{ruler}",
                    "--dataset-spec",
                    f"babilong:external_jsonl:{babilong}",
                    "--expected-dataset-sha256",
                    f"ruler={sha256_file(ruler)}",
                    "--expected-dataset-sha256",
                    f"babilong={sha256_file(babilong)}",
                    "--limit",
                    "8",
                    "--budgets",
                    "96",
                    "--provider",
                    "openai_compatible",
                    "--model",
                    "gpt-5.4-mini",
                    "--max-estimated-cost",
                    "5",
                    "--learned-retention-model",
                    "model.json",
                    "--learned-retention-training-manifest",
                    "training.json",
                    "--offline-manifest",
                    "offline.json",
                    "--expected-offline-sha256",
                    "a" * 64,
                    "--cache-dir",
                    str(root / "cache"),
                    "--output-dir",
                    str(output),
                ]
            )

            with self.assertRaisesRegex(ValueError, "passed preflight manifest"):
                execute(args)

            self.assertFalse(output.exists())

    def test_embedded_external_dataset_scrubs_only_local_path_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "name": "case",
                        "query": "Where?",
                        "answer": "there",
                        "context": [{"name": "context.txt", "text": "keep /tmp in task text"}],
                        "metadata": {
                            "source_path": "/workspace/private/raw.jsonl",
                            "dataset_path": "C:\\private\\dataset.jsonl",
                        },
                    }
                )
                + "\n"
            )
            spec = DatasetSpec("pilot", "external_jsonl", source)

            records = copy_dataset_sources(root / "bundle", [spec])
            embedded = root / "bundle" / records[0]["path"]
            payload = json.loads(embedded.read_text())

            self.assertEqual(payload["metadata"]["source_path"], "<portable-source>/raw.jsonl")
            self.assertEqual(
                payload["metadata"]["dataset_path"],
                "<portable-source>/dataset.jsonl",
            )
            self.assertEqual(payload["context"][0]["text"], "keep /tmp in task text")
            self.assertNotEqual(records[0]["source_sha256"], records[0]["sha256"])

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

    def test_real_model_gate_requires_one_returned_model_identifier_per_row(self) -> None:
        rows = [
            result_row(
                policy,
                selected_pointer="root.1" if policy == "pairwise_tournament" else "root.0",
            )
            for policy in MATCHED_POLICIES
        ]

        missing = budget_diagnostics(
            rows,
            budget=96,
            expected_tasks=1,
            require_response_model_identifier=True,
            configured_model_alias="gpt-5.4-mini",
        )
        self.assertFalse(missing["eligible"])
        self.assertEqual(len(missing["response_model_identifier_violations"]), len(MATCHED_POLICIES))

        for row in rows:
            row["retention_stats"]["response_model_identifiers"] = ["gpt-5.4-mini"]
        alias_only = budget_diagnostics(
            rows,
            budget=96,
            expected_tasks=1,
            require_response_model_identifier=True,
            configured_model_alias="gpt-5.4-mini",
        )
        self.assertFalse(alias_only["eligible"])
        self.assertEqual(
            len(alias_only["response_model_identifier_violations"]),
            len(MATCHED_POLICIES),
        )

        for row in rows:
            row["retention_stats"]["response_model_identifiers"] = [
                "gpt-5.4-mini-2026-07-01"
            ]
        bound = budget_diagnostics(
            rows,
            budget=96,
            expected_tasks=1,
            require_response_model_identifier=True,
            configured_model_alias="gpt-5.4-mini",
        )
        self.assertTrue(bound["eligible"])
        self.assertEqual(
            bound["observed_response_model_identifiers"],
            ["gpt-5.4-mini-2026-07-01"],
        )

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

    def test_pilot_binds_passed_offline_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            offline_commit = "a" * 40
            loom_commit = "b" * 40
            model_hash = "c" * 64
            training_hash = "d" * 64
            payload = {
                "phase": "offline",
                "status": "passed",
                "selected_budget": 96,
                "gate_checks": {"complete": True},
                "release_audit": {"ok": True},
                "repositories": {
                    "nanorlm": {"commit": offline_commit},
                    "loom": {"commit": loom_commit},
                },
                "configuration": {
                    "learned_model_sha256": model_hash,
                    "learned_training_manifest": {
                        "sha256": training_hash,
                        "validation": {"training_repository_commit": offline_commit},
                    },
                },
                "task_manifest": {"sha256": "e" * 64},
                "checksums": "checksums.txt",
            }
            source_root = root / "offline"
            source_root.mkdir()
            task_artifact = source_root / "task_manifest.json"
            task_artifact.write_text('{"tasks": []}\n')
            payload["artifact_inventory"] = [
                {
                    "path": "task_manifest.json",
                    "bytes": task_artifact.stat().st_size,
                    "sha256": sha256_file(task_artifact),
                }
            ]
            source = source_root / "manifest.json"
            source.write_text(json.dumps(payload, sort_keys=True))
            (source_root / "release_audit.json").write_text(
                json.dumps(payload["release_audit"], sort_keys=True)
            )
            checksum_path = source_root / "checksums.txt"
            checksum_path.write_text(
                "\n".join(
                    f"{sha256_file(path)}  {path.relative_to(source_root).as_posix()}"
                    for path in sorted(source_root.rglob("*"))
                    if path.is_file() and path != checksum_path
                )
                + "\n"
            )

            evidence = copy_and_validate_offline_manifest(
                source,
                root / "pilot",
                expected_manifest_sha256=sha256_file(source),
                expected_budget=96,
                expected_nanorlm_commit=offline_commit,
                expected_loom_commit=loom_commit,
                learned_model_sha256=model_hash,
                learned_training_manifest_sha256=training_hash,
            )

            self.assertTrue(evidence["ok"])
            self.assertEqual(evidence["offline_nanorlm_commit"], offline_commit)
            self.assertTrue((root / "pilot" / evidence["path"]).is_file())

            with self.assertRaisesRegex(ValueError, "offline manifest SHA-256 mismatch"):
                copy_and_validate_offline_manifest(
                    source,
                    root / "wrong-hash",
                    expected_manifest_sha256="0" * 64,
                    expected_budget=96,
                    expected_nanorlm_commit=offline_commit,
                    expected_loom_commit=loom_commit,
                    learned_model_sha256=model_hash,
                    learned_training_manifest_sha256=training_hash,
                )

            with self.assertRaisesRegex(ValueError, "different memory budget"):
                copy_and_validate_offline_manifest(
                    source,
                    root / "rejected",
                    expected_manifest_sha256=sha256_file(source),
                    expected_budget=128,
                    expected_nanorlm_commit=offline_commit,
                    expected_loom_commit=loom_commit,
                    learned_model_sha256=model_hash,
                    learned_training_manifest_sha256=training_hash,
                )

            task_artifact.write_text("tampered\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                copy_and_validate_offline_manifest(
                    source,
                    root / "tampered",
                    expected_manifest_sha256=sha256_file(source),
                    expected_budget=96,
                    expected_nanorlm_commit=offline_commit,
                    expected_loom_commit=loom_commit,
                    learned_model_sha256=model_hash,
                    learned_training_manifest_sha256=training_hash,
                )

            task_artifact.write_text('{"tasks": []}\n')
            with self.assertRaisesRegex(ValueError, "different nanoRLM commit"):
                copy_and_validate_offline_manifest(
                    source,
                    root / "wrong-commit",
                    expected_manifest_sha256=sha256_file(source),
                    expected_budget=96,
                    expected_nanorlm_commit="f" * 40,
                    expected_loom_commit=loom_commit,
                    learned_model_sha256=model_hash,
                    learned_training_manifest_sha256=training_hash,
                )

    def test_hosted_run_binds_passed_same_commit_preflight_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "preflight"
            source.mkdir()
            nanorlm_commit = "a" * 40
            loom_commit = "b" * 40
            offline_hash = "c" * 64
            task_hash = "d" * 64
            configuration = {
                "provider": "openai_compatible",
                "model": "gpt-5.4-mini",
                "budget": 96,
            }
            datasets = [
                {
                    "label": "ruler",
                    "dataset": "external_jsonl",
                    "embedded": True,
                    "path": "datasets/ruler.jsonl",
                    "source_sha256": "e" * 64,
                    "sha256": "f" * 64,
                    "normalization": "canonical_json_and_portable_local_path_metadata_v1",
                }
            ]
            dataset_path = source / "datasets" / "ruler.jsonl"
            dataset_path.parent.mkdir()
            dataset_path.write_text('{"name": "case"}\n')
            task_path = source / "task_manifest.json"
            task_path.write_text('{"tasks": []}\n')
            prior_path = source / "prior_evidence" / "offline_manifest.json"
            prior_path.parent.mkdir()
            prior_path.write_text('{"status": "passed"}\n')
            inventory_paths = [dataset_path, task_path, prior_path]
            payload = {
                "phase": "pilot_preflight",
                "requested_phase": "pilot",
                "preflight_only": True,
                "network_calls_issued": 0,
                "status": "passed",
                "gate_checks": {"complete": True},
                "release_audit": {"ok": True},
                "repositories": {
                    "nanorlm": {"commit": nanorlm_commit},
                    "loom": {"commit": loom_commit},
                },
                "selected_budget": 96,
                "task_manifest": {"sha256": task_hash},
                "configuration": configuration,
                "datasets": datasets,
                "prior_offline_evidence": {"ok": True, "sha256": offline_hash},
                "artifact_inventory": [
                    {
                        "path": path.relative_to(source).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                    for path in inventory_paths
                ],
                "checksums": "checksums.txt",
            }
            manifest = source / "manifest.json"
            manifest.write_text(json.dumps(payload, sort_keys=True))
            (source / "release_audit.json").write_text(
                json.dumps(payload["release_audit"], sort_keys=True)
            )
            checksum_path = source / "checksums.txt"
            checksum_path.write_text(
                "\n".join(
                    f"{sha256_file(path)}  {path.relative_to(source).as_posix()}"
                    for path in sorted(source.rglob("*"))
                    if path.is_file() and path != checksum_path
                )
                + "\n"
            )

            evidence = copy_and_validate_preflight_manifest(
                manifest,
                root / "actual",
                expected_manifest_sha256=sha256_file(manifest),
                phase="pilot",
                expected_nanorlm_commit=nanorlm_commit,
                expected_loom_commit=loom_commit,
                expected_budget=96,
                expected_task_manifest_sha256=task_hash,
                expected_configuration=configuration,
                expected_datasets=datasets,
                expected_offline_manifest_sha256=offline_hash,
            )

            self.assertTrue(evidence["ok"])
            self.assertEqual(evidence["checksum_index"]["verified_files"], 5)
            self.assertTrue(evidence["checksum_index"]["complete_coverage"])
            self.assertTrue((root / "actual" / evidence["path"]).is_file())

            complete_checksums = checksum_path.read_text()
            checksum_path.write_text(
                "\n".join(
                    line
                    for line in complete_checksums.splitlines()
                    if line.endswith("manifest.json") or line.endswith("release_audit.json")
                )
                + "\n"
            )
            with self.assertRaisesRegex(ValueError, "does not cover the release bundle"):
                copy_and_validate_preflight_manifest(
                    manifest,
                    root / "incomplete",
                    expected_manifest_sha256=sha256_file(manifest),
                    phase="pilot",
                    expected_nanorlm_commit=nanorlm_commit,
                    expected_loom_commit=loom_commit,
                    expected_budget=96,
                    expected_task_manifest_sha256=task_hash,
                    expected_configuration=configuration,
                    expected_datasets=datasets,
                    expected_offline_manifest_sha256=offline_hash,
                )
            checksum_path.write_text(complete_checksums)

            manifest.write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                copy_and_validate_preflight_manifest(
                    manifest,
                    root / "rejected",
                    expected_manifest_sha256=evidence["sha256"],
                    phase="pilot",
                    expected_nanorlm_commit=nanorlm_commit,
                    expected_loom_commit=loom_commit,
                    expected_budget=96,
                    expected_task_manifest_sha256=task_hash,
                    expected_configuration=configuration,
                    expected_datasets=datasets,
                    expected_offline_manifest_sha256=offline_hash,
                )

    def test_hosted_family_audit_requires_declared_benchmark_identity(self) -> None:
        ruler_spec = DatasetSpec("ruler", "external_jsonl", Path("ruler.jsonl"))
        example = BenchmarkExample(
            name="case",
            query="What?",
            context=[ContextBlock(name="context.txt", text="alpha")],
            answer="alpha",
            must_contain=["alpha"],
            metadata={"benchmark": "RULER"},
        )

        self.assertTrue(hosted_family_audit([(ruler_spec, example)])["ok"])
        example.metadata["benchmark"] = "Other"
        self.assertFalse(hosted_family_audit([(ruler_spec, example)])["ok"])

    def test_pilot_configuration_and_cost_reservation_are_frozen(self) -> None:
        args = SimpleNamespace(
            phase="pilot",
            provider="openai_compatible",
            model="gpt-5.4-mini",
            base_url="",
            depth=3,
            max_output_tokens=512,
            seed=0,
            start_index=0,
            limit=8,
            max_estimated_cost=5.0,
            cache_dir="/tmp/bound-cache",
        )
        specs = [
            DatasetSpec("ruler", "external_jsonl", Path("ruler.jsonl")),
            DatasetSpec("babilong", "external_jsonl", Path("babilong.jsonl")),
        ]
        validate_phase_configuration(args, specs, [96])
        with self.assertRaisesRegex(ValueError, "frozen 96-token budget"):
            validate_phase_configuration(args, specs, [128])
        with self.assertRaisesRegex(ValueError, "ordered ruler and babilong"):
            validate_phase_configuration(
                args,
                [
                    DatasetSpec("foo", "external_jsonl", Path("ruler.jsonl")),
                    DatasetSpec("bar", "external_jsonl", Path("babilong.jsonl")),
                ],
                [96],
            )

        example = BenchmarkExample(
            name="case",
            query="What is the code?",
            context=[ContextBlock(name="context.txt", text="word " * 3000)],
            answer="alpha",
            must_contain=["alpha"],
        )
        reservation = conservative_cost_upper_bound(
            [(specs[0], example)],
            provider="openai_compatible",
            model="gpt-5.4-mini",
            base_url=None,
            budget=96,
            depth=3,
            max_output_tokens=512,
        )
        self.assertGreater(reservation["logical_policy_upper_bound_usd"], 0.0)
        self.assertLess(reservation["logical_policy_upper_bound_usd"], 5.0)

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
