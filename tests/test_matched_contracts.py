import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bench import build_pairbench
from scripts.run_matched_retention import (DatasetSpec, audit_trace_bindings, budget_diagnostics,
    determinism_check, example_record, git_snapshot, run_budget)
from scripts.train_learned_retention import repository_record


class MatchedContractTests(unittest.TestCase):
    def test_manifest_result_trace_replay_and_determinism_share_task_identity(self):
        spec = DatasetSpec('pairbench', 'pairbench')
        examples = build_pairbench(n=2, seed=0)
        # Display names do not identify source tasks.
        examples[1].name = examples[0].name
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_budget(phase='offline', specs=[spec], examples={spec.label: examples},
                budget=128, budget_root=root, provider='heuristic', model='demo/heuristic', base_url=None,
                learned_model=None, seed=0, depth=3, max_output_tokens=512, max_estimated_cost=None,
                response_cache_dir=None, response_cache_namespace_value='')
            expected = {example_record(spec, i, example)['task_id'] for i, example in enumerate(examples)}
            self.assertEqual({row['task_id'] for row in result['rows']}, expected)
            self.assertEqual(result['diagnostics']['observed_tasks'], 2)
            self.assertTrue(audit_trace_bindings(root, result['rows'])['ok'])
            check = determinism_check(result, first_spec=spec, first_example=examples[0], provider='heuristic',
                model='demo/heuristic', base_url=None, learned_model=None, seed=0, depth=3, max_output_tokens=512)
            self.assertTrue(check['ok'], check)
            path = next(root.glob('reports/*/loom_traces/*/*.jsonl'))
            events = [json.loads(line) for line in path.read_text().splitlines()]
            events[0]['task_id'] = 'wrong-task'
            path.write_text('\n'.join(json.dumps(event) for event in events))
            self.assertFalse(audit_trace_bindings(root, result['rows'])['ok'])
            rows = result['rows']
            rows[0]['completed'] = False
            self.assertFalse(budget_diagnostics(rows, budget=128, expected_tasks=2)['eligible'])
            self.assertTrue(budget_diagnostics(rows, budget=128, expected_tasks=2)['incomplete_tasks'])
            self.assertTrue(budget_diagnostics(rows+[rows[0]], budget=128, expected_tasks=2)['duplicate_policy_rows'])

    def test_git_unavailable_or_status_failure_never_claims_clean(self):
        with patch('subprocess.run', side_effect=FileNotFoundError('git')):
            self.assertFalse(repository_record()['clean'])
            self.assertFalse(git_snapshot(Path('.'))['clean'])
        def fail_status(args, **kwargs):
            return subprocess.CompletedProcess(args, 1 if 'status' in args else 0, stdout='' if 'status' in args else 'a'*40)
        with patch('subprocess.run', side_effect=fail_status):
            self.assertFalse(repository_record()['clean'])
            self.assertFalse(git_snapshot(Path('.'))['clean'])
