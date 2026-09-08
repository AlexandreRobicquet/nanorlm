import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nanorlm.bench import build_pairbench
from scripts.run_matched_retention import build_parser, validate_phase_configuration, portable_example
from scripts.run_matched_retention import (DatasetSpec, audit_trace_bindings, budget_diagnostics,
    determinism_check, example_record, git_snapshot, run_budget)
from scripts.train_learned_retention import repository_record


class MatchedContractTests(unittest.TestCase):
    def test_manifest_result_trace_replay_and_determinism_share_task_identity(self):
        spec = DatasetSpec('pairbench', 'pairbench')
        examples = build_pairbench(n=2, seed=0)
        # Display names do not identify source tasks.
        examples[1].name = examples[0].name
        examples[0].context[0].metadata['source_path'] = '/workspace/private/raw.jsonl'
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
            self.assertNotIn('/workspace/private', json.dumps(result['rows']))
            self.assertIn('<portable-source>/raw.jsonl', json.dumps(result['rows']))
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

    def test_offline_configuration_cannot_substitute_easier_evidence(self):
        from dataclasses import replace
        specs = [DatasetSpec('dossierbench','dossierbench'), DatasetSpec('ruler-synthetic','ruler_synthetic'), DatasetSpec('babilong-synthetic','babilong_synthetic')]
        for key, value in {'limit':1, 'start_index':1, 'depth':1, 'max_output_tokens':128,
                           'seed':2, 'model':'other', 'provider':'openai_compatible',
                           'base_url':'http://localhost', 'max_estimated_cost':1}.items():
            args = build_parser().parse_args(['--output-dir','unused'])
            setattr(args,key,value)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError,'frozen development configuration'):
                validate_phase_configuration(args,specs,[96,128,192])
        args = build_parser().parse_args(['--output-dir','unused'])
        with self.assertRaisesRegex(ValueError,'three development families'):
            validate_phase_configuration(args,[DatasetSpec('pairbench','pairbench')],[96,128,192])

    def test_training_configuration_cannot_substitute_datasets_or_optimizer(self):
        from scripts.run_matched_retention import FROZEN_TRAINING, validate_training_configuration
        payload = {'training':dict(FROZEN_TRAINING),'datasets':[
            {'dataset':dataset,'seed':seed,'budget':80 if dataset=='dossierbench' else 90,
             'examples':12,'status':'included','trajectories':12}
            for dataset in FROZEN_TRAINING['datasets'] for seed in [0,1]]}
        validate_training_configuration(payload)
        for key in FROZEN_TRAINING:
            changed = {**payload,'training':{**payload['training'],key:None}}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError,'frozen protocol configuration'):
                validate_training_configuration(changed)
        with self.assertRaisesRegex(ValueError,'frozen dataset slices'):
            validate_training_configuration({**payload,'datasets':payload['datasets'][:-1]})

    def test_all_absolute_metadata_paths_are_portable_without_altering_task_text(self):
        from scripts.run_matched_retention import portable_value, portable_example
        from nanorlm.bench import BenchmarkExample
        from nanorlm import ContextBlock
        metadata = {key:'/workspace/private/raw.jsonl' for key in ['file_path','input_path','working_directory','filePath','cwd','arbitrary_metadata_field']}
        metadata['nested'] = {'list':['/mnt/private/file.txt', r'C:\Users\private\file.txt']}
        value = {'query':'Explain /workspace/private/raw.jsonl','metadata':metadata}
        portable = portable_value(value)
        self.assertEqual(portable['query'],value['query'])
        self.assertNotIn('/workspace/',json.dumps(portable['metadata']))
        self.assertNotIn('/mnt/',json.dumps(portable['metadata']))
        example = BenchmarkExample('case','query',[ContextBlock('block','original text',metadata)],'answer',[])
        sanitized = portable_example(example)
        self.assertEqual(sanitized.context[0].text,'original text')
        self.assertNotIn('/workspace/',json.dumps(sanitized.context[0].metadata))
