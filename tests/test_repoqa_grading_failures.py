import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.evaluate_repoqa import file_hash
from scripts.grade_repoqa import grade_experiment


class GradingFailureTests(unittest.TestCase):
    def test_timeout_is_persisted_and_both_failed_or_interrupted_attempts_block_retry(self):
        class TimedOutBackend:
            calls = 0

            def __init__(self, *_args):
                self.spent = 0.0
                self.ledger = []
                self.failed_request = False

            def _chat_text(self, *_args):
                type(self).calls += 1
                self.failed_request = True
                raise TimeoutError('provider may already have accepted the request')

            def response_model_identifiers(self):
                return []

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root/'dataset.json'
            dataset.write_text(json.dumps({'tasks':[
                {'id':'task', 'question':'question', 'expected_facts':['fact'], 'reference_spans':[]}]}))
            experiment = root/'experiment'
            experiment.mkdir()
            (experiment/'experiment.json').write_text(json.dumps({
                'dataset_sha256':file_hash(dataset), 'experiment_sha256':'experiment'}))
            rows = []
            for strategy in ('lexical','full','retention'):
                case = experiment/f'task--{strategy}'
                case.mkdir()
                (case/'run.json').write_text('{}')
                (case/'answer.json').write_text(json.dumps({'claims':[], 'uncertainties':[]}))
                rows.append({'task_id':'task', 'strategy':strategy, 'directory':case.name,
                             'run_sha256':file_hash(case/'run.json')})
            (experiment/'results.json').write_text(json.dumps({'experiment_sha256':'experiment','rows':rows}))
            output = root/'grades'
            with (patch('scripts.grade_repoqa.MeteredBackend', TimedOutBackend),
                  patch('scripts.grade_repoqa.resolved_api_key', return_value='test-only'),
                  patch('scripts.grade_repoqa.verified_receipt', return_value={'status':'answered'}),
                  patch('scripts.grade_repoqa.load_evidence', return_value={'spans':[]})):
                with self.assertRaisesRegex(ValueError, 'recorded'):
                    grade_experiment(dataset, experiment, output)
                receipt_path = output/'task--lexical.json'
                receipt = json.loads(receipt_path.read_text())
                self.assertTrue(receipt['request_failed'])
                self.assertTrue(receipt['failed_request_billing_unknown'])
                self.assertEqual(receipt['usage_ledger'], [])
                with self.assertRaisesRegex(ValueError, 'billing reconciliation'):
                    grade_experiment(dataset, experiment, output)
                self.assertEqual(TimedOutBackend.calls, 1)
                receipt_path.unlink()
                (output/'task--lexical.pending.json').write_text('{}')
                with self.assertRaisesRegex(ValueError, 'interrupted grading request'):
                    grade_experiment(dataset, experiment, output)
                self.assertEqual(TimedOutBackend.calls, 1)
