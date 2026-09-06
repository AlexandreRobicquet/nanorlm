"""Evidence integrity at the paid experiment boundary."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.evaluate_repoqa import validate_dataset, verified_receipt


class EvaluationIntegrityTests(unittest.TestCase):
    def test_reference_excerpts_must_match_pinned_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            (parent / 'repo').mkdir()
            (parent / 'repo/code.py').write_text('first\nsecond\n')
            data = {'schema': 'nanorlm-repoqa-eval-v1',
                    'repositories': {'repo': {'commit': 'abc'}},
                    'tasks': [{'id': 'repo-01', 'repository': 'repo', 'reference_spans': [
                        {'path': 'code.py', 'start': 2, 'end': 2, 'text': 'second\n',
                         'file_sha256': hashlib.sha256(b'first\nsecond\n').hexdigest()}]}]}
            with patch('scripts.evaluate_repoqa.git_value', side_effect=lambda root, *args: 'abc' if args[0] == 'rev-parse' else ''):
                validate_dataset(data, parent)
                data['tasks'][0]['reference_spans'][0]['text'] = 'invented\n'
                with self.assertRaisesRegex(ValueError, 'excerpt mismatch'):
                    validate_dataset(data, parent)

    def test_dirty_source_and_duplicate_tasks_are_rejected(self):
        data = {'schema': 'nanorlm-repoqa-eval-v1', 'repositories': {'repo': {'commit': 'abc'}},
                'tasks': [{'id': 'repo-01', 'repository': 'repo', 'reference_spans': []}] * 2}
        with patch('scripts.evaluate_repoqa.git_value', return_value='abc'):
            with self.assertRaisesRegex(ValueError, 'clean pinned'):
                validate_dataset(data, Path('/unused'))
        with patch('scripts.evaluate_repoqa.git_value', side_effect=lambda root, *args: 'abc' if args[0] == 'rev-parse' else ''):
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                validate_dataset(data, Path('/unused'))

    def test_resume_rejects_tampered_receipt_and_foreign_case(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            task = {'id': 'repo-01', 'question': 'where?'}
            (directory / 'binding.json').write_text(json.dumps(
                {'experiment_sha256': 'abc', 'task_id': 'repo-01', 'strategy': 'lexical'}))
            (directory / 'checksums.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'checksum inventory'):
                verified_receipt(directory, task, 'lexical', 'abc')
            with self.assertRaisesRegex(ValueError, 'binding mismatch'):
                verified_receipt(directory, task, 'full', 'abc')
