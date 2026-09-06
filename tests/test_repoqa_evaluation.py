"""Evidence integrity at the paid experiment boundary."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.evaluate_repoqa import validate_dataset, verified_receipt
from scripts.grade_repoqa import normalize_grade, validate_grade


class EvaluationIntegrityTests(unittest.TestCase):
    def test_grading_distinguishes_missing_contradicted_and_supported_content(self):
        grade = {'facts': [
            {'index':1,'covered_by_answer':True,'contradicted_by_answer':False,'claim_indices':[1],'reason':'Explicit correct value.'},
            {'index':2,'covered_by_answer':False,'contradicted_by_answer':True,'claim_indices':[2],'reason':'Wrong value.'},
            {'index':3,'covered_by_answer':False,'contradicted_by_answer':False,'claim_indices':[],'reason':'Not mentioned.'}],
            'claims': [
                {'index':1,'citation_verdict':'supports','materially_incorrect':False,'reason':'Code agrees.'},
                {'index':2,'citation_verdict':'contradicts','materially_incorrect':True,'reason':'Code disagrees.'}],
            'reference_concern':''}
        normalized = normalize_grade(grade, 3, 2)
        self.assertEqual([row['correct'] for row in normalized['facts']], [True,False,False])
        self.assertEqual([row['supported'] for row in normalized['claims']], [True,False])
        grade['claims'][1]['materially_incorrect'] = False
        normalized = normalize_grade(grade, 3, 2)
        self.assertTrue(normalized['claims'][1]['materially_incorrect'])
        self.assertTrue(normalized['claims'][1]['incorrect_from_reference_contradiction'])
        grade['facts'][0]['claim_indices'] = []
        with self.assertRaisesRegex(ValueError, 'identify candidate claims'):
            normalize_grade(grade, 3, 2)

    def test_malformed_grading_rows_fail_with_a_recordable_validation_error(self):
        with self.assertRaisesRegex(ValueError, 'schema mismatch'):
            validate_grade({'facts':['not an object']}, 1, 0)
        with self.assertRaisesRegex(ValueError, 'coverage flags'):
            normalize_grade({'facts':[None]}, 1, 0)

    def test_grader_cannot_omit_claims_or_use_truthy_strings(self):
        grade = {'facts': [{'index': 1, 'correct': True, 'reason': 'source'}],
                 'claims': [{'index': 1, 'supported': True, 'materially_incorrect': False, 'reason': 'source'}],
                 'reference_concern': ''}
        validate_grade(grade, 1, 1)
        with self.assertRaisesRegex(ValueError, 'count mismatch'):
            validate_grade(grade, 1, 2)
        grade['facts'][0]['correct'] = 'false'
        with self.assertRaisesRegex(ValueError, 'schema mismatch'):
            validate_grade(grade, 1, 1)

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
