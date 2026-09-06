import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from nanorlm import RLMConfig
from repoqa import MeteredBackend, digest
from scripts.batch_repoqa import Transport, submit


class BatchTransportTests(unittest.TestCase):
    def test_queue_limit_preserves_unsubmitted_requests(self):
        requests = {str(index): {'custom_id': str(index), 'method': 'POST', 'url': '/v1/chat/completions',
                    'body': {'model': 'gpt-4.1-mini-2025-04-14', 'max_completion_tokens': 100,
                             'messages': [{'role': 'user', 'content': 'x' * size}]}}
                    for index, size in enumerate([90_000, 90_000, 45_000])}
        encoder = SimpleNamespace(encode=lambda value, **kwargs: range(len(value)), name='test')
        fake = SimpleNamespace(encoding_for_model=lambda model: encoder)
        with tempfile.TemporaryDirectory() as temporary, patch.dict('sys.modules', {'tiktoken': fake}), \
                patch('importlib.metadata.version', return_value='test'), \
                patch('scripts.batch_repoqa.api', side_effect=[{'id': 'file-test'}, {'id': 'batch-test', 'status': 'validating'}]):
            root = Path(temporary)
            submit(root, requests, {}, 'gpt-4.1-mini-2025-04-14', 5)
            sent = [json.loads(line)['custom_id'] for line in (root/'round-01/input.jsonl').read_text().splitlines()]
            self.assertEqual(sent, ['0', '2'])
            self.assertEqual(set(requests), {'0','1','2'})
            self.assertLessEqual(json.loads((root/'round-01/submission.json').read_text())['queued_input_tokens'],140_000)

    def backend(self):
        return MeteredBackend(RLMConfig(model='gpt-4.1-mini-2025-04-14'), .25)

    def test_pending_inspections_defer_the_retention_final_answer(self):
        pending = {}
        transport = Transport('case-01', 'retention', {}, pending)
        backend = self.backend()
        backend.stage = 'inspect'
        result = transport.chat(backend, 'worker', 'source')
        self.assertEqual(json.loads(result['content'])['confidence'], 0.0)
        backend.stage = 'answer'
        transport.chat(backend, 'answer', 'placeholder memory')
        self.assertTrue(transport.missing)
        self.assertEqual(len(pending), 1)
        self.assertEqual(next(iter(pending.values()))['body']['messages'][0]['content'], 'worker')

    def test_repeated_requests_keep_separate_billable_identities(self):
        pending = {}
        transport = Transport('case-01', 'lexical', {}, pending)
        backend = self.backend()
        transport.chat(backend, 'answer', 'source')
        transport.chat(backend, 'answer', 'source')
        self.assertEqual(len(pending), 2)
        self.assertEqual([key.rsplit('.',1)[1] for key in pending], ['1','2'])

    def test_response_binding_and_usage_are_preserved(self):
        pending = {}
        backend = self.backend()
        Transport('case-01', 'lexical', {}, pending).chat(backend, 'answer', 'source')
        cid, request = next(iter(pending.items()))
        response = {'request_sha256': digest(request['body']), 'response': {'response': {
            'status_code': 200, 'body': {'model': 'gpt-4.1-mini-2025-04-14',
                'choices': [{'message': {'content': '{"claims":[],"uncertainties":[]}'}}],
                'usage': {'prompt_tokens': 123, 'completion_tokens': 45}}}}}
        transport = Transport('case-01', 'lexical', {cid: response}, {})
        result = transport.chat(backend, 'answer', 'source')
        self.assertEqual((result['usage'].prompt_tokens,result['usage'].completion_tokens,result['usage'].calls),(123,45,1))
        self.assertFalse(transport.missing)
        response['request_sha256'] = 'tampered'
        with self.assertRaisesRegex(ValueError, 'response/request mismatch'):
            Transport('case-01','lexical',{cid:response},{}).chat(backend,'answer','source')
