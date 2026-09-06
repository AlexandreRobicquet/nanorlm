import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanorlm import HeuristicBackend, RLMConfig, Usage
from repoqa import (MeteredBackend, load_evidence, rank_chunks, run_question, scan_repository,
                    seal_evidence, text_hash, validate_answer)


class FakeAnswerBackend(HeuristicBackend):
    spent = 0.0001
    failed_request = False
    ledger = []
    stage = 'answer'
    def _chat_text(self, system, user):
        import re
        source = re.search(r'\[(s_[0-9a-f]+)\]', user).group(1)
        self.ledger = [{'stage':self.stage,'usage':{'prompt_tokens':100,'completion_tokens':20,'calls':1},'estimated_usd':.0001}]
        return {'content':json.dumps({'claims':[{'text':'The default retry limit is three.', 'citations':[source]}], 'uncertainties':[]}), 'usage':Usage(100,20,1)}
    def response_model_identifiers(self):
        return ['test-model']


class RepoQuestionTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'requires Unix named pipes')
    def test_named_pipe_is_omitted_without_waiting_for_a_writer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            os.mkfifo(root/'events.py')
            (root/'valid.py').write_text('LIMIT = 3\n')
            result = subprocess.run([sys.executable, '-c',
                'import json,sys;from pathlib import Path;from repoqa import scan_repository;'
                'print(json.dumps(scan_repository(Path(sys.argv[1]))))', str(root)],
                capture_output=True, text=True, check=True, timeout=5,
                cwd=Path(__file__).resolve().parents[1])
            scan = json.loads(result.stdout)
            self.assertEqual([file['path'] for file in scan['files']], ['valid.py'])
            self.assertEqual(scan['omitted_files'], [{'path':'events.py','reason':'non_regular_file'}])

    def test_non_utf8_git_filename_does_not_abort_other_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'valid.py').write_text('LIMIT = 3\n')
            real_run = subprocess.run
            def git_output(args, **kwargs):
                if 'ls-files' in args:
                    return real_run([sys.executable, '-c',
                        "import sys;sys.stdout.buffer.write(b'bad\\xff.py\\x00valid.py\\x00')"], **kwargs)
                return real_run(args, **kwargs)
            with patch('repoqa.subprocess.run', side_effect=git_output):
                scan = scan_repository(root)
            self.assertEqual([file['path'] for file in scan['files']], ['valid.py'])
            self.assertEqual(scan['omitted_files'][0]['reason'], 'non_utf8_path')
            self.assertEqual(scan['omitted_files'][0]['path_bytes_hex'], b'bad\xff.py'.hex())
            json.dumps(scan, ensure_ascii=False).encode('utf-8')

    def source(self, root):
        repo=root/'repo';repo.mkdir()
        (repo/'retry.py').write_bytes(b'MAX_RETRIES = 3\r\n# override with retry_limit\r\ndef retry(retry_limit=MAX_RETRIES):\r\n    return retry_limit\r\n')
        (repo/'test_retry.py').write_text('from retry import retry\ndef test_override():\n    assert retry(5) == 5\n')
        return repo

    def test_exact_spans_and_source_hashes_preserve_crlf_unicode_and_long_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);repo=self.source(root)
            (repo/'unicode.py').write_text('# '+'文'*8000+'\n')
            scan=scan_repository(repo)
            for span in scan['chunks']:
                text=(repo/span['path']).read_bytes().decode()
                self.assertEqual(text[span['char_start']:span['char_end']],span['text'])
                self.assertEqual(text_hash(text),span['source_sha256'])
                self.assertEqual(text_hash(span['text']),span['text_sha256'])
                self.assertLessEqual(span['estimated_tokens'],1024)
            selected=rank_chunks('retry limit override tests',scan['chunks'])
            self.assertIn(selected[0]['path'],{'retry.py','test_retry.py'})

    def test_no_network_preview_skips_sensitive_generated_and_symlink_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);repo=self.source(root)
            (repo/'.env').write_text('secret')
            (repo/'secret.py').write_text('sk-'+'x'*30)
            (repo/'escape.py').symlink_to(root/'outside.py')
            (root/'outside.py').write_text('secret')
            (repo/'node_modules').mkdir();(repo/'node_modules/bad.py').write_text('not source')
            with patch('urllib.request.urlopen',side_effect=AssertionError('network')):
                run=run_question(repository=repo,question='retry limit',output=root/'answer',model='gpt-4.1-mini',preview=True)
            self.assertEqual(run['estimated_usd'],0)
            evidence=load_evidence(root/'answer/evidence.json')
            self.assertEqual({span['path'] for span in evidence['spans']},{'retry.py','test_retry.py'})
            self.assertTrue({'sensitive_filename','secret_pattern','symlink'} <= {r['reason'] for r in evidence['omitted_files']})
            run_question(repository=None,question='retry limit',output=root/'reuse',evidence_in=root/'answer/evidence.json')
            self.assertEqual((root/'answer/evidence.json').read_bytes(),(root/'reuse/evidence.json').read_bytes())

    def test_git_scope_excludes_untracked_files_but_uses_working_tree_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo=self.source(Path(tmp))
            subprocess.run(['git','init','-q',str(repo)],check=True)
            subprocess.run(['git','-C',str(repo),'add','retry.py'],check=True)
            (repo/'retry.py').write_text('MAX_RETRIES = 9\n')
            scan=scan_repository(repo)
            self.assertEqual([f['path'] for f in scan['files']],['retry.py'])
            self.assertEqual(scan['chunks'][0]['text'],'MAX_RETRIES = 9\n')
            self.assertFalse(scan['repository']['working_tree_clean'])

    def test_retention_preview_exposes_every_source_sent_to_inspection(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);repo=self.source(root)
            for index in range(8):
                (repo/f'context_{index}.py').write_text('# retry limit '+('configuration '*30)+'\n')
            args={'repository':repo,'question':'retry limit','strategy':'retention',
                  'model':'gpt-4.1-mini','context_budget':250,'candidate_budget':2000}
            with patch('urllib.request.urlopen',side_effect=AssertionError('network')):
                run_question(**args,output=root/'preview',preview=True)
            preview=load_evidence(root/'preview/evidence.json')
            inspected=[]
            def completion(_question, context):
                inspected.extend(block.name for block in context)
                return SimpleNamespace(kept_items=[],completed=True,stop_reasons=[],
                                       retention_stats={},trace=SimpleNamespace(jsonl=''))
            with (patch('repoqa.MeteredBackend',return_value=FakeAnswerBackend()),
                  patch('repoqa.resolved_api_key',return_value='test'),
                  patch('repoqa.RLM') as engine):
                engine.return_value.completion.side_effect=completion
                run_question(**args,output=root/'actual')
            self.assertEqual(preview['stage'],'candidates')
            self.assertEqual([span['id'] for span in preview['spans']],inspected)
            self.assertGreater(sum(span['estimated_tokens'] for span in preview['spans']),250)

    def test_citations_and_evidence_fail_closed(self):
        with self.assertRaisesRegex(ValueError,'citations'):
            validate_answer({'claims':[{'text':'unsupported','citations':['invented']}],'uncertainties':[]},[])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);repo=self.source(root)
            run_question(repository=repo,question='retry',output=root/'out')
            path=root/'out/evidence.json';payload=json.loads(path.read_text());payload['spans'][0]['text']='tampered';path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError,'checksum'):
                load_evidence(path)
            with self.assertRaisesRegex(ValueError,'empty'):
                run_question(repository=repo,question='retry',output=root/'out')

    def test_full_context_refuses_truncation_and_saves_failure_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);repo=self.source(root)
            with self.assertRaisesRegex(ValueError,'refusing to truncate'):
                run_question(repository=repo,question='retry',output=root/'out',strategy='full',context_budget=1)
            run=json.loads((root/'out/run.json').read_text())
            self.assertEqual(run['status'],'failed')
            self.assertEqual(run['estimated_usd'],0)

    def test_model_answer_and_retention_produce_cited_bundle(self):
        for strategy in ['lexical','full','retention']:
            with self.subTest(strategy=strategy), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);repo=self.source(root)
                with patch('repoqa.MeteredBackend',return_value=FakeAnswerBackend()),patch('repoqa.resolved_api_key',return_value='test'):
                    run=run_question(repository=repo,question='retry limit',output=root/'out',strategy=strategy,model='gpt-4.1-mini')
                self.assertEqual(run['status'],'answered')
                self.assertIn(r'retry\.py:',(root/'out/answer.md').read_text())
                self.assertEqual(run['response_models'],['test-model'])
                self.assertTrue((root/'out/checksums.json').is_file())
                if strategy=='retention':
                    self.assertTrue((root/'out/retention-trace.jsonl').is_file())

    def test_cost_cap_stops_before_network(self):
        backend=MeteredBackend(RLMConfig(model='gpt-4.1-mini'),0.000001)
        with patch('urllib.request.urlopen',side_effect=AssertionError('network')):
            with self.assertRaisesRegex(ValueError,'cost cap'):
                backend._chat_text('system','source')
        self.assertEqual(backend.spent,0)
        self.assertFalse(backend.failed_request)

    def test_repair_prompt_receives_the_schema_error(self):
        from nanorlm import OpenAICompatibleBackend
        class Backend(OpenAICompatibleBackend):
            prompts = []
            def _chat_text(self, system, user):
                self.prompts.append(user)
                return {'content': json.dumps({'confidence':90 if len(self.prompts)==1 else .9}), 'usage':Usage(10,2,1)}
        backend=Backend(RLMConfig(model='test'))
        data,usage=backend._chat_json('test','probability', 'source', required_keys=['confidence'])
        self.assertEqual(data['confidence'],.9)
        self.assertIn('finite number in [0, 1]',backend.prompts[1])
        self.assertIn('Original output instructions: probability',backend.prompts[1])
        self.assertEqual(usage.calls,2)

    def test_every_retention_policy_preserves_citable_source_spans(self):
        for policy in ['keep_recent','summary_only','single_critic_topk','pairwise_tournament','learned_retention']:
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);repo=self.source(root)
                with patch('repoqa.MeteredBackend',return_value=FakeAnswerBackend()),patch('repoqa.resolved_api_key',return_value='test'):
                    result=run_question(repository=repo,question='retry limit',output=root/'out',strategy='retention',model='gpt-4.1-mini',retention_policy=policy)
                self.assertEqual(result['status'],'answered')
                self.assertTrue(load_evidence(root/'out/evidence.json')['spans'])

    def test_markdown_claims_cannot_escape_their_citation(self):
        from repoqa import render_answer
        evidence={'spans':[{'id':'s_abc','path':'test.py','line_start':1,'line_end':2}],
                  'coverage':{'selected_spans':1,'scanned_spans':1,'selected_files':1,'scanned_files':1},'omitted_files':[]}
        answer={'claims':[{'text':'Supported claim\n- Uncited claim [link](https://example.com) <script>', 'citations':['s_abc']}], 'uncertainties':[]}
        text=render_answer('question',answer,evidence,{'status':'answered','latency_ms':1})
        self.assertEqual(sum(line.startswith('- ') for line in text.splitlines()),1)
        self.assertNotIn('<script>',text)
        self.assertNotIn('[link](https://example.com)',text)
        self.assertIn('sources.md#s_abc',text)
