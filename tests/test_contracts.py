"""Boundary regressions for real inputs, independent of benchmark winners."""
import hashlib
import json
import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bench
from inspection_replay import InspectionReplayBackend
from nanorlm import ContextBlock, HeuristicBackend, InspectionResult, RLM, RLMConfig, Usage, extract_json_object

POLICIES = ['keep_recent', 'summary_only', 'single_critic_topk', 'pairwise_tournament', 'learned_retention']


class OversizedBackend(HeuristicBackend):
    def inspect(self, query, documents, depth, branch):
        return InspectionResult('evidence ' * 100, [], 'evidence', .9, usage=Usage(1, 1, 1))


class ContractTests(unittest.TestCase):
    def test_root_leaf_and_recursive_memory_respect_all_budgets(self):
        for policy in POLICIES:
            for budget in [0, 10, 80]:
                for context in ['short input', [ContextBlock('a', 'word ' * 100), ContextBlock('b', 'word ' * 100)]]:
                    with self.subTest(policy=policy, budget=budget, context_type=type(context)):
                        result = RLM(RLMConfig(model='demo/heuristic', memory_budget_tokens=budget, max_depth=4, retention_policy=policy), backend=OversizedBackend()).completion('evidence', context)
                        self.assertLessEqual(sum(item.tokens for item in result.kept_items), budget)
                        self.assertTrue(all(step['after_tokens'] <= budget for step in result.retention_decisions))
                        self.assertTrue(result.retention_decisions)

    def test_large_single_document_splits_losslessly_with_source_spans(self):
        text = 'release evidence line\n' * 300
        inspected = []
        class Recorder(HeuristicBackend):
            def inspect(self, query, documents, depth, branch):
                inspected.extend(documents)
                return InspectionResult('release evidence', [], '', .5, usage=Usage(calls=1))
        result = RLM(RLMConfig(model='demo/heuristic', max_depth=12, max_steps=256, max_leaf_tokens=80), backend=Recorder()).completion('release', text)
        self.assertEqual(''.join(block.text for block in inspected), text)
        self.assertTrue(all(block.tokens <= 80 for block in inspected))
        self.assertGreater(len(inspected), 1)
        self.assertTrue(result.completed)
        for block in inspected:
            m = block.metadata
            self.assertEqual(text[m['char_start']:m['char_end']], block.text)
            self.assertEqual(m['source_sha256'], hashlib.sha256(text.encode()).hexdigest())

    def test_exhausted_steps_are_disclosed_without_incrementing_past_limit(self):
        result = RLM(RLMConfig(model='demo/heuristic', max_depth=12, max_steps=3)).completion('release', [ContextBlock(str(i), 'release ' * 100) for i in range(8)])
        self.assertFalse(result.completed)
        self.assertIn('max_steps', result.stop_reasons)
        self.assertLessEqual(result.retention_stats['steps_used'], 3)
        self.assertTrue(result.retention_stats['omitted_blocks'])

    def test_depth_limit_cannot_send_oversized_inspection(self):
        with patch.object(HeuristicBackend, 'inspect', side_effect=AssertionError('oversized request')):
            result = RLM(RLMConfig(model='demo/heuristic', max_depth=0, max_leaf_tokens=32)).completion('word', 'word ' * 1000)
        self.assertFalse(result.completed)
        self.assertIn('max_depth', result.stop_reasons)

    def test_valid_json_strings_handle_braces_and_escapes(self):
        for summary in ['literal } brace', 'literal { brace', 'escaped " quote and \\ {brace}', 'nested {"a": 2}']:
            payload = {'summary': summary, 'evidence': [], 'answer_candidate': '', 'confidence': .9}
            self.assertEqual(extract_json_object('JSON:\n' + json.dumps(payload)), payload)

    def test_duplicate_case_names_keep_every_trace_and_stable_replay_checksum(self):
        examples = [bench.BenchmarkExample('duplicate', 'evidence', [ContextBlock('a', 'evidence ' + value)], value, [value]) for value in ['first', 'second', 'second']]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = bench.run_dataset(examples, 'keep_recent', output_dir=root/'bundle', inspection_replay_dir=root/'replay')
            self.assertEqual(len(list((root/'bundle/trace_examples/keep_recent').glob('*.jsonl'))), 3)
            self.assertEqual(len(list((root/'bundle/loom_traces/keep_recent').glob('*.jsonl'))), 3)
            for row in result['results']:
                replay = row['retention_stats']['inspection_replay']
                files = list((root/'replay').rglob(replay['store_file']))
                self.assertEqual(len(files), 1)
                self.assertEqual(hashlib.sha256(files[0].read_bytes()).hexdigest(), replay['store_sha256'])

    def test_case_names_never_become_paths_and_symlink_outputs_fail_before_inspection(self):
        example = bench.BenchmarkExample('../../escape', 'evidence', [ContextBlock('a', 'evidence')], 'evidence', ['evidence'])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench.run_dataset([example], 'keep_recent', output_dir=root/'bundle')
            self.assertFalse((root/'bundle/escape.jsonl').exists())
            self.assertEqual(len(list((root/'bundle/trace_examples/keep_recent').glob('*.jsonl'))), 1)
            (root/'outside').mkdir()
            (root/'attack').mkdir()
            (root/'attack/trace_examples').symlink_to(root/'outside', target_is_directory=True)
            with patch.object(HeuristicBackend, 'inspect', side_effect=AssertionError('network must not begin')):
                with self.assertRaises(ValueError):
                    bench.run_dataset([example], 'keep_recent', output_dir=root/'attack')

    def test_replay_and_cache_latency_do_not_enter_quality_reward(self):
        example = bench.BenchmarkExample('case', 'evidence', [ContextBlock('a', 'evidence')], 'evidence', ['evidence'])
        result = RLM(RLMConfig(model='demo/heuristic')).completion('evidence', example.context)
        with tempfile.TemporaryDirectory() as tmp:
            with patch('bench.time.perf_counter', side_effect=[0, .25]):
                # Only the harness clock is patched: the engine result is identical.
                with patch('bench.run_policy_case', return_value=result):
                    first = bench.run_dataset([example], 'keep_recent')['results'][0]
            with patch('bench.run_policy_case', return_value=result), patch('bench.time.perf_counter', side_effect=[0, .001]):
                second = bench.run_dataset([example], 'keep_recent')['results'][0]
        self.assertEqual(first['reward_score'], second['reward_score'])
        self.assertNotEqual(first['latency_ms'], second['latency_ms'])

    def test_empty_namespace_is_distinct_from_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = InspectionReplayBackend(HeuristicBackend(), Path(tmp)/'store.json', namespace={})
            self.assertEqual(backend.namespace, {})

    def test_explicit_task_identity_is_exported_and_exact_match_is_honest(self):
        example = bench.BenchmarkExample('case', 'evidence', [ContextBlock('a', 'evidence first')], 'first', ['first'], task_id='task_manifest_identity')
        with tempfile.TemporaryDirectory() as tmp:
            summary = bench.run_dataset([example], 'keep_recent', output_dir=tmp)
            events = [json.loads(line) for line in next(Path(tmp).glob('loom_traces/*/*.jsonl')).read_text().splitlines()]
        self.assertEqual(summary['results'][0]['task_id'], example.task_id)
        self.assertEqual({event['task_id'] for event in events}, {example.task_id})
        self.assertEqual(events[-1]['metadata']['containment_accuracy'], 1)
        self.assertEqual(events[-1]['metadata']['exact_match'], 0)

class AdditionalBoundaryTests(unittest.TestCase):
    def test_unchanged_replay_stats_do_not_reread_store_and_detect_later_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'store.json'
            backend = InspectionReplayBackend(HeuristicBackend(), path)
            backend.inspect('release', [ContextBlock('a', 'release evidence')], 0, 'root')
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(Path, 'read_bytes', side_effect=AssertionError('unnecessary store read')):
                self.assertEqual(backend.replay_stats()['store_sha256'], expected)
                self.assertEqual(backend.replay_stats()['store_sha256'], expected)
            path.write_text(path.read_text() + '\n')
            self.assertNotEqual(backend.replay_stats()['store_sha256'], expected)

    def test_provider_rejects_invalid_shapes_and_oversized_inputs_before_network(self):
        from nanorlm import OpenAICompatibleBackend
        backend = OpenAICompatibleBackend(RLMConfig(model='gpt-4.1-mini', max_input_tokens=300))
        for payload in [{'score': float('nan')}, {'score': 11}, {'confidence': None}, {'confidence': '0.5'}, {'evidence': 'wrong'}, {'summary': {}}]:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                backend._parse_json_payload(json.dumps(payload), list(payload))
        with patch('urllib.request.urlopen', side_effect=AssertionError('must not send')):
            with self.assertRaisesRegex(ValueError, 'max_input_tokens'):
                backend._chat_text('system', 'x' * 1000)

    def test_broken_custom_policy_cannot_escape_engine_budget(self):
        class BrokenPolicy:
            name = 'broken'
            def select(self, query, candidates, budget):
                return candidates
        with self.assertRaisesRegex(ValueError, 'exceeded the memory budget'):
            RLM(RLMConfig(model='demo/heuristic', memory_budget_tokens=1), backend=OversizedBackend(), policy=BrokenPolicy()).completion('evidence', 'input')

    def test_non_ascii_and_long_identifiers_consume_budget(self):
        from nanorlm import estimate_tokens, split_context_blocks
        for text in ['文' * 400, 'long_identifier' * 100, '{}[]()' * 200]:
            blocks = split_context_blocks([ContextBlock('a', text)], 40)
            self.assertGreater(len(blocks), 1)
            self.assertEqual(''.join(block.text for block in blocks), text)
            self.assertTrue(all(estimate_tokens(block.text) <= 40 for block in blocks))

    def test_context_binding_survives_reused_caller_task_identity(self):
        first = bench.BenchmarkExample('same', 'query', [ContextBlock('a', 'first')], 'first', ['first'], task_id='reused')
        second = bench.BenchmarkExample('same', 'query', [ContextBlock('a', 'second')], 'second', ['second'], task_id='reused')
        self.assertNotEqual(bench.case_artifact_stem(first, 'data', 0), bench.case_artifact_stem(second, 'data', 0))


if __name__ == '__main__':
    unittest.main()
