import unittest
import json
import tempfile
from pathlib import Path
from scripts.report_repoqa import adjudicate_grade, batch_accounting, select_strategy
from scripts.evaluate_repoqa import file_hash


class SelectionTests(unittest.TestCase):
    def test_batch_costs_count_all_calls_and_measure_shared_turnaround(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, start, end in [(1,100,110),(2,130,150)]:
                directory = root/f'round-{index:02d}'
                directory.mkdir()
                cid = f'case--retention.request.{index}'
                (directory/'input.jsonl').write_text(json.dumps({'custom_id':cid,'body':{'model':'gpt-4.1-mini-2025-04-14'}})+'\n')
                (directory/'submission.json').write_text(json.dumps({'input_sha256':file_hash(directory/'input.jsonl')}))
                (directory/'batch.json').write_text(json.dumps({'status':'completed','created_at':start,'completed_at':end}))
                (directory/'output.jsonl').write_text(json.dumps({'custom_id':cid,'response':{'body':{'usage':{'prompt_tokens':100,'completion_tokens':10}}}})+'\n')
            result = batch_accounting(root,'gpt-4.1-mini-2025-04-14')['case--retention']
            self.assertEqual(result['calls'],2)
            self.assertAlmostEqual(result['normal_price_usd'],.000112)
            self.assertAlmostEqual(result['batch_estimated_usd'],.000056)
            self.assertEqual(result['batch_availability_ms'],50_000)

    def test_malformed_grade_requires_explicit_bound_adjudication(self):
        original = {'error':'invalid JSON', 'requires_adjudication':True}
        replacement = {'facts':[{'index':1,'correct':True,'reason':'source confirms'}],
                       'claims':[{'index':1,'supported':True,'materially_incorrect':False,'reason':'cited code confirms'}],
                       'reference_concern':''}
        with self.assertRaisesRegex(ValueError, 'complete replacement_grade'):
            adjudicate_grade(original, {}, 'abc', 1, 1)
        audit = {'receipt_sha256':'abc','reason':'Audited the raw answer and actual citations.', 'replacement_grade':replacement}
        self.assertEqual(adjudicate_grade(original,audit,'abc',1,1), replacement)
        self.assertTrue(original['requires_adjudication'])
        with self.assertRaisesRegex(ValueError, 'not bound'):
            adjudicate_grade(original,audit,'different',1,1)
        with self.assertRaisesRegex(ValueError, 'count mismatch'):
            adjudicate_grade(original,audit,'abc',2,1)

    def summaries(self, lexical=18, full=19, retention=19):
        return {name: {'fully_correct': quality, 'citation_precision': .95, 'estimated_usd': cost}
                for name, quality, cost in [('lexical',lexical,.1),('full',full,.4),('retention',retention,1)]}

    def test_cheapest_close_quality_wins(self):
        self.assertEqual(select_strategy(self.summaries())[1], 'lexical')

    def test_retention_needs_three_extra_correct_questions(self):
        eligible, selected = select_strategy(self.summaries(retention=20))
        self.assertNotIn('retention', eligible)
        self.assertEqual(selected, 'full')
        self.assertEqual(select_strategy(self.summaries(retention=22))[1], 'retention')

    def test_low_citation_support_cannot_win_on_cost(self):
        summaries = self.summaries()
        summaries['lexical']['citation_precision'] = .80
        self.assertEqual(select_strategy(summaries)[1], 'full')
