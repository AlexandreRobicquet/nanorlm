import unittest
from scripts.grade_grouped_repoqa import group_cases, request_body
from scripts.grade_repoqa import grading_packet


class GroupedGradingTests(unittest.TestCase):
    def test_groups_never_expose_competing_answers_to_the_same_question(self):
        cases=[{'id':str(i),'task_id':str(i//3)} for i in range(81)]
        groups=group_cases(cases)
        self.assertEqual(sorted(c['id'] for g in groups for c in g),sorted(c['id'] for c in cases))
        self.assertTrue(all(1<=len(g)<=2 for g in groups))
        self.assertTrue(all(len({c['task_id'] for c in g})==len(g) for g in groups))
        self.assertEqual(groups,group_cases(cases))

    def test_each_claim_gets_only_its_actual_citations_and_counts_are_constrained(self):
        def span(sid,text):return {'id':sid,'path':'code.py','line_start':1,'line_end':1,'text':text}
        packet=grading_packet({'question':'where?','expected_facts':['one'],'reference_spans':[]},
            {'claims':[{'text':'first','citations':['a']},{'text':'second','citations':['b']}],'uncertainties':[]},
            {'spans':[span('a','first code'),span('b','second code'),span('c','uncited code')]})
        self.assertEqual(packet['candidate_claims'][0]['cited_excerpts'],[span('a','first code')])
        self.assertEqual(packet['candidate_claims'][1]['cited_excerpts'],[span('b','second code')])
        body=request_body({'blind_id':packet})
        schema=body['response_format']['json_schema']['schema']
        self.assertEqual(schema['required'],['blind_id'])
        self.assertFalse(schema['additionalProperties'])
        self.assertEqual(schema['properties']['blind_id']['properties']['claims']['minItems'],2)
        self.assertEqual(schema['properties']['blind_id']['properties']['facts']['maxItems'],1)
