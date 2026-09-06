import unittest
from scripts.report_repoqa import select_strategy


class SelectionTests(unittest.TestCase):
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
