"""Independent arithmetic and evidence-boundary checks on descriptive exports."""
import csv
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent


def rows(name):
    with (ROOT / name).open() as stream:
        return list(csv.DictReader(stream))


class AuditTests(unittest.TestCase):
    def test_input_bytes_pinned_and_no_fitting_claim(self):
        receipt = json.loads((ROOT / 'receipt.json').read_text())
        self.assertIs(receipt['fit'], False)
        self.assertIs(receipt['strict_asof_qualified'], False)
        for source in receipt['inputs']:
            self.assertEqual(hashlib.sha256(Path(source['path']).read_bytes()).hexdigest(), source['sha256'])

    def test_races_are_single_cases_with_independent_primary_recalculation(self):
        polls = rows('selected-polls.csv')
        comparisons = rows('race-comparisons.csv')
        keys = [(row['variant'], row['comparison'], row['race_id']) for row in comparisons]
        self.assertEqual(len(keys), len(set(keys)))
        for row in comparisons:
            if row['variant'] != 'field_end' or row['comparison'] != 'all_pollsters':
                continue
            race = [poll for poll in polls if poll['race_id'] == row['race_id']]
            late = [float(poll['dem_share']) for poll in race if 1 <= int(poll['end_lead']) <= 7]
            prior = [float(poll['dem_share']) for poll in race if 8 <= int(poll['end_lead']) <= 14]
            actual = float(race[0]['actual_dem_share'])
            first, last = sum(prior)/len(prior), sum(late)/len(late)
            sign = 1 if last > first else -1
            self.assertAlmostEqual(float(row['trend_pp']), abs(100*(last-first)), places=10)
            self.assertAlmostEqual(float(row['performance_pp']), sign*100*(actual-last), places=10)

    def test_every_summary_recomputed_and_cycle_totals_match(self):
        comparisons = rows('race-comparisons.csv')
        summaries = rows('summary.csv')
        for summary in summaries:
            selected = [row for row in comparisons if row['variant']==summary['variant'] and row['comparison']==summary['comparison'] and float(row['trend_pp'])>=float(summary['threshold_pp']) and (summary['cycle']=='all' or row['cycle']==summary['cycle'])]
            misses = [float(row['performance_pp']) for row in selected]
            self.assertEqual(int(summary['n']), len(misses))
            self.assertEqual(int(summary['below']), sum(value < -1 for value in misses))
            self.assertEqual(int(summary['within']), sum(-1 <= value <= 1 for value in misses))
            self.assertEqual(int(summary['above']), sum(value > 1 for value in misses))

    def test_sample_universe_and_availability(self):
        polls = rows('selected-polls.csv')
        keys = [(row['race_id'], row['sample_id']) for row in polls]
        self.assertEqual(len(keys), len(set(keys)))
        for row in polls:
            self.assertIn(row['population'], ('LV','RV'))
            self.assertLess(row['available_date'][:10], row['election_date'][:10])
            self.assertLess(row['end_date'][:10], row['election_date'][:10])
            self.assertEqual(row['publication_date_verified'], 'False')
        excluded = {row['race_id'] for row in rows('truth-2022-event-party-joins.csv') if row['accepted']=='False'}
        self.assertFalse(excluded & {row['race_id'] for row in rows('race-comparisons.csv')})


if __name__=='__main__':
    unittest.main()
