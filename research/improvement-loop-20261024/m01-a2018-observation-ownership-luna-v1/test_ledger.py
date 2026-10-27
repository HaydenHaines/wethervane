import csv, hashlib, json, unittest
from pathlib import Path
import build_ledger as bl

class LedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ledger=bl.build()
        with (bl.OUT/'factor-membership.csv').open() as f: cls.members=list(csv.DictReader(f))
    def test_frozen_pins_and_selection(self):
        context,plan,problem,hashes=bl.load_inputs()
        self.assertEqual(context['identity'],plan['context_identity'])
        self.assertEqual(plan['provider_contract']['composition']['config']['selections'][4][1], 'profile_turnout_response_observed_total_gamma_v1')
        self.assertEqual(plan['provider_contract']['composition']['config']['selections'][5][1], 'candidate_iid_profile_choice_v1')
        self.assertEqual([p['mass'] for p in plan['parents']],[.125]*8)
        self.assertFalse(plan['national_complete']);self.assertFalse(plan['forecast_qualified'])
    def test_global_factor_counts_and_zero_observed_total_families(self):
        c=self.ledger['counts']
        self.assertEqual((c['events'],c['contests']),(54,37))
        self.assertEqual((c['official_count_observations'],c['participation_report_observations']),(0,0))
        self.assertEqual((c['survey_blocks'],c['survey_record_rows']),(65,7968))
        self.assertEqual((c['poll_blocks'],c['poll_rows'],c['ranked_poll_rows']),(1,274,0))
        self.assertEqual(c['factor_membership_rows'],8242)
    def test_survey_record_membership_matches_factor_source_contract(self):
        context,plan,problem,_=bl.load_inputs()
        survey_rows=[m for m in self.members if m['factor_family']=='survey_potential']
        self.assertEqual(len(survey_rows),7968)
        by_source={}
        for s in problem['surveys']:
            ids=[r['record_id'] for r in s['rows']]
            by_source.setdefault(s['source_id'],[]).extend(ids)
        for ids in by_source.values(): self.assertEqual(len(ids),len(set(ids)))
        # This verifies source-scoped token uniqueness only, not distinct people.
        self.assertTrue(all('not proven' in m['unit_as_encoded'] for m in survey_rows))
        self.assertTrue(all(m['source_id'].startswith('CES') for m in survey_rows))
        by_source=self.ledger['survey_block_counts_by_source']
        self.assertEqual((by_source['CES2014_historical_training']['blocks'],by_source['CES2014_historical_training']['records'],by_source['CES2014_historical_training']['contest_specific_blocks']),(31,3662,31))
        self.assertEqual((by_source['CES2016_historical_training']['blocks'],by_source['CES2016_historical_training']['records'],by_source['CES2016_historical_training']['no_contest_blocks']),(34,4306,34))
    def test_poll_membership_and_joint_covariance_block(self):
        problem=json.loads(bl.PROBLEM.read_text()); block=problem['polls'][0]
        poll=[m for m in self.members if m['factor_family']=='correlated_poll_mvn']
        self.assertEqual(len(poll),274)
        self.assertEqual((len(block['covariance']),len(block['covariance'][0])),(274,274))
        self.assertEqual(len({m['joint_group'] for m in poll}),1)
        self.assertEqual(self.ledger['poll_likelihood_semantics']['ordinary_paired_logit_rows'],274)
        self.assertEqual(self.ledger['poll_likelihood_semantics']['ranked_rows_in_A2018_block'],0)
        self.assertEqual(len({m['observation_token'] for m in poll}),274)
    def test_target_and_no_value_leakage(self):
        t=self.ledger['target']
        self.assertEqual((t['contest_id'],t['event_id'],t['poll_row_count']),('contest_az2018_regular','event_az2018_general',20))
        self.assertEqual((t['survey_block_count'],t['official_count_observation_count'],t['participation_report_count']),(0,0,0))
        self.assertEqual(set(t['candidate_record_fields']),{'id','bloc','incumbent'})
        self.assertEqual(len(t['candidate_ids']),10)
        self.assertTrue(self.ledger['cancellation_and_identification']['event_scale_cancels'])
        self.assertTrue(self.ledger['cancellation_and_identification']['contest_scalar_response_cancels'])
        self.assertIn('indirect',self.ledger['cancellation_and_identification']['why_turnout_not_identified_by_ratio'])
        with (bl.OUT/'factor-membership.csv').open() as f: fields=set(next(csv.reader(f)))
        self.assertFalse({'logit_value','mask','count','covariance_value'} & fields)
        self.assertEqual(self.ledger['plan_status']['forecast_qualified'],False)
    def test_source_codes_assign_one_factor_per_survey_and_poll_block(self):
        code=bl.COMPONENTS.read_text()
        self.assertIn('scope.factor(name,',code)
        self.assertIn('survey:',code)
        self.assertIn('poll:',code)
        self.assertIn('pm.MvNormal(',code)
        self.assertIn('pm.NegativeBinomial(',code)
        self.assertIn('pm.Poisson(',code)
        self.assertIn('eventG_unobserved',code)
        self.assertIn('name = f"survey_likelihood_',code)
        self.assertIn('name = f"poll_likelihood_',code)
        self.assertIn('means.append(pt.log(rates[first]) - pt.log(rates[second]))',code)
        self.assertIn('response[name] = pm.Deterministic',code)
        self.assertIn('pm.math.sigmoid(logits[name])',code)
        self.assertIn('cov=np.array(block["covariance"])',code)

if __name__=='__main__': unittest.main(verbosity=2)
