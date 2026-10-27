import copy, json, tempfile, unittest
from unittest.mock import patch
from pathlib import Path
import build_ledger as bl

class ClosureLedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context,cls.plan,cls.problem,cls.hashes=bl.load_actual()
        cls.component_text=bl.COMPONENTS.read_text()
        cls.rank_text=bl.RANK.read_text()
        cls.open_text=bl.OPEN.read_text()
    def validate_pair(self, change):
        p=copy.deepcopy(self.problem); change(p)
        c=copy.deepcopy(self.context); c['problem']=p
        return bl.validate_closure(c,self.plan,p,self.component_text,self.rank_text,self.open_text)
    def test_actual_pinned_closure_conditions(self):
        closure=bl.validate_closure(self.context,self.plan,self.problem,self.component_text,self.rank_text,self.open_text)
        self.assertTrue(closure['open_person_absent'])
        self.assertEqual((closure['ranked_observation_count'],closure['runoff_observation_count'],closure['core_ranked_observation_count']),(0,0,0))
        self.assertEqual((closure['ranked_poll_channel_count'],closure['ordinary_poll_row_count']),(0,274))
        self.assertTrue(closure['survey_loop_eventG_free'])
        self.assertTrue(closure['ordinary_poll_formula_verified'])
        self.assertTrue(closure['contest_response_is_scalar'])
    def test_added_official_count_blocks_conditional_prior_claim(self):
        with self.assertRaisesRegex(ValueError,'official counts/reports'):
            self.validate_pair(lambda p:p['official_counts'].append({'id':'fixture'}))
    def test_added_participation_report_blocks_conditional_prior_claim(self):
        with self.assertRaisesRegex(ValueError,'official counts/reports'):
            self.validate_pair(lambda p:p['participation_reports'].append({'id':'fixture'}))
    def test_present_open_person_blocks_conditional_prior_claim(self):
        c=copy.deepcopy(self.context);c['extension']['open_person']={'fixture':True}
        with self.assertRaisesRegex(ValueError,'open_person is present'):
            bl.validate_closure(c,self.plan,c['problem'],self.component_text,self.rank_text,self.open_text)
    def test_ranked_extension_observation_blocks_conditional_prior_claim(self):
        c=copy.deepcopy(self.context);c['extension']['ranked_runoff']['ranked'][0]['observations'].append({'id':'fixture'})
        with self.assertRaisesRegex(ValueError,'ranked observation list'):
            bl.validate_closure(c,self.plan,c['problem'],self.component_text,self.rank_text,self.open_text)
    def test_runoff_observation_blocks_conditional_prior_claim(self):
        c=copy.deepcopy(self.context);c['extension']['ranked_runoff']['runoffs'][0]['observations'].append({'id':'fixture'})
        with self.assertRaisesRegex(ValueError,'runoff observation list'):
            bl.validate_closure(c,self.plan,c['problem'],self.component_text,self.rank_text,self.open_text)
    def test_core_ranked_observation_blocks_conditional_prior_claim(self):
        with self.assertRaisesRegex(ValueError,'core ranked measurement'):
            self.validate_pair(lambda p:p['ranked_measurement']['ranked'][0]['observations'].append({'id':'fixture'}))
    def test_ranked_poll_channel_blocks_ordinary_cancellation_claim(self):
        poll_id=self.problem['polls'][0]['rows'][0]['observation_id']
        with self.assertRaisesRegex(ValueError,'ranked poll channels'):
            self.validate_pair(lambda p:p['ranked_measurement']['poll_channels'].__setitem__(poll_id,{}))
    def test_source_pin_mismatch_fails_closed(self):
        with patch.dict(bl.EXPECTED,{str(bl.COMPONENTS):'0'*64}):
            with self.assertRaisesRegex(ValueError,'source/receipt drift'):
                bl.load_actual()
    def test_source_counterexamples_downgrade_closure(self):
        altered=self.component_text.replace('for survey in c.data["surveys"]:',
            'for survey in c.data["surveys"]:\n                _eventG_closure_fixture = v["eventG"]',1)
        with self.assertRaisesRegex(ValueError,'Survey likelihood now depends'):
            bl.validate_model_source(self.problem,altered,self.rank_text,self.open_text)
        altered_poll=self.component_text.replace('means.append(pt.log(rates[first]) - pt.log(rates[second]))',
            'means.append(pt.log(rates[first] * v["eventG"][0]) - pt.log(rates[second]))',1)
        with self.assertRaisesRegex(ValueError,'source no longer matches closure contract'):
            bl.validate_model_source(self.problem,altered_poll,self.rank_text,self.open_text)
    def test_scratch_output_has_no_survey_record_tokens(self):
        # Tokens are examined only locally and never inserted into output or diagnostics.
        tokens=[r['record_id'] for s in self.problem['surveys'] for r in s['rows']]
        with tempfile.TemporaryDirectory(prefix='a2018-ledger-luna-v2-test-') as d:
            ledger=bl.build(d)
            files=list(Path(d).iterdir())
            self.assertEqual([p.name for p in files],['aggregate-ledger.json'])
            payload=files[0].read_bytes()
            self.assertFalse(any(token.encode() in payload for token in tokens))
            self.assertNotIn('record_id',ledger['target'].get('candidate_record_fields',[]))
    def test_v1_aggregate_projection_is_unchanged(self):
        with tempfile.TemporaryDirectory(prefix='a2018-ledger-luna-v2-match-') as d:
            ledger=bl.build(d)
            v1=json.loads(bl.V1_LEDGER.read_text())
            self.assertEqual(bl.aggregate_projection(ledger),bl.aggregate_projection(v1))
            self.assertTrue(ledger['predecessor_aggregate_comparison']['matches_v1_aggregate'])
            self.assertEqual(ledger['counts']['factor_membership_rows'],8242)
            self.assertEqual(ledger['eventG_likelihood_closure']['event_ids'],sorted(e['id'] for e in self.problem['events']))
            self.assertEqual(ledger['eventG_likelihood_closure']['status'],'verified_source_algebra_conditional_on_shared_kappa')

if __name__=='__main__': unittest.main(verbosity=2)
