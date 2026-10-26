import unittest
from test_validator import fixture
from validator import validate


class ExplicitTargetChecks(unittest.TestCase):
    def test_historical_empty_targets_rejected(self):
        ledger=fixture()
        ledger['evaluation_event_keys']=[]
        result=validate(ledger)
        self.assertEqual(result['status'],'unresolved_or_rejected')
        self.assertIn('evaluation_event_keys: nonempty list required',result['errors'])

    def test_retrospective_explicit_empty_targets_allowed(self):
        ledger=fixture()
        ledger['mode']='retrospective_joint'
        ledger['evaluation_event_keys']=[]
        result=validate(ledger)
        self.assertEqual(result['status'],'contract_checks_passed')
        self.assertFalse(result['authenticated'])
        self.assertFalse(result['fit_authorized'])

    def test_retrospective_missing_targets_still_rejected(self):
        ledger=fixture()
        ledger['mode']='retrospective_joint'
        del ledger['evaluation_event_keys']
        self.assertEqual(validate(ledger)['status'],'unresolved_or_rejected')

    def test_historical_explicit_target_detects_reuse(self):
        ledger=fixture()
        ledger['records'][0]['outcome_label_event_keys']=ledger['evaluation_event_keys'][:]
        self.assertIn('evaluation outcome used as training input',validate(ledger)['errors'])

if __name__=='__main__':
    unittest.main()
