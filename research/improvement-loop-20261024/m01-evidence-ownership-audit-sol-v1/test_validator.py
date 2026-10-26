import unittest
from validator import validate

PIN = dict(path='synthetic-evidence.json', sha256='a'*64)

def fixture():
    records=[]
    for name in ('poll', 'survey'):
        records.append(dict(id=name, source=PIN, factor_owner=name, canonical_atoms_status='established',
            canonical_atoms=[name+':record:1'], outcome_label_event_keys=[],
            availability_status='verified', verified_available_at='2022-10-01T00:00:00Z', historical_version_status='verified'))
    return dict(mode='historical_replay', cutoff='2022-10-15T00:00:00Z',records=records,
        evaluation_event_keys=['2022:TX:senate:regular'], complete_vector_status='bound', vector_manifest=PIN,
        dependencies=[dict(records=['poll','survey'], respondents=dict(status='disjoint_established',evidence=PIN),
            source_error=dict(status='jointly_modeled',evidence=PIN,adapter=PIN,conditional_law_id='synthetic_shared_event_v1'))])

class Checks(unittest.TestCase):
    def test_synthetic_complete(self):
        result=validate(fixture())
        self.assertEqual(result['status'],'contract_checks_passed')
        self.assertFalse(result['authenticated'])
        self.assertFalse(result['fit_authorized'])
    def test_missing_is_not_disjoint(self):
        x=fixture(); x['dependencies']=[]
        self.assertIn('missing dependence assessment',str(validate(x)['errors']))
    def test_different_ids_do_not_resolve_respondents(self):
        x=fixture(); x['dependencies'][0]['respondents']['status']='unknown'
        self.assertIn('respondents unresolved',str(validate(x)['errors']))
    def test_same_atom_under_alias(self):
        x=fixture(); x['records'][1]['canonical_atoms']=x['records'][0]['canonical_atoms']
        self.assertIn('duplicate evidence atom',str(validate(x)['errors']))
    def test_target_label_reuse(self):
        x=fixture(); x['records'][0]['outcome_label_event_keys']=x['evaluation_event_keys']
        self.assertIn('evaluation outcome used',str(validate(x)['errors']))
    def test_calibration_official_label_reuse(self):
        x=fixture()
        for r in x['records']:
            r['outcome_label_event_keys']=['2020:TX:senate:regular']
        self.assertIn('outcome label reused across factors',str(validate(x)['errors']))
    def test_proxy_publication_rejected(self):
        x=fixture(); x['records'][0]['availability_status']='field_end_proxy'
        self.assertIn('historical availability unresolved',str(validate(x)['errors']))
    def test_later_vintage_rejected(self):
        x=fixture(); x['records'][0]['verified_available_at']='2022-10-16T00:00:00Z'
        self.assertIn('available after cutoff',str(validate(x)['errors']))
    def test_joint_model_requires_adapter(self):
        x=fixture(); del x['dependencies'][0]['source_error']['adapter']
        self.assertIn('conditional adapter required',str(validate(x)['errors']))
    def test_retrospective_does_not_claim_asof(self):
        x=fixture(); x['mode']='retrospective_joint'; x['records'][0]['availability_status']='unknown'
        self.assertEqual(validate(x)['status'],'contract_checks_passed')
    def test_unknown_atoms_rejected(self):
        x=fixture(); x['records'][0]['canonical_atoms_status']='unknown'
        self.assertIn('canonical atoms unresolved',str(validate(x)['errors']))

if __name__ == '__main__':
    unittest.main()
