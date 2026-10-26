import copy
import unittest
from test_validator import fixture
from validator import validate


class MalformedChecks(unittest.TestCase):
    def reject(self, value):
        result = validate(value)
        self.assertEqual(result['status'], 'unresolved_or_rejected')
        self.assertTrue(result['errors'])
        self.assertFalse(result['authenticated'])
        self.assertFalse(result['fit_authorized'])

    def mutate(self, path, value):
        ledger = fixture()
        target = ledger
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        self.reject(ledger)

    def test_non_mapping_ledger(self):
        for value in (None, [], 'ledger', 1, False):
            with self.subTest(value=value):
                self.reject(value)

    def test_missing_required_containers(self):
        for key in ('records', 'dependencies', 'evaluation_event_keys'):
            ledger=fixture(); del ledger[key]
            with self.subTest(key=key):
                self.reject(ledger)

    def test_invalid_container_types(self):
        for key in ('records', 'dependencies', 'evaluation_event_keys'):
            for value in (None, {}, 'text', 1, True, ()): 
                with self.subTest(key=key,value=value):
                    self.mutate((key,),value)
        self.mutate(('records',),[])

    def test_invalid_record(self):
        for value in (None, [], 'record', 1, {}):
            with self.subTest(value=value):
                self.mutate(('records',0),value)
        for value in (None, [], {}, 1, '', ' '):
            self.mutate(('records',0,'id'),value)
            self.mutate(('records',0,'factor_owner'),value)

    def test_invalid_atom_and_label_lists(self):
        for field in ('canonical_atoms', 'outcome_label_event_keys'):
            for value in (None, {}, 'text', [None], [{}], [[]], [1], [''], [' '], ['same','same']):
                with self.subTest(field=field,value=value):
                    self.mutate(('records',0,field),value)
        self.mutate(('records',0,'canonical_atoms'),[])
        for value in ([{}], [[]], [1], [''], ['same','same']):
            self.mutate(('evaluation_event_keys',),value)

    def test_invalid_edge(self):
        for value in (None, [], 'edge', 1, {}):
            self.mutate(('dependencies',0),value)
        for value in (None, {}, 'poll', [None], [{}], [[]], ['poll'], ['poll','poll'],
                      ['poll','unknown'], ['poll','survey','extra']):
            self.mutate(('dependencies',0,'records'),value)

    def test_invalid_axis_claims(self):
        for axis in ('respondents','source_error'):
            for value in (None, [], 'claim', 1, {}):
                self.mutate(('dependencies',0,axis),value)

    def test_invalid_pin_fields(self):
        for value in (None, [], 'pin', {}, dict(path=[],sha256='a'*64),
                      dict(path=' ',sha256='a'*64),dict(path='x',sha256='z'*64),
                      dict(path='x',sha256=['a']*64)):
            self.mutate(('records',0,'source'),value)
            self.mutate(('vector_manifest',),value)
        self.mutate(('dependencies',0,'source_error','conditional_law_id'),['law'])

    def test_duplicate_ids_and_edges(self):
        self.mutate(('records',1,'id'),'poll')
        ledger=fixture(); ledger['dependencies'].append(copy.deepcopy(ledger['dependencies'][0]))
        self.reject(ledger)

    def test_naive_and_invalid_dates(self):
        for value in (None, 1, [], {}, '', '2022-10-01', '2022-10-01T00:00:00',
                      '2022-99-99T00:00:00Z','2022-10-01T00:00:00+25:00'):
            self.mutate(('cutoff',),value)
            self.mutate(('records',0,'verified_available_at'),value)

    def test_timezone_offsets_compare_instants(self):
        # Same date locally; the first source is after cutoff in UTC.
        ledger=fixture(); ledger['cutoff']='2022-10-01T00:00:00Z'
        ledger['records'][0]['verified_available_at']='2022-09-30T20:00:00-05:00'
        self.reject(ledger)
        ledger['records'][0]['verified_available_at']='2022-10-01T03:00:00+03:00'
        self.assertEqual(validate(ledger)['status'],'contract_checks_passed')

if __name__ == '__main__':
    unittest.main()
