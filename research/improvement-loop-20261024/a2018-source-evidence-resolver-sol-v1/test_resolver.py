"""Local metadata tests. No record token or source values printed/exported."""
import copy,json,tempfile,unittest
from pathlib import Path
import resolver as r
class ResolverTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.bindings=r.load_bindings();cls.problem=json.loads(r.read_pinned(cls.bindings['problem']))
 def test_actual_aggregate_and_blocked_fields(self):
  out=r.resolve();self.assertEqual(out['summary']['survey_memberships'],7968);self.assertEqual(out['summary']['poll_rows'],274);self.assertEqual(out['summary']['target_poll_rows'],20);self.assertEqual(out['summary']['survey_contest_joint_answer_blocks'],31);self.assertEqual(out['summary']['survey_no_contest_community_turnout_blocks'],34)
  self.assertEqual(out['claim_checker']['full_row_audit'],'not_invoked_missing_source_measurement_and_chronology');self.assertFalse(out['original_source_observation_atoms_authenticated']);self.assertFalse(out['actual_compiled_factor_graph_authenticated']);self.assertFalse(out['summary']['covariance_positive_definite_verified']);self.assertFalse(out['respondent_or_sample_independence_verified'])
 def test_all_roles_tamper_rejects_before_resolution(self):
  for item in self.bindings.values():
   def reader(path):
    raw=Path(path).read_bytes();return raw+b' 'if path==item['path']else raw
   with self.assertRaisesRegex(r.Rejected,'pinned_source_bytes_differ'):r.resolve(reader)
 def test_missing_source_and_record_key_rejects(self):
  for kind in ['source','record','schema']:
   p=copy.deepcopy(self.problem)
   if kind=='source':p['surveys'][0]['source_id']='invented-missing-source'
   if kind=='record':p['surveys'][0]['rows'][0].pop('record_id')
   if kind=='schema':p['surveys'][0]['rows'][0]['invented_field']=True
   with self.assertRaises(r.Rejected):r.inspect_prepared(p)
 def test_duplicate_local_records_and_poll_measurements_reject(self):
  p=copy.deepcopy(self.problem);p['surveys'][0]['rows'].append(copy.deepcopy(p['surveys'][0]['rows'][0]))
  with self.assertRaisesRegex(r.Rejected,'duplicate_source_scoped'):r.inspect_prepared(p)
  p=copy.deepcopy(self.problem);p['polls'][0]['rows'][1]['observation_id']=p['polls'][0]['rows'][0]['observation_id']
  with self.assertRaisesRegex(r.Rejected,'duplicate_poll'):r.inspect_prepared(p)
 def test_aliases_remain_unresolved_not_distinct_people(self):
  p=copy.deepcopy(self.problem);sid=p['surveys'][0]['source_id'];original=next(s for s in p['sources']if s['id']==sid);alias=copy.deepcopy(original);alias['id']='invented-alias';p['sources'].append(alias);p['surveys'][0]['source_id']=alias['id']
  out=r.inspect_prepared(p);self.assertGreater(out['used_source_alias_hash_groups'],0);self.assertEqual(out['alias_lineage_status'],'unresolved_alias_groups')
 def test_covariance_malformed_asymmetric_nonfinite_diagonal_reject(self):
  for kind in ['shape','asymmetry','nonfinite','diagonal']:
   p=copy.deepcopy(self.problem);c=p['polls'][0]['covariance']
   if kind=='shape':c[0].pop()
   if kind=='asymmetry':c[0][1]+=1
   if kind=='nonfinite':c[0][0]=float('nan')
   if kind=='diagonal':c[0][0]=0
   with self.assertRaises(r.Rejected):r.inspect_prepared(p)
 def test_order_digest_changes_and_invalid_candidate_target_rejects(self):
  baseline=r.inspect_prepared(self.problem);p=copy.deepcopy(self.problem);p['polls'][0]['rows'].reverse();changed=r.inspect_prepared(p)
  self.assertNotEqual(changed['poll_ordered_vector_covariance_sha256'],baseline['poll_ordered_vector_covariance_sha256'])
  p=copy.deepcopy(self.problem);p['polls'][0]['rows'][0]['first_candidate_id']='invented-outside-target'
  with self.assertRaisesRegex(r.Rejected,'poll_candidate_target_mismatch'):r.inspect_prepared(p)
 def test_indefinite_covariance_not_claimed_pd(self):
  p=copy.deepcopy(self.problem);cov=p['polls'][0]['covariance'];n=len(cov)
  for i in range(n):
   for j in range(n):cov[i][j]=float(i==j)
  cov[0][1]=cov[1][0]=2.
  out=r.inspect_prepared(p);self.assertFalse(out['covariance_positive_definite_verified'])
 def test_result_contains_no_survey_or_poll_tokens(self):
  out=r.resolve();encoded=json.dumps(out)
  self.assertFalse(any(row['record_id']in encoded for s in self.problem['surveys']for row in s['rows']))
  self.assertFalse(any(row['observation_id']in encoded for b in self.problem['polls']for row in b['rows']))
 def test_original_sources_preserved(self):
  for item in self.bindings.values():r.read_pinned(item)
if __name__=='__main__':unittest.main(verbosity=2)
