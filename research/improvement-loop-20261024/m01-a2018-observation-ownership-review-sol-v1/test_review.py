"""Local-only identifier comparison; output never includes record or poll tokens."""
import csv,json,hashlib,unittest
from pathlib import Path
from fractions import Fraction as F
ROOT=Path('/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/m01-a2018-observation-ownership-luna-v1')
BASE=Path('/home/hayden/projects/wethervane-redesign-20261006/integration/research/senate-stack-integration-20261020')
class ReviewTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.context=json.loads((BASE/'national-multicycle-execution-astra-v1/A2018/context.json').read_text());cls.problem=cls.context['problem'];cls.ledger=json.loads((ROOT/'ledger.json').read_text())
 def test_complete_memberships_match_in_memory_only(self):
  expected=[];p=self.problem;contests={c['id']:c for c in p['contests']}
  for survey in p['surveys']:
   for row in survey['rows']:expected.append(('survey_likelihood_'+survey['id'],'survey_potential',str(row['record_id']),survey['source_id'],survey['event_id'],survey['contest_id']or''))
  for block in p['polls']:
   for row in block['rows']:expected.append(('poll_likelihood_'+block['id'],'correlated_poll_mvn',str(row['observation_id']),'A2018_polls',contests[row['contest_id']]['event_id'],row['contest_id']))
  fields=['factor_id','factor_family','observation_token','source_id','event_id','contest_id']
  with (ROOT/'factor-membership.csv').open()as stream:actual=[tuple(r[k]for k in fields)for r in csv.DictReader(stream)]
  self.assertTrue(sorted(expected)==sorted(actual),'membership mismatch; identifiers suppressed');self.assertEqual(len(actual),8242)
  self.assertEqual(len({x[0]for x in actual}),66)
 def test_extension_likelihood_closure_not_just_base_counts(self):
  ext=self.context['extension'];rr=ext['ranked_runoff'];self.assertIsNone(ext['open_person'])
  self.assertEqual(len(rr['ranked']),1);self.assertEqual(len(rr['runoffs']),1)
  self.assertTrue(all(not x['observations']for x in rr['ranked']+rr['runoffs']))
  self.assertEqual(len(self.problem['events']),54);self.assertEqual(self.problem['official_counts'],[]);self.assertEqual(self.problem['participation_reports'],[])
  for name in ['mixed-count-graph-astra-v1/rank_contribution.py','mixed-count-graph-astra-v1/open_contribution.py']:
   relative='research/senate-stack-integration-20261020/'+name
   self.assertEqual(hashlib.sha256((BASE/name).read_bytes()).hexdigest(),self.context['descriptor']['source_inventory'][relative])
 def test_covariance_shape_and_target_aggregate(self):
  block=self.problem['polls'][0];self.assertEqual(len(block['covariance']),274);self.assertTrue(all(len(r)==274 for r in block['covariance']))
  self.assertEqual(sum(x['contest_id']=='contest_az2018_regular'for x in block['rows']),20)
  self.assertEqual(self.ledger['poll_likelihood_semantics']['ranked_rows_in_A2018_block'],0)
 def test_exact_scalar_cancellation_is_not_profile_invariance(self):
  intensities=[F(1),F(2)];D=[F(1,4),F(3,4)];R=[F(3,4),F(1,4)]
  expected=sum(i*p for i,p in zip(intensities,D))/sum(i*p for i,p in zip(intensities,R))
  for g,r in [(F(1),F(1)),(F(13),F(1,7)),(F(1,100),F(9,10))]:
   self.assertEqual(sum(g*r*i*p for i,p in zip(intensities,D))/sum(g*r*i*p for i,p in zip(intensities,R)),expected)
  changed=[F(2),F(1)];other=sum(i*p for i,p in zip(changed,D))/sum(i*p for i,p in zip(changed,R));self.assertNotEqual(other,expected)
if __name__=='__main__':unittest.main(verbosity=2)
