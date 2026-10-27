"""Local metadata-only adversarial checks; never print source row tokens."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import build_ledger as bl

class ReviewTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.c,cls.plan,cls.p,cls.hashes=bl.load_actual();cls.component=bl.COMPONENTS.read_text();cls.rank=bl.RANK.read_text();cls.open=bl.OPEN.read_text()
 def validate(self,c):return bl.validate_closure(c,self.plan,c['problem'],self.component,self.rank,self.open)
 def test_missing_and_malformed_extension_fail_closed(self):
  for change in [lambda c:c['extension'].pop('ranked_runoff'),lambda c:c['extension']['ranked_runoff'].pop('ranked'),lambda c:c['extension']['ranked_runoff']['runoffs'][0].pop('observations'),lambda c:c['problem']['ranked_measurement'].__setitem__('poll_channels',None)]:
   c=copy.deepcopy(self.c);change(c)
   with self.assertRaises((ValueError,KeyError)):self.validate(c)
 def test_all_contribution_pin_drifts_reject(self):
  original=bl.sha
  for bad in [bl.COMPONENTS,bl.RANK,bl.OPEN]:
   with patch.object(bl,'sha',side_effect=lambda path: '0'*64 if path==bad else original(path)):
    with self.assertRaisesRegex(ValueError,'drift'):bl.load_actual()
 def test_inventory_and_plan_link_mismatch_reject(self):
  loads=json.loads;context_text=bl.CONTEXT.read_text();plan_text=bl.PLAN.read_text()
  rel='research/senate-stack-integration-20261020/mixed-count-graph-astra-v1/rank_contribution.py'
  for kind in ['context','plan']:
   def altered(text,*args,**kwargs):
    v=loads(text,*args,**kwargs)
    if kind=='context'and text==context_text:v['descriptor']['source_inventory'][rel]='0'*64
    if kind=='plan'and text==plan_text:v['source_pins']['research/senate-stack-integration-20261020/national-problem-assembly-astra-v1/components.py']='0'*64
    return v
   with patch.object(bl.json,'loads',side_effect=altered):
    with self.assertRaises(ValueError):bl.load_actual()
 def test_aggregates_beyond_projection_and_no_token_emission(self):
  with tempfile.TemporaryDirectory()as root:
   v2=bl.build(root);v1=json.loads(bl.V1_LEDGER.read_text())
   for key in ['survey_block_counts_by_source','poll_row_counts_by_event','counts','parent_masses','target','survey_blocks','poll_blocks','sources','events']:
    self.assertTrue(v2[key]==v1[key],'aggregate mismatch; source identifiers suppressed')
   raw=(Path(root)/'aggregate-ledger.json').read_bytes()
   self.assertFalse(any(row['record_id'].encode()in raw for s in self.p['surveys']for row in s['rows']))
 def test_shape_helper_is_not_independent_source_authentication(self):
  # Existing exact-file hash gate is essential: a substring guard is not a proof.
  altered=self.component.replace('logits[name] = baseline + race[i]', 'logits[name] = baseline + race[i] + v["turnout_logit"][c.event_index[contest["event_id"]]][:, :, None]')
  self.assertNotEqual(altered,self.component)
  result=bl.validate_model_source(self.p,altered,self.rank,self.open)
  self.assertTrue(result['contest_response_is_scalar'])

if __name__=='__main__':unittest.main(verbosity=2)
