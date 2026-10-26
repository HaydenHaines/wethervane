import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import design
import tiny_count as c

class DesignTests(unittest.TestCase):
 def test_ast_source_order_bundle_and_scientific_guards_preserved(self):
  result=design.describe();self.assertFalse(result['allow_actual_tx_execution']);self.assertEqual(result['source_analysis']['exact_bundle_keys'],design.BUNDLE_KEYS)
  self.assertEqual(result['gradient_contract']['dimensions'],662);self.assertEqual(result['gradient_contract']['E_shape'],[3,24])
  self.assertFalse(result['gradient_contract']['parent_mass_in_logp']);self.assertEqual(result['gradient_contract']['original_parent_masses'],[.125]*8)
  self.assertTrue(all(v is None for v in result['future_required_inputs'].values()))
 def test_changed_expected_source_fails_before_any_scientific_import(self):
  with tempfile.TemporaryDirectory() as temp:
   folder=Path(temp);plan=json.loads((design.ORIGINAL/'plan.json').read_text());plan['source_pins'][0]['sha256']='0'*64
   (folder/'plan.json').write_text(json.dumps(plan));(folder/'experiment.py').write_bytes((design.ORIGINAL/'experiment.py').read_bytes())
   with patch.object(design,'ORIGINAL',folder):
    with self.assertRaises(AssertionError):design.describe()
 def test_single_memo_tiny_stream_matches_memory_oracle_without_loading(self):
  with tempfile.TemporaryDirectory() as temp:
   result=c.count_fixture(Path(temp)/'run',16384)
   oracle=pickle.dumps(c.invented(16384),protocol=5)
   self.assertEqual(result['counter']['accepted_bytes'],len(oracle));self.assertEqual(result['counter']['accepted_prefix_sha256'],hashlib.sha256(oracle).hexdigest())
   self.assertEqual(result['stats']['fsync_calls'],2*result['counter']['write_calls'])
   self.assertEqual([p.name for p in (Path(temp)/'run').iterdir()],['progress.json'])
   separate=sum(len(pickle.dumps(value,protocol=5)) for value in c.invented(16384).values())
   self.assertGreater(separate,len(oracle))
 def test_prewrite_cap_is_not_complete_size_or_restore_evidence(self):
  with tempfile.TemporaryDirectory() as temp:
   result=c.count_fixture(Path(temp)/'run',65536,budget=32)
   self.assertEqual(result['status'],'tiny_prefix_capped_not_complete');self.assertLessEqual(result['counter']['accepted_bytes'],32)
   self.assertGreater(result['counter']['attempted_end'],32);self.assertFalse(result['public_completion_receipt'])
 def test_progress_fsync_failure_never_returns_a_complete_observation(self):
  with tempfile.TemporaryDirectory() as temp:
   with patch.object(c.os,'fsync',side_effect=OSError('invented fsync failure')):
    with self.assertRaises(OSError):c.count_fixture(Path(temp)/'run',128)
   self.assertTrue((Path(temp)/'run/failure.json').exists());self.assertFalse((Path(temp)/'run/success.json').exists())
 def test_unknown_actual_object_input_not_supported(self):
  with self.assertRaises(ValueError):c.invented('/owner/actual-graph')
  with self.assertRaises(ValueError):c.invented(65537)
 def test_tiny_perwrite_overhead_and_locally_observed_child(self):
  observations=[]
  with tempfile.TemporaryDirectory() as temp:
   for size in (0,1024,65536):
    for durable in (False,True):observations.append(c.count_fixture(Path(temp)/(str(size)+'-'+str(durable)),size,durable=durable))
   code='from pathlib import Path;import json,sys;import tiny_count;print(json.dumps(tiny_count.count_fixture(Path(sys.argv[1]),1024)))'
   child=subprocess.Popen([sys.executable,'-B','-c',code,str(Path(temp)/'child')],cwd=design.HERE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
   actual_pid=child.pid;stdout,stderr=child.communicate(timeout=5);self.assertEqual(child.returncode,0,stderr)
   returned=json.loads(stdout);self.assertEqual(returned['status'],'tiny_pickler_returned')
   print(json.dumps(dict(schema='TinySizeDesignMeasurementsV1',observations=observations,
    locally_observed_controller=dict(controller_pid=__import__('os').getpid(),child_pid=actual_pid,actual_returncode=child.returncode,reaped=True,stdout_sha256=hashlib.sha256(stdout.encode()).hexdigest(),provenance='Popen direct child observed locally; not external scientific authorization'),actual_tx=False)))

if __name__=='__main__':unittest.main(verbosity=2)
