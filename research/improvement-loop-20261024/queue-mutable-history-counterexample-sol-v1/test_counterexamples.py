import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import unittest

HERE=Path(__file__).resolve().parent
class Counterexamples(unittest.TestCase):
 results=[];start=time.monotonic()
 def run_case(self,name):
  result=subprocess.run([sys.executable,'-B',str(HERE/'probe.py'),name],capture_output=True,text=True,timeout=20)
  self.assertEqual(result.returncode,0,result.stderr)
  observation=json.loads(result.stdout);self.assertEqual(observation['cleanup']['live_owned_attempts_after_cleanup'],{})
  self.assertTrue(observation['cleanup']['temporary_domain_removed'])
  self.assertTrue(observation['cleanup']['recorded_child_terminals'])
  for terminal in observation['cleanup']['recorded_child_terminals'].values():
   self.assertEqual(terminal['return_code'],0);self.assertFalse(terminal['timed_out']);self.assertTrue(terminal['recorded_cleanup_verified'])
  self.results.append(observation)
 def test_v2_detects_changed_unvisited_readonly_historical_receipt(self):self.run_case('v2-history')
 def test_incremental_does_not_visit_unrelated_changed_history(self):self.run_case('incremental-history')
 def test_incremental_trusted_index_can_be_forged_to_free_live_capacity(self):self.run_case('incremental-forged-row')
 def test_v2_malformed_release_without_result_frees_live_capacity(self):self.run_case('v2-forged-release')
 def test_v1_malformed_release_without_result_frees_live_capacity(self):self.run_case('v1-forged-release')
 @classmethod
 def tearDownClass(cls):
  usage=resource.getrusage(resource.RUSAGE_CHILDREN)
  print(json.dumps(dict(schema='QueueMutableHistoryCounterexamplesV1',cases=cls.results,wall_seconds=time.monotonic()-cls.start,
    reaped_child_cpu_seconds=usage.ru_utime+usage.ru_stime,scientific_jobs=0,priority='nice 15',all_temporary_domains_removed=True),indent=2))

if __name__=='__main__':unittest.main(verbosity=2)
