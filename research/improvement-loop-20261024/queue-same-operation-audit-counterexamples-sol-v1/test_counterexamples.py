"""Fresh actual tiny queue histories; no synthetic SQL/released-history insertion."""
import copy,os,subprocess,sys,time,unittest
from pathlib import Path
import execution as ex
import experiment_queue as q
import test_experiment_queue as fixtures

class AuditCounterexamples(unittest.TestCase):
 def setUp(self):
  self.fx=fixtures.QueueTests(methodName='runTest');self.fx.setUp();self.addCleanup(self.fx.doCleanups)
  self.job=self.fx.job('tiny');self.fx.admit([self.job]);self.fx.run_queue();self.root=self.fx.root
  self.assertEqual(self.fx.result('tiny')['outcome'],'accepted_execution')
 def audit(self):return ex.ownership_audit(self.root/'executions',q.ownership_expected(self.root))
 def locked(self):
  _,fd=q.dispatcher(self.root);self.addCleanup(os.close,fd);return fd
 def test_same_D_same_size_mtime_input_edit_requires_live_refresh(self):
  self.locked();first=self.audit();source=self.fx.source;raw=source.read_bytes();before=source.stat();self.assertFalse(first['historical_input_drift'])
  # Edit fixture input bytes only after its tiny job completed; do not execute the changed script.
  changed=raw.replace(b'import hashlib',b'import hashliB',1);self.assertNotEqual(changed,raw);self.assertEqual(len(changed),len(raw))
  try:
   source.write_bytes(changed);os.utime(source,ns=(before.st_atime_ns,before.st_mtime_ns));fresh=self.audit()
   self.assertTrue(fresh['historical_input_drift']);self.assertEqual(fresh['evidence'],first['evidence']);self.assertFalse(copy.deepcopy(first)['historical_input_drift'])
  finally:source.write_bytes(raw)
 def test_same_D_same_bytes_job_lock_replacement_requires_identity(self):
  self.locked();first=self.audit();spec=self.job['execution_spec'];lock=ex.job_directory(self.root/'executions',spec['cycle_id'],spec['job_key'])/'lock';saved=self.fx.base/'original-lock';old=ex.lock_identity(lock);raw=lock.read_bytes()
  lock.rename(saved)
  try:
   lock.write_bytes(raw);self.assertNotEqual(ex.lock_identity(lock),old)
   with self.assertRaisesRegex(ex.ExecutionError,'job lock identity changed'):self.audit()
   self.assertTrue(first['evidence'])
  finally:os.replace(saved,lock)
 def test_same_D_namespace_edit_is_not_protected_by_cached_success(self):
  self.locked();first=self.audit();path=next((self.root/'executions/jobs').glob('*/claim-*.json'));raw=path.read_bytes()
  try:
   path.write_bytes(b'{}')
   with self.assertRaises(Exception):self.audit()
   self.assertTrue(first['evidence'])
  finally:path.write_bytes(raw)
 def test_cooperating_manager_blocks_on_D_not_on_input_files(self):
  self.locked();self.audit();start=time.monotonic()
  code='import experiment_queue as q,sys; q.status(sys.argv[1])'
  child=subprocess.run([sys.executable,'-B','-c',code,str(self.root)],capture_output=True,text=True,timeout=3)
  self.assertNotEqual(child.returncode,0);self.assertIn('LockBusy',child.stderr);self.assertLess(time.monotonic()-start,3)
  self.assertFalse(self.audit()['historical_input_drift'])
if __name__=='__main__':unittest.main(verbosity=2)
