"""Small real-executor integration; every owned fixture is cleaned up."""
import json,os,subprocess,sys,unittest
from unittest.mock import patch
import receipt_projection
from pathlib import Path
import experiment_queue as q
import finite_intake as f
from test_experiment_queue import QueueTests as Fixture

class FiniteTests(unittest.TestCase):
 def setUp(self):
  self.fx=Fixture(methodName='runTest');self.fx.setUp();self.addCleanup(self.fx.doCleanups);self.root=self.fx.root
 def plan(self,name='q0',mode='seed'):
  return dict(schema='LocalExperimentQueuePlanV1',queue_id=name,jobs=[self.fx.job(name,mode)])
 def child(self,operation,point=None,plan=None):
  code='import json,sys;from pathlib import Path;import experiment_queue as q;r=Path(sys.argv[1]);p=json.loads(sys.argv[2]);op=sys.argv[3];q.enqueue(r,p,q.digest(p)) if op=="enqueue" else q.seal(r,True) if op=="seal" else q.finalize(r)'
  env=dict(os.environ);env.pop('FINITE_QUEUE_CRASH',None)
  if point:env['FINITE_QUEUE_CRASH']=point
  return subprocess.run([sys.executable,'-B','-c',code,str(self.root),json.dumps(plan),operation],env=env,capture_output=True,text=True)
 def accepted(self,plan):return q.enqueue(self.root,plan,q.digest(plan))
 def done(self):return self.fx.run_queue()
 def test_missing_db_cannot_reset_unmaterialized_intent(self):
  p=self.plan();self.assertEqual(self.child('enqueue','intent_after_commit',p).returncode,91)
  (self.root/'intake.sqlite').unlink()
  with self.assertRaises(Exception):q.initialize(self.root,2)
 def test_same_process_projection_errors_recover(self):
  q.seal(self.root,True)
  for i in range(3):
   with patch.object(receipt_projection.os,'link',side_effect=OSError('invented publication failure')):
    with self.assertRaises(OSError):q.finalize(self.root)
  self.assertEqual(len(list(self.root.glob('receipt-run-*.tmp'))),3)
  self.assertEqual(q.finalize(self.root),q.finalize(self.root))
 def test_current_source_and_old_schema_rejected_without_writes(self):
  store=self.root/'store.json';original=store.read_bytes();value=json.loads(original);value['schema']='LocalExperimentQueueStoreV3';store.write_text(json.dumps(value));before={str(p.relative_to(self.root)):p.read_bytes()for p in self.root.rglob('*')if p.is_file()}
  with self.assertRaises(Exception):q.work(self.root,self.fx.base/'must-not-exist',.01,.01)
  self.assertFalse((self.fx.base/'must-not-exist').exists());self.assertEqual(before,{str(p.relative_to(self.root)):p.read_bytes()for p in self.root.rglob('*')if p.is_file()});store.write_bytes(original)
 def test_corruption_after_finalization_is_not_accepted(self):
  p=self.plan();self.accepted(p);q.seal(self.root);self.done();pin=q.finalize(self.root);raw=Path(pin['path']).read_bytes();result=self.root/'queues'/'q0'/'jobs'/'q0'/'result.json';result.write_text('{}')
  with self.assertRaises(Exception):q.finalize(self.root)
  self.assertEqual(Path(pin['path']).read_bytes(),raw)
 def test_real_dependency_and_open_idle(self):
  p=dict(schema='LocalExperimentQueuePlanV1',queue_id='q0',jobs=[self.fx.job('a'),self.fx.job('b',producer='a')]);self.accepted(p);self.done()
  with self.assertRaises(Exception):q.finalize(self.root)
  q.seal(self.root);pin=q.finalize(self.root);doc=json.loads(Path(pin['path']).read_text());self.assertEqual(doc['disposition'],'all_succeeded');self.assertEqual(q.finalize(self.root),pin)
 def test_real_failure_blocks_dependency(self):
  p=dict(schema='LocalExperimentQueuePlanV1',queue_id='q0',jobs=[self.fx.job('a','nonzero'),self.fx.job('b',producer='a')]);self.accepted(p);q.seal(self.root);self.done();pin=q.finalize(self.root);self.assertEqual(json.loads(Path(pin['path']).read_text())['disposition'],'terminal_with_failures');self.assertEqual(self.fx.result('b')['outcome'],'blocked_predecessor')
 def test_admission_crashes_repair_after_seal(self):
  for n,point in enumerate(['intent_after_commit','admission_after_scaffold','admission_after_plan','admission_after_receipt','accepted_before_commit']):
   # independent fresh store per crash position; no workers launched in this test.
   self.root=self.fx.base/('store'+str(n));q.initialize(self.root,1);p=self.plan('job'+str(n));r=self.child('enqueue',point,p);self.assertEqual(r.returncode,91,r.stderr);self.assertEqual(f.info(self.root)['admitting'],1);q.seal(self.root)
   with self.assertRaises(Exception):q.finalize(self.root)
   with self.assertRaises(Exception):q.tick(self.root)
   self.assertFalse((self.root/'executions').exists())
   with self.assertRaises(Exception):self.accepted(self.plan('other'+str(n)))
   self.accepted(p);self.assertEqual(f.info(self.root)['admitting'],0);self.assertTrue((self.root/'queues'/p['queue_id']/'jobs'/p['jobs'][0]['id']).is_dir())
 def test_precommit_rolls_back_and_empty_policy(self):
  p=self.plan();self.assertEqual(self.child('enqueue','intent_before_commit',p).returncode,91);self.assertEqual(f.info(self.root)['intents'],0)
  self.assertEqual(self.child('seal','seal_before_commit').returncode,91);self.assertEqual(f.info(self.root)['state'],'OPEN')
  q.seal(self.root)
  with self.assertRaises(Exception):q.finalize(self.root)
 def test_projection_crashes_idempotent(self):
  for n,point in enumerate(['final_before_commit','final_after_commit','receipt_before_publish','receipt_after_publish','receipt_after_dirsync']):
   self.root=self.fx.base/('final'+str(n));q.initialize(self.root,1);q.seal(self.root,True);r=self.child('finalize',point);self.assertEqual(r.returncode,91,r.stderr);pin=q.finalize(self.root);self.assertEqual(q.finalize(self.root),pin);self.assertEqual(json.loads(Path(pin['path']).read_text())['disposition'],'explicit_empty')
 def test_live_and_corrupt_release_block_final(self):
  p=self.plan('hold','hold');self.accepted(p);q.seal(self.root);q.tick(self.root)
  self.fx.wait(lambda:(self.fx.base/'output-hold'/'receipt.json').exists())
  directory=self.root/'queues'/'hold'/'jobs'/'hold';(directory/'release.json').write_text('{}')
  with self.assertRaises(Exception):q.finalize(self.root)
  self.assertGreater(q.tick(self.root)['reserved'],0)
 def test_corrupt_accepted_scaffold_rejected(self):
  p=self.plan();self.accepted(p);(self.root/'queues'/'q0'/'jobs'/'q0').rmdir()
  with self.assertRaises(Exception):q.tick(self.root)
 def test_concurrent_enqueue_seal(self):
  for n in range(6):
   self.root=self.fx.base/('race'+str(n));q.initialize(self.root,1);p=self.plan('racejob'+str(n));gate=self.fx.base/('gate'+str(n))
   code='import json,sys,time;from pathlib import Path;import experiment_queue as q;r=Path(sys.argv[1]);g=Path(sys.argv[2]);p=json.loads(sys.argv[3]);op=sys.argv[4]\nwhile not g.exists():time.sleep(.001)\ntry:\n q.enqueue(r,p,q.digest(p)) if op=="enqueue" else q.seal(r,True)\nexcept Exception:sys.exit(3)'
   kids=[subprocess.Popen([sys.executable,'-B','-c',code,str(self.root),str(gate),json.dumps(p),op],stdout=subprocess.PIPE,stderr=subprocess.PIPE)for op in ['enqueue','seal']];gate.touch();outputs=[k.communicate(timeout=10)for k in kids];codes=[k.returncode for k in kids];self.assertEqual(codes[1],0);self.assertIn(codes[0],[0,3]);self.assertEqual(f.info(self.root)['intents'],int(codes[0]==0));self.assertEqual(f.info(self.root)['state'],'SEALED')

if __name__=='__main__':unittest.main()
