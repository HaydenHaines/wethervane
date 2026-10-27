"""Actual tiny subprocess integration, no scientific workload."""
import copy,json,os,subprocess,sys,time,unittest,sqlite3
from pathlib import Path
from unittest.mock import patch
import experiment_queue as q
import execution as ex
from test_experiment_queue import QueueTests as Fixture

class IntegrationTests(unittest.TestCase):
 def setUp(self):
  self.fx=Fixture(methodName='runTest');self.fx.setUp();self.root=self.fx.root;self.addCleanup(self.fx.doCleanups)
 def plan(self,name):return dict(schema='LocalExperimentQueuePlanV1',queue_id=name,jobs=[self.fx.job(name)])
 def run_queue(self):return self.fx.run_queue()
 def test_real_all_preparation_deaths_repair_same_reservation(self):
  for n,point in enumerate(['before_intent','after_intent','after_directory','after_specification','after_prepared','after_output_claim','after_job_claim','after_ownership_ack','before_executor_launch']):
   p=self.plan('j'+str(n));q.enqueue(self.root,p,q.digest(p));env=dict(os.environ);env['OWNERSHIP_CRASH']=point;env['FINITE_QUEUE_CRASH']=point
   child=subprocess.run([sys.executable,'-B','-c','import experiment_queue as q,sys;q.tick(sys.argv[1])',str(self.root)],env=env,capture_output=True,text=True);self.assertIn(child.returncode,[91,93],child.stderr)
   directory=self.root/'queues'/p['queue_id']/'jobs'/p['jobs'][0]['id'];reservation=ex.pin(directory/'reservation.json');self.run_queue();self.assertEqual(ex.pin(directory/'reservation.json'),reservation);self.assertEqual(q.document(directory/'result.json')['outcome'],'accepted_execution')
   self.assertEqual(len(list((self.root/'executions'/'attempts'/p['jobs'][0]['id']).glob('launch-intent.json'))),1)
 def test_publish_oserror_keeps_partial_pending_then_replays(self):
  p=self.plan('partial');q.enqueue(self.root,p,q.digest(p));publish=ex.publish;once=[]
  def fail(path,value):
   result=publish(path,value)
   if Path(path).name=='prepared.json'and not once:once.append(True);raise OSError('invented acknowledgment error')
   return result
  with patch.object(ex,'publish',side_effect=fail):state=q.tick(self.root)
  self.assertEqual(state['reserved'],1);self.assertFalse((self.root/'queues'/'partial'/'jobs'/'partial'/'result.json').exists());self.run_queue()
 def test_released_ownership_mutations_cannot_finalize_or_release_again(self):
  p=self.plan('a');q.enqueue(self.root,p,q.digest(p));self.run_queue();q.seal(self.root)
  expected=[self.root/'executions'/'ownership-store.json',self.root/'executions'/'ownership-intents'/'a.json',self.root/'executions'/'attempts'/'a'/'prepared.json']
  expected+=list((self.root/'executions'/'outputs').iterdir());expected+=list((self.root/'executions'/'jobs').glob('*/claim-*.json'))
  for path in expected:
   raw=path.read_bytes();path.write_text('{}')
   config,fd=q.dispatcher(self.root)
   try:
    with self.assertRaises(Exception):q.verified_result(self.root/'queues'/'a'/'jobs'/'a',p['jobs'][0])
   finally:os.close(fd)
   self.assertFalse((self.root/'final-run.json').exists());path.write_bytes(raw)
  # Verify finalization separately so sticky quarantine cannot mask the per-artifact checks.
  expected[-1].write_text('{}')
  with self.assertRaises(Exception):q.finalize(self.root)
  self.assertFalse((self.root/'final-run.json').exists())
 def test_final_pins_effective_spec_and_ack(self):
  p=dict(schema='LocalExperimentQueuePlanV1',queue_id='chain',jobs=[self.fx.job('a'),self.fx.job('b',producer='a')]);q.enqueue(self.root,p,q.digest(p));self.run_queue();q.seal(self.root);pin=q.finalize(self.root);final=q.document(pin['path']);paths={v['path']for v in final['evidence']}
  for name in ['a','b']:
   intent=self.root/'executions'/'ownership-intents'/(name+'.json');self.assertIn(str(intent),paths);bound=q.document(self.root/'queues'/'chain'/'jobs'/name/'binding.json');self.assertEqual(q.document(intent)['specification'],bound['execution_spec']);self.assertIn(str(self.root/'queues'/'chain'/'jobs'/name/'ownership-ack.json'),paths)
  self.assertNotIn(q.SHA_ARGUMENT,q.document(self.root/'executions'/'ownership-intents'/'b.json')['specification']['argv']);self.assertEqual(q.finalize(self.root),pin)
 def test_direct_prepare_and_launch_excluded_from_managed_domain(self):
  s=self.fx.job('rogue')['execution_spec'];before=list(self.root.iterdir())
  with self.assertRaisesRegex(ValueError,'dispatcher'):ex.prepare(self.root/'executions',s,q.digest(s))
  self.assertEqual(list(self.root.iterdir()),before)
  config,fd=q.dispatcher(self.root)
  try:
   with self.assertRaisesRegex(ValueError,'reservation'):ex.prepare(self.root/'executions',s,q.digest(s))
  finally:os.close(fd)
  p=self.plan('a');q.enqueue(self.root,p,q.digest(p));original=ex.launch
  with patch.object(ex,'launch',side_effect=SystemExit('before launch')):
   with self.assertRaises(SystemExit):q.tick(self.root)
  with self.assertRaisesRegex(ValueError,'dispatcher'):original(self.root/'executions','a')
  self.run_queue()
 def held(self,path):
  code='import fcntl,sys;f=open(sys.argv[1],"a");fcntl.flock(f,fcntl.LOCK_EX);print("ready",flush=True);sys.stdin.read()'
  p=subprocess.Popen([sys.executable,'-B','-c',code,str(path)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True);self.assertEqual(p.stdout.readline().strip(),'ready');return p
 def release_holder(self,p):
  p.stdin.close();p.wait(5);p.stdout.close();p.stderr.close()
 def test_bounded_ownership_and_dispatcher_contention(self):
  p=self.plan('a');q.enqueue(self.root,p,q.digest(p));self.run_queue();p2=self.plan('b');q.enqueue(self.root,p2,q.digest(p2))
  holder=self.held(self.root/'executions'/'ownership.lock')
  try:
   start=time.monotonic();state=q.tick(self.root);self.assertLess(time.monotonic()-start,3);self.assertFalse((self.root/'executions'/'attempts'/'b'/'launch-intent.json').exists())
  finally:self.release_holder(holder)
  self.run_queue();holder=self.held(self.root/'dispatcher.lock')
  try:
   with self.assertRaises(ex.LockBusy):q.tick(self.root)
  finally:self.release_holder(holder)
 def test_historical_input_drift_reported_not_global_preparation_block(self):
  # Preserve the original regression, and additionally require the final audit observation.
  case=Fixture(methodName='test_prepared_input_drift_defers_without_terminal_and_unrelated_job_runs');case.setUp()
  try:
   case.test_prepared_input_drift_defers_without_terminal_and_unrelated_job_runs();q.seal(case.root);pin=q.finalize(case.root);doc=q.document(pin['path']);self.assertIn('first',doc['observation']['ownership']['historical_input_drift']);self.assertEqual(doc['disposition'],'terminal_with_failures')
  finally:case.doCleanups()
 def test_exact_publication_retry_reasserts_file_and_directory_sync(self):
  path=self.fx.base/'record.json';ex.publish_exact(path,dict(value='same'));calls=[];sync=ex.os.fsync
  def observe(fd):calls.append(fd);return sync(fd)
  with patch.object(ex.os,'fsync',side_effect=observe):ex.publish_exact(path,dict(value='same'))
  self.assertEqual(len(calls),2)
  with patch.object(ex.os,'fsync',side_effect=OSError('invented uncertain sync')):
   with self.assertRaises(OSError):ex.publish_exact(path,dict(value='same'))
  ex.publish_exact(path,dict(value='same'))
 def test_symlinked_managed_execution_root_cannot_downgrade(self):
  elsewhere=self.fx.base/'elsewhere';elsewhere.mkdir();(self.root/'executions').symlink_to(elsewhere,target_is_directory=True);s=self.fx.job('rogue')['execution_spec']
  with self.assertRaisesRegex(ValueError,'namespace alias'):ex.prepare(self.root/'executions',s,q.digest(s))
  self.assertFalse((elsewhere/'ownership-store.json').exists())
 def test_held_dispatcher_and_reservation_do_not_bypass_admission_drift(self):
  p=self.plan('a');q.enqueue(self.root,p,q.digest(p))
  with patch.object(ex,'launch',side_effect=SystemExit('before launch')):
   with self.assertRaises(SystemExit):q.tick(self.root)
  (self.root/'queues'/'a'/'receipt.json').write_text('accidentally corrupted admission')
  config,fd=q.dispatcher(self.root)
  try:
   with self.assertRaises((ValueError,OSError)):ex.launch(self.root/'executions','a')
  finally:os.close(fd)
  self.assertFalse((self.root/'executions'/'attempts'/'a'/'launch-intent.json').exists())
 def test_corrupt_managed_marker_cannot_downgrade_before_first_prepare(self):
  marker=self.root/'execution-domain.json';value=q.document(marker);value['execution_root']=str(self.fx.base/'elsewhere');marker.write_text(json.dumps(value));s=self.fx.job('rogue')['execution_spec']
  with self.assertRaisesRegex(ValueError,'execution-root mismatch'):ex.prepare(self.root/'executions',s,q.digest(s))
  self.assertFalse((self.root/'executions').exists())
 def test_unrelated_parent_marker_does_not_claim_standalone_domain(self):
  s=self.fx.job('standalone')['execution_spec'];root=self.root/'independent-executor';ex.prepare(root,s,q.digest(s));self.assertIsNone(q.document(root/'ownership-store.json')['domain'])
 def test_sqlite_contention_is_bounded_without_seal_commit(self):
  db=sqlite3.connect(self.root/'intake.sqlite',isolation_level=None);db.execute('BEGIN IMMEDIATE')
  try:
   start=time.monotonic()
   with self.assertRaises(sqlite3.OperationalError):q.seal(self.root,True)
   self.assertLess(time.monotonic()-start,3)
  finally:db.rollback();db.close()
  self.assertEqual(q.finite_intake.info(self.root)['state'],'OPEN')
 def test_post_final_ownership_mutation_preserves_original_receipt(self):
  p=self.plan('a');q.enqueue(self.root,p,q.digest(p));self.run_queue();q.seal(self.root);pin=q.finalize(self.root);raw=Path(pin['path']).read_bytes();intent=self.root/'executions'/'ownership-intents'/'a.json';intent.write_text('{}')
  with self.assertRaises(Exception):q.finalize(self.root)
  self.assertEqual(Path(pin['path']).read_bytes(),raw)
 def test_unknown_external_claim_and_missing_domain_fail_closed(self):
  domain=self.root/'execution-domain.json';raw=domain.read_bytes();domain.unlink();s=self.fx.job('rogue')['execution_spec']
  with self.assertRaisesRegex(ValueError,'marker'):ex.prepare(self.root/'executions',s,q.digest(s))
  domain.write_bytes(raw);p=self.plan('a');q.enqueue(self.root,p,q.digest(p));self.run_queue();q.seal(self.root)
  unknown=self.root/'executions'/'attempts'/'unknown';unknown.mkdir()
  with self.assertRaises(ValueError):q.finalize(self.root)

if __name__=='__main__':unittest.main()
