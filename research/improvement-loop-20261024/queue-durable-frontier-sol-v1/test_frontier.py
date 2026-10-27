"""Real tiny commands, SQLite transaction death and competing local writers only."""
import copy,json,multiprocessing as mp,os,resource,sqlite3,tempfile,time,unittest
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CPU,(40,45));resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,)*2);resource.setrlimit(resource.RLIMIT_FSIZE,(16*1024**2,)*2)
import dispatch as d
import test_experiment_queue as fixtures

def die_admit(root,plan,stage):d.enqueue(root,plan,d.q.digest(plan),lambda at:os._exit(84)if at==stage else None)
def die_tick(root,stage):d.tick(root,lambda at:os._exit(85)if at==stage else None)
def competing(root,plan,messages):
 try:d.enqueue(root,plan,d.q.digest(plan));messages.put('accepted')
 except (ValueError,sqlite3.IntegrityError):messages.put('rejected')
def tick_once(root,messages):
 try:messages.put(d.tick(root))
 except BaseException as e:messages.put({'error':repr(e)})
class FrontierTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.base=Path(self.tmp.name);self.root=d.initialize(self.base/'queue');self.f=fixtures.QueueTests('runTest');self.f.base=self.base;self.f.root=self.root;self.f.source=self.base/'owner.py';self.f.source.write_text(fixtures.FIXTURE);self.addCleanup(self.f.cleanup)
 def plan(self,name='plan',mode='seed'):
  return dict(schema='LocalExperimentQueuePlanV1',queue_id=name,jobs=[self.f.job('consumer',producer='producer'),self.f.job('producer',mode)])
 def saved(self,name):
  with d.opened(self.root/'admissions.sqlite')as db:return d.row_value(db,'releases',name)
 def release(self,name):
  until=time.monotonic()+10
  while time.monotonic()<until:
   observation=d.tick(self.root)
   if self.saved(name):return self.saved(name)
   time.sleep(.02)
  self.fail(repr(observation))
 def crash(self,fn,args,code):
  child=mp.Process(target=fn,args=args);child.start();child.join(8);self.assertFalse(child.is_alive());self.assertEqual(child.exitcode,code)
 def audit(self):
  with d.locked(self.root)as(root,db):return d.audit_frontier(db)
 def test_atomic_admission_death_and_exact_retry(self):
  for stage in ['evidence','owners','frontier_inserted','committed']:
   p=dict(schema='LocalExperimentQueuePlanV1',queue_id=stage,jobs=[self.f.job(stage)])
   self.crash(die_admit,(self.root,p,stage),84)
   with d.opened(self.root/'admissions.sqlite')as db:
    n=1 if stage=='committed'else 0
    for table in ['evidence','owners','pending']:self.assertEqual(db.execute('SELECT count(*) FROM '+table).fetchone()[0],n)
   d.enqueue(self.root,p,d.q.digest(p));self.assertEqual(self.audit()['pending'],1)
   # Separate fresh domain per boundary; no jobs run here.
   self.root=d.initialize(self.base/('queue-'+stage));self.f.root=self.root
 def test_real_chain_reserve_release_crashes_and_replay(self):
  p=self.plan();d.enqueue(self.root,p,d.q.digest(p))
  for stage in ['binding_published','frontier_reserved','reserved','prepared','launched']:self.crash(die_tick,(self.root,stage),85)
  # Wait for clean executor terminal without releasing through the dispatcher.
  until=time.monotonic()+8
  while time.monotonic()<until:
   state=d.ex.inspect(self.root/'executions','producer')
   if state['attempt_terminal']and not state['job_lock_held']and not state['session_members']and all(v['state']!='alive'for v in state['process_observation'].values()):break
   time.sleep(.02)
  else:self.fail('executor terminal timeout')
  for stage in ['result_published','release_published','release_inserted','frontier_released','released']:self.crash(die_tick,(self.root,stage),85)
  self.assertEqual(self.audit(),{'active':0,'pending':1,'released':1,'historical_input_audit':False})
  with d.opened(self.root/'admissions.sqlite')as db:self.assertEqual(db.execute('SELECT attempt,remaining,blocked FROM pending').fetchall(),[('consumer',0,None)])
  self.release('consumer');self.assertEqual((self.base/'output-consumer/data.txt').read_text(),'seed:consumed');self.assertEqual(self.audit()['released'],2);self.assertEqual(d.tick(self.root)['reserved'],0)
 def test_failed_producer_never_wakes_consumer(self):
  p=self.plan(mode='nonzero');d.enqueue(self.root,p,d.q.digest(p));self.release('producer');self.assertEqual(self.audit()['pending'],1)
  with d.opened(self.root/'admissions.sqlite')as db:self.assertEqual(db.execute('SELECT remaining,blocked FROM pending WHERE attempt=?',('consumer',)).fetchone(),(1,'failed producer producer'))
  self.assertEqual(d.tick(self.root)['reserved'],0);self.assertFalse((self.root/'executions/attempts/consumer').exists())
 def test_concurrent_duplicate_claim_and_capacity_single_launch(self):
  a=dict(schema='LocalExperimentQueuePlanV1',queue_id='a',jobs=[self.f.job('one')]);b=copy.deepcopy(a);b['queue_id']='b';messages=mp.Queue();clients=[mp.Process(target=competing,args=(self.root,p,messages))for p in [a,b]]
  for child in clients:child.start()
  for child in clients:child.join(8);self.assertEqual(child.exitcode,0)
  self.assertEqual(sorted(messages.get(timeout=2)for _ in clients),['accepted','rejected']);ticks=[mp.Process(target=tick_once,args=(self.root,messages))for _ in range(2)]
  for child in ticks:child.start()
  for child in ticks:child.join(8);self.assertEqual(child.exitcode,0)
  observed=[messages.get(timeout=2)for _ in ticks];self.assertTrue(all('error'not in row for row in observed));self.release('one');self.assertEqual(len(list((self.root/'executions/attempts').glob('*/launch-intent.json'))),1);self.assertEqual(self.audit()['released'],1)
 def test_two_ready_owners_competing_ticks_reserve_one(self):
  p=dict(schema='LocalExperimentQueuePlanV1',queue_id='two',jobs=[self.f.job('first'),self.f.job('second')]);d.enqueue(self.root,p,d.q.digest(p));messages=mp.Queue();clients=[mp.Process(target=tick_once,args=(self.root,messages))for _ in range(2)]
  for child in clients:child.start()
  for child in clients:child.join(8);self.assertEqual(child.exitcode,0)
  states=[messages.get(timeout=2)for _ in clients];self.assertTrue(all('error'not in row for row in states))
  with d.opened(self.root/'admissions.sqlite')as db:self.assertEqual(db.execute('SELECT count(*) FROM reservations').fetchone()[0],1);self.assertLessEqual(db.execute('SELECT count(*) FROM active').fetchone()[0],1)
  self.release('first');self.release('second');self.assertEqual(self.audit()['released'],2);self.assertEqual(len(list((self.root/'executions/attempts').glob('*/launch-intent.json'))),2)
 def test_failed_predecessor_leaves_grandchild_waiting_without_completion(self):
  p=dict(schema='LocalExperimentQueuePlanV1',queue_id='three-fail',jobs=[self.f.job('last',producer='middle'),self.f.job('middle',producer='first'),self.f.job('first','nonzero')]);d.enqueue(self.root,p,d.q.digest(p));self.release('first')
  with d.opened(self.root/'admissions.sqlite')as db:self.assertEqual(db.execute('SELECT attempt,remaining,blocked FROM pending ORDER BY seq').fetchall(),[('last',1,None),('middle',1,'failed producer first')])
  state=d.tick(self.root);self.assertEqual(state['reserved'],0);self.assertEqual(state['queue_completion'],'not_inferred_from_empty_frontier');self.assertEqual(self.audit()['pending'],2);self.assertFalse((self.root/'executions/attempts/middle').exists());self.assertFalse((self.root/'executions/attempts/last').exists())
 def test_selected_dependency_tamper_still_blocks(self):
  p=self.plan();d.enqueue(self.root,p,d.q.digest(p));self.release('producer');saved=self.saved('producer')[0];path=Path(saved['result']['path']);path.write_bytes(path.read_bytes()+b' ');observation=d.tick(self.root);self.assertEqual(observation['state'],'blocked_frontier_integrity');self.assertFalse((self.root/'executions/attempts/consumer').exists())
 def test_domains_and_projection_audit_reject_corruption(self):
  with self.assertRaises(ValueError):d.admission.opened(self.root/'admissions.sqlite')
  p=self.plan();d.enqueue(self.root,p,d.q.digest(p))
  with d.opened(self.root/'admissions.sqlite')as db:db.execute('UPDATE pending SET remaining=0 WHERE attempt=?',('consumer',))
  with self.assertRaisesRegex(ValueError,'remaining dependency'):self.audit()
  self.assertEqual(d.tick(self.root)['state'],'blocked_frontier_integrity')
if __name__=='__main__':unittest.main(verbosity=2)
