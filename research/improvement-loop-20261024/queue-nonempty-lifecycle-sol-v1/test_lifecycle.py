"""Tiny actual executor jobs and SQLite transaction process death; no model runs."""
import copy,json,multiprocessing as mp,os,resource,signal,tempfile,time,unittest
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CPU,(40,45));resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,)*2);resource.setrlimit(resource.RLIMIT_FSIZE,(16*1024**2,)*2)
import dispatch as d
import consumer
import test_experiment_queue as fixtures

def die(root,stage):d.tick(root,lambda at:os._exit(86)if at==stage else None)
def die_admit(root,p):d.enqueue(root,p,d.q.digest(p),lambda at:os._exit(87)if at=='frontier_inserted'else None)
def enqueue_child(root,p,messages):
 try:d.enqueue(root,p,d.q.digest(p));messages.put('admitted')
 except BaseException as e:messages.put(repr(e))
def tick_child(root,messages):
 try:messages.put(d.tick(root))
 except BaseException as e:messages.put({'error':repr(e)})
class LifecycleTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.base=Path(self.tmp.name);self.root=d.initialize(self.base/'q');self.f=fixtures.QueueTests('runTest');self.f.base=self.base;self.f.root=self.root;self.f.source=self.base/'owner.py';self.f.source.write_text(fixtures.FIXTURE);self.addCleanup(self.f.cleanup)
 def enqueue(self,jobs,name='plan'):
  p=dict(schema='LocalExperimentQueuePlanV1',queue_id=name,jobs=jobs);d.enqueue(self.root,p,d.q.digest(p));return p
 def state(self):
  with d.locked(self.root)as(root,db):return d.completion(db)
 def audit(self):
  with d.locked(self.root)as(root,db):return d.audit_frontier(db)
 def saved(self,attempt):
  with d.opened(self.root/'admissions.sqlite')as db:return d.row_value(db,'releases',attempt)
 def release(self,attempt):
  until=time.monotonic()+10
  while time.monotonic()<until:
   row=d.tick(self.root)
   if self.saved(attempt):return row
   time.sleep(.02)
  self.fail(repr(row))
 def terminal(self,attempt):
  until=time.monotonic()+8
  while time.monotonic()<until:
   state=d.ex.inspect(self.root/'executions',attempt)
   if state['attempt_terminal']and not state['job_lock_held']and not state['session_members']and all(v['state']!='alive'for v in state['process_observation'].values()):return
   time.sleep(.02)
  self.fail('executor terminal timeout')
 def crash(self,stage):
  child=mp.Process(target=die,args=(self.root,stage));child.start();child.join(8);self.assertFalse(child.is_alive());self.assertEqual(child.exitcode,86)
 def multiroot(self):
  right=self.f.job('right',producer='root1');spec=right['execution_spec'];spec['argv'].extend([str(self.base/'output-root2/receipt.json'),d.q.SHA_ARGUMENT]);right['dependencies'].append(dict(job_id='root2',role='receipt',path_arguments=[6],sha256_arguments=[7]))
  join=self.f.job('join',producer='left');spec=join['execution_spec'];spec['argv'].extend([str(self.base/'output-right/receipt.json'),d.q.SHA_ARGUMENT]);join['dependencies'].append(dict(job_id='right',role='receipt',path_arguments=[6],sha256_arguments=[7]))
  return [self.f.job('last',producer='join'),join,self.f.job('left',producer='root1'),right,self.f.job('root1','nonzero'),self.f.job('root2','nonzero')]
 def blocked(self):
  with d.opened(self.root/'admissions.sqlite')as db:return db.execute('SELECT attempt,cause,body,digest FROM blocked_terminal ORDER BY attempt').fetchall()
 def test_failure_closure_crash_atomicity_diamond_first_cause_no_repeat(self):
  self.enqueue(self.multiroot());d.tick(self.root);self.terminal('root1')
  for stage in ['release_inserted','descendant_blocked','lifecycle_updated','frontier_released']:
   self.crash(stage);self.assertFalse(self.saved('root1'));self.assertEqual(self.blocked(),[]);state=self.audit();self.assertEqual((state['active'],state['pending'],state['blocked_terminal']),(1,5,0));self.assertFalse(state['nonempty_current_admissions_all_terminal'])
  self.crash('released');rows=self.blocked();self.assertEqual([row[0]for row in rows],['join','last','left','right']);self.assertTrue(all(row[1]=='root1'for row in rows));self.assertEqual(self.audit()['pending'],1);self.assertFalse(self.state()['nonempty_current_admissions_all_terminal'])
  self.release('root2');self.assertEqual(self.blocked(),rows);state=self.audit();self.assertEqual((state['executed_terminal'],state['blocked_terminal'],state['failed_execution']),(2,4,2));self.assertTrue(state['nonempty_current_admissions_all_terminal']);self.assertFalse(state['nonempty_current_admissions_all_accepted_unqualified']);self.assertEqual(state['state'],'nonempty_current_admissions_terminal_with_failures_or_blocks')
  for _ in range(3):d.tick(self.root)
  self.assertEqual(self.blocked(),rows);self.assertEqual(self.audit(),state)
  for name in ['join','last','left','right']:self.assertFalse((self.root/'executions/attempts'/name).exists())
 def test_reservation_counter_crashes_replay_without_duplicate_launch(self):
  self.enqueue([self.f.job('only')]);self.crash('frontier_reserved');state=self.audit();self.assertEqual((state['pending'],state['active'],state['executed_terminal']),(1,0,0));self.assertFalse(state['nonempty_current_admissions_all_terminal']);self.crash('reserved');state=self.audit();self.assertEqual((state['pending'],state['active']),(0,1));self.assertFalse(state['nonempty_current_admissions_all_terminal']);self.crash('prepared');self.crash('launched');self.terminal('only');self.release('only');self.assertTrue(self.audit()['nonempty_current_admissions_all_accepted_unqualified']);self.assertEqual(len(list((self.root/'executions/attempts').glob('*/launch-intent.json'))),1)
 def test_success_completion_and_new_admission_reopens_snapshot(self):
  self.assertEqual(self.state()['state'],'empty_no_admitted_work');self.assertFalse(self.state()['nonempty_current_admissions_all_accepted_unqualified']);self.enqueue([self.f.job('c',producer='p'),self.f.job('p')]);self.assertFalse(self.state()['nonempty_current_admissions_all_terminal']);self.release('c');state=self.audit();self.assertEqual(state['state'],'nonempty_current_admissions_terminal_all_accepted_unqualified');self.assertEqual(state['accepted_execution'],2);self.assertTrue(state['nonempty_current_admissions_all_terminal'])
  self.enqueue([self.f.job('new')],'new-plan');self.assertFalse(self.state()['nonempty_current_admissions_all_terminal']);self.release('new');self.assertEqual(self.audit()['accepted_execution'],3)
 def test_admission_counter_rollback_and_waiting_corruption_not_complete(self):
  p=dict(schema='LocalExperimentQueuePlanV1',queue_id='crash',jobs=[self.f.job('a')]);child=mp.Process(target=die_admit,args=(self.root,p));child.start();child.join(8);self.assertEqual(child.exitcode,87);self.assertEqual(self.audit()['total'],0);d.enqueue(self.root,p,d.q.digest(p))
  with d.opened(self.root/'admissions.sqlite')as db:db.execute('UPDATE pending SET remaining=1')
  self.assertEqual(self.state()['state'],'nonempty_current_admissions_waiting_unresolved');self.assertFalse(self.state()['nonempty_current_admissions_all_terminal'])
  with self.assertRaisesRegex(ValueError,'remaining dependency'):self.audit()
 def test_concurrent_admission_tick_and_failure_uses_one_capacity(self):
  self.enqueue([self.f.job('desc',producer='bad'),self.f.job('bad','nonzero')]);messages=mp.Queue();new=dict(schema='LocalExperimentQueuePlanV1',queue_id='other',jobs=[self.f.job('other')]);clients=[mp.Process(target=enqueue_child,args=(self.root,new,messages)),mp.Process(target=tick_child,args=(self.root,messages)),mp.Process(target=tick_child,args=(self.root,messages))]
  for child in clients:child.start()
  for child in clients:child.join(8);self.assertEqual(child.exitcode,0)
  values=[messages.get(timeout=2)for _ in clients];self.assertIn('admitted',values);self.assertTrue(all(not isinstance(v,dict)or'error'not in v for v in values));self.release('bad');self.release('other');state=self.audit();self.assertEqual((state['total'],state['blocked_terminal'],state['accepted_execution'],state['failed_execution']),(3,1,1,1));self.assertTrue(state['nonempty_current_admissions_all_terminal']);self.assertFalse((self.root/'executions/attempts/desc').exists())
 def test_unknown_exit_retains_active_not_terminal_and_no_descendant_block(self):
  self.enqueue([self.f.job('desc',producer='hold'),self.f.job('hold','hold')]);d.tick(self.root);until=time.monotonic()+8
  while not(self.base/'output-hold/receipt.json').exists()and time.monotonic()<until:time.sleep(.02)
  self.assertTrue((self.base/'output-hold/receipt.json').exists());directory=self.root/'executions/attempts/hold';identity=d.q.document(directory/'supervisor.json')['identity'];fd=d.ex.open_pidfd(identity);signal.pidfd_send_signal(fd,signal.SIGKILL);os.close(fd);until=time.monotonic()+5
  while d.ex.observe(identity)['state']=='alive'and time.monotonic()<until:time.sleep(.02)
  d.ex.stop(self.root/'executions','hold');until=time.monotonic()+5
  while d.ex.inspect(self.root/'executions','hold')['job_lock_held']and time.monotonic()<until:time.sleep(.02)
  d.ex.recover(self.root/'executions','hold');obs=d.tick(self.root);self.assertEqual(obs['reserved'],1);self.assertIn('unknown_exit',obs['observations']['hold']);state=self.audit();self.assertFalse(state['nonempty_current_admissions_all_terminal']);self.assertEqual(state['blocked_terminal'],0);self.assertFalse(self.saved('hold'));self.assertFalse((self.root/'executions/attempts/desc').exists())
 def test_balanced_but_false_zero_counter_cannot_report_completion(self):
  self.enqueue([self.f.job('pending')])
  with d.opened(self.root/'admissions.sqlite')as db:db.execute('UPDATE counts SET total=0,pending=0')
  with self.assertRaisesRegex(ValueError,'pending counter presence mismatch'):self.state()
  with self.assertRaisesRegex(ValueError,'lifecycle counter projection mismatch'):self.audit()
 def test_direct_machine_consumer_refuses_empty_active_failed_blocked_and_legacy(self):
  empty=self.state();self.assertFalse(empty['nonempty_current_admissions_all_terminal']);self.assertNotIn('all_terminal',empty);self.assertNotIn('all_executions_accepted_unqualified',empty);self.assertFalse(consumer.accepted_unqualified_work(empty))
  self.enqueue([self.f.job('child',producer='bad'),self.f.job('bad','nonzero')]);self.assertFalse(consumer.accepted_unqualified_work(self.state()));self.crash('reserved');self.assertEqual(self.state()['state'],'nonempty_current_admissions_active');self.assertFalse(consumer.accepted_unqualified_work(self.state()));self.release('bad');failed=self.state();self.assertTrue(failed['nonempty_current_admissions_all_terminal']);self.assertFalse(consumer.accepted_unqualified_work(failed));self.assertEqual(failed['blocked_terminal'],1)
  # New independent fresh domain proves the same direct consumer accepts real completed work.
  self.root=d.initialize(self.base/'successful');self.f.root=self.root;self.enqueue([self.f.job('good')]);self.release('good');good=self.state();self.assertTrue(consumer.accepted_unqualified_work(good));self.assertNotIn('all_terminal',good)
  legacy=dict(good,all_terminal=True)
  with self.assertRaisesRegex(ValueError,'legacy ambiguous'):consumer.accepted_unqualified_work(legacy)
  false_empty=dict(empty,nonempty_current_admissions_all_terminal=True)
  with self.assertRaisesRegex(ValueError,'nonempty terminal flag'):consumer.accepted_unqualified_work(false_empty)
  self.enqueue([self.f.job('after')],'after');self.assertFalse(consumer.accepted_unqualified_work(self.state()))
 def test_failed_single_execution_is_nonempty_terminal_but_not_accepted(self):
  self.enqueue([self.f.job('failed-alone','nonzero')]);self.release('failed-alone');snapshot=self.audit();self.assertTrue(snapshot['nonempty_current_admissions_all_terminal']);self.assertEqual(snapshot['blocked_terminal'],0);self.assertEqual(snapshot['failed_execution'],1);self.assertFalse(consumer.accepted_unqualified_work(snapshot))
  with self.assertRaisesRegex(ValueError,'explicit nonempty lifecycle schema'):consumer.accepted_unqualified_work({'all_terminal':True})
 def test_source_domain_and_counters_not_historical_authentication(self):
  with self.assertRaises(ValueError):d.admission.opened(self.root/'admissions.sqlite')
  path=self.root/'dispatch.json';config=json.loads(path.read_text());config['sources'][0]=d.ex.pin(d.LAB/'queue-durable-frontier-sol-v1/dispatch.py');path.write_text(json.dumps(config))
  with self.assertRaisesRegex(ValueError,'source identity differs'):d.tick(self.root)
  self.assertFalse((self.root/'dispatcher.lock').exists())
if __name__=='__main__':unittest.main(verbosity=2)
