"""Invented sensors/driver only; production worker/sampling entry points are forbidden."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import pilot
from sensors import Uncertain,numbers,pressure

class Clock:
 def __init__(self):self.t=1.
 def __call__(self):return self.t
 def sleep(self,v):self.t+=v

def healthy(t):
 psi={k:{'some':{'avg10':0.}}for k in ['cpu','memory','io']}
 return dict(mode='invented_test_only',harness=dict(rss_bytes=pilot.MIB),monotonic=t,cpu_util=.1,cpu_count=8,load1=.1,host_memory_available=8*1024*pilot.MIB,host_memory_total=16*1024*pilot.MIB,host_pressure=copy.deepcopy(psi),owners=[],free_bytes=8*1024*pilot.MIB,free_inodes=100000,probe_fsync_ms=1.,scratch_allocated=1000,scratch_entries=10,cgroup=dict(identity={'memory_max':256*pilot.MIB},quota_cores=1.,ancestors=[dict(capacity_cores=8,other_used_cores=.1)],memory_current=32*pilot.MIB,memory_peak=32*pilot.MIB,parent_memory_remaining=4*1024*pilot.MIB,memory_events={'oom':0,'oom_kill':0,'high':0},pids_current=0,pids_events={'max':0},cpu={'usage_usec':int(t*1000),'throttled_usec':0},pressure=copy.deepcopy(psi),members=[],io_stat='invented',rss_sum=1,pss_sum=1))
class FakeSensors:
 mode='invented_test_only'
 def __init__(self,clock,change=None):self.admitted_events={'oom':0,'oom_kill':0,'high':0};self.clock=clock;self.change=change;self.count=0
 def sample(self):
  self.count+=1;self.clock.sleep(.01);s=healthy(self.clock())
  if self.change:s=self.change(s,self.count)
  return s
class FakeDriver:
 mode='invented_test_only'
 def __init__(self):self.admitted=0;self.done=0;self.calls=[];self.clean=True;self.exit=0;self.pending=False
 def set_gate(self,pin):self.gate=pin
 def start(self,op,args):
  self.calls.append((op,args))
  if op=='enqueue':self.admitted=args['n']
  if op=='tick':self.done=self.admitted
  return op
 def poll(self,op):
  if self.pending:return None
  state=dict(capacity=1,reserved=0,all_jobs_recorded=True)
  return dict(mode=self.mode,exit_code=self.exit,value=state)
 def launches(self):return self.done
 def verify_capacity(self,s):pilot.require(s['reserved']<=1,'capacity failed')
 def verify_terminal_prefix(self,n):pilot.require(self.done==n,'invented prefix differs')
 def counts(self):return dict(P=self.admitted,J=self.done,R=self.done,A=self.done,E=0,H=1,scientific_cycles=0)
 def cleanup(self):return dict(verified=self.clean,mode=self.mode)

class PilotTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.clock=Clock();self.driver=FakeDriver()
  self.addCleanup(patch.stopall);patch.object(pilot.RealDriver,'start',side_effect=AssertionError('actual worker forbidden in tests')).start();patch.object(pilot.RealSensors,'sample',side_effect=AssertionError('production sampling forbidden')).start()
 def run_fake(self,change=None):return pilot.Pilot(self.root,FakeSensors(self.clock,change),self.driver,self.clock,self.clock.sleep).run()
 def test_cumulative_prefixes_no_science_and_fake_status(self):
  r=self.run_fake();self.assertEqual(r['status'],'invented_test_completed');self.assertEqual(r['scientific_cycles'],0);self.assertEqual([args['n']for op,args in self.driver.calls if op=='enqueue'],list(range(1,9)))
  self.assertEqual([json.loads((self.root/f'prefix-{n}.json').read_text())['counts']['J']for n in [1,2,4,8]],[1,2,4,8]);self.assertLessEqual(r['actual_launches'],16)
 def test_first_cpu_interval_requires_warmup_not_queue_launch(self):
  def change(s,n):
   if n==1:s['cpu_util']=None
   return s
  self.assertEqual(self.run_fake(change)['status'],'invented_test_completed')
 def test_pid_limit_event_stops_admission(self):
  s=healthy(1);s['cgroup']['pids_events']['max']=1
  with self.assertRaises(pilot.Stop):pilot.Policy().check(s,'invented')
 def test_rss_and_per_cycle_storage_reserve(self):
  for kind in ['rss','space','inodes']:
   s=healthy(1)
   if kind=='rss':s['harness']['rss_bytes']=300*pilot.MIB
   if kind=='space':s['scratch_allocated']=127*pilot.MIB
   if kind=='inodes':s['scratch_entries']=9999
   with self.assertRaises(pilot.Stop):pilot.Policy().check(s,'invented')
 def test_expired_config_refuses_without_sensors(self):
  cfg=self.root/'expired.json';cfg.write_text(json.dumps({'fresh_at_unix':0}))
  with self.assertRaises(pilot.Stop):pilot.execute(cfg)
 def test_owner_start_aborts_no_further_admission(self):
  def change(s,n):
   if n>12:s['owners']=[{'pid':111,'start_ticks':1}]
   return s
  r=self.run_fake(change);self.assertEqual(r['status'],'aborted');self.assertLess(self.driver.admitted,8);self.assertIn('owner',r['error']['message'])
 def test_owner_queue_or_source_drift_fails_closed(self):
  def change(s,n):
   if n==7:raise Uncertain('owner state pin changed')
   return s
  r=self.run_fake(change);self.assertEqual(r['status'],'aborted');self.assertTrue((self.root/'pilot-terminal.json').exists())
 def test_aggregate_detached_memory_not_just_controller(self):
  def change(s,n):s['cgroup']['memory_current']=220*pilot.MIB;return s
  r=self.run_fake(change);self.assertEqual(r['status'],'aborted');self.assertEqual(self.driver.calls,[])
 def test_pressure_latency_storage_and_parent_headroom(self):
  for key in ['pressure','latency','space','cpu_parent']:
   s=healthy(1.)
   if key=='pressure':s['cgroup']['pressure']['memory']['some']['avg10']=2
   if key=='latency':s['probe_fsync_ms']=100
   if key=='space':s['free_bytes']=1
   if key=='cpu_parent':s['cgroup']['ancestors'][0]['other_used_cores']=7.5
   with self.assertRaises(pilot.Stop):pilot.Policy().check(s,'invented')
 def test_missing_and_nonfinite_sensor(self):
  for value in [None,float('nan'),float('inf')]:
   s=healthy(1);s['cpu_util']=value
   with self.assertRaises(pilot.Stop):pilot.Policy().check(s,'invented')
 def test_memory_event_and_cpu_reset_stop(self):
  p=pilot.Policy();p.check(healthy(1),'first');s=healthy(2);s['cgroup']['memory_events']['oom_kill']=1
  with self.assertRaises(pilot.Stop):p.check(s,'second')
  p=pilot.Policy();p.check(healthy(3),'first')
  with self.assertRaises(pilot.Stop):p.check(healthy(2),'reset')
 def test_deadline_and_cleanup_uncertainty_preserved(self):
  self.driver.pending=True;self.driver.clean=False;r=self.run_fake();self.assertEqual(r['status'],'cleanup_uncertain');self.assertIn('deadline',r['error']['message'])
 def test_nonzero_worker_not_success(self):
  self.driver.exit=7;r=self.run_fake();self.assertEqual(r['status'],'aborted');self.assertEqual(self.driver.admitted,0)
 def test_modes_cannot_mix(self):
  self.driver.mode='real_unmodified_queue'
  with self.assertRaises(pilot.Stop):pilot.Pilot(self.root,FakeSensors(self.clock),self.driver)
 def test_journal_exclusive_and_source_scratch_overlap(self):
  pilot.journal(self.root,'once.json',{'v':1})
  with self.assertRaises(FileExistsError):pilot.journal(self.root,'once.json',{'v':2})
  with self.assertRaises(pilot.Stop):pilot.fresh_scratch(self.root,self.root/'source')
 def test_fresh_scratch_is_exclusive_and_outside_source(self):
  source=self.root/'fake-source';source.mkdir();parent=self.root/'scratch';parent.mkdir();first=pilot.fresh_scratch(parent,source);second=pilot.fresh_scratch(parent,source);self.assertNotEqual(first,second);self.assertEqual(first.stat().st_mode&0o777,0o700)
 def test_invented_proc_file_parsers(self):
  p=self.root/'stat';p.write_text('usage_usec 1\nthrottled_usec 0\n');self.assertEqual(numbers(p)['usage_usec'],1);p.write_text('some avg10=1.00 avg60=0.00 avg300=0.00 total=2\n');self.assertEqual(pressure(p)['some']['avg10'],1.)

if __name__=='__main__':unittest.main()
