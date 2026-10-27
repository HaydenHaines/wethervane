"""Targeted reproduced blockers, using only invented children/cgroups and local IPC."""
import json,os,socket,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import pilot
from test_pilot import Clock,FakeDriver,FakeSensors,healthy

class Process:
 pid=123456
 def __init__(self,wait_fails=False):self.returncode=None;self.wait_fails=wait_fails;self.waits=0
 def poll(self):return self.returncode
 def wait(self,timeout=None):
  self.waits+=1
  if self.wait_fails:raise pilot.subprocess.TimeoutExpired('invented',timeout)
  self.returncode=-9;return -9

class V2Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.cg=self.root/'fake-cgroup';self.cg.mkdir()
  for name,value in {'cgroup.events':'populated 0\n','cgroup.procs':'','cgroup.kill':'never written','cpu.stat':'usage_usec 0\n','memory.peak':'0','io.stat':''}.items():(self.cg/name).write_text(value)
  self.driver=pilot.RealDriver(self.root,self.root/'source',self.cg);self.driver.set_gate({'path':'invented gate'})
  self.parent=dict(pid=os.getpid(),start_ticks=1,boot_id='invented',state='S')
  self.addCleanup(patch.stopall);patch.object(pilot.RealSensors,'sample',side_effect=AssertionError('production sensors forbidden')).start()
 def test_direct_worker_no_capability_cannot_join_or_change_nice(self):
  request=self.root/'handmade.json';request.write_text(json.dumps({'root':str(self.root),'cgroup':str(self.cg)}))
  with patch.object(pilot.os,'nice')as nice,patch.object(Path,'write_text',side_effect=AssertionError('any join/write forbidden')):
   with self.assertRaisesRegex(pilot.Stop,'capability'):pilot.worker(request)
  nice.assert_not_called();self.assertEqual((self.cg/'cgroup.kill').read_text(),'never written')
 def test_granted_request_missing_admission_still_cannot_join(self):
  root=self.root/'queue-pilot-invented';root.mkdir(mode=0o700);request=root/'request.json';request.write_text(json.dumps({'root':str(root),'cgroup':str(self.cg),'source':str(self.root/'source')}));grant={'request':pilot.fingerprint(request),'admission':{'path':str(root/'admission.json')}}
  with patch.object(pilot,'receive_grant',return_value=grant),patch.object(pilot.os,'nice')as nice,patch.object(Path,'write_text',side_effect=AssertionError('mutation forbidden')):
   with self.assertRaises(FileNotFoundError):pilot.worker(request,100)
  nice.assert_not_called()
 def test_child_handle_and_creator_pidfd_retained_on_birth_failure(self):
  child=Process();fd=os.open('/dev/null',os.O_RDONLY)
  def identity(pid):
   if pid==os.getpid():return self.parent
   raise PermissionError('invented birth failure')
  with patch.object(pilot.subprocess,'Popen',return_value=child),patch.object(pilot.os,'pidfd_open',return_value=fd),patch.object(pilot,'birth',side_effect=identity),patch.object(pilot.signal,'pidfd_send_signal')as signal:
   with self.assertRaises(PermissionError):self.driver.start('control',{})
   self.assertEqual(len(self.driver.children),1);self.assertIs(self.driver.children[0]['process'],child);self.assertIsNone(self.driver.children[0]['identity']);self.assertGreater(child.waits,0);signal.assert_called_once()
   self.assertTrue(self.driver.cleanup()['verified'])
  with self.assertRaises(OSError):os.fstat(fd)
 def test_pidfd_failure_keeps_unknown_handle_without_guessed_signal(self):
  child=Process(wait_fails=True)
  with patch.object(pilot.subprocess,'Popen',return_value=child),patch.object(pilot.os,'pidfd_open',side_effect=PermissionError('invented pidfd unavailable')),patch.object(pilot,'birth',return_value=self.parent),patch.object(pilot.signal,'pidfd_send_signal')as signal:
   with self.assertRaises(PermissionError):self.driver.start('control',{})
   self.assertEqual(len(self.driver.children),1);self.assertFalse(self.driver.cleanup()['verified']);signal.assert_not_called()
 def test_unknown_descendant_refuses_even_when_direct_members_empty(self):
  (self.cg/'unknown-child').mkdir();(self.cg/'cgroup.events').write_text('populated 1\n')
  with patch.object(pilot.signal,'pidfd_send_signal')as signal:
   result=self.driver.cleanup()
  self.assertFalse(result['verified']);self.assertFalse(result['recursive_kill']);signal.assert_not_called();self.assertEqual((self.cg/'cgroup.kill').read_text(),'never written')
 def test_descendant_appearing_after_wait_refuses_before_member_signals(self):
  (self.cg/'cgroup.events').write_text('populated 1\n');clock=[0]
  def now():clock[0]+=1;return clock[0]
  with patch.object(self.driver,'_leaf_only',side_effect=[None,pilot.Stop('new descendant')]),patch.object(pilot.time,'monotonic',side_effect=now),patch.object(pilot.time,'sleep'),patch.object(pilot.signal,'pidfd_send_signal')as signal:
   with self.assertRaisesRegex(pilot.Stop,'descendant'):self.driver.cleanup()
  signal.assert_not_called();self.assertEqual((self.cg/'cgroup.kill').read_text(),'never written')
 def test_first_new_oom_during_cpu_warmup_never_rebases(self):
  clock=Clock();driver=FakeDriver()
  def mutate(sample,n):sample['cgroup']['memory_events']['oom_kill']=1;sample['cpu_util']=None if n==1 else .1;return sample
  result=pilot.Pilot(self.root,FakeSensors(clock,mutate),driver,clock,clock.sleep).run();self.assertEqual(result['status'],'aborted');self.assertEqual(driver.calls,[])
 def test_warmup_preserves_independent_pressure_checks(self):
  p=pilot.Policy();s=healthy(1);s['cpu_util']=None;s['host_pressure']['memory']['some']['avg10']=2
  with self.assertRaises(pilot.Stop):p.seed(s)
  self.assertEqual(p.events['oom_kill'],0)
 def test_ancestor_counter_reset_is_not_clamped_to_zero(self):
  from sensors import other_cpu_cores,Uncertain
  with self.assertRaises(Uncertain):other_cpu_cores(100,50,20,25,1)
  with self.assertRaises(Uncertain):other_cpu_cores(100,150,20,10,1)
  self.assertAlmostEqual(other_cpu_cores(0,2000000,0,500000,1),1.5)
 def test_exact_sample_mode_required(self):
  s=healthy(1);s['mode']='real_unmodified_queue'
  with self.assertRaisesRegex(pilot.Stop,'mode'):pilot.Policy().check(s,'invented')
 def test_real_policy_cannot_omit_immutable_admitted_events(self):
  with self.assertRaises(pilot.Stop):pilot.Policy('real_unmodified_queue')
 def test_local_socket_grant_links_exact_parent_child_and_request(self):
  request=self.root/'request.json';request.write_text('{}');real_parent=os.getpid();fake_child=765432
  expected_parent=dict(pid=real_parent,start_ticks=7,boot_id='invented');expected_child=dict(pid=fake_child,start_ticks=8,boot_id='invented')
  grant=dict(schema='PilotWorkerGrantV2',request=pilot.fingerprint(request),parent=expected_parent,child=expected_child,issued_monotonic=time.monotonic())
  a,b=socket.socketpair()
  try:
   a.sendall((json.dumps(grant)+'\n').encode())
   with patch.object(pilot.os,'getppid',return_value=real_parent),patch.object(pilot.os,'getpid',return_value=fake_child),patch.object(pilot,'birth',side_effect=lambda pid:expected_parent if pid==real_parent else expected_child):
    self.assertEqual(pilot.receive_grant(b.fileno(),request)['child']['pid'],fake_child)
  finally:a.close();b.close()
 def test_grant_rejects_stale_wrong_peer_birth_and_changed_request(self):
  request=self.root/'request.json';request.write_text('{}');real_parent=os.getpid();fake_child=765432
  parent=dict(pid=real_parent,start_ticks=7,boot_id='invented');child=dict(pid=fake_child,start_ticks=8,boot_id='invented')
  for case in ['stale','peer','birth','request']:
   grant=dict(schema='PilotWorkerGrantV2',request=pilot.fingerprint(request),parent=dict(parent),child=dict(child),issued_monotonic=time.monotonic())
   if case=='stale':grant['issued_monotonic']-=3
   if case=='peer':grant['parent']['pid']+=1
   if case=='birth':grant['child']['start_ticks']+=1
   if case=='request':grant['request']['sha256']='0'*64
   a,b=socket.socketpair()
   try:
    a.sendall((json.dumps(grant)+'\n').encode())
    with patch.object(pilot.os,'getppid',return_value=real_parent),patch.object(pilot.os,'getpid',return_value=fake_child),patch.object(pilot,'birth',side_effect=lambda pid:parent if pid==real_parent else child):
     with self.assertRaises(pilot.Stop):pilot.receive_grant(b.fileno(),request)
   finally:a.close();b.close()
 def test_empty_group_does_not_hide_known_live_fixture(self):
  expected=dict(pid=444,start_ticks=2,boot_id='invented',state='S')
  with patch.object(self.driver,'_recorded_identities',return_value={444:expected}),patch.object(pilot,'birth',return_value=expected),patch.object(pilot.signal,'pidfd_send_signal')as signal:
   result=self.driver.cleanup()
  self.assertFalse(result['verified']);signal.assert_not_called()
 def test_closed_controller_or_non_socket_capability_refuses(self):
  a,b=socket.socketpair();a.close()
  try:
   with self.assertRaises(pilot.Stop):pilot.receive_grant(b.fileno(),self.root/'absent')
  finally:b.close()
  with (self.root/'ordinary').open('w')as file:
   with self.assertRaisesRegex(pilot.Stop,'socket'):pilot.receive_grant(file.fileno(),self.root/'absent')

if __name__=='__main__':unittest.main()
