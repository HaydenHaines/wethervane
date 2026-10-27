"""Mock-only counterexamples: no real subprocess, cgroup or queue calls."""
import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import pilot

class Counterexamples(unittest.TestCase):
 def test_post_popen_birth_exception_drops_cleanup_handle(self):
  with tempfile.TemporaryDirectory()as name:
   root=Path(name);cg=root/'fake-cgroup';cg.mkdir();d=pilot.RealDriver(root,root/'source',cg)
   process=type('InventedProcess',(),{'pid':123456})()
   with patch.object(pilot.subprocess,'Popen',return_value=process)as spawn,patch.object(pilot,'birth',side_effect=PermissionError('invented proc failure')):
    with self.assertRaises(PermissionError):d.start('control',{})
   self.assertEqual(spawn.call_count,1);self.assertEqual(d.children,[])
 def test_new_descendant_skips_member_authentication_before_kill_write(self):
  with tempfile.TemporaryDirectory()as name:
   root=Path(name);cg=root/'fake-cgroup';cg.mkdir();(cg/'unknown-child').mkdir()
   for n,v in {'cgroup.events':'populated 1\n','cgroup.procs':'','cgroup.kill':'','cpu.stat':'usage_usec 0\n','memory.peak':'0','io.stat':''}.items():(cg/n).write_text(v)
   d=pilot.RealDriver(root,root/'source',cg);clock=[0]
   def now():clock[0]+=1;return clock[0]
   with patch.object(pilot.time,'monotonic',side_effect=now),patch.object(pilot.time,'sleep'):
    result=d.cleanup()
   self.assertEqual((cg/'cgroup.kill').read_text(),'1');self.assertFalse(result['verified'])
 def test_direct_worker_attempts_join_before_admission_exists(self):
  import json
  with tempfile.TemporaryDirectory()as name:
   root=Path(name);cg=root/'invented-group';cg.mkdir();request=root/'handmade-request.json'
   request.write_text(json.dumps({'root':str(root),'cgroup':str(cg),'cgroup_identity':[cg.stat().st_dev,cg.stat().st_ino]}))
   self.assertFalse((root/'admission.json').exists())
   class JoinAttempt(Exception):pass
   def intercept(path,text,*args,**kwargs):
    self.assertEqual(path,cg/'cgroup.procs');raise JoinAttempt('intercepted before any real cgroup write')
   with patch.object(pilot.os,'nice'),patch.object(Path,'write_text',intercept):
    with self.assertRaises(JoinAttempt):pilot.worker(request)
 def test_first_warmup_baselines_new_oom_event_instead_of_rejecting(self):
  from test_pilot import Clock,FakeSensors,FakeDriver
  with tempfile.TemporaryDirectory()as name:
   clock=Clock();driver=FakeDriver()
   def mutate(sample,n):
    sample['cgroup']['memory_events']['oom_kill']=1
    if n==1:sample['cpu_util']=None
    return sample
   sensor=FakeSensors(clock,mutate)
   with patch.object(pilot.RealDriver,'start',side_effect=AssertionError('real worker forbidden')),patch.object(pilot.RealSensors,'sample',side_effect=AssertionError('real sensors forbidden')):
    result=pilot.Pilot(name,sensor,driver,clock,clock.sleep).run()
   self.assertEqual(result['status'],'invented_test_completed')
 def test_mode_argument_does_not_validate_each_sample_mode(self):
  # Policy accepts either label; caller's mode equality is constructor-only.
  from test_pilot import healthy
  s=healthy(1);s['mode']='real_unmodified_queue'
  pilot.Policy().check(s,'fake-test')

if __name__=='__main__':unittest.main(verbosity=2)
