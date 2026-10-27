"""Entirely invented filesystem/grant preflight; no worker, cgroup, process or queue execution."""
import json,os,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import pilot
from test_pilot import healthy

class ReviewTests(unittest.TestCase):
 def build(self,base):
  here=base/'invented-harness';here.mkdir();source=base/'invented-source';source.mkdir();cg=base/'ordinary-cgroup-files';cg.mkdir();owner=base/'protected-owner';owner.mkdir();guard=owner/'state';guard.write_text('idle')
  scratch=base/'scratch';scratch.mkdir();root=scratch/'queue-pilot-invented';root.mkdir(mode=0o700)
  for n in ['pilot.py','sensors.py']:(here/n).write_text('invented source, not executed')
  (here/'queue-pins.json').write_text(json.dumps({'source':str(source),'pins':[]}))
  for n,v in {'cpu.max':'100000 100000','memory.max':str(256*pilot.MIB),'pids.max':'16','memory.events':'oom 0\noom_kill 0\nhigh 0\n','pids.events':'max 0\n','cgroup.procs':''}.items():(cg/n).write_text(v)
  cfg={'source':str(source),'cgroup':str(cg),'scratch_parent':str(scratch),'fresh_at_unix':time.time(),'protected_roots':[str(owner)],'protected_files':[pilot.fingerprint(guard)],'protected_identities':[],'harness_pins':[pilot.fingerprint(here/n)for n in ['pilot.py','sensors.py','queue-pins.json']]}
  admission={'schema':'PilotAdmissionV2','config':cfg,'admitted_at_unix':time.time(),'source_pins':cfg['harness_pins'],'runtime':pilot.fingerprint(Path(sys.executable).resolve())}
  leaf={'schema':'PilotLeafAdmissionV2','leaf':{'device':cg.stat().st_dev,'inode':cg.stat().st_ino,'path':str(cg),'mount':str(base),'cpu_max':'100000 100000','memory_max':256*pilot.MIB,'pids_max':16,'admitted_memory_events':{'oom':0,'oom_kill':0,'high':0}}}
  sample=healthy(time.monotonic());sample['mode']='real_unmodified_queue'
  sensor={'phase':'before-control','mode':'real_unmodified_queue','sample':sample}
  request={'root':str(root),'source':str(source),'cgroup':str(cg),'cgroup_identity':[cg.stat().st_dev,cg.stat().st_ino],'operation':'control','args':{}}
  for name,data in [('admission.json',admission),('leaf-admission.json',leaf),('00001-sensor.json',sensor),('request.json',request)]: (root/name).write_text(json.dumps(data))
  grant={key:pilot.fingerprint(root/name)for key,name in [('admission','admission.json'),('leaf_admission','leaf-admission.json'),('sensor','00001-sensor.json'),('request','request.json')]}
  return here,cg,root,grant
 def test_positive_readonly_preflight_and_drift_negatives(self):
  for case in ['normal','owner-change','limits-change','event-change','new-descendant','stale-resource','wrong-mode']:
   with self.subTest(case=case),tempfile.TemporaryDirectory()as name:
    base=Path(name);here,cg,root,grant=self.build(base)
    if case=='owner-change':(base/'protected-owner/state').write_text('changed')
    if case=='limits-change':(cg/'memory.max').write_text(str(512*pilot.MIB))
    if case=='event-change':(cg/'memory.events').write_text('oom 0\noom_kill 1\nhigh 0\n')
    if case=='new-descendant':(cg/'unknown').mkdir()
    if case in ['stale-resource','wrong-mode']:
     f=root/'00001-sensor.json';v=json.loads(f.read_text())
     if case=='stale-resource':v['sample']['monotonic']-=3
     else:v['sample']['mode']='invented_test_only'
     f.write_text(json.dumps(v));grant['sensor']=pilot.fingerprint(f)
    with patch.object(pilot,'HERE',here),patch.object(pilot,'cgroup_mount',return_value=base),patch.object(pilot.RealSensors,'owner_processes',return_value=[]),patch.object(pilot.os,'nice',side_effect=AssertionError('mutation forbidden')):
     if case=='normal':self.assertEqual(pilot.worker_preflight(root/'request.json',grant)[1],root)
     else:
      with self.assertRaises((pilot.Stop,pilot.Uncertain)):pilot.worker_preflight(root/'request.json',grant)
    self.assertEqual((cg/'cgroup.procs').read_text(),'')

if __name__=='__main__':unittest.main(verbosity=2)
