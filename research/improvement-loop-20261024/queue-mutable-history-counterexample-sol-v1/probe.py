"""Bounded actual-implementation corruption counterexamples. Temporary jobs only."""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
from unittest.mock import patch

HERE=Path(__file__).resolve().parent;LAB=HERE.parent
CASES=['v2-history','incremental-history','incremental-forged-row','v2-forged-release','v1-forged-release']
SOURCE='''import hashlib,json,pathlib,sys,time
out,mode=pathlib.Path(sys.argv[1]),sys.argv[2]
out.mkdir();data=out/'data.txt';data.write_text('tiny')
(out/'receipt.json').write_text(json.dumps(dict(schema='CounterexampleFixtureV1',data=dict(path=str(data),sha256=hashlib.sha256(b'tiny').hexdigest(),size=4))))
if mode=='hold':
 deadline=time.monotonic()+5
 while time.monotonic()<deadline and not (out.parent/'release-workers').exists():time.sleep(.01)
'''


def pin(path):
 raw=Path(path).read_bytes();return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),size=len(raw))


class Domain:
 def __init__(self,case):
  self.case=case;self.incremental=case.startswith('incremental')
  if self.incremental:
   sys.path.insert(0,str(LAB/'queue-v2-incremental-assessment-sol-v1'));import dispatch
   self.d=dispatch;self.q=dispatch.q;self.ex=dispatch.ex
  else:
   directory=LAB if case.startswith('v1') else LAB/'queue-v2'
   sys.path.insert(0,str(directory));import experiment_queue as q;import execution as ex
   self.q=q;self.ex=ex
 def __enter__(self):
  self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name);self.root=self.base/'queue';self.source=self.base/'owner.py';self.source.write_text(SOURCE)
  if self.incremental:self.d.initialize(self.root)
  else:self.q.initialize(self.root,1)
  self.cleanup=None;return self
 def job(self,name,mode='seed'):
  spec=dict(schema='LocalExecutionSpecV1',cycle_id='fixture',job_key=name,attempt_id=name,
   argv=[sys.executable,str(self.source),str(self.base/('output-'+name)),mode],executable=self.ex.executable_pin(sys.executable),
   cwd=str(self.base),environment={},input_pins=[self.ex.pin(self.source)],output_directory=str(self.base/('output-'+name)),
   native_receipts=[dict(path='receipt.json',required_fields=dict(schema='CounterexampleFixtureV1'))],timeout_seconds=7,grace_seconds=.1,previous_attempt=None,retry_reason='')
  plan=dict(schema='LocalExperimentQueuePlanV1',queue_id='plan-'+name,jobs=[dict(id=name,execution_spec=spec,dependencies=[],outputs=[dict(role='receipt',path='receipt.json',minimum_bytes=1)],resume_prelaunch=None)])
  (self.d.enqueue if self.incremental else self.q.enqueue)(self.root,plan,self.q.digest(plan))
 def tick(self):return (self.d.tick if self.incremental else self.q.tick)(self.root)
 def release_exists(self,name):
  if self.incremental:
   with self.d.admission.opened(self.root/'admissions.sqlite') as db:return db.execute('SELECT 1 FROM releases WHERE attempt=?',(name,)).fetchone() is not None
  return (self.root/f'queues/plan-{name}/jobs/{name}/release.json').exists()
 def wait_release(self,name):
  deadline=time.monotonic()+5
  while time.monotonic()<deadline:
   state=self.tick()
   if self.release_exists(name):return state
   time.sleep(.02)
  raise AssertionError(state)
 def wait_started(self,name):
  receipt=self.base/f'output-{name}/receipt.json';deadline=time.monotonic()+4
  while not receipt.exists() and time.monotonic()<deadline:time.sleep(.01)
  assert receipt.exists(),name+' child did not pass launch gate'
 def live(self):
  result={}
  for directory in (self.root/'executions/attempts').glob('*'):
   _,spec,prepared=self.ex.load(self.root/'executions',directory.name)
   observations={}
   for kind in ('supervisor','child'):
    path=directory/(kind+'.json')
    if path.exists():observations[kind]=self.ex.observe(self.q.document(path)['identity'])
   child=observations.get('child');members=self.ex.session_members(child['expected']) if child else []
   try:fd,_=self.ex.lock(self.root/'executions',spec['cycle_id'],spec['job_key'])
   except ValueError:held=True
   else:os.close(fd);held=False
   alive=[kind for kind,o in observations.items() if o['state']=='alive']
   if held or members or alive:result[directory.name]=dict(job_lock_held=held,alive_records=alive,session_member_count=len(members))
  return result
 def __exit__(self,kind,error,tb):
  (self.base/'release-workers').write_text('release only these temporary fixtures')
  deadline=time.monotonic()+8
  while self.live() and time.monotonic()<deadline:time.sleep(.02)
  survivors=self.live();signals=0
  if survivors:
   # Only this fresh temporary domain's recorded births may be signaled.
   for attempt in survivors:
    for role in ('child','supervisor'):
     path=self.root/'executions/attempts'/attempt/(role+'.json')
     if path.exists():
      identity=self.q.document(path)['identity']
      if self.ex.observe(identity)['state']=='alive':
       fd=self.ex.open_pidfd(identity)
       try:signal.pidfd_send_signal(fd,signal.SIGKILL);signals+=1
       finally:os.close(fd)
   deadline=time.monotonic()+3
   while self.live() and time.monotonic()<deadline:time.sleep(.02)
  for process in self.ex._SUPERVISORS:process.wait(timeout=3)
  terminal_records={}
  for directory in (self.root/'executions/attempts').glob('*'):
   path=directory/'terminal.json'
   if path.exists():
    value=self.q.document(path)
    terminal_records[directory.name]=dict(return_code=value['return_code'],process_outcome=value['process_outcome'],
     timed_out=value['timed_out'],recorded_cleanup_verified=value['cleanup']['verified'],terminal_pin=pin(path))
  self.cleanup=dict(live_owned_attempts_after_cleanup=self.live(),fallback_signals=signals,
                    recorded_child_terminals=terminal_records,
                    temp_file_bytes_before_removal=sum(p.stat().st_size for p in self.base.rglob('*') if p.is_file()))
  assert not self.cleanup['live_owned_attempts_after_cleanup'],self.cleanup
  self.temp.cleanup();self.cleanup['temporary_domain_removed']=not self.base.exists()


def historical(case):
 with Domain(case) as f:
  f.job('a');f.wait_release('a');target=f.base/'output-a/receipt.json';expected=f.ex.pin(target)
  cached_expected_root=f.q.digest(expected)
  target.chmod(0o444);replacement=target.with_name('owner-replacement.tmp');replacement.write_bytes(target.read_bytes()+b' ');os.replace(replacement,target)
  changed=f.ex.pin(target);assert changed!=expected
  f.job('b');hits=0;original_pin=f.ex.pin
  def observed(path):
   nonlocal hits
   if Path(path)==target:hits+=1
   return original_pin(path)
  with patch.object(f.ex,'pin',observed):state=f.tick()
  launched=(f.root/'executions/attempts/b/launch-intent.json').exists()
  if f.incremental:assert launched and hits==0;f.wait_release('b')
  else:
   assert not launched and hits>0
   assert state['queues']['plan-a']['a']['outcome']=='quarantined_integrity' and state['reserved']==1
  result=dict(case=case,mutation='owner can replace a chmod0444 file through writable parent directory',accidental_corruption=False,
   expected_pin=expected,actual_changed_pin=changed,cached_expected_root_unchanged=f.q.digest(expected)==cached_expected_root,
   next_tick_historical_receipt_pin_calls=hits,unrelated_b_launched=launched,observation=state)
 result['cleanup']=f.cleanup;return result


def forged(case):
 with Domain(case) as f:
  f.job('a','hold');f.job('b','hold');initial=f.tick();f.wait_started('a')
  assert not (f.root/'executions/attempts/b').exists() and initial['reserved']==1
  before=f.live();assert set(before)=={'a'},before
  if f.incremental:
   # Explicit malicious/unauthorized raw SQL writer. No dispatch API authorizes
   # this row. Triggers do not stop INSERT of a never-verified release.
   value=dict(schema='DependencyReleaseRecordV1',attempt='a',result=dict(path=str(f.root/'evidence/a/result.json'),sha256='0'*64,size=0),release=dict(path=str(f.root/'evidence/a/release.json'),sha256='0'*64,size=0))
   with f.d.admission.opened(f.root/'admissions.sqlite') as db:
    db.execute('BEGIN IMMEDIATE');db.execute('INSERT INTO releases VALUES (?,?,?)',('a',f.q.canonical(value),f.q.digest(value)));db.execute('COMMIT')
   mutation='owner-writable SQLite: unauthorized forged committed release row with canonical digest, no result/release files'
  else:
   directory=f.root/'queues/plan-a/jobs/a';assert not (directory/'result.json').exists()
   (directory/'release.json').write_text('{"forged":true}')
   mutation='owner-writable receipt store: unauthorized malformed release.json inserted while result.json absent and worker live'
  state=f.tick();f.wait_started('b');after=f.live();assert set(after)=={'a','b'},after
  assert state['reserved']==1 and not (f.base/'release-workers').exists()
  result=dict(case=case,mutation=mutation,accidental_corruption=False,legitimate_release_publisher_invoked=False,
   configured_capacity=1,live_before=before,live_after=after,observation_reserved=state['reserved'],observation=state,
   actual_live_attempts=len(after),capacity_exceeded=True,result_missing_for_a=not ((f.root/'evidence/a/result.json') if f.incremental else (f.root/'queues/plan-a/jobs/a/result.json')).exists())
 result['cleanup']=f.cleanup;return result


def main(case):
 start=time.monotonic();result=historical(case) if case.endswith('history') else forged(case)
 result['wall_seconds']=time.monotonic()-start;print(json.dumps(result))


if __name__=='__main__':
 assert len(sys.argv)==2 and sys.argv[1] in CASES
 main(sys.argv[1])
