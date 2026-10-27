import hashlib,json,os,signal,subprocess,sys,tempfile,time,unittest,shutil
from pathlib import Path
import execution as ex
from ledger import canonical

FIXTURE="""import pathlib,sys,time
p=pathlib.Path(sys.argv[1]);p.mkdir();(p/'receipt.json').write_text('{"status":"fixture"}')
if sys.argv[2]=='hold':time.sleep(30)
if sys.argv[2]=='fail':sys.exit(7)
"""
class OwnershipTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name);self.root=self.base/'store';self.source=self.base/'fixture.py';self.source.write_text(FIXTURE);self.addCleanup(self.tmp.cleanup);self.addCleanup(self.cleanup)
 def cleanup(self):
  identities=[]
  for name in ['supervisor.json','child.json']:
   for p in self.root.glob('attempts/*/'+name):
    value=json.loads(p.read_text());identity=value['identity'];identities.append(identity)
    if ex.observe(identity)['state']=='alive':
     fd=ex.open_pidfd(identity)
     try:signal.pidfd_send_signal(fd,signal.SIGKILL)
     finally:os.close(fd)
  deadline=time.monotonic()+5
  while any(ex.observe(i)['state']=='alive'for i in identities) and time.monotonic()<deadline:time.sleep(.01)
  self.assertFalse(any(ex.observe(i)['state']=='alive'for i in identities),'owned fixture survived cleanup')
  for p in ex._SUPERVISORS:
   if p.poll()is not None:p.wait()
 def spec(self,name,path=None,mode='ok',job=None,previous=None):
  return dict(schema='LocalExecutionSpecV1',cycle_id='cycle',job_key=job or name,attempt_id=name,argv=[sys.executable,str(self.source),str(path or self.base/name),mode],executable=ex.executable_pin(sys.executable),cwd=str(self.base),environment={},input_pins=[ex.pin(self.source)],output_directory=str(path or self.base/name),native_receipts=[dict(path='receipt.json',required_fields={'status':'fixture'})],timeout_seconds=5,grace_seconds=.05,previous_attempt=previous,retry_reason='explicit fixture retry'if previous else '')
 def prepare(self,s):return ex.prepare(self.root,s,hashlib.sha256(canonical(s).encode()).hexdigest())
 def child(self,s,point=None,gate=None):
  env=dict(os.environ);env.pop('OWNERSHIP_CRASH',None)
  if point:env['OWNERSHIP_CRASH']=point
  code='import json,sys,hashlib,time;from pathlib import Path;import execution as e;from ledger import canonical;s=json.loads(sys.argv[2]);g=sys.argv[3]\nwhile g and not Path(g).exists():time.sleep(.001)\ne.prepare(sys.argv[1],s,hashlib.sha256(canonical(s).encode()).hexdigest())'
  return subprocess.Popen([sys.executable,'-B','-c',code,str(self.root),json.dumps(s),str(gate)if gate else''],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 def wait(self,predicate):
  deadline=time.monotonic()+8
  while time.monotonic()<deadline:
   value=predicate()
   if value:return value
   time.sleep(.02)
  self.fail('fixture timeout')
 def terminal(self,name):return self.wait(lambda:ex.inspect(self.root,name)['terminal_receipt'])
 def test_containment_both_orders_exact_and_sibling(self):
  for reverse in [False,True]:
   self.root=self.base/('store'+str(reverse));a,b=self.base/'owned',self.base/'owned'/'child'
   if reverse:a,b=b,a
   self.prepare(self.spec('first',a))
   for path in [a,b]:
    with self.assertRaises(ValueError):self.prepare(self.spec('second',path))
   self.prepare(self.spec('sibling',self.base/'owned-other'))
 def test_real_concurrent_conflicts_and_identical_replay(self):
  for n,paths in enumerate([('owned','owned/child'),('owned/child','owned'),('owned','owned')]):
   self.root=self.base/('race'+str(n));gate=self.base/('gate'+str(n));kids=[self.child(self.spec('j'+str(i),self.base/path),gate=gate)for i,path in enumerate(paths)];gate.touch();outputs=[k.communicate(timeout=10)for k in kids];codes=[k.returncode for k in kids];self.assertEqual(sorted(codes),[0,1],outputs)
  self.root=self.base/'identical';gate=self.base/'identicalgate';s=self.spec('same');kids=[self.child(s,gate=gate)for _ in range(2)];gate.touch();outputs=[k.communicate(timeout=10)for k in kids];self.assertEqual([k.returncode for k in kids],[0,0],outputs);self.assertEqual(len(list((self.root/'ownership-intents').glob('*.json'))),1)
 def test_crash_claims_repair_without_loss(self):
  for n,point in enumerate(['after_intent','after_directory','after_specification','after_prepared','after_output_claim','after_job_claim']):
   self.root=self.base/('crash'+str(n));s=self.spec('a',self.base/'owned');p=self.child(s,point);out=p.communicate(timeout=10);self.assertEqual(p.returncode,93,out)
   with self.assertRaises(ValueError):self.prepare(self.spec('nested',self.base/'owned'/'child'))
   pin=self.prepare(s);self.assertEqual(self.prepare(s),pin);self.assertFalse((self.root/'attempts'/'a'/'launch-intent.json').exists())
 def test_pending_intent_retains_job_identity(self):
  s=self.spec('a',job='same');p=self.child(s,'after_intent');p.communicate(timeout=10);self.assertEqual(p.returncode,93)
  with self.assertRaises(ValueError):self.prepare(self.spec('b',job='same'))
  changed=dict(s,argv=s['argv']+['changed'])
  with self.assertRaises(ValueError):self.prepare(changed)
  self.prepare(s)
 def test_real_success_replay_live_and_terminal_retry(self):
  s=self.spec('a',mode='fail',job='same');pin=self.prepare(s);ex.launch(self.root,'a');terminal=self.terminal('a');self.assertEqual(ex.inspect(self.root,'a')['attempt_terminal']['return_code'],7);self.assertEqual(self.prepare(s),pin)
  nxt=self.spec('b',job='same',previous={'attempt_id':'a','receipt':terminal});self.prepare(nxt);ex.launch(self.root,'b');self.terminal('b');self.assertEqual(ex.inspect(self.root,'b')['attempt_terminal']['return_code'],0)
  with self.assertRaises(ValueError):self.prepare(self.spec('fork',job='same',previous={'attempt_id':'a','receipt':terminal}))
 def test_live_holds_claim_and_unknown_recovery(self):
  s=self.spec('hold',mode='hold');pin=self.prepare(s);ex.launch(self.root,'hold');directory=self.root/'attempts'/'hold';self.wait(lambda:(self.base/'hold'/'receipt.json').exists());self.assertEqual(self.prepare(s),pin)
  with self.assertRaises(ValueError):self.prepare(self.spec('nested',self.base/'hold'/'child'))
  identity=json.loads((directory/'supervisor.json').read_text())['identity'];fd=ex.open_pidfd(identity);signal.pidfd_send_signal(fd,signal.SIGKILL);os.close(fd);self.wait(lambda:ex.observe(identity)['state']!='alive')
  with self.assertRaises(ValueError):ex.recover(self.root,'hold')
  ex.stop(self.root,'hold');self.wait(lambda:not ex.inspect(self.root,'hold')['job_lock_held']);terminal=ex.recover(self.root,'hold');self.assertIsNone(ex.inspect(self.root,'hold')['attempt_terminal']['return_code']);self.assertEqual(ex.recover(self.root,'hold'),terminal)
 def test_partial_corruption_and_missing_store_fail_closed(self):
  s=self.spec('a');p=self.child(s,'after_prepared');p.communicate(timeout=10);self.assertEqual(p.returncode,93);(self.root/'attempts'/'a'/'prepared.json').write_text('{}')
  with self.assertRaises(ValueError):self.prepare(s)
  (self.root/'ownership-store.json').unlink()
  with self.assertRaises(ValueError):self.prepare(self.spec('b'))
 def test_mixed_import_and_alternate_source_refuse_before_mutation(self):
  alt=self.base/'alternate';alt.mkdir()
  for name in ['execution.py','ledger.py','run_receipts.py']:shutil.copy2(Path(ex.__file__).parent/name,alt/name)
  s=self.spec('a');self.prepare(s);before={str(p.relative_to(self.root)):p.read_bytes()for p in self.root.rglob('*')if p.is_file()}
  code='import json,sys,hashlib;sys.path.insert(0,sys.argv[1]);import execution as e;from ledger import canonical;s=json.loads(sys.argv[3]);e.prepare(sys.argv[2],s,hashlib.sha256(canonical(s).encode()).hexdigest())'
  result=subprocess.run([sys.executable,'-B','-c',code,str(alt),str(self.root),json.dumps(s)],capture_output=True,text=True);self.assertNotEqual(result.returncode,0);self.assertEqual(before,{str(p.relative_to(self.root)):p.read_bytes()for p in self.root.rglob('*')if p.is_file()})
  mixed=self.base/'mixed-root';code='import sys,json,hashlib;sys.path.insert(0,sys.argv[1]);import ledger;sys.path.insert(0,sys.argv[2]);import execution as e;s=json.loads(sys.argv[4]);e.prepare(sys.argv[3],s,hashlib.sha256(ledger.canonical(s).encode()).hexdigest())'
  result=subprocess.run([sys.executable,'-B','-c',code,str(alt),str(Path(ex.__file__).parent),str(mixed),json.dumps(s)],capture_output=True,text=True);self.assertNotEqual(result.returncode,0);self.assertFalse(mixed.exists())
 def test_external_deletion_of_sole_intent_is_documented_limit(self):
  # External deletion is not a supported crash transition: no redundant witness.
  s=self.spec('a',self.base/'owned');p=self.child(s,'after_intent');p.communicate(timeout=10);self.assertEqual(p.returncode,93)
  (self.root/'ownership-intents'/'a.json').unlink()
  self.prepare(self.spec('nested',self.base/'owned'/'child'))
  self.assertFalse((self.root/'attempts'/'a').exists())
 def test_missing_intent_cannot_free_existing_claim(self):
  s=self.spec('a',self.base/'owned');self.prepare(s);(self.root/'ownership-intents'/'a.json').unlink()
  with self.assertRaises(ValueError):self.prepare(self.spec('nested',self.base/'owned'/'child'))
  with self.assertRaises(ValueError):ex.inspect(self.root,'a')
 def test_before_intent_crash_and_concurrent_siblings(self):
  s=self.spec('a');p=self.child(s,'before_intent');p.communicate(timeout=10);self.assertEqual(p.returncode,93);self.prepare(s)
  self.root=self.base/'siblings';gate=self.base/'siblinggate';kids=[self.child(self.spec('j'+str(i),self.base/path),gate=gate)for i,path in enumerate(['owned','owned-other'])];gate.touch();outputs=[k.communicate(timeout=10)for k in kids];self.assertEqual([k.returncode for k in kids],[0,0],outputs)
 def test_unadopted_store_and_source_drift(self):
  self.root.mkdir();(self.root/'attempts').mkdir()
  before=list(self.root.iterdir())
  with self.assertRaises(ValueError):self.prepare(self.spec('a'))
  self.assertEqual(list(self.root.iterdir()),before)
  self.root=self.base/'new';self.prepare(self.spec('a'));binding=self.root/'ownership-store.json';value=json.loads(binding.read_text());value['sources'][0]['sha256']='0'*64;binding.write_text(json.dumps(value))
  with self.assertRaises(ValueError):ex.launch(self.root,'a')

if __name__=='__main__':unittest.main()
