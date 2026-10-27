import tempfile,unittest,multiprocessing as mp,json,os,sqlite3
from pathlib import Path
import core

def worker(root,op,args,barrier=None,queue=None,fail=None):
 if barrier:barrier.wait()
 try:
  result=getattr(core,op)(root,*args,**({'fail':fail}if fail else {}))
  if queue:queue.put(('ok',result))
 except Exception as e:
  if queue:queue.put((type(e).__name__,str(e)))
  else:raise
def held_worker(root,op,args,point,ready,release,queue):
 original=core.crash
 def pause(selected,actual):
  if actual==point:
   ready.set()
   if not release.wait(5):raise RuntimeError('pause timeout')
  return original(selected,actual)
 core.crash=pause
 worker(root,op,args,queue=queue)

class Lifecycle(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)/'store';core.initialize(self.root);core.create_run(self.root,'r');self.input=Path(self.tmp.name)/'input';self.input.write_text('immutable fixture input');self.payload={'input':core.filepin(self.input)}
 def tearDown(self):self.tmp.cleanup()
 def complete(self,job='j',disposition='succeeded'):
  evidence=Path(self.tmp.name)/('e-'+job);evidence.write_text('tiny fixture result; not executor evidence');terminal=Path(self.tmp.name)/('t-'+job+'.json');terminal.write_bytes(core.canonical({'schema':'InventedWorkflowTerminalV1','run':'r','job':job,'payload_sha256':core.digest(self.payload),'disposition':disposition,'evidence':[core.filepin(evidence)]}));pin=core.filepin(terminal);core.record_terminal(self.root,'r',job,pin);return terminal,evidence
 def die(self,op,args,point):
  p=mp.Process(target=worker,args=(self.root,op,args),kwargs={'fail':point});p.start();p.join(5);self.assertFalse(p.is_alive());self.assertEqual(p.exitcode,91)
 def test_open_cannot_finalize_and_pending_blocks_sealed(self):
  core.enqueue(self.root,'r','j',self.payload)
  with self.assertRaises(core.Rejected):core.finalize(self.root,'r')
  core.seal(self.root,'r')
  with self.assertRaises(core.Rejected):core.finalize(self.root,'r')
  self.assertEqual(core.status(self.root,'r')['state'],'SEALED')
 def test_empty_policy_and_failure_disposition(self):
  core.seal(self.root,'r')
  with self.assertRaises(core.Rejected):core.finalize(self.root,'r')
  core.create_run(self.root,'empty','explicit_empty');core.seal(self.root,'empty');pin=core.finalize(self.root,'empty');self.assertEqual(json.loads(Path(pin['path']).read_text())['disposition'],'explicit_empty')
  core.create_run(self.root,'r2');core.enqueue(self.root,'r2','j',self.payload);core.seal(self.root,'r2')
 def test_exact_retry_and_rejected_postseal_no_mutation(self):
  core.enqueue(self.root,'r','j',self.payload);self.complete(disposition='failed');core.seal(self.root,'r');pin=core.finalize(self.root,'r');self.assertEqual(json.loads(Path(pin['path']).read_text())['disposition'],'terminal_with_failures')
  before=(self.root/'intake.sqlite').read_bytes();self.assertTrue(core.enqueue(self.root,'r','j',self.payload)['already_accepted'])
  with self.assertRaises(core.Rejected):core.enqueue(self.root,'r','late',self.payload)
  other=Path(self.tmp.name)/'other';other.write_text('different valid input')
  with self.assertRaises(core.Rejected):core.enqueue(self.root,'r','j',{'input':core.filepin(other)})
  self.assertEqual((self.root/'intake.sqlite').read_bytes(),before)
 def test_real_enqueue_seal_race(self):
  # Both OS processes cross the barrier; DB/authoritative flock decide order.
  wins={0:0,1:0}
  for index in range(12):
   run='race'+str(index);core.create_run(self.root,run);barrier=mp.Barrier(2);q=mp.Queue()
   a=mp.Process(target=worker,args=(self.root,'enqueue',(run,'j',self.payload),barrier,q));b=mp.Process(target=worker,args=(self.root,'seal',(run,),barrier,q));a.start();b.start();a.join(5);b.join(5)
   self.assertFalse(a.is_alive()or b.is_alive());self.assertEqual((a.exitcode,b.exitcode),(0,0));out=[q.get(timeout=1),q.get(timeout=1)];state=core.status(self.root,run);self.assertEqual(state['state'],'SEALED');self.assertIn(state['count'],[0,1]);self.assertEqual(sum(x[0]=='Rejected'for x in out),1-state['count']);wins[state['count']]+=1;q.close();q.join_thread()
  print('real_concurrent_enqueue_seal_outcomes',wins)
 def test_enqueue_commit_crashes(self):
  self.die('enqueue',('r','j',self.payload),'enqueue_before_commit');self.assertEqual(core.status(self.root,'r')['count'],0)
  self.die('enqueue',('r','j',self.payload),'enqueue_after_commit');self.assertEqual(core.status(self.root,'r')['count'],1);self.assertTrue(core.enqueue(self.root,'r','j',self.payload)['already_accepted'])
 def test_seal_commit_crashes(self):
  self.die('seal',('r',),'seal_before_commit');self.assertEqual(core.status(self.root,'r')['state'],'OPEN')
  self.die('seal',('r',),'seal_after_commit');self.assertEqual(core.status(self.root,'r')['state'],'SEALED')
  with self.assertRaises(core.Rejected):core.enqueue(self.root,'r','late',self.payload)
 def test_final_db_and_projection_crashes(self):
  for point in ['final_before_commit','final_after_commit','receipt_before_publish','receipt_after_publish','receipt_after_dirsync']:
   with tempfile.TemporaryDirectory()as tmp:
    root=Path(tmp)/'s';core.initialize(root);core.create_run(root,'r','explicit_empty');core.seal(root,'r')
    p=mp.Process(target=worker,args=(root,'finalize',('r',)),kwargs={'fail':point});p.start();p.join(5);self.assertEqual(p.exitcode,91);self.assertFalse(p.is_alive())
    state=core.status(root,'r');self.assertEqual(state['state'],'SEALED'if point=='final_before_commit'else'FINALIZED');pin=core.finalize(root,'r');data=Path(pin['path']).read_bytes();self.assertEqual(core.finalize(root,'r'),pin);self.assertEqual(core.status(root,'r')['final_sha256'],pin['sha256']);self.assertEqual(Path(pin['path']).read_bytes(),data)
 def test_terminal_commit_crash(self):
  core.enqueue(self.root,'r','j',self.payload);t,_=self.complete();pin=core.filepin(t)
  # New accepted job has no terminal; use same correctly bound fixture adjusted ID.
  core.enqueue(self.root,'r','k',self.payload);v=json.loads(t.read_text());v['job']='k';t2=Path(self.tmp.name)/'tk';t2.write_bytes(core.canonical(v));p2=core.filepin(t2)
  self.die('record_terminal',('r','k',p2),'terminal_before_commit');core.seal(self.root,'r')
  with self.assertRaises(core.Rejected):core.finalize(self.root,'r')
  self.die('record_terminal',('r','k',p2),'terminal_after_commit');core.finalize(self.root,'r')
 def test_drift_missing_corrupt_and_wrong_identity_fail_closed(self):
  core.enqueue(self.root,'r','j',self.payload);t,e=self.complete();core.seal(self.root,'r');pin=core.finalize(self.root,'r')
  old=e.read_bytes();e.write_text('changed')
  with self.assertRaises(core.Integrity):core.finalize(self.root,'r')
  e.write_bytes(old);Path(pin['path']).write_text('corrupt projection')
  with self.assertRaises(core.Integrity):core.finalize(self.root,'r')
  self.assertEqual(core.status(self.root,'r')['state'],'FINALIZED')
 def test_admission_set_and_source_binding_corruption(self):
  core.enqueue(self.root,'r','j',self.payload);core.seal(self.root,'r')
  with sqlite3.connect(self.root/'intake.sqlite')as db:db.execute('DELETE FROM jobs')
  with self.assertRaises(core.Integrity):core.status(self.root,'r')
  with sqlite3.connect(self.root/'intake.sqlite')as db:db.execute('UPDATE metadata SET value=?',('{}',))
  with self.assertRaises(core.Integrity):core.status(self.root,'r')
 def test_wrong_terminal_identity_and_missing_receipt(self):
  core.enqueue(self.root,'r','j',self.payload);t,_=self.complete();v=json.loads(t.read_text());v['job']='wrong';wrong=Path(self.tmp.name)/'wrong';wrong.write_bytes(core.canonical(v))
  with self.assertRaises(core.Integrity):core.record_terminal(self.root,'r','j',core.filepin(wrong))
  core.seal(self.root,'r');t.unlink()
  with self.assertRaises(FileNotFoundError):core.finalize(self.root,'r')
  self.assertEqual(core.status(self.root,'r')['state'],'SEALED')
 def test_uncertain_directory_fsync_no_ack_then_recovery(self):
  from unittest.mock import patch
  core.create_run(self.root,'e','explicit_empty');core.seal(self.root,'e')
  with patch.object(core,'fsyncdir',side_effect=OSError('injected uncertain directory fsync')):
   with self.assertRaises(OSError):core.finalize(self.root,'e')
  self.assertEqual(core.status(self.root,'e')['state'],'FINALIZED')
  pin=core.finalize(self.root,'e');self.assertEqual(pin,core.finalize(self.root,'e'))
 def test_contended_both_serialization_orders(self):
  for first in ['enqueue','seal']:
   run='held'+first;core.create_run(self.root,run);ready=mp.Event();release=mp.Event();q=mp.Queue()
   args=(run,'j',self.payload)if first=='enqueue'else(run,)
   a=mp.Process(target=held_worker,args=(self.root,first,args,first+'_before_commit',ready,release,q));a.start();self.assertTrue(ready.wait(5))
   second='seal'if first=='enqueue'else'enqueue';other=(run,)if second=='seal'else(run,'j',self.payload)
   b=mp.Process(target=worker,args=(self.root,second,other,None,q));b.start();release.set();a.join(5);b.join(5);self.assertFalse(a.is_alive()or b.is_alive());self.assertEqual((a.exitcode,b.exitcode),(0,0))
   values=[q.get(timeout=1),q.get(timeout=1)];state=core.status(self.root,run);self.assertEqual(state['state'],'SEALED');self.assertEqual(state['count'],1 if first=='enqueue'else 0);self.assertEqual(sum(v[0]=='Rejected'for v in values),0 if first=='enqueue'else 1);q.close();q.join_thread()
 def test_repeated_same_process_prepublication_errors_recover(self):
  from unittest.mock import patch
  core.create_run(self.root,'e','explicit_empty');core.seal(self.root,'e');real_link=core.os.link
  for index in range(3):
   with patch.object(core.os,'link',side_effect=OSError('injected prepublication ordinary exception')):
    with self.assertRaises(OSError):core.finalize(self.root,'e')
   self.assertEqual(core.status(self.root,'e')['state'],'FINALIZED');self.assertFalse((self.root/'final-e.json').exists())
  leftovers=list(self.root.glob('receipt-e-*.tmp'));self.assertEqual(len(leftovers),3);self.assertEqual(len({p.name for p in leftovers}),3)
  expected=leftovers[0].read_bytes();self.assertTrue(all(p.read_bytes()==expected for p in leftovers));pin=core.finalize(self.root,'e');self.assertEqual(Path(pin['path']).read_bytes(),expected);self.assertEqual(core.finalize(self.root,'e'),pin)
if __name__=='__main__':unittest.main()
