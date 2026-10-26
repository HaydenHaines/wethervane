"""Invented objects only. Retain all uncertain/corrupt fixture files."""
import dataclasses,hashlib,json,threading,time,unittest,uuid,multiprocessing as mp
from pathlib import Path
import session as s
HERE=Path(__file__).resolve().parent;ROOT=HERE/('test-run-'+uuid.uuid4().hex);ROOT.mkdir();REPORT=[]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def make():return s.Session(digest(HERE/'engine.py'),digest(HERE/'session.py'))
def request():return dict(schema='InventedCountingRequestV1',fixture='nested_v1',payload_bytes=4096,budget_bytes=65536,fault='none')
class Tests(unittest.TestCase):
 def test_normal_explicit_ack_and_immutable_snapshot(self):
  issuer=make();ack=issuer.execute(ROOT/'normal',request());snapshot=issuer.verify(ack)
  with self.assertRaises(TypeError):snapshot.files['new']=b'bad'
  with self.assertRaises(TypeError):snapshot.manifest['files']['request.json']['bytes']=0
  with self.assertRaises(ValueError):issuer.verify(dataclasses.asdict(ack))
  with self.assertRaises(ValueError):issuer.verify(dataclasses.replace(ack))
  with self.assertRaises(ValueError):make().verify(ack)
  REPORT.append(dict(test='normal',source=snapshot.provenance['source_sha256'],files=len(snapshot.files),copy_and_restart_rejected=True))
 def test_forked_process_cannot_adopt_retained_session(self):
  issuer=make();ack=issuer.execute(ROOT/'fork-adoption',request());recv,send=mp.get_context('fork').Pipe(duplex=False)
  def child():
   try:issuer.verify(ack)
   except ValueError:send.send('rejected');send.close();return
   send.send('incorrectly accepted');send.close()
  proc=mp.get_context('fork').Process(target=child);proc.start();send.close()
  try:
   self.assertTrue(recv.poll(2));self.assertEqual(recv.recv(),'rejected');proc.join(2);self.assertFalse(proc.is_alive());self.assertEqual(proc.exitcode,0)
  finally:
   recv.close()
   if proc.is_alive():proc.terminate();proc.join(1)
  REPORT.append(dict(test='forked_session_adoption',rejected=True))
 def test_uncertain_supervisor_success_cannot_be_rehashed_to_ack(self):
  issuer=make();root=ROOT/'uncertain'
  with self.assertRaises(OSError):issuer.execute(root,request(),test_fault='supervisor_post_rename_fsync')
  control=root.with_name(root.name+'.supervisor');receipt=control/'success.json';self.assertTrue(receipt.exists());self.assertEqual(len(issuer._issued),0)
  raw=json.loads(receipt.read_text());manifest_sha=raw['manifest_sha256']
  # Bounded V3 counterexample preserved: raw files/rehashing appear consistent.
  issuer.engine.verify_complete(root,manifest_sha,receipt,digest(receipt))
  with self.assertRaises(ValueError):issuer.verify(dict(path=str(receipt),sha256=digest(receipt)))
  REPORT.append(dict(test='uncertain_success',receipt_visible=True,raw_v3_file_check_accepts=True,v4_ack_issued=False,receipt_sha256=digest(receipt)))
 def test_concurrent_readers_block_cooperating_writer(self):
  issuer=make();ack=issuer.execute(ROOT/'cooperating',request());root=Path(ack.root);original=(root/'count-candidate.json').read_bytes();ready=[threading.Event(),threading.Event()];release=threading.Event();attempted=threading.Event();entered=threading.Event();snapshots=[];errors=[]
  def reader(i):
   try:
    def hook(stage,name):
     if stage=='lock_acquired':ready[i].set();self.assertTrue(release.wait(2))
    snapshots.append(issuer.verify(ack,test_hook=hook))
   except BaseException as e:errors.append(repr(e))
  readers=[threading.Thread(target=reader,args=(i,))for i in range(2)]
  for thread in readers:thread.start()
  self.assertTrue(all(event.wait(2)for event in ready))
  writer=threading.Thread(target=issuer.write_fixture_metadata,args=(ack,'count-candidate.json',original+b' ',attempted,entered));writer.start();self.assertTrue(attempted.wait(1));time.sleep(.05);self.assertFalse(entered.is_set());release.set()
  for thread in readers:thread.join(3);self.assertFalse(thread.is_alive())
  writer.join(3);self.assertFalse(writer.is_alive());self.assertTrue(entered.is_set());self.assertFalse(errors);self.assertEqual(len(snapshots),2);self.assertTrue(all(x.files['output/count-candidate.json']==original for x in snapshots))
  with self.assertRaises(ValueError):issuer.verify(ack)
  REPORT.append(dict(test='cooperating_race',readers=2,writer_blocked_until_readers_release=True,snapshots_remain_original=True,changed_directory_next_verify_rejected=True))
 def test_noncooperating_after_read_mutation_does_not_change_returned_snapshot(self):
  issuer=make();ack=issuer.execute(ROOT/'noncooperating-after',request());path=Path(ack.root)/'count-candidate.json';original=path.read_bytes();changed=False
  def hook(stage,name):
   nonlocal changed
   if stage=='after_read'and name=='output/count-candidate.json'and not changed:path.write_bytes(original+b' ');changed=True
  snapshot=issuer.verify(ack,test_hook=hook);self.assertTrue(changed);self.assertEqual(snapshot.files['output/count-candidate.json'],original);self.assertNotEqual(path.read_bytes(),original)
  with self.assertRaises(ValueError):issuer.verify(ack)
  REPORT.append(dict(test='uncooperative_after_read',snapshot_valid_original_bytes=True,live_directory_changed=True,no_live_directory_claim=True))
 def test_noncooperating_before_read_change_rejects(self):
  issuer=make();ack=issuer.execute(ROOT/'noncooperating-before',request());path=Path(ack.root)/'count-candidate.json';original=path.read_bytes();changed=False
  def hook(stage,name):
   nonlocal changed
   if stage=='before_read'and name=='output/count-candidate.json'and not changed:path.write_bytes(original+b' ');changed=True
  with self.assertRaises(ValueError):issuer.verify(ack,test_hook=hook)
  REPORT.append(dict(test='uncooperative_before_read',changed_captured_bytes_rejected=True))
if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));(HERE/'test-result.json').write_text(json.dumps(dict(schema='V4CallerAckSnapshotTestsV1',root=str(ROOT),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),report=REPORT,scope='invented fixtures only; no deserialize or graph'),indent=2)+'\n');raise SystemExit(not result.wasSuccessful())
