"""Invented objects only; preserve every temporary fixture attempt for review."""
import hashlib,json,pickle,signal,unittest,uuid,os,time,multiprocessing as mp,select
from pathlib import Path
import counting as c
HERE=Path(__file__).resolve().parent
ROOT=HERE/('test-run-'+uuid.uuid4().hex);ROOT.mkdir()
RUNS=[]

def interrupted_controller(root,request):
 try:c.run(root,request,c.sha(c.canonical(request)+b'\n'),c.pin(c.SOURCE)['sha256'])
 except BaseException as error:
  c.atomic_json(Path(root)/'controller-outcome.json',dict(type=type(error).__name__,message=str(error)))
  raise SystemExit(84)

class Tests(unittest.TestCase):
 def request(self,**changes):
  r=dict(schema='InventedCountingRequestV1',fixture='nested_v1',payload_bytes=4096,budget_bytes=65536,fault='none');r.update(changes);return r
 def run_case(self,name,**changes):
  r=self.request(**changes);out=c.run(ROOT/name,r,c.sha(c.canonical(r)+b'\n'),c.pin(c.SOURCE)['sha256']);RUNS.append(out);return out,r
 def test_stream_budget_before_accepting_write(self):
  sink=c.CountingSink(4);self.assertEqual(sink.write(b'abc'),3)
  with self.assertRaises(c.BudgetExceeded):sink.write(b'de')
  self.assertEqual(sink.accepted,3);self.assertEqual(sink.h.hexdigest(),hashlib.sha256(b'abc').hexdigest());self.assertEqual(sink.attempted_end,5)
 def test_complete_counts_match_independent_tiny_pickle_bytes(self):
  out,r=self.run_case('complete');raw=pickle.dumps(c.invented(r),protocol=5)
  self.assertEqual(out['counter']['accepted_bytes'],len(raw));self.assertEqual(out['counter']['accepted_prefix_sha256'],c.sha(raw))
  manifest=c.verify_complete(out['root'],out['manifest_pin']['sha256']);self.assertFalse(manifest['deserialization_authorized'])
  progress=json.loads((Path(out['root'])/'progress.json').read_text());self.assertEqual(progress['nice'],15);self.assertEqual(progress['limits']['cpu'],[2,3]);self.assertEqual(progress['limits']['file'],[1048576,1048576])
  self.assertFalse(any(p.suffix in ['.pkl','.pickle']for p in Path(out['root']).iterdir()))
 def test_budget_failure_retained_without_complete_marker(self):
  out,r=self.run_case('budget-failure',budget_bytes=64);self.assertFalse(out['complete']);root=Path(out['root']);self.assertTrue((root/'failure.json').exists());self.assertFalse((root/'complete.json').exists())
 def test_interrupted_prefix_retained(self):
  out,r=self.run_case('write-failure',fault='raise_after_write');self.assertFalse(out['complete']);counter=json.loads((Path(out['root'])/'progress.json').read_text())['counter'];self.assertGreater(counter['accepted_bytes'],0);self.assertFalse((Path(out['root'])/'complete.json').exists())
 def test_death_after_candidate_never_publishes_complete(self):
  out,r=self.run_case('metadata-death',fault='exit_after_metadata');self.assertEqual(out['terminal']['returncode'],73);self.assertTrue((Path(out['root'])/'count-candidate.json').exists());self.assertFalse((Path(out['root'])/'complete.json').exists())
 def test_soft_cpu_limit_sigxcpu_terminates_only_toy_child(self):
  out,r=self.run_case('cpu-cap',fault='cpu_before_count');self.assertIn(out['terminal']['returncode'],[-signal.SIGXCPU,-signal.SIGKILL]);self.assertFalse(out['complete']);self.assertEqual(out['terminal']['termination_reason'],'cpu_soft_limit_sigxcpu');self.assertTrue(out['terminal']['cleanup']['verified']);self.assertFalse((Path(out['root'])/'complete.json').exists())
 def test_source_request_duplicate_identity_fail_closed(self):
  r=self.request();bad=ROOT/'bad-source'
  with self.assertRaises(ValueError):c.run(bad,r,c.sha(c.canonical(r)+b'\n'),'0'*64)
  self.assertFalse(bad.exists())
  with self.assertRaises(ValueError):c.run(ROOT/'bad-request',r,'0'*64,c.pin(c.SOURCE)['sha256'])
  out,r=self.run_case('duplicate')
  with self.assertRaises(FileExistsError):c.run(out['root'],r,c.sha(c.canonical(r)+b'\n'),c.pin(c.SOURCE)['sha256'])
 def test_complete_metadata_tamper_rejected_and_preserved(self):
  out,r=self.run_case('tamper');path=Path(out['root'])/'count-candidate.json';old=path.read_bytes();path.write_bytes(old+b' ')
  with self.assertRaises(ValueError):c.verify_complete(out['root'],out['manifest_pin']['sha256'])
  (Path(out['root'])/'tamper-note.json').write_text(json.dumps(dict(intentional_test_mutation='count-candidate trailing byte',original_sha256=c.sha(old))))
 def test_v1_complete_schema_not_silently_adopted(self):
  out,r=self.run_case('version-refusal');root=Path(out['root']);manifest=json.loads((root/'complete.json').read_text());manifest['schema']='LocallyGeneratedCompleteCountingManifestV1';(root/'complete.json').write_bytes(c.canonical(manifest)+b'\n')
  with self.assertRaises(ValueError):c.verify_complete(root,c.pin(root/'complete.json')['sha256'])
  (root/'version-test-note.json').write_text(json.dumps(dict(intentional_test_mutation='completion schema V1; recomputed caller SHA still cannot authorize V2 adoption')))
 def test_launcher_receipt_write_exception_tears_down_child(self):
  root=ROOT/'launch-receipt-exception';request=self.request(fault='cpu_before_count');original=c.atomic_json
  def fail_launch(path,value):
   if Path(path).name=='launch.json':raise OSError('invented launcher receipt write failure')
   return original(path,value)
  c.atomic_json=fail_launch
  try:
   with self.assertRaisesRegex(OSError,'receipt write failure'):c.run(root,request,c.sha(c.canonical(request)+b'\n'),c.pin(c.SOURCE)['sha256'])
  finally:c.atomic_json=original
  terminal=json.loads((root/'terminal.json').read_text());self.assertEqual(terminal['termination_reason'],'controller_exception');self.assertTrue(terminal['cleanup']['verified']);self.assertTrue(terminal['cleanup']['reaped']);self.assertIn('SIGTERM',terminal['cleanup']['signals']);self.assertFalse((root/'complete.json').exists());self.assertFalse(Path('/proc/'+str(terminal['pid'])).exists())
 def test_real_controller_sigterm_tears_down_child_and_records_terminal(self):
  root=ROOT/'controller-sigterm';request=self.request(fault='cpu_before_count');controller=mp.Process(target=interrupted_controller,args=(root,request));controller.start();deadline=time.monotonic()+5;child_fd=None
  try:
   while time.monotonic()<deadline:
    if (root/'launch.json').exists()and(root/'progress.json').exists():break
    self.assertTrue(controller.is_alive());time.sleep(.01)
   self.assertTrue((root/'progress.json').exists());pid=json.loads((root/'launch.json').read_text())['pid'];child_fd=os.pidfd_open(pid)
   os.kill(controller.pid,signal.SIGTERM);controller.join(5);self.assertFalse(controller.is_alive());self.assertEqual(controller.exitcode,84)
   poller=select.poll();poller.register(child_fd,select.POLLIN);self.assertTrue(poller.poll(1000),'owned worker exit must be observed via pidfd')
   terminal=json.loads((root/'terminal.json').read_text());self.assertEqual(terminal['termination_reason'],'controller_interruption');self.assertEqual(terminal['controller_error']['signal'],signal.SIGTERM);self.assertTrue(terminal['cleanup']['verified']);self.assertTrue(terminal['cleanup']['reaped']);self.assertFalse((root/'complete.json').exists())
  finally:
   if child_fd is not None:os.close(child_fd)
   if controller.is_alive():controller.terminate();controller.join(2)

if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
 (HERE/'test-result.json').write_text(json.dumps(dict(schema='InventedCountingSinkTestResultsV1',test_root=str(ROOT),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),runs=RUNS,scope='invented fixtures only; no deserialization, actual graph, compile or sampling'),indent=2)+'\n');raise SystemExit(not result.wasSuccessful())
