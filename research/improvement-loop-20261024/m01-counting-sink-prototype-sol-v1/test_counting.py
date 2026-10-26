"""Invented objects only; preserve every temporary fixture attempt for review."""
import hashlib,json,pickle,signal,unittest,uuid
from pathlib import Path
import counting as c
HERE=Path(__file__).resolve().parent
ROOT=HERE/('test-run-'+uuid.uuid4().hex);ROOT.mkdir()
RUNS=[]
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
 def test_hard_cpu_cap_terminates_only_toy_child(self):
  out,r=self.run_case('cpu-cap',fault='cpu_before_count');self.assertIn(out['terminal']['returncode'],[-signal.SIGXCPU,-signal.SIGKILL]);self.assertFalse(out['complete']);self.assertFalse((Path(out['root'])/'complete.json').exists())
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
if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
 (HERE/'test-result.json').write_text(json.dumps(dict(schema='InventedCountingSinkTestResultsV1',test_root=str(ROOT),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),runs=RUNS,scope='invented fixtures only; no deserialization, actual graph, compile or sampling'),indent=2)+'\n');raise SystemExit(not result.wasSuccessful())
