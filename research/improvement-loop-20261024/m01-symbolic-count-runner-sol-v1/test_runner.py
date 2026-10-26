import ast,dataclasses,hashlib,json,os,signal,subprocess,sys,tempfile,time,unittest,multiprocessing as mp
from pathlib import Path
import controller as c
from common import pin,Writer
HERE=Path(__file__).resolve().parent;LAB=HERE.parent
REPORT=[]
def race_creator(root,pipe):
 original=Path.exists;intercepted=False
 def observed(path):
  nonlocal intercepted
  result=original(path)
  if path==Path(root) and not intercepted:
   intercepted=True;pipe.send('ready');pipe.recv()
  return result
 from unittest.mock import patch
 with patch.object(Path,'exists',observed):
  try:
   token=c.Controller(c.sources()).fixture(root,128);pipe.send(('ack',token.observation['returncode']))
  except ValueError as error:pipe.send(('rejected',str(error)))
 pipe.close()

class Tests(unittest.TestCase):
 def test_exact_scientific_guard_calls_and_seven_root_single_stream(self):
  original=ast.parse((LAB/'m01-parent-gradient-after-replacement-plan-sol-v1/experiment.py').read_text());new=ast.parse((HERE/'pipeline.py').read_text())
  original=next(n for n in original.body if isinstance(n,ast.FunctionDef)and n.name=='symbolic');new=next(n for n in new.body if isinstance(n,ast.FunctionDef)and n.name=='symbolic')
  guards=lambda node:[ast.dump(n,include_attributes=False)for n in ast.walk(node)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='need'and not any(isinstance(a,ast.Constant)and a.value=='new exclusive output directory required'for a in n.args)]
  self.assertEqual(guards(original),guards(new))
  old_dump=next(n for n in ast.walk(original)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='dump')
  counted=next(n for n in ast.walk(new)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='count_bundle')
  self.assertEqual(ast.dump(old_dump.args[0],include_attributes=False),ast.dump(counted.args[0],include_attributes=False))
  self.assertFalse(any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=='pickle'and n.func.attr in ('dump','dumps','load','loads')for n in ast.walk(new)))
  plan=json.loads((HERE/'plan.json').read_text());self.assertEqual(plan['runner_sources'],c.sources());self.assertEqual(plan['gradient_contract']['dimensions'],662)
 def test_normal_tiny_count_and_ephemeral_snapshot_reject_copy_restart(self):
  with tempfile.TemporaryDirectory()as temp:
   root=Path(temp)/'normal';issuer=c.Controller(c.sources());token=issuer.fixture(root,4096);result=json.loads(token.files['receipts/result.json'])
   self.assertFalse(result['pickle_payload_persisted']);self.assertEqual(result['stream_count']['status'],'single_pickler_returned')
   worker=json.loads(token.files['receipts/worker-terminal.json']);self.assertEqual(worker['nice'],15);self.assertEqual(worker['limits']['cpu'],[2,3]);self.assertTrue(token.observation['reaped'])
   with self.assertRaises(TypeError):token.files['fake']=b'bad'
   with self.assertRaises(TypeError):token.observation['peaks']['rss']=0
   with self.assertRaises(ValueError):issuer.verify_retained(dataclasses.replace(token))
   with self.assertRaises(ValueError):c.Controller(c.sources()).verify_retained(token)
   original=token.files['receipts/result.json'];(root/'receipts/result.json').write_bytes(original+b' ')
   self.assertEqual(issuer.verify_retained(token).files['receipts/result.json'],original)
   self.assertFalse(list(root.rglob('*.pkl')));REPORT.append(dict(test='normal',returncode=token.observation['returncode'],reaped=True,count=result['stream_count'],scope='snapshot bytes remain valid, live-directory assertion absent'))
 def test_logical_cap_and_serialization_failure_preserve_nonzero_receipts(self):
  for fault in ('cap','raise'):
   with tempfile.TemporaryDirectory()as temp:
    root=Path(temp)/fault;issuer=c.Controller(c.sources())
    with self.assertRaises(ValueError):issuer.fixture(root,65536,fault)
    terminal=json.loads((root/'terminal.json').read_text());self.assertNotEqual(terminal['returncode'],0);self.assertTrue(terminal['reaped']);self.assertFalse(issuer.issued)
    self.assertTrue((root/'receipts/worker-failure.json').exists());self.assertFalse((root/'receipts/result.json').exists())
    REPORT.append(dict(test=fault,terminal=terminal))
 def test_uncertain_controller_terminal_fsync_has_no_reconstructable_ack(self):
  with tempfile.TemporaryDirectory()as temp:
   root=Path(temp)/'uncertain';issuer=c.Controller(c.sources())
   with self.assertRaises(OSError):issuer.fixture(root,128,'terminal_fsync')
   self.assertTrue((root/'terminal.json').exists());self.assertFalse(issuer.issued)
   with self.assertRaises(ValueError):issuer.verify_retained(dict(path=str(root/'terminal.json'),sha256=pin(root/'terminal.json')['sha256']))
   REPORT.append(dict(test='uncertain_fsync',terminal_visible=True,issued_acknowledgments=0))
 def test_competing_creators_cannot_mutate_winning_attempt(self):
  with tempfile.TemporaryDirectory()as temp:
   root=Path(temp)/'race';context=mp.get_context('fork');pairs=[context.Pipe()for _ in range(2)]
   children=[context.Process(target=race_creator,args=(str(root),pair[1]))for pair in pairs]
   for child in children:child.start()
   for parent,childend in pairs:childend.close();self.assertTrue(parent.poll(3));self.assertEqual(parent.recv(),'ready')
   for parent,_ in pairs:parent.send('go')
   outcomes=[]
   for parent,_ in pairs:self.assertTrue(parent.poll(6));outcomes.append(parent.recv());parent.close()
   for child in children:child.join(3);self.assertEqual(child.exitcode,0)
   self.assertEqual(sorted(x[0]for x in outcomes),['ack','rejected'])
   self.assertFalse((root/'controller-failure.json').exists());self.assertEqual(json.loads((root/'terminal.json').read_text())['returncode'],0)
   REPORT.append(dict(test='competing_creators',outcomes=outcomes,winner_mutated_by_loser=False))
 def test_actual_gate_missing_technical_evidence_creates_no_attempt(self):
  with tempfile.TemporaryDirectory()as temp:
   root=Path(temp)/'never-created'
   with self.assertRaises((ValueError,TypeError)):c.Controller(c.sources()).actual(root,pin(HERE/'plan.json'),None)
   self.assertFalse(root.exists());self.assertFalse(Path(str(root)+'.lock').exists())
 def test_nonblocking_capture_rejects_fifo(self):
  from common import read_regular
  with tempfile.TemporaryDirectory()as temp:
   path=Path(temp)/'fifo';os.mkfifo(path)
   start=time.monotonic()
   with self.assertRaises(ValueError):read_regular(path)
   self.assertLess(time.monotonic()-start,1)
 def test_toy_wall_watchdog_reaps_direct_child_without_ack(self):
  with tempfile.TemporaryDirectory()as temp:
   root=Path(temp)/'wall';issuer=c.Controller(c.sources())
   with self.assertRaises(ValueError):issuer.fixture(root,128,'hold')
   terminal=json.loads((root/'terminal.json').read_text());self.assertEqual(terminal['reason'],'wall_watchdog');self.assertTrue(terminal['reaped']);self.assertNotEqual(terminal['returncode'],0);self.assertFalse(issuer.issued)
   self.assertFalse(Path('/proc/'+str(terminal['child_pid'])).exists());REPORT.append(dict(test='wall_watchdog',terminal=terminal))
 def test_real_controller_sigterm_reaps_its_direct_toy_worker(self):
  with tempfile.TemporaryDirectory()as temp:
   root=Path(temp)/'interrupt'
   proc=subprocess.Popen([sys.executable,'-B',str(HERE/'controller.py'),'--fixture','--fault','hold','--output',str(root)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
   deadline=time.monotonic()+5
   while not (root/'launch.json').exists()and proc.poll()is None and time.monotonic()<deadline:time.sleep(.01)
   self.assertTrue((root/'launch.json').exists());proc.terminate();stdout,stderr=proc.communicate(timeout=6)
   self.assertNotEqual(proc.returncode,0);terminal=json.loads((root/'terminal.json').read_text())
   self.assertTrue(terminal['reaped']);self.assertNotEqual(terminal['returncode'],0)
   self.assertFalse(Path('/proc/'+str(terminal['child_pid'])).exists());REPORT.append(dict(test='controller_sigterm',terminal=terminal,controller_returncode=proc.returncode))

if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));print(json.dumps(dict(schema='SymbolicCountRunnerToyTestsV1',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),report=REPORT,actual_graph_executed=False,scientific_imports=False),indent=2));raise SystemExit(not result.wasSuccessful())
