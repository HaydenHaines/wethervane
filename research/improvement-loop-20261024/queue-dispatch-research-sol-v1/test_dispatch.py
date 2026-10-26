import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import tempfile
import time
import unittest
import dispatch as d
import test_experiment_queue as fixtures


def die(root,stage):
    d.tick(root,lambda at:os._exit(73) if at==stage else None)


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.root=d.initialize(self.base/'queue')
        self.f=fixtures.QueueTests('runTest');self.f.base=self.base;self.f.root=self.root
        self.f.source=self.base/'owner.py';self.f.source.write_text(fixtures.FIXTURE.replace('time.sleep(60)','time.sleep(.4)'))
        self.addCleanup(self.f.cleanup)

    def plan(self,name,mode='seed'):
        return dict(schema='LocalExperimentQueuePlanV1',queue_id=name,jobs=[self.f.job(name,mode)])

    def admit(self,p):return d.enqueue(self.root,p,d.q.digest(p))

    def wait_release(self,attempt):
        deadline=time.monotonic()+8
        while time.monotonic()<deadline:
            state=d.tick(self.root)
            with d.admission.opened(self.root/'admissions.sqlite') as db:
                row=db.execute('SELECT body FROM releases WHERE attempt=?',(attempt,)).fetchone()
            if row:return json.loads(row[0])
            time.sleep(.02)
        self.fail(repr(state))

    def wait_terminal_without_releasing(self,attempt):
        deadline=time.monotonic()+8
        while time.monotonic()<deadline:
            state=d.ex.inspect(self.root/'executions',attempt)
            if state['attempt_terminal'] and not state['job_lock_held'] and not state['session_members'] and all(o['state']!='alive' for o in state['process_observation'].values()):
                return
            time.sleep(.02)
        self.fail('executor terminal/cleanup did not settle: '+repr(state))

    def crash(self,stage):
        child=mp.Process(target=die,args=(self.root,stage));child.start();child.join(8)
        self.assertFalse(child.is_alive());self.assertEqual(child.exitcode,73)

    def test_real_launch_cleanup_and_atomic_caller_crashes(self):
        for stage in ('reserved','prepared','launched','release_inserted','released'):
            with self.subTest(stage=stage):
                p=self.plan(stage);self.admit(p)
                if stage in ('release_inserted','released'):
                    d.tick(self.root)
                    deadline=time.monotonic()+5
                    while time.monotonic()<deadline:
                        s=d.ex.inspect(self.root/'executions',stage)
                        if s['attempt_terminal'] and not s['job_lock_held']:break
                        time.sleep(.02)
                self.crash(stage)
                result=self.wait_release(stage)
                self.assertEqual(result['outcome'],'accepted_execution')
                self.assertEqual(result['attempt'],stage)
                self.assertEqual(d.tick(self.root)['reserved'],0)

    def test_live_source_and_terminal_drift_keep_capacity(self):
        self.admit(self.plan('held','hold'));d.tick(self.root)
        self.assertEqual(d.tick(self.root)['reserved'],1)
        original=self.f.source.read_text();self.f.source.write_text(original+'\n# drift')
        self.assertIn('retained_integrity',d.tick(self.root)['observations']['held'])
        self.f.source.write_text(original)
        self.wait_terminal_without_releasing('held')
        terminal=self.root/'executions/attempts/held/terminal.json'
        old=terminal.read_text();value=json.loads(old);value['attempt_id']='wrong';terminal.write_text(json.dumps(value))
        self.assertIn('retained_integrity',d.tick(self.root)['observations']['held'])
        terminal.write_text(old)
        self.assertEqual(self.wait_release('held')['outcome'],'accepted_execution')

    def test_registered_spec_identity_drift_and_partial_scaffold(self):
        self.admit(self.plan('identity'));self.crash('prepared')
        path=self.root/'executions/attempts/identity/specification.json'
        old=path.read_bytes();path.write_bytes(old+b' ')
        state=d.tick(self.root);self.assertEqual(state['reserved'],1)
        self.assertIn('retained_integrity',state['observations']['identity'])
        path.write_bytes(old);self.wait_release('identity')
        self.admit(self.plan('partial'));self.crash('reserved')
        (self.root/'executions/attempts/partial').mkdir()
        state=d.tick(self.root);self.assertEqual(state['reserved'],1)
        self.assertIn('retained_integrity',state['observations']['partial'])

    def test_process_failure_and_explicit_executor_retry(self):
        first=self.plan('failed','nonzero');self.admit(first)
        result=self.wait_release('failed');self.assertEqual(result['outcome'],'failed_process')
        retry=self.plan('retry');spec=retry['jobs'][0]['execution_spec']
        spec['job_key']='failed'
        spec['previous_attempt']=dict(attempt_id='failed',receipt=result['executor_terminal'])
        spec['retry_reason']='Explicit synthetic nonzero retry with new output and attempt'
        self.admit(retry);self.crash('prepared')
        prior=Path(result['executor_terminal']['path']);old=prior.read_bytes();prior.write_bytes(old+b' ')
        state=d.tick(self.root);self.assertEqual(state['reserved'],1)
        self.assertIn('retained_integrity',state['observations']['retry'])
        prior.write_bytes(old)
        self.assertEqual(self.wait_release('retry')['outcome'],'accepted_execution')

    def test_active_input_and_reservation_corruption_retain_capacity(self):
        pinned=self.base/'explicit-input.txt';pinned.write_text('original')
        p=self.plan('input','hold');p['jobs'][0]['execution_spec']['input_pins'].append(d.ex.pin(pinned))
        self.admit(p);d.tick(self.root)
        pinned.write_text('drift')
        state=d.tick(self.root);self.assertEqual(state['reserved'],1)
        self.assertIn('retained_integrity',state['observations']['input'])
        pinned.write_text('original')
        with d.admission.opened(self.root/'admissions.sqlite') as db:
            body=db.execute("SELECT body FROM reservations WHERE attempt='input'").fetchone()[0]
            db.execute('DROP TRIGGER reservation_update')
            db.execute("UPDATE reservations SET body=body||' ' WHERE attempt='input'")
        state=d.tick(self.root);self.assertEqual(state['reserved'],1)
        self.assertIn('reservation digest',state['observations']['input'])
        with d.admission.opened(self.root/'admissions.sqlite') as db:
            db.execute("UPDATE reservations SET body=? WHERE attempt='input'",(body,))
        self.wait_release('input')

    def test_capacity_and_unverified_cleanup_are_retained(self):
        self.admit(self.plan('first','hold'));self.admit(self.plan('second'))
        d.tick(self.root)
        self.assertEqual(d.tick(self.root)['reserved'],1)
        self.assertFalse((self.root/'executions/attempts/second').exists())
        self.wait_terminal_without_releasing('first')
        path=self.root/'executions/attempts/first/terminal.json'
        old=path.read_text();terminal=json.loads(old)
        terminal['cleanup']=dict(verified=False,remaining=['unresolved'])
        path.write_text(json.dumps(terminal))
        state=d.tick(self.root);self.assertEqual(state['reserved'],1)
        self.assertIn('unverified cleanup',state['observations']['first'])
        self.assertFalse((self.root/'executions/attempts/second').exists())
        path.write_text(old);self.wait_release('first');self.wait_release('second')

    def test_unknown_exit_never_releases(self):
        self.f.source.write_text(fixtures.FIXTURE)
        self.admit(self.plan('unknown','hold'));d.tick(self.root)
        directory=self.root/'executions/attempts/unknown'
        deadline=time.monotonic()+5
        while not (self.base/'output-unknown/receipt.json').exists() and time.monotonic()<deadline:time.sleep(.02)
        self.assertTrue((self.base/'output-unknown/receipt.json').exists(), 'child must pass executor gate before supervisor kill')
        identity=d.q.document(directory/'supervisor.json')['identity']
        fd=d.ex.open_pidfd(identity);signal.pidfd_send_signal(fd,signal.SIGKILL);os.close(fd)
        deadline=time.monotonic()+5
        while d.ex.observe(identity)['state']=='alive' and time.monotonic()<deadline:time.sleep(.02)
        d.ex.stop(self.root/'executions','unknown')
        deadline=time.monotonic()+5
        while d.ex.inspect(self.root/'executions','unknown')['job_lock_held'] and time.monotonic()<deadline:time.sleep(.02)
        d.ex.recover(self.root/'executions','unknown')
        state=d.tick(self.root)
        self.assertEqual(state['reserved'],1)
        self.assertEqual(state['observations']['unknown'],'retained_unknown_exit')

    def test_dependencies_and_old_receipts_not_adopted(self):
        p=self.plan('dep');p['jobs'][0]['dependencies']=[dict(job_id='foreign')]
        with self.assertRaises(ValueError):self.admit(p)
        with self.assertRaises(FileNotFoundError): d.tick(self.base/'old-domain')


if __name__=='__main__':unittest.main(verbosity=2)
