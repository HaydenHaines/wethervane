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


def die(root,stage):d.tick(root,lambda at:os._exit(79) if at==stage else None)


class DependencyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.root=d.initialize(self.base/'queue')
        self.f=fixtures.QueueTests('runTest');self.f.base=self.base;self.f.root=self.root
        self.f.source=self.base/'owner.py';self.f.source.write_text(fixtures.FIXTURE)
        self.addCleanup(self.f.cleanup)

    def chain(self,name='chain',mode='seed'):
        # Consumer first deliberately tests readiness, not insertion order.
        return dict(schema='LocalExperimentQueuePlanV1',queue_id=name,
                    jobs=[self.f.job('consumer',producer='producer'),self.f.job('producer',mode)])

    def admit(self,p):return d.enqueue(self.root,p,d.q.digest(p))

    def released(self,attempt):
        with d.admission.opened(self.root/'admissions.sqlite') as db:return d.row_value(db,'releases',attempt)

    def wait_release(self,attempt):
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            state=d.tick(self.root)
            saved=self.released(attempt)
            if saved:return d.q.document(saved[0]['result']['path'])
            time.sleep(.02)
        self.fail(repr(state))

    def terminal(self,attempt):
        deadline=time.monotonic()+8
        while time.monotonic()<deadline:
            state=d.ex.inspect(self.root/'executions',attempt)
            if state['attempt_terminal'] and not state['job_lock_held'] and not state['session_members'] and all(o['state']!='alive' for o in state['process_observation'].values()):return
            time.sleep(.02)
        self.fail(repr(state))

    def crash(self,stage):
        child=mp.Process(target=die,args=(self.root,stage));child.start();child.join(10)
        self.assertFalse(child.is_alive());self.assertEqual(child.exitcode,79)

    def test_real_chain_effective_spec_and_successful_predecessor_repeat(self):
        self.admit(self.chain());self.wait_release('consumer')
        self.assertEqual((self.base/'output-consumer/data.txt').read_text(),'seed:consumed')
        bound=d.q.document(self.root/'evidence/consumer/binding.json')
        dependency=bound['dependencies'][0];spec=bound['execution_spec']
        self.assertEqual(spec['argv'][5],dependency['output']['sha256'])
        self.assertTrue(all(dependency[k] in spec['input_pins'] for k in ('result','release','terminal','output')))
        self.assertEqual(d.tick(self.root)['reserved'],0)
        # Explicit success repeat is allowed, not inferred scientific necessity.
        repeated=self.f.job('repeat');s=repeated['execution_spec'];s['job_key']='consumer'
        prior=self.released('consumer')[0];old=d.q.document(prior['result']['path'])
        s['previous_attempt']=dict(attempt_id='consumer',receipt=old['executor_terminal']);s['retry_reason']='Explicit synthetic successful repeat'
        self.admit(dict(schema='LocalExperimentQueuePlanV1',queue_id='repeat',jobs=[repeated]))
        self.assertEqual(self.wait_release('repeat')['outcome'],'accepted_execution')

    def test_transitive_three_job_chain_and_native_failure(self):
        plan=dict(schema='LocalExperimentQueuePlanV1',queue_id='three',jobs=[
            self.f.job('last',producer='middle'),self.f.job('middle',producer='first'),self.f.job('first')])
        self.admit(plan);self.wait_release('last')
        self.assertEqual((self.base/'output-last/data.txt').read_text(),'seed:consumed:consumed')
        plan=dict(schema='LocalExperimentQueuePlanV1',queue_id='bad-native',jobs=[
            self.f.job('blocked',producer='bad'),self.f.job('bad','wrong_metadata')])
        self.admit(plan)
        self.assertEqual(self.wait_release('bad')['outcome'],'failed_native_outputs')
        state=d.tick(self.root);self.assertEqual(state['reserved'],0)
        self.assertIn('producer failed',state['observations']['blocked'])
        self.assertFalse((self.root/'executions/attempts/blocked').exists())

    def test_producer_nonzero_never_authorizes_consumer(self):
        self.admit(self.chain(mode='nonzero'))
        self.assertEqual(self.wait_release('producer')['outcome'],'failed_process')
        state=d.tick(self.root);self.assertEqual(state['reserved'],0)
        self.assertIn('producer failed',state['observations']['consumer'])
        self.assertFalse((self.root/'executions/attempts/consumer').exists())

    def test_selected_producer_receipt_result_release_terminal_tamper(self):
        self.admit(self.chain());self.wait_release('producer')
        targets=[self.base/'output-producer/receipt.json',self.root/'evidence/producer/result.json',
                 self.root/'evidence/producer/release.json',self.root/'executions/attempts/producer/terminal.json']
        for path in targets:
            with self.subTest(path=path.name):
                old=path.read_bytes();path.write_bytes(old+b' ')
                state=d.tick(self.root)
                self.assertEqual(state['state'],'blocked_pre_reservation')
                self.assertFalse((self.root/'executions/attempts/consumer').exists())
                self.assertFalse(self.released('consumer'))
                path.write_bytes(old)
        with d.admission.opened(self.root/'admissions.sqlite') as db:
            old=db.execute("SELECT body FROM releases WHERE attempt='producer'").fetchone()[0]
            db.execute('DROP TRIGGER release_update');db.execute("UPDATE releases SET body=body||' ' WHERE attempt='producer'")
        self.assertEqual(d.tick(self.root)['state'],'blocked_pre_reservation')
        with d.admission.opened(self.root/'admissions.sqlite') as db:db.execute("UPDATE releases SET body=? WHERE attempt='producer'",(old,))
        self.wait_release('consumer')

    def test_active_binding_and_launch_dependency_recheck(self):
        self.admit(self.chain());self.wait_release('producer');self.crash('reserved')
        own=self.root/'evidence/consumer/binding.json';saved=own.read_bytes();own.write_bytes(saved+b' ')
        state=d.tick(self.root);self.assertEqual(state['reserved'],1)
        self.assertIn('retained_integrity',state['observations']['consumer'])
        self.assertFalse((self.root/'executions/attempts/consumer').exists());own.write_bytes(saved)
        path=self.root/'evidence/producer/release.json';old=path.read_bytes();path.write_bytes(old+b' ')
        state=d.tick(self.root);self.assertEqual(state['reserved'],1)
        self.assertIn('retained_integrity',state['observations']['consumer'])
        self.assertFalse((self.root/'executions/attempts/consumer').exists())
        path.write_bytes(old)
        # Change dependency after executor preparation, before launch. This exact
        # callback exercises the final synchronous dependency recheck.
        def drift(stage):
            if stage=='prepared':path.write_bytes(old+b' ')
        state=d.tick(self.root,drift)
        self.assertEqual(state['reserved'],1);self.assertIn('retained_integrity',state['observations']['consumer'])
        self.assertFalse((self.root/'executions/attempts/consumer/launch-intent.json').exists())
        path.write_bytes(old);self.wait_release('consumer')

    def test_caller_crashes_around_dependency_publications(self):
        self.admit(self.chain());self.wait_release('producer')
        for stage in ('binding_published','reserved','prepared','launched'):
            self.crash(stage)
        self.terminal('consumer')
        for stage in ('result_published','release_published','release_inserted','released'):
            self.crash(stage)
            if stage!='released':
                self.assertFalse(self.released('consumer'))
        self.assertEqual(self.wait_release('consumer')['outcome'],'accepted_execution')
        self.assertEqual(d.tick(self.root)['reserved'],0)

    def test_uncommitted_producer_publications_do_not_authorize_consumer(self):
        self.admit(self.chain());d.tick(self.root);self.terminal('producer')
        self.crash('release_published')
        self.assertFalse(self.released('producer'))
        self.assertFalse((self.root/'executions/attempts/consumer').exists())
        path=self.root/'evidence/producer/release.json';old=path.read_bytes();path.write_bytes(old+b' ')
        state=d.tick(self.root);self.assertEqual(state['reserved'],1)
        self.assertIn('retained_integrity',state['observations']['producer'])
        self.assertFalse(self.released('producer'))
        self.assertFalse((self.root/'executions/attempts/consumer').exists())
        path.write_bytes(old);self.wait_release('consumer')
        self.assertEqual((self.base/'output-consumer/data.txt').read_text(),'seed:consumed')

    def test_unknown_producer_retains_capacity_no_false_readiness(self):
        self.admit(self.chain(mode='hold'));d.tick(self.root)
        directory=self.root/'executions/attempts/producer';deadline=time.monotonic()+8
        while not (self.base/'output-producer/receipt.json').exists() and time.monotonic()<deadline:time.sleep(.02)
        self.assertTrue((self.base/'output-producer/receipt.json').exists())
        identity=d.q.document(directory/'supervisor.json')['identity'];fd=d.ex.open_pidfd(identity)
        signal.pidfd_send_signal(fd,signal.SIGKILL);os.close(fd)
        deadline=time.monotonic()+5
        while d.ex.observe(identity)['state']=='alive' and time.monotonic()<deadline:time.sleep(.02)
        d.ex.stop(self.root/'executions','producer')
        deadline=time.monotonic()+5
        while d.ex.inspect(self.root/'executions','producer')['job_lock_held'] and time.monotonic()<deadline:time.sleep(.02)
        d.ex.recover(self.root/'executions','producer')
        state=d.tick(self.root);self.assertEqual(state['reserved'],1)
        self.assertIn('unknown_exit',state['observations']['producer'])
        self.assertFalse(self.released('producer'));self.assertFalse((self.root/'executions/attempts/consumer').exists())

    def test_unknown_cross_plan_edge_rejected_and_old_domain_refused(self):
        plan=self.chain();plan['jobs']=plan['jobs'][:1]
        with self.assertRaises(ValueError):self.admit(plan)
        path=self.root/'dispatch.json';old=path.read_bytes();value=json.loads(old);value['schema']='IndependentDispatchResearchV1';path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):d.tick(self.root)
        path.write_bytes(old)


if __name__=='__main__':unittest.main(verbosity=2)
