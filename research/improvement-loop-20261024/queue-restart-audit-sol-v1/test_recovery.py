import copy
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
import probe

spec=importlib.util.spec_from_file_location('patched_queue',probe.HERE/'patched/experiment_queue.py')
patched=importlib.util.module_from_spec(spec);spec.loader.exec_module(patched)


def tick_then_die(root):
    patched.tick(root);os._exit(83)


class RecoveryTests(unittest.TestCase):
    def test_original_failure_preserved_and_fixed_same_actual_boundary(self):
        original=probe.partial_probe(probe.original_q,probe.LAB)
        fixed=probe.partial_probe(patched,probe.HERE/'patched')
        self.assertTrue(original['contender_admitted']);self.assertIsNotNone(original['original_exact_repair_error'])
        self.assertFalse(original['all_jobs_recorded'])
        self.assertFalse(fixed['contender_admitted']);self.assertIsNotNone(fixed['contender_error'])
        self.assertIsNone(fixed['original_exact_repair_error']);self.assertEqual(fixed['partial_admissions'],[])
        self.assertTrue(fixed['all_jobs_recorded']);self.assertEqual(fixed['actual_launches'],1)
        print(json.dumps(dict(original=original,fixed=fixed)))

    def test_twelve_serial_adaptive_admissions_with_competing_killed_tick_clients(self):
        with tempfile.TemporaryDirectory() as temporary:
            f=probe.domain(Path(temporary),patched)
            try:
                receipts=[]
                for i in range(12):
                    # The next plan is constructed only after its predecessor
                    # result has been authenticated; this is infrastructure-only.
                    if i:
                        previous=f.root/f'queues/cycle{i-1}/jobs/job{i-1}/result.json'
                        self.assertEqual(patched.document(previous)['outcome'],'accepted_execution')
                        receipts.append(patched.executor.pin(previous))
                    job=f.job('job'+str(i))
                    if i:job['execution_spec']['input_pins'].append(receipts[-1])
                    plan=dict(schema='LocalExperimentQueuePlanV1',queue_id='cycle'+str(i),jobs=[job])
                    admission=patched.enqueue(f.root,plan,patched.digest(plan))
                    self.assertEqual(admission,patched.enqueue(f.root,plan,patched.digest(plan)))
                    clients=[mp.Process(target=tick_then_die,args=(f.root,)) for _ in range(2)]
                    for client in clients:client.start()
                    for client in clients:client.join(8);self.assertEqual(client.exitcode,83)
                    state=probe.complete(patched,f.root)
                    self.assertTrue(state['all_jobs_recorded']);self.assertEqual(state['reserved'],0)
                self.assertEqual(len(list((f.root/'executions/attempts').glob('*/launch-intent.json'))),12)
                self.assertEqual(len(list(f.root.glob('queues/*/jobs/*/release.json'))),12)
                # The original immediate historical-byte gate is preserved.
                oldest=f.root/'queues/cycle0/plan.json';old=oldest.read_bytes();oldest.write_bytes(old+b' ')
                self.assertTrue(patched.tick(f.root)['quarantined_admissions'])
                oldest.write_bytes(old);self.assertTrue(patched.status(f.root)['all_jobs_recorded'])
                print(json.dumps(dict(serial_cycles=12,killed_competing_clients=24,actual_launches=12,reservations=0,historical_plan_drift_quarantined=True)))
            finally:f.cleanup()

    def test_partial_admission_preserves_claims_but_allows_unrelated_admission(self):
        with tempfile.TemporaryDirectory() as temporary:
            f=probe.domain(Path(temporary),patched)
            try:
                plan=dict(schema='LocalExperimentQueuePlanV1',queue_id='partial',jobs=[f.job('first')])
                child=mp.Process(target=probe.killed_admission,args=(str(probe.HERE/'patched'),str(f.root),plan));child.start();child.join(8);self.assertEqual(child.exitcode,82)
                other=dict(schema='LocalExperimentQueuePlanV1',queue_id='other',jobs=[f.job('second')])
                for field in ('attempt_id','output_directory'):
                    collision=copy.deepcopy(other);collision['queue_id']='collision-'+field
                    collision['jobs'][0]['execution_spec'][field]=plan['jobs'][0]['execution_spec'][field]
                    with self.assertRaisesRegex(ValueError,'belongs to a partial admission'):
                        patched.enqueue(f.root,collision,patched.digest(collision))
                patched.enqueue(f.root,other,patched.digest(other))
                patched.enqueue(f.root,plan,patched.digest(plan))
                self.assertTrue(probe.complete(patched,f.root)['all_jobs_recorded'])
            finally:f.cleanup()

    def test_missing_or_corrupt_partial_scaffold_rejects_new_admission(self):
        with tempfile.TemporaryDirectory() as temporary:
            f=probe.domain(Path(temporary),patched)
            try:
                pending=f.root/'queues/incomplete';pending.mkdir()
                plan=dict(schema='LocalExperimentQueuePlanV1',queue_id='fresh',jobs=[f.job('fresh')])
                with self.assertRaisesRegex(ValueError,'unresolved admission scaffold'):
                    patched.enqueue(f.root,plan,patched.digest(plan))
                (pending/'plan.json').write_text('{corrupt')
                with self.assertRaises(ValueError):patched.enqueue(f.root,plan,patched.digest(plan))
                self.assertFalse((f.root/'queues/fresh').exists())
                self.assertFalse(patched.status(f.root)['all_jobs_recorded'])
            finally:f.cleanup()

    def test_receipt_before_job_directory_death_exact_retry_and_tick_repair(self):
        with tempfile.TemporaryDirectory() as temporary:
            f=probe.domain(Path(temporary),patched)
            try:
                plan=dict(schema='LocalExperimentQueuePlanV1',queue_id='receipt-first',jobs=[f.job('fresh')])
                child=mp.Process(target=probe.killed_admission,args=(str(probe.HERE/'patched'),str(f.root),plan,'receipt.json'))
                child.start();child.join(8);self.assertEqual(child.exitcode,82)
                receipt=patched.executor.pin(f.root/'queues/receipt-first/receipt.json')
                self.assertFalse((f.root/'queues/receipt-first/jobs/fresh').exists())
                self.assertEqual(patched.enqueue(f.root,plan,patched.digest(plan)),receipt)
                self.assertTrue(probe.complete(patched,f.root)['all_jobs_recorded'])
                self.assertEqual(len(list((f.root/'executions/attempts').glob('*/launch-intent.json'))),1)
            finally:f.cleanup()


if __name__=='__main__':unittest.main(verbosity=2)
