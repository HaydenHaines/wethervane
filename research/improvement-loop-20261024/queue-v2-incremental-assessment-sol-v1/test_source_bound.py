import copy
import json
import multiprocessing as mp
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import dispatch as d
import test_dependency as inherited


def admission_death(root,plan,stage):d.enqueue(root,plan,d.q.digest(plan),lambda at:os._exit(84) if at==stage else None)


def competing_admission(root,plan,messages):
    try:d.enqueue(root,plan,d.q.digest(plan));messages.put('accepted')
    except sqlite3.IntegrityError:messages.put('duplicate')


class SourceBoundTests(inherited.DependencyTests):
    # Select only these new methods explicitly; ancestor methods are exercised
    # separately in the bounded suite below.
    def test_atomic_admission_crash_recovery_has_no_unsealed_claim_loss(self):
        for stage in ('evidence','owners','committed'):
            with self.subTest(stage=stage):
                plan=dict(schema='LocalExperimentQueuePlanV1',queue_id=stage,jobs=[self.f.job(stage)])
                child=mp.Process(target=admission_death,args=(self.root,plan,stage));child.start();child.join(8);self.assertEqual(child.exitcode,84)
                with d.admission.opened(self.root/'admissions.sqlite') as db:
                    expected=1 if stage=='committed' else 0
                    self.assertEqual(db.execute('SELECT count(*) FROM evidence WHERE id=?',(stage,)).fetchone()[0],expected)
                    self.assertEqual(db.execute('SELECT count(*) FROM owners WHERE admission=?',(stage,)).fetchone()[0],expected)
                d.enqueue(self.root,plan,d.q.digest(plan))
                self.assertEqual(d.enqueue(self.root,plan,d.q.digest(plan)),d.q.digest(plan))
                collision=copy.deepcopy(plan);collision['queue_id']='collision-'+stage
                with self.assertRaises(sqlite3.IntegrityError):d.enqueue(self.root,collision,d.q.digest(collision))
        with d.admission.opened(self.root/'admissions.sqlite') as db:
            self.assertEqual(db.execute('SELECT count(*) FROM evidence').fetchone()[0],3)
            self.assertEqual(db.execute('SELECT count(*) FROM owners').fetchone()[0],3)

    def test_competing_same_identity_admissions_and_real_dispatch_once(self):
        plan=dict(schema='LocalExperimentQueuePlanV1',queue_id='one',jobs=[self.f.job('same')]);other=copy.deepcopy(plan);other['queue_id']='two'
        messages=mp.Queue();clients=[mp.Process(target=competing_admission,args=(self.root,p,messages)) for p in (plan,other)]
        for child in clients:child.start()
        for child in clients:child.join(8);self.assertEqual(child.exitcode,0)
        self.assertEqual(sorted([messages.get(timeout=2),messages.get(timeout=2)]),['accepted','duplicate'])
        self.assertEqual(self.wait_release('same')['outcome'],'accepted_execution')
        self.assertEqual(len(list((self.root/'executions/attempts').glob('*/launch-intent.json'))),1)
        self.assertEqual(d.tick(self.root)['reserved'],0)

    def test_changed_source_roster_refused_without_lock_or_dispatch_mutation(self):
        path=self.root/'dispatch.json';config=json.loads(path.read_text())
        config['sources'][0]=d.ex.pin(d.LAB/'queue-dependency-research-sol-v1/dispatch.py');path.write_text(json.dumps(config))
        before={str(p.relative_to(self.root)):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        with self.assertRaisesRegex(ValueError,'source identity differs'):d.tick(self.root)
        after={str(p.relative_to(self.root)):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before,after);self.assertFalse((self.root/'dispatcher.lock').exists())

    def test_mixed_actual_imports_and_actual_v2_queue_domain_refused(self):
        code='''import importlib,sys;from pathlib import Path
old,new,name,root=sys.argv[1:];sys.path.insert(0,old);importlib.import_module(name);sys.path.insert(0,new);import dispatch;dispatch.initialize(Path(root))
'''
        for name in ('experiment_queue','execution','ledger','run_receipts'):
            root=self.base/('mixed-'+name)
            r=subprocess.run([sys.executable,'-B','-c',code,str(d.LAB),str(d.HERE),name,str(root)],capture_output=True,text=True,timeout=10)
            self.assertNotEqual(r.returncode,0);self.assertFalse(root.exists())
        root=self.base/'actual-v2';d.q.initialize(root,1)
        with self.assertRaises(FileNotFoundError):d.tick(root)
        self.assertFalse((root/'dispatcher.lock').exists())


if __name__=='__main__':
    own=['test_atomic_admission_crash_recovery_has_no_unsealed_claim_loss','test_competing_same_identity_admissions_and_real_dispatch_once','test_changed_source_roster_refused_without_lock_or_dispatch_mutation','test_mixed_actual_imports_and_actual_v2_queue_domain_refused']
    original=['test_active_binding_and_launch_dependency_recheck','test_caller_crashes_around_dependency_publications','test_producer_nonzero_never_authorizes_consumer','test_unknown_producer_retains_capacity_no_false_readiness','test_selected_producer_receipt_result_release_terminal_tamper']
    suite=unittest.TestSuite([*(SourceBoundTests(n) for n in own),*(inherited.DependencyTests(n) for n in original)])
    result=unittest.TextTestRunner(verbosity=2).run(suite);raise SystemExit(not result.wasSuccessful())
