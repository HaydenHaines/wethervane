import copy
import json
import itertools
import multiprocessing as mp
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
import core
import test_experiment_queue as fixtures


def killed(path, plan, stage):
    core.admit(path, plan, core.queue.digest(plan), lambda at: os._exit(71) if at == stage else None)


def competing(path, plan, result):
    try:
        core.admit(path, plan, core.queue.digest(plan))
        result.put('accepted')
    except sqlite3.IntegrityError:
        result.put('duplicate')


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.db = self.base/'admissions.sqlite'
        core.initialize(self.db)
        self.f = fixtures.QueueTests('runTest')
        self.f.base = self.base
        self.f.source = self.base/'owner.py'
        self.f.source.write_text(fixtures.FIXTURE)

    def plan(self, name):
        return dict(schema='LocalExperimentQueuePlanV1', queue_id=name, jobs=[self.f.job(name)])

    def admit(self, p):
        return core.admit(self.db, p, core.queue.digest(p))

    def test_exact_retry_fresh_inputs_changed_identity_and_resumption(self):
        p = self.plan('a')
        self.assertEqual(self.admit(p), self.admit(p))
        self.f.source.write_text('drift')
        with self.assertRaises(ValueError): self.admit(p)
        self.f.source.write_text(fixtures.FIXTURE)
        p['jobs'][0]['execution_spec']['retry_reason'] = 'changed'
        with self.assertRaises(ValueError): self.admit(p)
        p = self.plan('b'); p['jobs'][0]['resume_prelaunch'] = {}
        with self.assertRaises(ValueError): self.admit(p)
        with self.assertRaises(ValueError): core.admit(self.db, self.plan('c'), 'wrong')

    def test_duplicates_nested_paths_and_symlink(self):
        p = self.plan('a'); self.admit(p)
        for field in ('attempt_id', 'output_directory'):
            b = self.plan('b'); b['jobs'][0]['execution_spec'][field] = p['jobs'][0]['execution_spec'][field]
            with self.assertRaises(sqlite3.IntegrityError): self.admit(b)
        b = self.plan('b'); b['jobs'][0]['execution_spec']['output_directory'] += '/../output-a/child'
        with self.assertRaises(ValueError): self.admit(b)
        b['jobs'][0]['execution_spec']['output_directory'] = str(self.base/'output-a/child')
        with self.assertRaises(ValueError): self.admit(b)
        link = self.base/'alias'; link.symlink_to(self.base, target_is_directory=True)
        b['jobs'][0]['execution_spec']['output_directory'] = str(link/'out')
        with self.assertRaises(ValueError): self.admit(b)

    def nested_plan(self, name, suffixes):
        jobs = [self.f.job(name+'-job'+str(i)) for i in range(len(suffixes))]
        for job, suffix in zip(jobs, suffixes):
            job['execution_spec']['output_directory'] = str(self.base/suffix)
        return dict(schema='LocalExperimentQueuePlanV1', queue_id=name, jobs=jobs)

    def test_nonadjacent_same_plan_nested_regression(self):
        # '-' sorts before '/', so the old adjacent-only sorted check missed this.
        for i, order in enumerate(itertools.permutations(['owned','owned-other','owned/child'])):
            with self.subTest(order=order):
                with self.assertRaises(ValueError): self.admit(self.nested_plan('batch'+str(i),order))
        for order in (['owned','owned/child'], ['owned/child','owned'], ['owned','owned']):
            with self.subTest(order=order):
                with self.assertRaises(ValueError): self.admit(self.nested_plan('exact-reversed',order))
        self.assertEqual(core.operational_status(self.db)['admissions'],0)
        # Similar prefixes without ancestry remain valid.
        self.admit(self.nested_plan('valid',['owned-other','owned']))
        self.assertEqual(core.audit(self.db)['admissions'],1)

    def test_cross_plan_nested_both_admission_orders(self):
        for i, order in enumerate((['owned','owned/child'],['owned/child','owned'])):
            with self.subTest(order=order):
                db = self.base/('cross'+str(i)+'.sqlite'); core.initialize(db)
                a=self.nested_plan('first'+str(i),[order[0],'owned-other'])
                b=self.nested_plan('second'+str(i),[order[1]])
                core.admit(db,a,core.queue.digest(a))
                with self.assertRaises(ValueError): core.admit(db,b,core.queue.digest(b))
                self.assertEqual(core.audit(db)['admissions'],1)

    def test_structural_audit_rejects_consistent_nested_projection(self):
        # Represent a legacy/hostile database with perfectly matching evidence and
        # index rows: projection equality alone must not certify this invariant.
        p=self.nested_plan('legacy',['owned','owned-other','owned/child'])
        with core.opened(self.db) as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('INSERT INTO evidence VALUES (?,?,?)',(p['queue_id'],core.queue.digest(p),core.queue.canonical(p)))
            db.executemany('INSERT INTO owners VALUES (?,?,?,?)',[(j['execution_spec']['attempt_id'],j['execution_spec']['output_directory'],p['queue_id'],j['id']) for j in p['jobs']])
            db.execute('COMMIT')
        self.assertEqual(core.operational_status(self.db)['ownership_claims'],3)
        with self.assertRaisesRegex(ValueError,'nested output ownership'): core.audit(self.db)

    def test_process_death_atomic_boundaries(self):
        for stage in ('evidence', 'owners', 'committed'):
            p = self.plan(stage)
            child = mp.Process(target=killed, args=(self.db,p,stage)); child.start(); child.join(10)
            self.assertFalse(child.is_alive()); self.assertEqual(child.exitcode,71)
            expected = 1 if stage == 'committed' else 0
            with core.opened(self.db) as db:
                self.assertEqual(db.execute('SELECT count(*) FROM evidence WHERE id=?',(stage,)).fetchone()[0],expected)
                self.assertEqual(db.execute('SELECT count(*) FROM owners WHERE admission=?',(stage,)).fetchone()[0],expected)
            self.admit(p)
            self.assertEqual(core.audit(self.db)['structural_audit'],'passed')

    def test_v1_store_refused_without_migration(self):
        # V1 and V2 share table shape; the explicit version gate must still deny
        # adopting historical claims certified by the defective V1 invariant.
        with core.opened(self.db) as db:
            db.execute("UPDATE meta SET schema='IncrementalAdmissionResearchV1'")
        with self.assertRaisesRegex(ValueError,'unsupported store'): core.opened(self.db)
        with self.assertRaisesRegex(ValueError,'unsupported store'): self.admit(self.plan('new'))
        with self.assertRaisesRegex(ValueError,'unsupported store'): core.audit(self.db)
        with self.assertRaises(FileExistsError): core.initialize(self.db)

    def test_competing_claims(self):
        a = self.plan('a'); b = self.plan('b')
        b['jobs'][0]['execution_spec']['attempt_id'] = 'a'
        results = mp.Queue()
        children = [mp.Process(target=competing,args=(self.db,p,results)) for p in (a,b)]
        for c in children: c.start()
        for c in children: c.join(10); self.assertEqual(c.exitcode,0)
        self.assertEqual(sorted([results.get(timeout=2),results.get(timeout=2)]),['accepted','duplicate'])
        self.assertEqual(core.operational_status(self.db)['ownership_claims'],1)

    def test_mutable_index_trust_and_explicit_audit(self):
        self.admit(self.plan('a'))
        with core.opened(self.db) as db:
            with self.assertRaises(sqlite3.IntegrityError): db.execute("DELETE FROM owners")
            db.execute('DROP TRIGGER owners_delete'); db.execute('DELETE FROM owners')
        self.assertEqual(core.operational_status(self.db)['ownership_claims'],0)
        with self.assertRaises(ValueError): core.audit(self.db)

    def test_thousand_serial_synthetic_admissions(self):
        start = time.monotonic()
        for i in range(1000): self.admit(self.plan('p'+str(i)))
        self.assertEqual(core.operational_status(self.db)['admissions'],1000)
        self.assertEqual(core.audit(self.db)['admissions'],1000)
        print(json.dumps(dict(synthetic_admissions=1000, seconds=time.monotonic()-start, executions=0)))


if __name__ == '__main__': unittest.main(verbosity=2)
