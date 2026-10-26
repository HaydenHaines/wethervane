import copy
import json
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
