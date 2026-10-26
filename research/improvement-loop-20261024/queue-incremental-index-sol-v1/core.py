"""Opt-in admission research core. No dispatch, capacity, migration or execution API.

Trust: SQLite bytes and cooperating writers; operational status is not an audit.
"""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import experiment_queue as queue

SCHEMA = 'IncrementalAdmissionResearchV1'


def connect(path):
    db = sqlite3.connect(path, timeout=30, isolation_level=None)
    db.execute('PRAGMA foreign_keys=ON')
    db.execute('PRAGMA synchronous=FULL')
    # Rollback journal deliberately keeps evidence and indexes in one atomic file.
    return db


def initialize(path):
    """Fresh database only; caller must durably provision its parent directory."""
    path = Path(path).absolute()
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    try:
        with connect(path) as db:
            db.executescript('''BEGIN IMMEDIATE;
            CREATE TABLE meta(schema TEXT NOT NULL);
            INSERT INTO meta VALUES ('IncrementalAdmissionResearchV1');
            CREATE TABLE evidence(id TEXT PRIMARY KEY, digest TEXT UNIQUE NOT NULL, body TEXT NOT NULL);
            CREATE TABLE owners(attempt TEXT PRIMARY KEY, output TEXT UNIQUE NOT NULL,
              admission TEXT NOT NULL REFERENCES evidence(id), job TEXT NOT NULL,
              UNIQUE(admission,job));
            CREATE TRIGGER evidence_update BEFORE UPDATE ON evidence BEGIN SELECT RAISE(ABORT,'immutable evidence'); END;
            CREATE TRIGGER evidence_delete BEFORE DELETE ON evidence BEGIN SELECT RAISE(ABORT,'immutable evidence'); END;
            CREATE TRIGGER owners_update BEFORE UPDATE ON owners BEGIN SELECT RAISE(ABORT,'immutable ownership'); END;
            CREATE TRIGGER owners_delete BEFORE DELETE ON owners BEGIN SELECT RAISE(ABORT,'immutable ownership'); END;
            COMMIT;''')
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except BaseException:
        # Leave a failed bootstrap visible; never silently replace or adopt it.
        raise


def opened(path):
    if not Path(path).is_file():
        raise ValueError('explicit initialize required')
    db = connect(path)
    if db.execute('SELECT schema FROM meta').fetchall() != [(SCHEMA,)]:
        db.close()
        raise ValueError('unsupported store')
    return db


def admission_rows(plan):
    rows = []
    for job in plan['jobs']:
        if job['resume_prelaunch'] is not None:
            raise ValueError('resumption unsupported; original queue proof is not adopted')
        spec = job['execution_spec']
        output = Path(spec['output_directory'])
        # A stable lexical identity avoids silently changing symlink aliases later.
        if str(output.resolve()) != str(output) or output.exists():
            raise ValueError('fresh canonical output path required')
        rows.append((spec['attempt_id'], str(output), plan['queue_id'], job['id']))
    outputs = sorted(row[1] for row in rows)
    for a, b in zip(outputs, outputs[1:]):
        if Path(b).is_relative_to(a):
            raise ValueError('nested output ownership')
    return rows


def admit(path, plan, expected, fault=lambda stage: None):
    """Atomically commit canonical evidence + indexed claims; idempotent exact retry.

    fault is a test seam, called within the transaction and after durable commit.
    """
    body = queue.canonical(plan)
    plan = json.loads(body)  # one immutable logical snapshot of caller-owned values
    digest = hashlib.sha256(body.encode()).hexdigest()
    if digest != expected:
        raise ValueError('current plan digest mismatch')
    db = opened(path)
    try:
        db.execute('BEGIN IMMEDIATE')
        # Recheck all current executable/input bytes under admission serialization,
        # even for an exact retry. No historical input scan is implied.
        queue.validate_plan(plan)
        rows = admission_rows(plan)
        old = db.execute('SELECT digest,body FROM evidence WHERE id=?', (plan['queue_id'],)).fetchone()
        if old:
            if old != (digest, body):
                raise ValueError('changed admission identity')
            actual = db.execute('SELECT attempt,output,admission,job FROM owners WHERE admission=? ORDER BY job', (plan['queue_id'],)).fetchall()
            if actual != sorted(rows, key=lambda r: r[3]):
                raise ValueError('current admission ownership mismatch')
            db.execute('COMMIT')
            return digest
        # Ancestor ownership queries use indexed exact parent lookups. Descendant
        # prefix query has a bounded range on canonical absolute path strings.
        for attempt, output, admission, job in rows:
            for parent in Path(output).parents:
                if db.execute('SELECT 1 FROM owners WHERE output=?', (str(parent),)).fetchone():
                    raise ValueError('ancestor output owned')
            if db.execute('SELECT 1 FROM owners WHERE output>=? AND output<? LIMIT 1', (output+'/', output+'0')).fetchone():
                raise ValueError('descendant output owned')
        db.execute('INSERT INTO evidence VALUES (?,?,?)', (plan['queue_id'], digest, body))
        fault('evidence')
        db.executemany('INSERT INTO owners VALUES (?,?,?,?)', rows)
        fault('owners')
        db.execute('COMMIT')
        fault('committed')
        return digest
    finally:
        db.close()  # uncommitted transitions roll back, including after client death


def operational_status(path):
    with opened(path) as db:
        return dict(schema=SCHEMA, trust='trusted_index_not_historical_audit',
                    admissions=db.execute('SELECT count(*) FROM evidence').fetchone()[0],
                    ownership_claims=db.execute('SELECT count(*) FROM owners').fetchone()[0],
                    dispatch='unsupported', qualification='unassessed')


def audit(path):
    """Full structural audit; does not revalidate historic mutable scientific inputs.

    External rollback/deletion of a consistent whole DB needs an independent anchor.
    """
    with opened(path) as db:
        db.execute('BEGIN')
        if db.execute('PRAGMA integrity_check').fetchall() != [('ok',)]:
            raise ValueError('database integrity')
        expected = []
        for identity, digest, body in db.execute('SELECT id,digest,body FROM evidence'):
            plan = json.loads(body)
            if queue.canonical(plan) != body or queue.digest(plan) != digest or plan['queue_id'] != identity:
                raise ValueError('evidence digest/identity')
            for job in plan['jobs']:
                s = job['execution_spec']
                expected.append((s['attempt_id'], s['output_directory'], identity, job['id']))
        actual = db.execute('SELECT attempt,output,admission,job FROM owners').fetchall()
        if sorted(expected) != sorted(actual):
            raise ValueError('ownership projection mismatch')
        return dict(structural_audit='passed', admissions=len(list(db.execute('SELECT id FROM evidence'))),
                    historical_inputs='not_checked', rollback_protection='absent')
