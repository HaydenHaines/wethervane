"""Research independent-job dispatcher. Trusted mutable SQLite; no old receipts/DAGs."""
from contextlib import contextmanager
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
sys.path.insert(0, str(LAB))
import execution as ex
import experiment_queue as q
loader = importlib.util.spec_from_file_location('admission_v2', LAB/'queue-incremental-index-sol-v2/core.py')
admission = importlib.util.module_from_spec(loader)
loader.loader.exec_module(admission)


def initialize(root):
    root = Path(root).resolve()
    root.mkdir()  # fresh dedicated domain only; capacity fixed at one
    admission.initialize(root/'admissions.sqlite')
    with admission.opened(root/'admissions.sqlite') as db:
        db.executescript('''BEGIN IMMEDIATE;
        CREATE TABLE reservations(attempt TEXT PRIMARY KEY REFERENCES owners(attempt), body TEXT NOT NULL, digest TEXT NOT NULL);
        CREATE TABLE releases(attempt TEXT PRIMARY KEY REFERENCES reservations(attempt), body TEXT NOT NULL, digest TEXT NOT NULL);
        CREATE TRIGGER reservation_update BEFORE UPDATE ON reservations BEGIN SELECT RAISE(ABORT,'immutable'); END;
        CREATE TRIGGER reservation_delete BEFORE DELETE ON reservations BEGIN SELECT RAISE(ABORT,'immutable'); END;
        CREATE TRIGGER release_update BEFORE UPDATE ON releases BEGIN SELECT RAISE(ABORT,'immutable'); END;
        CREATE TRIGGER release_delete BEFORE DELETE ON releases BEGIN SELECT RAISE(ABORT,'immutable'); END;
        COMMIT;''')
    ex.publish(root/'dispatch.json',dict(schema='IndependentDispatchResearchV1',root=str(root),capacity=1,
        sources=[ex.pin(p) for p in [Path(__file__),LAB/'queue-incremental-index-sol-v2/core.py',LAB/'experiment_queue.py',LAB/'execution.py',LAB/'ledger.py',LAB/'run_receipts.py']]))
    return root


@contextmanager
def locked(root):
    root=Path(root).resolve()
    config=q.document(root/'dispatch.json')
    ex.require(config['schema']=='IndependentDispatchResearchV1' and config['root']==str(root) and config['capacity']==1,'unsupported dispatch domain')
    for pin in config['sources']: ex.verify_pin(pin)
    fd=os.open(root/'dispatcher.lock',os.O_CREAT|os.O_RDWR,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX)
        db=admission.opened(root/'admissions.sqlite')
        try: yield root, db
        finally: db.close()
    finally: os.close(fd)


def enqueue(root,plan,expected):
    ex.require(all(not j['dependencies'] and j['resume_prelaunch'] is None for j in plan['jobs']),
               'DAG dependency binding and prelaunch resumption unsupported')
    with locked(root): return admission.admit(Path(root)/'admissions.sqlite',plan,expected)


def selected(db,attempt):
    row=db.execute('SELECT e.id,e.digest,e.body,o.output,o.job FROM owners o JOIN evidence e ON e.id=o.admission WHERE o.attempt=?',(attempt,)).fetchone()
    ex.require(row is not None,'missing admitted owner')
    identity,digest,body,output,job_id=row
    plan=json.loads(body)
    ex.require(q.canonical(plan)==body and q.digest(plan)==digest and plan['queue_id']==identity,'active admission identity/digest')
    jobs=q.validate_plan(plan)  # all pins in selected current plan are fresh
    job=jobs[job_id]
    ex.require(not job['dependencies'] and job['resume_prelaunch'] is None,'unsupported active graph/resumption')
    spec=job['execution_spec']
    if spec['previous_attempt'] is not None:
        ex.verify_pin(spec['previous_attempt']['receipt'])
    ex.require(spec['attempt_id']==attempt and spec['output_directory']==output and str(Path(output).resolve())==output,'active ownership identity')
    return job, digest


def record(db,table,attempt,value):
    db.execute('INSERT INTO '+table+' VALUES (?,?,?)',(attempt,q.canonical(value),q.digest(value)))


def reconcile(root,db,attempt,body,digest,fault):
    reservation=json.loads(body)
    ex.require(q.canonical(reservation)==body and q.digest(reservation)==digest,'reservation digest')
    job,plan_digest=selected(db,attempt)
    spec=job['execution_spec']
    expected=dict(schema='IndependentDispatchReservationV1',attempt=attempt,admission_digest=plan_digest,job=job)
    ex.require(reservation==expected,'reservation identity/spec changed')
    execution_root=root/'executions'; directory=ex.attempt(execution_root,attempt)
    if not directory.exists():
        # Only an absent executor attempt may be prepared. Partial scaffolds fail
        # closed in load/registered; never delete or synthesize cleanup evidence.
        ex.prepare(execution_root,spec,q.digest(spec))
        fault('prepared')
    _,actual,_=ex.load(execution_root,attempt,verify_inputs=True)
    ex.require(actual==spec,'executor spec differs from reserved exact spec')
    ex.registered(execution_root,directory,spec)
    state=ex.inspect(execution_root,attempt)
    if not (directory/'launch-intent.json').exists():
        ex.require(state['state']=='prepared' and not state['job_lock_held'] and not state['session_members'] and
                   all(o['state']!='alive' for o in state['process_observation'].values()),'unlaunched attempt not unused')
        ex.launch(execution_root,attempt)
        fault('launched')
        return 'retained_launch_or_live'
    terminal=state['attempt_terminal']
    if (not terminal or state['job_lock_held'] or state['session_members'] or
        any(o['state']=='alive' for o in state['process_observation'].values())):
        return 'retained_live_or_unresolved'
    ex.require(terminal['cleanup']['verified'],'unverified cleanup')
    if terminal['return_code'] is None: return 'retained_unknown_exit'
    outcome=('failed_timeout' if terminal['timed_out'] else 'failed_process' if terminal['return_code']!=0 else
             'failed_native_outputs' if terminal['native_outputs']!='metadata_verified_science_unassessed' else 'accepted_execution')
    outputs={}
    if outcome=='accepted_execution':
        for declaration in job['outputs']:
            path=Path(spec['output_directory'])/declaration['path']
            ex.require(path.resolve().is_relative_to(Path(spec['output_directory'])),'escaped output')
            pin=ex.pin(path)
            ex.require(pin in terminal['native_receipts'] and pin['size']>=declaration['minimum_bytes'],'output gate drift/failure')
            outputs[declaration['role']]=pin
    # Result and release are one record/transaction. No result-before-release
    # window exists; a crash before COMMIT leaves capacity reserved.
    result=dict(schema='IndependentDispatchReleaseV1',attempt=attempt,reservation_sha256=digest,
                executor_terminal=state['terminal_receipt'],outcome=outcome,outputs=outputs,qualification='unassessed')
    db.execute('BEGIN IMMEDIATE')
    record(db,'releases',attempt,result)
    fault('release_inserted')
    db.execute('COMMIT')
    fault('released')
    return outcome


def tick(root,fault=lambda stage:None):
    with locked(root) as (root,db):
        active=db.execute('SELECT r.attempt,r.body,r.digest FROM reservations r LEFT JOIN releases l ON l.attempt=r.attempt WHERE l.attempt IS NULL').fetchall()
        ex.require(len(active)<=1,'capacity history invalid')
        if not active:
            row=db.execute('SELECT o.attempt FROM owners o LEFT JOIN reservations r ON r.attempt=o.attempt WHERE r.attempt IS NULL ORDER BY o.rowid LIMIT 1').fetchone()
            if row:
                attempt=row[0]
                try:
                    job,digest=selected(db,attempt)
                    # Existing external executor attempts cannot be adopted.
                    ex.require(not ex.attempt(root/'executions',attempt).exists(),'external attempt exists before reservation')
                    value=dict(schema='IndependentDispatchReservationV1',attempt=attempt,admission_digest=digest,job=job)
                    db.execute('BEGIN IMMEDIATE');record(db,'reservations',attempt,value);db.execute('COMMIT')
                    fault('reserved')
                    active=[(attempt,q.canonical(value),q.digest(value))]
                except (OSError,ValueError,KeyError,TypeError) as error:
                    return dict(state='blocked_pre_reservation',error=str(error),reserved=0)
        observations={}
        for attempt,body,digest in active:
            try: observations[attempt]=reconcile(root,db,attempt,body,digest,fault)
            except (OSError,ValueError,KeyError,TypeError) as error:
                observations[attempt]='retained_integrity: '+str(error)
        reserved=db.execute('SELECT count(*) FROM reservations r LEFT JOIN releases l ON l.attempt=r.attempt WHERE l.attempt IS NULL').fetchone()[0]
        return dict(schema='IndependentDispatchOperationalObservationV1',trust='trusted_mutable_index_no_historical_audit',reserved=reserved,observations=observations,qualification='unassessed')
