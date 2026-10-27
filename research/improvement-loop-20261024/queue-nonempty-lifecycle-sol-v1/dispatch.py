"""Research within-plan DAG dispatcher; trusted mutable SQLite, no old receipts."""
from contextlib import contextmanager
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
sys.path.insert(0, str(LAB / "queue-v2"))
import execution as ex
import experiment_queue as q
loader = importlib.util.spec_from_file_location('admission_v2', LAB/'queue-incremental-index-sol-v2/core.py')
admission = importlib.util.module_from_spec(loader)
loader.loader.exec_module(admission)


def current_sources():
    expected = LAB / 'queue-v2'
    ex.require(Path(q.__file__).resolve() == expected / 'experiment_queue.py', 'wrong queue validator source')
    ex.require(admission.queue is q, 'admission validator import differs')
    ex.require(Path(admission.__file__).resolve() == LAB / 'queue-incremental-index-sol-v2/core.py', 'wrong admission source')
    return [ex.pin(Path(__file__)), ex.pin(Path(admission.__file__))] + q.current_sources()


def initialize(root):
    sources = current_sources()
    root = Path(root).resolve()
    root.mkdir()  # fresh dedicated domain only; capacity fixed at one
    admission.initialize(root/'admissions.sqlite')
    with admission.opened(root/'admissions.sqlite') as db:
        db.executescript('''BEGIN IMMEDIATE;
        UPDATE meta SET schema='NonemptyCurrentAdmissionsLifecycleResearchV1';
        CREATE TABLE pending(seq INTEGER PRIMARY KEY AUTOINCREMENT, attempt TEXT UNIQUE NOT NULL REFERENCES owners(attempt), remaining INTEGER NOT NULL CHECK(remaining>=0), blocked TEXT CHECK(blocked IS NULL));
        CREATE INDEX ready_frontier ON pending(seq,attempt) WHERE remaining=0 AND blocked IS NULL;
        CREATE TABLE dependency_edges(producer TEXT NOT NULL REFERENCES owners(attempt), consumer TEXT NOT NULL REFERENCES owners(attempt), PRIMARY KEY(producer,consumer));
        CREATE INDEX consumers_by_attempt ON dependency_edges(consumer,producer);
        CREATE TABLE active(slot INTEGER PRIMARY KEY CHECK(slot=1), attempt TEXT UNIQUE NOT NULL REFERENCES reservations(attempt));
        CREATE TABLE blocked_terminal(attempt TEXT PRIMARY KEY REFERENCES owners(attempt),cause TEXT NOT NULL REFERENCES releases(attempt),body TEXT NOT NULL,digest TEXT NOT NULL);
        CREATE TRIGGER blocked_update BEFORE UPDATE ON blocked_terminal BEGIN SELECT RAISE(ABORT,'immutable blocked terminal'); END;
        CREATE TRIGGER blocked_delete BEFORE DELETE ON blocked_terminal BEGIN SELECT RAISE(ABORT,'immutable blocked terminal'); END;
        CREATE TABLE counts(slot INTEGER PRIMARY KEY CHECK(slot=1),total INTEGER NOT NULL,pending INTEGER NOT NULL,active INTEGER NOT NULL,released INTEGER NOT NULL,blocked INTEGER NOT NULL,accepted INTEGER NOT NULL,failed INTEGER NOT NULL,CHECK(total>=0 AND pending>=0 AND active BETWEEN 0 AND 1 AND released>=0 AND blocked>=0 AND accepted>=0 AND failed>=0 AND total=pending+active+released+blocked AND released=accepted+failed));
        INSERT INTO counts VALUES(1,0,0,0,0,0,0,0);
        CREATE TABLE reservations(attempt TEXT PRIMARY KEY REFERENCES owners(attempt), body TEXT NOT NULL, digest TEXT NOT NULL);
        CREATE TABLE releases(attempt TEXT PRIMARY KEY REFERENCES reservations(attempt), body TEXT NOT NULL, digest TEXT NOT NULL);
        CREATE TRIGGER reservation_update BEFORE UPDATE ON reservations BEGIN SELECT RAISE(ABORT,'immutable'); END;
        CREATE TRIGGER reservation_delete BEFORE DELETE ON reservations BEGIN SELECT RAISE(ABORT,'immutable'); END;
        CREATE TRIGGER release_update BEFORE UPDATE ON releases BEGIN SELECT RAISE(ABORT,'immutable'); END;
        CREATE TRIGGER release_delete BEFORE DELETE ON releases BEGIN SELECT RAISE(ABORT,'immutable'); END;
        COMMIT;''')
    ex.publish(root/'dispatch.json',dict(schema='SourceBoundNonemptyLifecycleResearchV1',root=str(root),capacity=1,
        sources=sources))
    return root


def opened(path):
    ex.require(Path(path).is_file(),'explicit frontier initialize required')
    db=admission.connect(path)
    if db.execute('SELECT schema FROM meta').fetchall()!=[('NonemptyCurrentAdmissionsLifecycleResearchV1',)]:
        db.close();raise ValueError('unsupported frontier database')
    return db


@contextmanager
def locked(root):
    sources=current_sources()
    root=Path(root).resolve()
    config=q.document(root/'dispatch.json')
    ex.require(config['schema']=='SourceBoundNonemptyLifecycleResearchV1' and config['root']==str(root) and config['capacity']==1,'unsupported dispatch domain')
    ex.require(config['sources']==sources, 'store source identity differs from running dispatcher')
    for pin in config['sources']: ex.verify_pin(pin)
    fd=os.open(root/'dispatcher.lock',os.O_CREAT|os.O_RDWR,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX)
        db=opened(root/'admissions.sqlite')
        try: yield root, db
        finally: db.close()
    finally: os.close(fd)


class NotReady(ValueError):
    """Known absence/failure of a predecessor, not authorization to launch."""


def enqueue(root,plan,expected,fault=lambda stage:None):
    # New admission contract: evidence, immutable claims, edges and pending are one commit.
    body=q.canonical(plan);plan=json.loads(body);digest=q.digest(plan)
    ex.require(digest==expected,'current plan digest mismatch')
    with locked(root) as (root,db):
        try:
            db.execute('BEGIN IMMEDIATE');q.validate_plan(plan);rows=admission.admission_rows(plan)
            old=db.execute('SELECT digest,body FROM evidence WHERE id=?',(plan['queue_id'],)).fetchone()
            if old:
                ex.require(old==(digest,body),'changed admission identity')
                actual=db.execute('SELECT attempt,output,admission,job FROM owners WHERE admission=? ORDER BY job',(plan['queue_id'],)).fetchall()
                ex.require(actual==sorted(rows,key=lambda r:r[3]),'admission ownership changed')
                db.execute('COMMIT');return digest
            for attempt,output,identity,job in rows:
                for parent in Path(output).parents:
                    ex.require(not db.execute('SELECT 1 FROM owners WHERE output=?',(str(parent),)).fetchone(),'ancestor output owned')
                ex.require(not db.execute('SELECT 1 FROM owners WHERE output>=? AND output<? LIMIT 1',(output+'/',output+'0')).fetchone(),'descendant output owned')
            db.execute('INSERT INTO evidence VALUES(?,?,?)',(plan['queue_id'],digest,body));fault('evidence')
            db.executemany('INSERT INTO owners VALUES(?,?,?,?)',rows);fault('owners')
            jobs={job['id']:job['execution_spec']['attempt_id']for job in plan['jobs']}
            for job in plan['jobs']:
                attempt=jobs[job['id']];producers={jobs[edge['job_id']]for edge in job['dependencies']}
                db.execute('INSERT INTO pending(attempt,remaining) VALUES(?,?)',(attempt,len(producers)))
                db.executemany('INSERT INTO dependency_edges VALUES(?,?)',[(producer,attempt)for producer in sorted(producers)])
            db.execute('UPDATE counts SET total=total+?,pending=pending+? WHERE slot=1',(len(rows),len(rows)));fault('frontier_inserted');db.execute('COMMIT');fault('committed');return digest
        except BaseException:
            if db.in_transaction:db.execute('ROLLBACK')
            raise


def finish_frontier(db,attempt,outcome,fault):
    ex.require(db.execute('SELECT attempt FROM active WHERE slot=1').fetchone()==(attempt,),'release does not own capacity slot')
    accepted=outcome=='accepted_execution';new_blocked=0
    if accepted:
        db.execute('UPDATE pending SET remaining=remaining-1 WHERE attempt IN (SELECT consumer FROM dependency_edges WHERE producer=?)',(attempt,))
    else:
        affected=[row[0]for row in db.execute("WITH RECURSIVE descendants(attempt) AS (SELECT consumer FROM dependency_edges WHERE producer=? UNION SELECT e.consumer FROM dependency_edges e JOIN descendants d ON e.producer=d.attempt) SELECT attempt FROM descendants ORDER BY attempt",(attempt,))]
        cause_digest=row_value(db,'releases',attempt)[1]
        for consumer in affected:
            if db.execute('SELECT 1 FROM blocked_terminal WHERE attempt=?',(consumer,)).fetchone():continue
            ex.require(db.execute('SELECT 1 FROM pending WHERE attempt=?',(consumer,)).fetchone()is not None,'failed descendant already reserved/executed or missing')
            value=dict(schema='FailedDependencyBlockedTerminalV1',attempt=consumer,cause_attempt=attempt,cause_release_sha256=cause_digest,reason='transitive_failed_dependency',qualification='not_executed_science_unassessed')
            db.execute('INSERT INTO blocked_terminal VALUES(?,?,?,?)',(consumer,attempt,q.canonical(value),q.digest(value)));db.execute('DELETE FROM pending WHERE attempt=?',(consumer,));new_blocked+=1;fault('descendant_blocked')
    db.execute('DELETE FROM active WHERE slot=1 AND attempt=?',(attempt,))
    db.execute('UPDATE counts SET active=active-1,released=released+1,pending=pending-?,blocked=blocked+?,accepted=accepted+?,failed=failed+? WHERE slot=1',(new_blocked,new_blocked,int(accepted),int(not accepted)));fault('lifecycle_updated')


def reserve_frontier(db,attempt,value,fault):
    db.execute('BEGIN IMMEDIATE')
    try:
        ex.require(db.execute('SELECT attempt FROM active').fetchone()is None,'capacity occupied')
        row=db.execute('SELECT remaining,blocked FROM pending WHERE attempt=?',(attempt,)).fetchone();ex.require(row==(0,None),'selected job not ready')
        record(db,'reservations',attempt,value);db.execute('INSERT INTO active VALUES(1,?)',(attempt,));db.execute('DELETE FROM pending WHERE attempt=?',(attempt,));db.execute('UPDATE counts SET pending=pending-1,active=active+1 WHERE slot=1');fault('frontier_reserved');db.execute('COMMIT');fault('reserved')
    except BaseException:
        if db.in_transaction:db.execute('ROLLBACK')
        raise


def selected(db,attempt):
    row=db.execute('SELECT e.id,e.digest,e.body,o.output,o.job FROM owners o JOIN evidence e ON e.id=o.admission WHERE o.attempt=?',(attempt,)).fetchone()
    ex.require(row is not None,'missing admitted owner')
    identity,digest,body,output,job_id=row;plan=json.loads(body)
    ex.require(q.canonical(plan)==body and q.digest(plan)==digest and plan['queue_id']==identity,'active admission identity/digest')
    job=q.validate_plan(plan)[job_id]
    ex.require(job['resume_prelaunch'] is None,'unsupported resumption')
    spec=job['execution_spec']
    if spec['previous_attempt'] is not None:ex.verify_pin(spec['previous_attempt']['receipt'])
    ex.require(spec['attempt_id']==attempt and spec['output_directory']==output and str(Path(output).resolve())==output,'active ownership identity')
    return job,digest,identity


def row_value(db,table,attempt):
    row=db.execute('SELECT body,digest FROM '+table+' WHERE attempt=?',(attempt,)).fetchone()
    if row is None:return None
    body,digest=row;value=json.loads(body)
    ex.require(q.canonical(value)==body and q.digest(value)==digest,table+' digest')
    return value,digest


def record(db,table,attempt,value):
    db.execute('INSERT INTO '+table+' VALUES (?,?,?)',(attempt,q.canonical(value),q.digest(value)))


def files(root,attempt):
    directory=root/'evidence'/attempt
    ex.ensure_directory(directory)
    ex.require({p.name for p in directory.iterdir()} <= {'binding.json','result.json','release.json'},'unknown evidence scaffold')
    return directory


def publish_exact(path,value):
    if path.exists():ex.require(path.read_bytes()==q.canonical(value).encode(),'immutable publication changed: '+path.name)
    else:ex.publish(path,value)
    return ex.pin(path)


def binding(root,db,attempt):
    import copy
    job,plan_digest,identity=selected(db,attempt)
    spec=copy.deepcopy(job['execution_spec']);pins={p['path']:p for p in spec['input_pins']}
    ex.require(len(pins)==len(spec['input_pins']),'duplicate base input paths')
    dependencies=[]
    for edge in job['dependencies']:
        row=db.execute('SELECT attempt FROM owners WHERE admission=? AND job=?',(identity,edge['job_id'])).fetchone()
        ex.require(row is not None,'missing within-plan producer owner')
        producer=row[0]
        verified=verified_release(root,db,producer)
        if verified['result']['outcome']!='accepted_execution':raise NotReady('producer failed: '+producer)
        output=verified['result']['outputs'][edge['role']]
        dependency=dict(edge=edge,producer_attempt=producer,result=verified['record']['result'],
                        release=verified['record']['release'],terminal=verified['result']['executor_terminal'],output=output)
        for item in (dependency['result'],dependency['release'],dependency['terminal'],output):
            ex.verify_pin(item)
            ex.require(item['path'] not in pins or pins[item['path']]==item,'conflicting dependency input pin')
            pins[item['path']]=item
        for index in edge['sha256_arguments']:spec['argv'][index]=output['sha256']
        dependencies.append(dependency)
    spec['input_pins']=list(pins.values())
    ex.validate(spec)
    return dict(schema='DependencyExecutionBindingV1',attempt=attempt,admission_digest=plan_digest,
                job=job,dependencies=dependencies,execution_spec=spec,specification_sha256=q.digest(spec))


def reserved_binding(root,db,attempt):
    saved=row_value(db,'reservations',attempt)
    ex.require(saved is not None,'missing producer reservation')
    reservation,digest=saved
    current=binding(root,db,attempt)
    path=files(root,attempt)/'binding.json'
    ex.verify_pin(reservation['binding'])
    ex.require(reservation==dict(schema='DependencyReservationV1',attempt=attempt,binding=ex.pin(path)), 'reservation binding/identity')
    ex.require(q.document(path)==current,'frozen dependency binding differs from current authenticated producer')
    return current,digest


def observed_result(root,attempt,bound,reservation_digest):
    spec=bound['execution_spec'];execution_root=root/'executions'
    _,actual,_=ex.load(execution_root,attempt,verify_inputs=True)
    ex.require(actual==spec,'executor differs from exact bound spec')
    ex.registered(execution_root,ex.attempt(execution_root,attempt),spec)
    state=ex.inspect(execution_root,attempt);terminal=state['attempt_terminal']
    if (not terminal or state['job_lock_held'] or state['session_members'] or
        any(o['state']=='alive' for o in state['process_observation'].values())):
        raise NotReady('live_or_unresolved: '+attempt)
    ex.require(terminal['cleanup']['verified'],'unverified cleanup')
    if terminal['return_code'] is None:raise NotReady('unknown_exit: '+attempt)
    outcome=('failed_timeout' if terminal['timed_out'] else 'failed_process' if terminal['return_code']!=0 else
             'failed_native_outputs' if terminal['native_outputs']!='metadata_verified_science_unassessed' else 'accepted_execution')
    outputs={}
    if outcome=='accepted_execution':
        for declaration in bound['job']['outputs']:
            path=Path(spec['output_directory'])/declaration['path']
            ex.require(path.resolve().is_relative_to(Path(spec['output_directory'])),'escaped output')
            item=ex.pin(path)
            ex.require(item in terminal['native_receipts'] and item['size']>=declaration['minimum_bytes'],'output gate changed')
            outputs[declaration['role']]=item
    return dict(schema='DependencyJobResultV1',attempt=attempt,reservation_sha256=reservation_digest,
                executor_terminal=state['terminal_receipt'],outcome=outcome,outputs=outputs,qualification='unassessed')


def verified_release(root,db,attempt):
    saved=row_value(db,'releases',attempt)
    if saved is None:raise NotReady('producer not released: '+attempt)
    record_value,_=saved
    bound,reservation_digest=reserved_binding(root,db,attempt)
    expected_result=observed_result(root,attempt,bound,reservation_digest)
    directory=files(root,attempt)
    for key in ('result','release'):ex.verify_pin(record_value[key])
    ex.require(record_value==dict(schema='DependencyReleaseRecordV1',attempt=attempt,
        result=ex.pin(directory/'result.json'),release=ex.pin(directory/'release.json')),'producer release record identity')
    ex.require(q.document(directory/'result.json')==expected_result,'producer result contradicts executor')
    expected_release=dict(schema='DependencyCapacityReleaseV1',attempt=attempt,reservation_sha256=reservation_digest,
                          result=record_value['result'],executor_terminal=expected_result['executor_terminal'])
    ex.require(q.document(directory/'release.json')==expected_release,'producer release chain mismatch')
    return dict(record=record_value,result=expected_result)


def reconcile(root,db,attempt,fault):
    bound,reservation_digest=reserved_binding(root,db,attempt)
    spec=bound['execution_spec'];execution_root=root/'executions';directory=ex.attempt(execution_root,attempt)
    if not directory.exists():
        ex.prepare(execution_root,spec,q.digest(spec));fault('prepared')
    _,actual,_=ex.load(execution_root,attempt,verify_inputs=True)
    ex.require(actual==spec,'executor differs from reserved binding')
    ex.registered(execution_root,directory,spec)
    state=ex.inspect(execution_root,attempt)
    if not (directory/'launch-intent.json').exists():
        ex.require(state['state']=='prepared' and not state['job_lock_held'] and not state['session_members'] and
                   all(o['state']!='alive' for o in state['process_observation'].values()),'unlaunched attempt not unused')
        # Re-derive producer chain after preparation and immediately before launch;
        # executor input gates independently check all exported dependency pins.
        reserved_binding(root,db,attempt)
        ex.launch(execution_root,attempt);fault('launched')
        return 'retained_launch_or_live'
    result=observed_result(root,attempt,bound,reservation_digest)
    directory=files(root,attempt)
    result_pin=publish_exact(directory/'result.json',result);fault('result_published')
    release=dict(schema='DependencyCapacityReleaseV1',attempt=attempt,reservation_sha256=reservation_digest,
                 result=result_pin,executor_terminal=result['executor_terminal'])
    release_pin=publish_exact(directory/'release.json',release);fault('release_published')
    value=dict(schema='DependencyReleaseRecordV1',attempt=attempt,result=result_pin,release=release_pin)
    db.execute('BEGIN IMMEDIATE')
    try:
        record(db,'releases',attempt,value);fault('release_inserted');finish_frontier(db,attempt,result['outcome'],fault);fault('frontier_released');db.execute('COMMIT');fault('released')
    except BaseException:
        if db.in_transaction:db.execute('ROLLBACK')
        raise
    return result['outcome']


def tick(root,fault=lambda stage:None):
    with locked(root) as (root,db):
        # These queries visit only the live slot and partial ready index, never history.
        active=db.execute('SELECT attempt FROM active').fetchall();ex.require(len(active)<=1,'capacity frontier invalid');observations={}
        if not active:
            candidate=db.execute('SELECT attempt FROM pending WHERE remaining=0 AND blocked IS NULL ORDER BY seq LIMIT 1').fetchone()
            if candidate:
                attempt=candidate[0]
                try:
                    # Cached readiness is a selection hint only: authenticate full dependency chain.
                    bound=binding(root,db,attempt);ex.require(not ex.attempt(root/'executions',attempt).exists(),'external attempt before reservation');directory=files(root,attempt);ex.require({p.name for p in directory.iterdir()}<={'binding.json'},'unexpected pre-reservation evidence');item=publish_exact(directory/'binding.json',bound);fault('binding_published');value=dict(schema='DependencyReservationV1',attempt=attempt,binding=item);reserve_frontier(db,attempt,value,fault);active=[(attempt,)]
                except (NotReady,OSError,ValueError,KeyError,TypeError)as error:return dict(state='blocked_frontier_integrity',error=str(error),attempt=attempt,reserved=0)
        for(attempt,)in active:
            try:observations[attempt]=reconcile(root,db,attempt,fault)
            except NotReady as error:observations[attempt]='retained: '+str(error)
            except (OSError,ValueError,KeyError,TypeError)as error:observations[attempt]='retained_integrity: '+str(error)
        reserved=db.execute('SELECT count(*) FROM active').fetchone()[0]
        return dict(schema='DurableFrontierOperationalObservationV1',trust='trusted_mutable_index_no_historical_audit',reserved=reserved,observations=observations,lifecycle=completion(db),qualification='unassessed')


def audit_frontier(db):
    """Explicit O(history+edges) structural audit, never an automatic idle-tick scan."""
    expected_owners=[];expected_edges=[]
    for identity,body_digest,body in db.execute('SELECT id,digest,body FROM evidence'):
        plan=json.loads(body);ex.require(q.canonical(plan)==body and q.digest(plan)==body_digest and plan['queue_id']==identity,'admission identity/digest changed');jobs={job['id']:job['execution_spec']['attempt_id']for job in plan['jobs']}
        for job in plan['jobs']:
            spec=job['execution_spec'];expected_owners.append((spec['attempt_id'],spec['output_directory'],identity,job['id']));expected_edges.extend((producer,spec['attempt_id'])for producer in {jobs[edge['job_id']]for edge in job['dependencies']})
    actual_owners=db.execute('SELECT attempt,output,admission,job FROM owners').fetchall();ex.require(sorted(actual_owners)==sorted(expected_owners),'owner projection mismatch');admission.disjoint_outputs([row[1]for row in actual_owners]);ex.require(sorted(db.execute('SELECT producer,consumer FROM dependency_edges').fetchall())==sorted(expected_edges),'dependency edge projection mismatch')
    owners={row[0]for row in actual_owners};reservations={row[0]for row in db.execute('SELECT attempt FROM reservations')};releases={row[0]for row in db.execute('SELECT attempt FROM releases')};active={row[0]for row in db.execute('SELECT attempt FROM active')};pending={row[0]:(row[1],row[2])for row in db.execute('SELECT attempt,remaining,blocked FROM pending')};blocked={row[0]:(row[1],row[2],row[3])for row in db.execute('SELECT attempt,cause,body,digest FROM blocked_terminal')}
    ex.require(active==reservations-releases and len(active)<=1,'active projection mismatch');ex.require(set(pending).isdisjoint(blocked)and set(pending)|set(blocked)==owners-reservations,'pending/blocked partition mismatch')
    successful=set();failed=set()
    for producer in releases:
        value,_=row_value(db,'releases',producer);result=q.document(value['result']['path']);(successful if result['outcome']=='accepted_execution'else failed).add(producer)
    def descendants(root):
        return {row[0]for row in db.execute('WITH RECURSIVE d(attempt) AS (SELECT consumer FROM dependency_edges WHERE producer=? UNION SELECT e.consumer FROM dependency_edges e JOIN d ON e.producer=d.attempt) SELECT attempt FROM d',(root,))}
    expected_blocked=set().union(*(descendants(root)for root in failed))if failed else set();ex.require(set(blocked)==expected_blocked,'failed descendant closure mismatch')
    for consumer,(cause,body,stored_digest)in blocked.items():
        value=json.loads(body);ex.require(q.canonical(value)==body and q.digest(value)==stored_digest,'blocked digest mismatch');ex.require(cause in failed and consumer in descendants(cause),'invalid blocked cause');expected=dict(schema='FailedDependencyBlockedTerminalV1',attempt=consumer,cause_attempt=cause,cause_release_sha256=row_value(db,'releases',cause)[1],reason='transitive_failed_dependency',qualification='not_executed_science_unassessed');ex.require(value==expected,'blocked causal record mismatch')
    for consumer,(remaining,marker)in pending.items():
        producers={row[0]for row in db.execute('SELECT producer FROM dependency_edges WHERE consumer=?',(consumer,))};ex.require(marker is None and remaining==len(producers-successful),'remaining dependency projection mismatch')
    expected=(len(owners),len(pending),len(active),len(releases),len(blocked),len(successful),len(failed));ex.require(db.execute('SELECT total,pending,active,released,blocked,accepted,failed FROM counts WHERE slot=1').fetchone()==expected,'lifecycle counter projection mismatch')
    return completion(db)


def completion(db):
    """Snapshot of all currently admitted owners, not a sealed service shutdown claim."""
    rows=db.execute('SELECT total,pending,active,released,blocked,accepted,failed FROM counts WHERE slot=1').fetchall();ex.require(len(rows)==1,'missing lifecycle counter')
    total,pending,active,released,blocked,accepted,failed=rows[0];ex.require(all(type(n)is int and n>=0 for n in rows[0])and active<=1 and total==pending+active+released+blocked and released==accepted+failed,'invalid lifecycle counters');ex.require(active==db.execute('SELECT count(*) FROM active').fetchone()[0],'active count mismatch')
    ex.require(bool(pending)==bool(db.execute('SELECT 1 FROM pending LIMIT 1').fetchone()),'pending counter presence mismatch');ex.require(bool(blocked)==bool(db.execute('SELECT 1 FROM blocked_terminal LIMIT 1').fetchone()),'blocked counter presence mismatch')
    terminal=total>0 and pending==0 and active==0
    if total==0:state='empty_no_admitted_work'
    elif terminal:state='nonempty_current_admissions_terminal_with_failures_or_blocks'if failed or blocked else'nonempty_current_admissions_terminal_all_accepted_unqualified'
    elif active:state='nonempty_current_admissions_active'
    elif db.execute('SELECT 1 FROM pending WHERE remaining=0 AND blocked IS NULL ORDER BY seq LIMIT 1').fetchone():state='nonempty_current_admissions_runnable'
    else:state='nonempty_current_admissions_waiting_unresolved'
    return dict(schema='NonemptyCurrentAdmissionsLifecycleV1',state=state,total=total,pending=pending,active=active,executed_terminal=released,blocked_terminal=blocked,accepted_execution=accepted,failed_execution=failed,nonempty_current_admissions_all_terminal=terminal,nonempty_current_admissions_all_accepted_unqualified=terminal and failed==0 and blocked==0,scope='current admissions snapshot; new admissions reopen work; not scientific qualification')
