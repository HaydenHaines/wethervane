"""Internal finite-generation adapter; caller MUST hold dispatcher.lock."""
import hashlib,json,os,sqlite3
from pathlib import Path
import execution as ex
from ledger import canonical
from run_receipts import decode
from receipt_projection import project

def crash(point):
    if os.environ.get('FINITE_QUEUE_CRASH')==point:os._exit(91)
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def doc(p):return decode(Path(p).read_bytes())
def require(v,m):ex.require(v,m)
def db(root):
    c=sqlite3.connect(str(Path(root)/'intake.sqlite'),timeout=.5,isolation_level=None);c.execute('PRAGMA synchronous=FULL');c.execute('BEGIN IMMEDIATE');return c
def initialize(root):
    root=Path(root).resolve();new=not(root/'intake.sqlite').exists()
    marker=root/'intake-created.json'
    require(not new or not marker.exists(),'previously initialized lifecycle DB is missing; no reset')
    require(not new or not any((root/'queues').iterdir()),'missing lifecycle DB with existing admissions; no adoption')
    c=db(root)
    try:
        c.execute('CREATE TABLE IF NOT EXISTS binding(value TEXT NOT NULL)');c.execute('CREATE TABLE IF NOT EXISTS run(state TEXT NOT NULL,allow_empty INTEGER NOT NULL,seal TEXT,final BLOB)');c.execute('CREATE TABLE IF NOT EXISTS intents(id TEXT PRIMARY KEY,plan TEXT NOT NULL,state TEXT NOT NULL,receipt TEXT)')
        binding={'schema':'FiniteQueueIntakeOwnedV1','root':str(root),'store':ex.pin(root/'store.json')};old=c.execute('SELECT value FROM binding').fetchone()
        if old:require(json.loads(old[0])==binding,'lifecycle store binding changed')
        else:c.execute('INSERT INTO binding VALUES(?)',(canonical(binding),));c.execute('INSERT INTO run VALUES("OPEN",0,NULL,NULL)')
        c.commit()
        fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:c.close()
    expected={'schema':'FiniteIntakeCreatedOwnedV1','store':ex.pin(root/'store.json')}
    if marker.exists():require(doc(marker)==expected,'intake marker drift')
    else:ex.publish(marker,expected)
def opened(root):
    root=Path(root).resolve();require((root/'intake.sqlite').is_file(),'missing intake database')
    require(doc(root/'intake-created.json')=={'schema':'FiniteIntakeCreatedOwnedV1','store':ex.pin(root/'store.json')},'missing or changed intake marker')
    c=db(root)
    try:
        old=c.execute('SELECT value FROM binding').fetchone();require(old and json.loads(old[0])=={'schema':'FiniteQueueIntakeOwnedV1','root':str(root),'store':ex.pin(root/'store.json')},'lifecycle binding drift');return c
    except BaseException:c.close();raise
def admissions(c):return [{'queue_id':i,'plan_sha256':digest(json.loads(p))}for i,p,s,r in c.execute('SELECT id,plan,state,receipt FROM intents ORDER BY id')]
def checkseal(c):
    row=c.execute('SELECT state,allow_empty,seal,final FROM run').fetchone();require(row is not None,'missing lifecycle row')
    if row[0]!='OPEN':require(row[2]==digest(admissions(c)),'sealed admission set drift')
    return row
def frontier(root):
    c=opened(root)
    try:
        checkseal(c);rows=c.execute('SELECT id,plan,state,receipt FROM intents').fetchall();known={x[0]for x in rows}
        require(all(p.is_dir()and p.name in known for p in (Path(root)/'queues').iterdir()),'unknown admission not in durable intent set')
        for i,p,s,r in rows:
            plan=json.loads(p);directory=Path(root)/'queues'/i;require(plan['queue_id']==i,'intent queue identity mismatch')
            if s=='ACCEPTED':
                ex.verify_pin(json.loads(r));require(doc(directory/'receipt.json')['plan']==ex.pin(directory/'plan.json'),'accepted plan pin drift');require(digest(doc(directory/'plan.json'))==digest(plan),'intent/file plan mismatch')
                require((directory/'jobs').is_dir()and {p.name for p in (directory/'jobs').iterdir()}=={j['id']for j in plan['jobs']},'accepted job scaffold differs')
            else:require(s=='ADMITTING','unknown intent state')
    finally:c.close()
def require_accepted(root):
    require(info(root)['admitting']==0,'ADMITTING intent requires exact repair before dispatch/reconciliation')
def evidence(root):
    pins=[]
    for directory in sorted((Path(root)/'queues').iterdir()):
        for name in ('plan.json','receipt.json'):
            pins.append(ex.pin(directory/name))
        for job in sorted((directory/'jobs').iterdir()):
            for name in ('binding.json','reservation.json','result.json','release.json','ownership-ack.json'):
                path=job/name
                if path.exists():pins.append(ex.pin(path))
            result=doc(job/'result.json')
            if result.get('executor_terminal'):pins.append(result['executor_terminal'])
            pins.extend(result.get('outputs',{}).values())
    for pin in pins:ex.verify_pin(pin)
    return pins
def info(root):
    c=opened(root)
    try:
        row=checkseal(c);return {'state':row[0],'intents':c.execute('SELECT COUNT(*) FROM intents').fetchone()[0],'admitting':c.execute('SELECT COUNT(*) FROM intents WHERE state="ADMITTING"').fetchone()[0],'seal':row[2],'acceptance':'metadata only; finalize revalidates original queue'}
    finally:c.close()
def enqueue(root,plan,config,admit):
    frontier(root);c=opened(root);key=plan['queue_id'];payload=canonical(plan)
    try:
        row=checkseal(c);old=c.execute('SELECT plan,state,receipt FROM intents WHERE id=?',(key,)).fetchone()
        if old:
            require(old[0]==payload,'changed plan: durable intent duplicate')
        else:
            require(row[0]=='OPEN','new post-seal intake rejected')
            # Original V3 preflight is read-only and runs before durable intent.
            admit(root,plan,digest(plan),config,False)
            for other, in c.execute('SELECT plan FROM intents WHERE state="ADMITTING"'):
                for a in json.loads(other)['jobs']:
                    for b in plan['jobs']:
                        x,y=a['execution_spec'],b['execution_spec'];require(x['attempt_id']!=y['attempt_id']and not (Path(x['output_directory']).is_relative_to(Path(y['output_directory'])) or Path(y['output_directory']).is_relative_to(Path(x['output_directory']))),'identity owned by durable pending intent')
            c.execute('INSERT INTO intents VALUES(?, ?, "ADMITTING",NULL)',(key,payload));crash('intent_before_commit');c.commit();crash('intent_after_commit')
    finally:c.close()
    # Intent owns attempts/outputs even if the process dies before any files.
    receipt=admit(root,plan,digest(plan),config,True)
    for job in plan['jobs']:ex.ensure_directory(Path(root)/'queues'/key/'jobs'/job['id'])
    c=opened(root)
    try:
        checkseal(c);c.execute('UPDATE intents SET state="ACCEPTED",receipt=? WHERE id=?',(canonical(receipt),key));crash('accepted_before_commit');c.commit();crash('accepted_after_commit')
    finally:c.close()
    frontier(root);return receipt

def seal(root,allow_empty=False):
    require(type(allow_empty)is bool,'explicit boolean empty policy');frontier(root);c=opened(root)
    try:
        row=checkseal(c)
        if row[0]=='OPEN':
            h=digest(admissions(c));c.execute('UPDATE run SET state="SEALED",allow_empty=?,seal=?',(int(allow_empty),h));crash('seal_before_commit');c.commit();crash('seal_after_commit')
        else:require(bool(row[1])==allow_empty,'sealed empty policy immutable')
    finally:c.close()
    return info(root)
def finalize(root,observation):
    c=opened(root)
    try:
        row=checkseal(c);require(row[0]!='OPEN','OPEN is idle, never final');require(c.execute('SELECT COUNT(*) FROM intents WHERE state="ADMITTING"').fetchone()[0]==0,'interrupted admission requires exact repair')
        admitted=admissions(c);require(admitted or row[1],'empty run requires explicit seal policy');require(observation['all_jobs_recorded']and not observation['reserved']and not observation['partial_admissions']and not observation['quarantined_admissions']and not observation['unknown_reservations'],'pending/live/corrupt reservations block finality')
        outcomes=[j['outcome']for jobs in observation['queues'].values()for j in jobs.values()];require(bool(outcomes)==bool(admitted),'observation/admission support mismatch')
        final={'schema':'FiniteQueueOwnedResearchFinalV1','store':ex.pin(Path(root)/'store.json'),'seal':row[2],'admissions':admitted,'evidence':evidence(root)+observation['ownership']['evidence'],'observation':observation,'disposition':'explicit_empty'if not admitted else'all_succeeded'if all(o=='accepted_execution'for o in outcomes)else'terminal_with_failures','qualification':'unassessed'}
        # V3 observations have no timestamps; canonical bytes remain deterministic.
        data=(canonical(final)+'\n').encode()
        if row[0]=='FINALIZED':require(bytes(row[3])==data,'finalized reconciled state changed')
        else:c.execute('UPDATE run SET state="FINALIZED",final=?',(data,));crash('final_before_commit');c.commit();crash('final_after_commit')
    finally:c.close()
    return project(root,'run',data,os.environ.get('FINITE_QUEUE_CRASH'))
