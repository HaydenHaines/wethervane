"""Finite intake bookkeeping core. NOT queue-v3 integration or execution evidence."""
import contextlib,fcntl,hashlib,json,os,sqlite3,secrets
from pathlib import Path

class Rejected(Exception):pass
class Integrity(Exception):pass
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def filepin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def verify(p):
 if filepin(p['path'])!=p:raise Integrity('pinned evidence drift')
def source():return filepin(Path(__file__))
def fsyncdir(p):
 fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def crash(selected,point):
 if selected==point:os._exit(91)
def connect(root):
 db=sqlite3.connect(str(Path(root)/'intake.sqlite'),timeout=10,isolation_level=None)
 db.execute('PRAGMA synchronous=FULL');db.execute('PRAGMA foreign_keys=ON')
 return db
@contextlib.contextmanager
def transaction(root,initial=False):
 root=Path(root).resolve()
 if not initial and not(root/'intake.sqlite').is_file():raise Integrity('missing store')
 with (root/'intake.lock').open('a+b')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  db=connect(root)
  try:
   db.execute('BEGIN IMMEDIATE')
   if not initial:
    m=json.loads(db.execute('SELECT value FROM metadata WHERE key="binding"').fetchone()[0])
    if m!={'schema':'FiniteIntakeCoreV2','root':str(root),'source':source()}:raise Integrity('source/root binding drift')
   yield db
  finally:
   if db.in_transaction:db.rollback()
   db.close()

def initialize(root):
 root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
 with transaction(root,True)as db:
  db.execute('CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
  db.execute('CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,state TEXT NOT NULL,empty_policy TEXT NOT NULL,seal TEXT,final BLOB)')
  db.execute('CREATE TABLE IF NOT EXISTS jobs(run TEXT NOT NULL,id TEXT NOT NULL,payload TEXT NOT NULL,terminal TEXT,PRIMARY KEY(run,id),FOREIGN KEY(run) REFERENCES runs(id))')
  expected={'schema':'FiniteIntakeCoreV2','root':str(root),'source':source()};row=db.execute('SELECT value FROM metadata WHERE key="binding"').fetchone()
  if row and json.loads(row[0])!=expected:raise Integrity('existing store source/root mismatch')
  db.execute('INSERT OR IGNORE INTO metadata VALUES(?,?)',('binding',canonical(expected).decode()));db.commit()
 fsyncdir(root)
def identifier(x):
 if not isinstance(x,str)or not x or len(x)>80 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-'for c in x):raise Rejected('invalid id')
def create_run(root,run,empty_policy='reject'):
 identifier(run)
 if empty_policy not in ['reject','explicit_empty']:raise Rejected('explicit empty policy required')
 with transaction(root)as db:
  row=db.execute('SELECT empty_policy FROM runs WHERE id=?',(run,)).fetchone()
  if row and row[0]!=empty_policy:raise Rejected('run policy immutable')
  db.execute('INSERT OR IGNORE INTO runs VALUES(?,?,?,NULL,NULL)',(run,'OPEN',empty_policy));db.commit()
def runrow(db,run):
 row=db.execute('SELECT state,empty_policy,seal,final FROM runs WHERE id=?',(run,)).fetchone()
 if row is None:raise Rejected('unknown run')
 return row

def jobs(db,run):return db.execute('SELECT id,payload,terminal FROM jobs WHERE run=? ORDER BY id',(run,)).fetchall()
def admission_list(db,run):return [{'id':j,'payload':json.loads(p)}for j,p,t in jobs(db,run)]
def checkseal(db,run,row):
 if row[0]!='OPEN'and row[2]!=digest(admission_list(db,run)):raise Integrity('sealed admission set drift')
def enqueue(root,run,job,payload,fail=None):
 identifier(job);body=canonical(payload).decode()
 if not isinstance(payload,dict)or set(payload)!= {'input'}:raise Rejected('explicit pinned input required')
 verify(payload['input'])
 with transaction(root)as db:
  row=runrow(db,run);checkseal(db,run,row);old=db.execute('SELECT payload FROM jobs WHERE run=? AND id=?',(run,job)).fetchone()
  if old:
   if old[0]!=body:raise Rejected('changed duplicate')
   # Exact retry acknowledges original membership even after seal, never adds work.
   return {'accepted':True,'already_accepted':True}
  if row[0]!='OPEN':raise Rejected('post-seal intake rejected without store mutation')
  db.execute('INSERT INTO jobs VALUES(?,?,?,NULL)',(run,job,body));crash(fail,'enqueue_before_commit');db.commit();crash(fail,'enqueue_after_commit')
 return {'accepted':True,'already_accepted':False}
def seal(root,run,fail=None):
 with transaction(root)as db:
  row=runrow(db,run);checkseal(db,run,row)
  if row[0]!='OPEN':return row[2]
  sealhash=digest(admission_list(db,run));db.execute('UPDATE runs SET state="SEALED",seal=? WHERE id=?',(sealhash,run));crash(fail,'seal_before_commit');db.commit();crash(fail,'seal_after_commit');return sealhash

# Terminal evidence is a deliberately invented workflow contract, NOT a trusted
# real executor receipt. A future V3 adapter must perform its real reconciliation.
def terminal_document(pin,run,job,payload):
 verify(pin);v=json.loads(Path(pin['path']).read_bytes())
 if set(v)!={'schema','run','job','payload_sha256','disposition','evidence'}or v['schema']!='InventedWorkflowTerminalV1'or v['run']!=run or v['job']!=job or v['payload_sha256']!=digest(payload)or v['disposition']not in ['succeeded','failed','blocked','cancelled']:raise Integrity('wrong terminal evidence')
 for p in v['evidence']:verify(p)
 return v

def record_terminal(root,run,job,pin,fail=None):
 with transaction(root)as db:
  row=runrow(db,run);checkseal(db,run,row);j=db.execute('SELECT payload,terminal FROM jobs WHERE run=? AND id=?',(run,job)).fetchone()
  if j is None:raise Rejected('unaccepted job')
  payload=json.loads(j[0]);verify(payload['input']);terminal_document(pin,run,job,payload)
  if j[1]:
   if json.loads(j[1])!=pin:raise Rejected('terminal replacement prohibited')
   return
  if row[0]=='FINALIZED':raise Integrity('finalized without terminal')
  db.execute('UPDATE jobs SET terminal=? WHERE run=? AND id=?',(canonical(pin).decode(),run,job));crash(fail,'terminal_before_commit');db.commit();crash(fail,'terminal_after_commit')
def project(root,run,data,fail=None):
 root=Path(root);path=root/('final-'+run+'.json')
 if path.exists():
  if path.read_bytes()!=data:raise Integrity('final projection differs; never overwrite')
  fsyncdir(root);return filepin(path)
 tmp=root/('receipt-'+run+'-'+str(os.getpid())+'-'+secrets.token_hex(16)+'.tmp')
 with tmp.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
 crash(fail,'receipt_before_publish')
 try:os.link(tmp,path) # Exclusive publication, no overwrite.
 except FileExistsError:
  if path.read_bytes()!=data:raise Integrity('conflicting final receipt')
 crash(fail,'receipt_after_publish');fsyncdir(root);crash(fail,'receipt_after_dirsync');tmp.unlink();fsyncdir(root)
 return filepin(path)
def finalize(root,run,fail=None):
 with transaction(root)as db:
  row=runrow(db,run);checkseal(db,run,row)
  if row[0]=='OPEN':raise Rejected('OPEN never final')
  alljobs=jobs(db,run)
  if not alljobs and row[1]!='explicit_empty':raise Rejected('empty sealed run needs explicit-empty policy')
  terminal=[]
  for job,p,t in alljobs:
   payload=json.loads(p);verify(payload['input'])
   if t is None:raise Rejected('unknown/pending job blocks finality')
   tp=json.loads(t);v=terminal_document(tp,run,job,payload);terminal.append({'id':job,'payload_sha256':digest(payload),'terminal':tp,'disposition':v['disposition']})
  final={'schema':'FiniteIntakeFinalV1','run':run,'store_source':source(),'seal_sha256':row[2],'accepted_count':len(alljobs),'disposition':'explicit_empty'if not alljobs else 'all_succeeded'if all(x['disposition']=='succeeded'for x in terminal)else 'terminal_with_failures','jobs':terminal,'qualification':'unassessed','trust':'mutable local SQLite and cooperating files; invented terminal contract only'}
  data=canonical(final)+b'\n'
  if row[0]=='FINALIZED':
   if bytes(row[3])!=data:raise Integrity('authoritative final row drift')
  else:
   db.execute('UPDATE runs SET state="FINALIZED",final=? WHERE id=?',(data,run));crash(fail,'final_before_commit');db.commit();crash(fail,'final_after_commit')
  # Keep same authoritative lock through projection; recovery always starts at DB.
  return project(root,run,data,fail)
def status(root,run):
 with transaction(root)as db:
  row=runrow(db,run);checkseal(db,run,row)
  return {'acceptance':'metadata only; finalize revalidates evidence/projection','state':row[0],'count':len(jobs(db,run)),'seal':row[2],'final_sha256':hashlib.sha256(bytes(row[3])).hexdigest()if row[3]is not None else None}
