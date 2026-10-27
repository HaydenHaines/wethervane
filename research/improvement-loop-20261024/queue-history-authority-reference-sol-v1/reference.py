"""Same-UID synthetic authority reference; no worker launcher, protected service or scientific admission."""
import copy,hashlib,json,os,re,sqlite3,uuid
from pathlib import Path
TRUST='trusted_mutable_same_uid_reference'
SCHEMA='HistoryAuthorityReferenceV1'
class Refused(ValueError):pass
def need(ok,code):
 if not ok:raise Refused(code)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def code_pin():
 p=Path(__file__).resolve();raw=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
def ident(x):return isinstance(x,str)and re.fullmatch('[A-Za-z0-9_.:-]{1,120}',x)is not None
def exact(x,keys):need(isinstance(x,dict)and set(x)==set(keys),'unknown_or_missing_fields')
def crash(stage):
 if os.environ.get('REFERENCE_CRASH')==stage:os._exit(93)
def head_shape(head):
 exact(head,['domain','sequence','hash'])
 need(isinstance(head['domain'],str)and re.fullmatch('[0-9a-f]{32}',head['domain'])and type(head['sequence'])is int and 0<=head['sequence']<=512 and isinstance(head['hash'],str)and re.fullmatch('[0-9a-f]{64}',head['hash']),'invalid_head_shape')
def fresh_state():return {'jobs':{},'active':None,'lease':1,'open':True}
def ready(state):return sorted(k for k,v in state['jobs'].items()if v['state']=='pending'and all(state['jobs'][x]['state']=='accepted_unqualified'for x in v['deps']))
def apply(state,action,args,event_hash):
 jobs=state['jobs']
 if action=='admit':
  exact(args,['job','output','deps','evidence_hash','target','unit']);job=args['job'];need(ident(job)and job not in jobs,'duplicate_or_invalid_job')
  need(state['open'],'intake_closed');need(len(jobs)<64,'reference_job_limit');output=args['output'];need(isinstance(output,str)and Path(output).is_absolute()and Path(output).anchor=='/'and os.path.normpath(output)==output,'canonical_output_required')
  for old in jobs.values():need(not(Path(output).is_relative_to(old['output'])or Path(old['output']).is_relative_to(output)),'output_ownership_conflict')
  deps=args['deps'];need(isinstance(deps,list)and len(deps)<=32 and all(ident(x)for x in deps)and len(deps)==len(set(deps))and all(x in jobs for x in deps),'dependency_missing_or_duplicate')
  need(ident(args['target'])and ident(args['unit'])and isinstance(args['evidence_hash'],str)and re.fullmatch('[0-9a-f]{64}',args['evidence_hash']),'evidence_target_unit_required')
  jobs[job]={**copy.deepcopy(args),'state':'pending'}
 elif action=='reserve':
  exact(args,['job','worker_token']);need(state['active']is None,'capacity_reserved');need(args['job']in ready(state)and ident(args['worker_token']),'job_not_ready_or_worker_unknown')
  state['active']={'job':args['job'],'worker_token':args['worker_token'],'reservation_hash':event_hash,'reserved_lease':state['lease']};jobs[args['job']]['state']='active'
 elif action=='release':
  exact(args,['evidence']);e=args['evidence'];exact(e,['schema','job','worker_token','reservation_hash','worker_status','cleanup_verified','outcome','terminal_sha256','cleanup_sha256','scientifically_qualified'])
  a=state['active'];need(a is not None,'no_active_reservation')
  need(e['schema']=='InventedCleanupEvidenceV1'and e['job']==a['job']and e['worker_token']==a['worker_token']and e['reservation_hash']==a['reservation_hash'],'release_binding_mismatch')
  need(e['worker_status']=='exited'and e['cleanup_verified']is True,'worker_or_cleanup_unknown')
  need(e['outcome']in ['accepted_unqualified','failed']and e['scientifically_qualified']is False,'outcome_or_qualification_invalid')
  need(all(isinstance(e[k],str)and re.fullmatch('[0-9a-f]{64}',e[k])for k in ['terminal_sha256','cleanup_sha256']),'cleanup_evidence_digest_required')
  jobs[a['job']]['state']=e['outcome'];state['active']=None
 elif action=='lease':
  exact(args,[]);state['lease']+=1  # fencing does NOT imply death/release
 elif action=='seal':
  exact(args,[]);need(jobs,'empty_batch_not_sealable');state['open']=False
 else:raise Refused('unknown_action')
 changed=True
 while changed:
  changed=False
  for job in jobs.values():
   if job['state']=='pending'and any(jobs[d]['state']in ['failed','blocked']for d in job['deps']):job['state']='blocked';changed=True

def receipt(config,event_hash,seq,action):return {'schema':'HistoricalCommitReceiptV1','domain':config['domain'],'sequence':seq,'hash':event_hash,'action':action,'historical_commit_only':True,'live_launch_authorization':False,'runtime_worker_evidence_authenticated':False,'trust':TRUST,'scientifically_qualified':False}

class Reference:
 def __init__(self,root):
  self.root=Path(root).absolute();need(self.root==self.root.resolve(),'root_alias_refused');self.path=self.root/'authority.sqlite';need(self.path.is_file(),'missing_reference_store')
 @classmethod
 def create(cls,root):
  root=Path(root).absolute();need(root==root.resolve(),'root_alias_refused');root.mkdir(mode=0o700,parents=False,exist_ok=False)
  path=root/'authority.sqlite';db=sqlite3.connect(path);db.execute('PRAGMA journal_mode=DELETE');db.execute('PRAGMA synchronous=FULL')
  db.executescript('CREATE TABLE meta(k TEXT PRIMARY KEY,body TEXT NOT NULL); CREATE TABLE events(seq INTEGER PRIMARY KEY,body TEXT NOT NULL,digest TEXT NOT NULL); CREATE TABLE responses(k TEXT PRIMARY KEY,request_hash TEXT NOT NULL,body TEXT NOT NULL); CREATE TABLE objects(digest TEXT PRIMARY KEY,body BLOB NOT NULL); CREATE TABLE jobs(job TEXT PRIMARY KEY,body TEXT NOT NULL); CREATE TABLE ready(job TEXT PRIMARY KEY); CREATE TABLE active(slot INTEGER PRIMARY KEY CHECK(slot=1),body TEXT NOT NULL); CREATE TABLE projection_head(slot INTEGER PRIMARY KEY CHECK(slot=1),body TEXT NOT NULL);')
  config={'schema':SCHEMA,'domain':uuid.uuid4().hex,'root':str(root),'source':code_pin(),'trust':TRUST};head={'domain':config['domain'],'sequence':0,'hash':digest(config)}
  db.execute('INSERT INTO meta VALUES(?,?)',('config',canonical(config)));db.execute('INSERT INTO meta VALUES(?,?)',('head',canonical(head)));cls._write_projection(db,fresh_state(),head);db.commit();db.close();return cls(root)
 def connect(self):
  db=sqlite3.connect(self.path,timeout=.5,isolation_level=None);db.execute('PRAGMA synchronous=FULL');return db
 @staticmethod
 def _write_projection(db,state,head):
  for table in ['jobs','ready','active','projection_head']:db.execute('DELETE FROM '+table)
  db.executemany('INSERT INTO jobs VALUES(?,?)',[(k,canonical(v))for k,v in sorted(state['jobs'].items())]);db.executemany('INSERT INTO ready VALUES(?)',[(x,)for x in ready(state)])
  if state['active']is not None:db.execute('INSERT INTO active VALUES(1,?)',(canonical(state['active']),))
  db.execute('INSERT INTO projection_head VALUES(1,?)',(canonical(head),))
 def _read(self,db,floor=None,projection=True):
  config=json.loads(db.execute("SELECT body FROM meta WHERE k='config'").fetchone()[0]);need(config.get('schema')==SCHEMA and config.get('root')==str(self.root)and config.get('trust')==TRUST,'domain_version_or_root_refused');need(config.get('source')==code_pin(),'source_version_drift')
  state=fresh_state();head={'domain':config['domain'],'sequence':0,'hash':digest(config)};history={0:head['hash']};responses={}
  for seq,body,h in db.execute('SELECT seq,body,digest FROM events ORDER BY seq'):
   event=json.loads(body);need(canonical(event)==body and seq==head['sequence']+1 and event['sequence']==seq and event['previous']==head['hash']and event['domain']==config['domain']and digest(event)==h,'event_chain_corrupt')
   need(event['key']not in responses and event['request_hash']==digest({'action':event['action'],'args':event['args']}),'event_idempotency_corrupt')
   if event['action']=='admit':
    obj=db.execute('SELECT body FROM objects WHERE digest=?',(event['args']['evidence_hash'],)).fetchone();need(obj is not None and hashlib.sha256(obj[0]).hexdigest()==event['args']['evidence_hash'],'referenced_object_corrupt')
   apply(state,event['action'],event['args'],h);head={'domain':config['domain'],'sequence':seq,'hash':h};history[seq]=h;responses[event['key']]=(event['request_hash'],receipt(config,h,seq,event['action']))
  stored=json.loads(db.execute("SELECT body FROM meta WHERE k='head'").fetchone()[0]);need(stored==head,'head_chain_mismatch')
  actual={k:(req,json.loads(body))for k,req,body in db.execute('SELECT k,request_hash,body FROM responses')};need(actual==responses,'idempotency_receipt_corrupt')
  if floor is not None:
   head_shape(floor);need(floor['domain']==head['domain']and type(floor['sequence'])is int and floor['sequence']in history and history[floor['sequence']]==floor['hash'],'caller_floor_rollback_or_fork')
  if projection:
   want_jobs={k:canonical(v)for k,v in state['jobs'].items()};have_jobs=dict(db.execute('SELECT job,body FROM jobs'))
   have_ready=sorted(x[0]for x in db.execute('SELECT job FROM ready'));have_active=[x[0]for x in db.execute('SELECT body FROM active')];have_heads=[x[0]for x in db.execute('SELECT body FROM projection_head')]
   need(have_jobs==want_jobs and have_ready==ready(state)and have_active==([]if state['active']is None else[canonical(state['active'])])and have_heads==[canonical(head)],'projection_corrupt_rebuild_required')
  return config,state,head,responses
 def _mutate(self,key,expected_head,lease,action,args,objects=()):
  need(ident(key),'idempotency_key_required');head_shape(expected_head);db=self.connect()
  try:
   db.execute('BEGIN IMMEDIATE');config,state,head,responses=self._read(db,expected_head)
   req=digest({'action':action,'args':args})
   if key in responses:
    need(responses[key][0]==req,'idempotency_body_conflict');db.rollback();return responses[key][1]
   need(expected_head==head and type(lease)is int and lease==state['lease'],'stale_head_or_lease')
   for h,raw in objects:
    need(type(raw)is bytes and hashlib.sha256(raw).hexdigest()==h,'object_bytes_mismatch');old=db.execute('SELECT body FROM objects WHERE digest=?',(h,)).fetchone();need(old is None or old[0]==raw,'object_key_conflict');db.execute('INSERT OR IGNORE INTO objects VALUES(?,?)',(h,raw))
   need(head['sequence']<512,'reference_event_limit')
   event={'domain':config['domain'],'sequence':head['sequence']+1,'previous':head['hash'],'key':key,'request_hash':req,'action':action,'args':args};h=digest(event);apply(state,action,args,h)
   db.execute('INSERT INTO events VALUES(?,?,?)',(event['sequence'],canonical(event),h));crash('after_event_insert')
   newhead={'domain':config['domain'],'sequence':event['sequence'],'hash':h};db.execute("UPDATE meta SET body=? WHERE k='head'",(canonical(newhead),));self._write_projection(db,state,newhead);crash('after_projection')
   result=receipt(config,h,event['sequence'],action);db.execute('INSERT INTO responses VALUES(?,?,?)',(key,req,canonical(result)));crash('before_commit');db.commit();crash('after_commit');return result
  finally:db.close()
 def admit(self,key,expected_head,lease,job,output,deps=(),evidence=b'invented',target='invented-target',unit='invented-unit'):
  need(type(evidence)is bytes and len(evidence)<=65536,'bounded_byte_evidence_required');h=hashlib.sha256(evidence).hexdigest();return self._mutate(key,expected_head,lease,'admit',{'job':job,'output':output,'deps':list(deps),'evidence_hash':h,'target':target,'unit':unit},[(h,evidence)])
 def reserve(self,key,head,lease,job,worker_token):return self._mutate(key,head,lease,'reserve',{'job':job,'worker_token':worker_token})
 def release(self,key,head,lease,evidence):return self._mutate(key,head,lease,'release',{'evidence':evidence})
 def advance_lease(self,key,head,lease):return self._mutate(key,head,lease,'lease',{})
 def seal(self,key,head,lease):return self._mutate(key,head,lease,'seal',{})
 def status(self,floor=None):
  db=self.connect()
  try:
   db.execute('BEGIN');_,s,h,_=self._read(db,floor);states=sorted(x['state']for x in s['jobs'].values());terminal=bool(states)and all(x in ['accepted_unqualified','failed','blocked']for x in states)
   return {'trust':TRUST,'head':h,'lease':s['lease'],'intake_open':s['open'],'ready':ready(s),'active':s['active'],'states':{k:v['state']for k,v in s['jobs'].items()},'nonempty_current_admissions_terminal':terminal,'all_current_admissions_accepted_unqualified':bool(states)and all(x=='accepted_unqualified'for x in states),'closed_nonempty_batch_terminal':not s['open']and terminal,'scientifically_qualified':False,'rollback_floor':'caller_held_prefix_checked'if floor else'none_not_rollback_protected','runtime_worker_evidence_authenticated':False}
  finally:db.close()
 def canonical_head(self,floor=None):
  db=self.connect()
  try:db.execute('BEGIN');_,s,h,_=self._read(db,floor,False);return h,s['lease']
  finally:db.close()
 def lookup(self,key,floor=None):
  db=self.connect()
  try:db.execute('BEGIN');_,_,_,responses=self._read(db,floor,False);return responses.get(key,(None,None))[1]
  finally:db.close()
 def rebuild(self,expected_head,lease,floor=None):
  head_shape(expected_head);db=self.connect()
  try:
   db.execute('BEGIN IMMEDIATE');_,s,h,_=self._read(db,floor,False);need(expected_head==h and type(lease)is int and lease==s['lease'],'stale_rebuild_head_or_lease');self._write_projection(db,s,h);crash('before_rebuild_commit');db.commit();crash('after_rebuild_commit');return {'head':h,'projection_rebuilt':True,'trust':TRUST}
  finally:db.close()
 def head_response(self,nonce,floor=None):
  need(ident(nonce),'nonce_required');h,lease=self.canonical_head(floor);return {'domain':h['domain'],'head':h,'lease':lease,'nonce':nonce,'floor_checked':floor,'response_authenticated':False,'trust':TRUST}

def check_response(response,nonce,domain,floor=None):
 head_shape(response['head'])
 if floor is not None:head_shape(floor)
 need(response['nonce']==nonce and response['domain']==domain and response['head']['domain']==domain,'nonce_or_domain_mismatch')
 if floor is not None:need(response['floor_checked']==floor and response['head']['sequence']>=floor['sequence']and(response['head']['sequence']!=floor['sequence']or response['head']['hash']==floor['hash']),'response_floor_mismatch')
 return {'consistent_reply_claim':True,'authenticated':False,'trust':TRUST}
