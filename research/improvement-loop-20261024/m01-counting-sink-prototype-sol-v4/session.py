"""V4 ephemeral caller acknowledgment and locked verified BYTE snapshots.

No acknowledgment reconstruction/loading API. Trusted in-process caller only;
not hostile Python/filesystem/power-loss protection. Engine is private transport.
"""
import contextlib,dataclasses,fcntl,hashlib,hmac,importlib.util,json,os,secrets,stat,threading
from pathlib import Path
from types import MappingProxyType
HERE=Path(__file__).resolve().parent
MAX_FILE=1024**2;MAX_SNAPSHOT=8*1024**2

def need(ok,message):
 if not ok:raise ValueError(message)
def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':')).encode()
@dataclasses.dataclass(frozen=True,eq=False)
class _Acknowledgment:
 issuer:str
 issuer_pid:int
 nonce:str
 root:str
 control:str
 manifest_sha:str
 receipt_sha:str
 engine_sha:str
 session_sha:str
 root_identity:tuple
 control_identity:tuple
 lock_identity:tuple
 mac:str
@dataclasses.dataclass(frozen=True)
class VerifiedSnapshot:
 files:object
 manifest:object
 provenance:object
 scope:str='Verified immutable captured bytes only, not a live-directory or durability assertion'

def freeze(value):
 if isinstance(value,dict):return MappingProxyType({k:freeze(v)for k,v in value.items()})
 if isinstance(value,(list,tuple)):return tuple(freeze(v)for v in value)
 return value

def identity(fd):
 s=os.fstat(fd);return(s.st_dev,s.st_ino)
@contextlib.contextmanager
def lock(root,exclusive):
 path=Path(root).with_name(Path(root).name+'.v4-lock');fd=os.open(path,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 try:
  need(stat.S_ISREG(os.fstat(fd).st_mode)and os.fstat(fd).st_nlink==1,'unsafe lock identity')
  fcntl.flock(fd,fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH);yield identity(fd)
 finally:os.close(fd)

def read_directory(path,allowed,hook=None,prefix='output'):
 fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);files={};total=0
 try:
  directory_identity=identity(fd);need(set(os.listdir(fd))==set(allowed),'ambiguous exact '+prefix+' roster')
  for name in sorted(allowed):
   if hook:hook('before_read',prefix+'/'+name)
   child=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
   try:
    before=os.fstat(child);need(stat.S_ISREG(before.st_mode)and before.st_size<=MAX_FILE,'invalid/oversize snapshot member')
    with os.fdopen(child,'rb',closefd=False)as stream:data=stream.read(MAX_FILE+1)
    after=os.fstat(child);need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'file changed during capture')
    need(len(data)==before.st_size,'short/oversize captured file');total+=len(data);need(total<=MAX_SNAPSHOT,'snapshot budget exceeded');files[prefix+'/'+name]=data
   finally:os.close(child)
   if hook:hook('after_read',prefix+'/'+name)
  need(set(os.listdir(fd))==set(allowed),'directory roster changed during capture');return files,directory_identity
 finally:os.close(fd)

class Session:
 def __init__(self,expected_engine_sha,expected_session_sha):
  self._owner_pid=os.getpid();self.engine_sha=expected_engine_sha;self.session_sha=expected_session_sha;self._check_sources()
  spec=importlib.util.spec_from_file_location('_v4_private_count_engine',HERE/'engine.py');engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine);self.engine=engine
  self._secret=secrets.token_bytes(32);self._issuer=secrets.token_hex(16);self._issued={};self.last_attempt=None
 def _check_sources(self):
  need(os.getpid()==self._owner_pid,'caller session/acknowledgment cannot be adopted by another process')
  need(sha((HERE/'engine.py').read_bytes())==self.engine_sha and sha(Path(__file__).read_bytes())==self.session_sha,'caller-selected source bytes changed')
 def _capture(self,context,hook=None):
  root=Path(context['root']);control=Path(context['control']);e=self.engine
  files,rid=read_directory(root,e.OUTPUT_FILES|{'complete.json'},hook,'output');other,cid=read_directory(control,e.CONTROL_FILES|{'success.json'},hook,'supervisor');files.update(other);need(sum(map(len,files.values()))<=MAX_SNAPSHOT,'combined snapshot budget exceeded')
  if 'root_identity'in context:need(rid==context['root_identity']and cid==context['control_identity'],'directory identity changed')
  raw=files['output/complete.json'];need(sha(raw)==context['manifest_sha'],'captured manifest differs from caller acknowledgment');manifest=json.loads(raw)
  raw=files['supervisor/success.json'];need(sha(raw)==context['receipt_sha'],'captured external receipt differs from caller acknowledgment');ack=json.loads(raw)
  need(manifest['schema']=='LocallyGeneratedCompleteCountingManifestV3'and manifest['terminal_exit']==0 and manifest['deserialization_authorized']is False,'invalid captured manifest')
  need(set(manifest['files'])==e.OUTPUT_FILES and set(ack['supervisor_files'])==e.CONTROL_FILES,'captured manifest roster mismatch')
  for name,pin in manifest['files'].items():
   b=files['output/'+name];need(sha(b)==pin['sha256']and len(b)==pin['bytes'],'captured output mismatch: '+name)
  for name,pin in ack['supervisor_files'].items():
   b=files['supervisor/'+name];need(sha(b)==pin['sha256']and len(b)==pin['bytes'],'captured supervisor mismatch: '+name)
  reap=json.loads(files['supervisor/terminal.json']);launch=json.loads(files['supervisor/launch.json']);result=json.loads(files['supervisor/controller-result.json']);worker=json.loads(files['output/terminal.json'])
  need(ack['schema']=='ReapedCountingControllerSuccessV3'and ack['output_root']==str(root)and ack['manifest_sha256']==context['manifest_sha'],'captured receipt/root mismatch')
  need(ack['controller_reaped']and ack['controller_returncode']==0 and reap['controller_reaped']and reap['controller_returncode']==0 and reap['controller_pid']==launch['controller_pid']==ack['controller_pid'],'captured controller reap identity mismatch')
  need(ack['source_sha256']==reap['source_sha256']==launch['source_sha256']==manifest['source_sha256']==self.engine_sha,'captured child source identity mismatch')
  need(ack['request_sha256']==reap['request_sha256']==launch['request_sha256']==manifest['request_sha256']==sha(files['output/request.json']),'captured request identity mismatch')
  need(result['complete']and result['manifest_pin']['sha256']==context['manifest_sha']and worker['returncode']==0 and worker['cleanup']['verified']and worker['cleanup']['reaped'],'captured result/worker cleanup mismatch')
  need(sha(files['output/source-snapshot.py'])==self.engine_sha,'captured source snapshot mismatch')
  # Return only immutable byte copies. No later reopen supplies interpreted data.
  return files,manifest,dict(root_identity=rid,control_identity=cid,controller_pid=ack['controller_pid'],source_sha256=self.engine_sha)
 def execute(self,root,request,test_fault=None):
  self._check_sources();need(threading.active_count()==1,'start invented controller before creating reader/writer threads')
  root=Path(root).resolve();need(root.is_relative_to(HERE)and root!=HERE,'new packet-local fixture root required')
  with lock(root,True)as lid:
   original=self.engine.atomic_json
   def uncertain_success(path,value):
    path=Path(path)
    if test_fault=='supervisor_post_rename_fsync'and path.name=='success.json':
     old_fsync=os.fsync
     def fail_dir(fd):
      if stat.S_ISDIR(os.fstat(fd).st_mode)and path.exists():raise OSError(5,'invented supervisor-success directory fsync failure after rename')
      return old_fsync(fd)
     os.fsync=fail_dir
     try:return original(path,value)
     finally:os.fsync=old_fsync
    return original(path,value)
   self.engine.atomic_json=uncertain_success
   try:result=self.engine.supervise(root,request,self.engine.sha(self.engine.canonical(request)+b'\n'),self.engine_sha)
   finally:self.engine.atomic_json=original
   self.last_attempt=result;need(result['complete'],'invented count/controller did not complete')
   context=dict(root=str(root),control=result['supervisor_directory'],manifest_sha=result['manifest_pin']['sha256'],receipt_sha=result['supervisor_receipt_pin']['sha256'])
   files,manifest,provenance=self._capture(context);body=dict(issuer=self._issuer,issuer_pid=self._owner_pid,nonce=secrets.token_hex(16),engine_sha=self.engine_sha,session_sha=self.session_sha,lock_identity=lid,**context,root_identity=provenance['root_identity'],control_identity=provenance['control_identity']);mac=hmac.new(self._secret,canonical(body),'sha256').hexdigest();ack=_Acknowledgment(**body,mac=mac)
   self._issued[id(ack)]=(ack,canonical(body));return ack
 def _authenticate(self,ack):
  self._check_sources();need(type(ack)is _Acknowledgment and id(ack)in self._issued and self._issued[id(ack)][0]is ack,'original caller-retained acknowledgment object required; files/digests/copies/restarts cannot reconstruct it')
  body=dataclasses.asdict(ack);mac=body.pop('mac');need(canonical(body)==self._issued[id(ack)][1]and hmac.compare_digest(mac,hmac.new(self._secret,canonical(body),'sha256').hexdigest()),'caller acknowledgment altered')
  return body
 def verify(self,ack,test_hook=None):
  body=self._authenticate(ack)
  with lock(body['root'],False)as lid:
   need(lid==tuple(body['lock_identity']),'cooperating lock identity changed')
   if test_hook:test_hook('lock_acquired','')
   files,manifest,provenance=self._capture(body,test_hook)
  return VerifiedSnapshot(MappingProxyType(files),freeze(manifest),freeze(dict(provenance,issuer=self._issuer,issuer_pid=self._owner_pid,acknowledgment_nonce=ack.nonce)))
 def write_fixture_metadata(self,ack,name,data,attempted=None,entered=None):
  body=self._authenticate(ack);need(name in self.engine.OUTPUT_FILES,'only invented fixture artifacts can be changed')
  if attempted:attempted.set()
  with lock(body['root'],True)as lid:
   need(lid==tuple(body['lock_identity']),'lock identity changed')
   if entered:entered.set()
   (Path(body['root'])/name).write_bytes(data)
