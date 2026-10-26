"""V3 invented-object counting ONLY. Never deserializes or writes serialized payload.

Caller-selected SHA expectations provide local byte identity, never permission to
load objects. All state lives in a new exclusive output directory.
"""
import argparse,hashlib,json,os,pickle,resource,signal,subprocess,sys,time,traceback,uuid,multiprocessing as mp
from pathlib import Path
SOURCE=Path(__file__).resolve()
OUTPUT_FILES=frozenset({'request.json','source-snapshot.py','process.log','progress.json','count-candidate.json','launch.json','terminal.json'})
CONTROL_FILES=frozenset({'launch.json','controller.log','controller-result.json','terminal.json'})
_SEAL_FAULT=None
_SEAL_FAULT_FIRED=False
CAPS=dict(wall_seconds=10,cpu_soft_seconds=2,cpu_hard_seconds=3,address_space_bytes=256*1024**2,individual_file_bytes=1024**2,rss_watchdog_bytes=128*1024**2,output_watchdog_bytes=4*1024**2,poll_seconds=.05)
def require(ok,message):
 if not ok:raise ValueError(message)
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def pin(path):
 b=Path(path).read_bytes();return dict(sha256=sha(b),bytes=len(b))
def atomic_json(path,value):
 path=Path(path)
 if path.name=='seal-failure.json'and _SEAL_FAULT=='seal_fsync_and_failure_write_fail':raise OSError(13,'injected filesystem denial of seal-failure evidence')
 temp=path.with_name(path.name+'.partial-'+uuid.uuid4().hex)
 with temp.open('xb')as stream:stream.write(canonical(value)+b'\n');stream.flush();os.fsync(stream.fileno())
 os.replace(temp,path)
 descriptor=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
 try:
  global _SEAL_FAULT_FIRED
  if path.name=='complete.json'and _SEAL_FAULT in ['seal_fsync_fail','seal_fsync_and_failure_write_fail']and not _SEAL_FAULT_FIRED:
   _SEAL_FAULT_FIRED=True;raise OSError(5,'injected directory fsync failure after complete rename')
  os.fsync(descriptor)
 finally:os.close(descriptor)
class BudgetExceeded(ValueError):pass
class CountingSink:
 def __init__(self,budget,observe=lambda state:None):
  require(type(budget)is int and budget>=0,'nonnegative integer budget required');self.budget=budget;self.accepted=0;self.attempted_end=0;self.calls=0;self.h=hashlib.sha256();self.observe=observe
 def snapshot(self):return dict(accepted_bytes=self.accepted,last_attempted_end=self.attempted_end,write_calls=self.calls,accepted_prefix_sha256=self.h.hexdigest(),budget_bytes=self.budget)
 def write(self,data):
  view=memoryview(data).cast('B');size=view.nbytes;self.attempted_end=self.accepted+size;self.calls+=1
  if self.attempted_end>self.budget:
   self.observe(self.snapshot());raise BudgetExceeded('write would exceed logical counting budget')
  for offset in range(0,size,64*1024):self.h.update(view[offset:offset+64*1024])
  self.accepted+=size;self.observe(self.snapshot());return size
 def flush(self):pass

def validate_request(request):
 require(type(request)is dict and set(request)=={'schema','fixture','payload_bytes','budget_bytes','fault'},'exact request fields required')
 require(request['schema']=='InventedCountingRequestV1'and request['fixture']=='nested_v1','invented fixture only; no graphs/files/object inputs')
 require(type(request['payload_bytes'])is int and 0<=request['payload_bytes']<=65536,'payload at most64KiB')
 require(type(request['budget_bytes'])is int and 0<=request['budget_bytes']<=1024**2,'logical budget at most1MiB')
 require(request['fault']in ['none','raise_after_write','exit_after_metadata','cpu_before_count','seal_fsync_fail','seal_fsync_and_failure_write_fail'],'unknown fault')
def invented(request):
 validate_request(request)
 return dict(schema='InventedCountingFixtureV1',items=[None,True,7,['a',3.25]],payload=bytes(i%251 for i in range(request['payload_bytes'])))
def worker(root,request_expected,source_expected):
 root=Path(root);request_raw=(root/'request.json').read_bytes();require(sha(request_raw)==request_expected,'worker request bytes changed');require(pin(SOURCE)['sha256']==source_expected,'worker source bytes changed');request=json.loads(request_raw);validate_request(request)
 state=dict(schema='CountingWorkerProgressV1',status='started',request_sha256=request_expected,source_sha256=source_expected,limits=dict(cpu=resource.getrlimit(resource.RLIMIT_CPU),address_space=resource.getrlimit(resource.RLIMIT_AS),file=resource.getrlimit(resource.RLIMIT_FSIZE)),nice=os.getpriority(os.PRIO_PROCESS,0))
 atomic_json(root/'progress.json',state)
 try:
  if request['fault']=='cpu_before_count':
   while True:pass
  def observe(snapshot):
   atomic_json(root/'progress.json',dict(state,status='counting',counter=snapshot))
   if request['fault']=='raise_after_write':raise RuntimeError('invented interruption after progress persisted')
  sink=CountingSink(request['budget_bytes'],observe)
  # Only deterministic invented fixture is accepted. No payload file is created.
  pickle.Pickler(sink,protocol=5).dump(invented(request))
  metadata=dict(schema='CompleteCountCandidateV3',status='pickler_returned',counter=sink.snapshot(),protocol=5,request_sha256=request_expected,source_sha256=source_expected,serialized_payload_persisted=False,deserialization_authorized=False,python=sys.version,executable=sys.executable)
  atomic_json(root/'count-candidate.json',metadata)
  if request['fault']=='exit_after_metadata':os._exit(73)
 except BaseException:
  atomic_json(root/'failure.json',dict(schema='CountingWorkerFailureV1',error=traceback.format_exc(),request_sha256=request_expected,source_sha256=source_expected));raise

class ControllerInterrupted(BaseException):
 def __init__(self,signum):self.signum=signum;super().__init__('controller interrupted by signal '+str(signum))

def run(root,request,request_expected,source_expected):
 require(pin(SOURCE)['sha256']==source_expected,'caller-selected source SHA mismatch');validate_request(request);raw=canonical(request)+b'\n';require(sha(raw)==request_expected,'caller-selected request SHA mismatch')
 root=Path(root).resolve();root.mkdir(exist_ok=False);(root/'request.json').write_bytes(raw);(root/'source-snapshot.py').write_bytes(SOURCE.read_bytes());require(pin(root/'source-snapshot.py')['sha256']==source_expected,'snapshot mismatch')
 def setup():
  os.setpriority(os.PRIO_PROCESS,0,15);resource.setrlimit(resource.RLIMIT_CPU,(2,3));resource.setrlimit(resource.RLIMIT_AS,(CAPS['address_space_bytes'],CAPS['address_space_bytes']));resource.setrlimit(resource.RLIMIT_FSIZE,(CAPS['individual_file_bytes'],CAPS['individual_file_bytes']));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
  if spawn_old_mask is not None:signal.pthread_sigmask(signal.SIG_SETMASK,spawn_old_mask)
 env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
 for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS']:env[key]='1'
 start=time.monotonic();rss=threads=output_peak=0;reason=None;child=None;pidfd=None;pending_error=None;pending_traceback=None;terminal_write_error=None;spawn_old_mask=None
 previous={sig:signal.getsignal(sig)for sig in [signal.SIGINT,signal.SIGTERM]}
 def interrupted(signum,frame):raise ControllerInterrupted(signum)
 for sig in previous:signal.signal(sig,interrupted)
 cleanup=dict(scope='dedicated direct child only; invented worker never spawns descendants',needed=False,signals=[],reaped=False,verified=False)
 try:
  with (root/'process.log').open('xb')as log:
   # Defer catchable interrupts until the parent has assigned its child handle.
   # The child restores the inherited old mask in setup before exec.
   spawn_old_mask=signal.pthread_sigmask(signal.SIG_BLOCK,set(previous))
   try:
    child=subprocess.Popen([sys.executable,'-B',str(SOURCE),'--worker',str(root),'--request-sha',request_expected,'--source-sha',source_expected],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,preexec_fn=setup)
    # Own unreaped direct child: pidfd cannot attach a reused PID while unreaped.
    pidfd=os.pidfd_open(child.pid)
   finally:signal.pthread_sigmask(signal.SIG_SETMASK,spawn_old_mask)
   atomic_json(root/'launch.json',dict(pid=child.pid,caps=CAPS,source_sha256=source_expected,request_sha256=request_expected,command=child.args))
   while child.poll()is None:
    try:
     data={x.split(':',1)[0]:x.split(':',1)[1].strip()for x in Path('/proc/'+str(child.pid)+'/status').read_text().splitlines()if ':'in x};rss=max(rss,int(data.get('VmRSS','0 kB').split()[0])*1024);threads=max(threads,int(data.get('Threads','0')))
    except FileNotFoundError:pass
    size=sum(p.stat().st_size for p in root.iterdir()if p.is_file());output_peak=max(output_peak,size)
    if time.monotonic()-start>CAPS['wall_seconds']:reason='wall_timeout'
    elif rss>CAPS['rss_watchdog_bytes']:reason='rss_watchdog_cap'
    elif size>CAPS['output_watchdog_bytes']:reason='disk_watchdog_cap'
    if reason:break
    time.sleep(CAPS['poll_seconds'])
 except BaseException as error:
  pending_error=error;pending_traceback=error.__traceback__;reason='controller_interruption'if isinstance(error,ControllerInterrupted)else'controller_exception'
 finally:
  # Catchable interruption must not interrupt the bounded teardown itself.
  # Repeated INT/TERM are temporarily ignored; SIGKILL/power loss cannot be caught.
  for sig in previous:signal.signal(sig,signal.SIG_IGN)
  try:
   if child is not None:
    try:
     if child.poll()is None:
      cleanup['needed']=True
      try:
       if pidfd is not None:signal.pidfd_send_signal(pidfd,signal.SIGTERM)
       else:child.terminate() # only Popen's own still-unreaped child
      except ProcessLookupError:pass # child raced to exit; still reap it below
      cleanup['signals'].append('SIGTERM')
      try:child.wait(timeout=1)
      except subprocess.TimeoutExpired:
       try:
        if pidfd is not None:signal.pidfd_send_signal(pidfd,signal.SIGKILL)
        else:child.kill()
       except ProcessLookupError:pass
       cleanup['signals'].append('SIGKILL');child.wait(timeout=1)
     code=child.wait(timeout=1);cleanup['reaped']=True;cleanup['verified']=child.poll()is not None
    except BaseException as cleanup_error:
     cleanup['error']=repr(cleanup_error);code=child.returncode
    if code==-signal.SIGXCPU and reason is None:reason='cpu_soft_limit_sigxcpu'
    terminal=dict(schema='CountingWorkerTerminalV3',pid=child.pid,returncode=code,termination_reason=reason,wall_seconds=time.monotonic()-start,peak_observed_rss=rss,peak_threads=threads,peak_output_bytes=output_peak,caps=CAPS,cleanup=cleanup,controller_error=None if pending_error is None else dict(type=type(pending_error).__name__,message=str(pending_error),signal=getattr(pending_error,'signum',None)))
    try:atomic_json(root/'terminal.json',terminal)
    except BaseException as error:
     terminal_write_error=error
     # Durable storage may itself be broken. Preserve a best-effort terminal on
     # the controller's inherited stderr, and never publish completion.
     os.write(2,b'COUNTING_TERMINAL_FALLBACK '+canonical(terminal)+b'\n')
   else:code=None;terminal=None
  finally:
   if pidfd is not None:os.close(pidfd)
   for sig,handler in previous.items():signal.signal(sig,handler)
 if child is None:
  if pending_error is not None:raise pending_error.with_traceback(pending_traceback)
  raise RuntimeError('no child spawned')
 require(cleanup['verified'],'owned-child cleanup unresolved; completion forbidden')
 if pending_error is not None:raise pending_error.with_traceback(pending_traceback)
 if terminal_write_error is not None:raise terminal_write_error
 if code!=0:return dict(status='failed_preserved',terminal=terminal,root=str(root),complete=False)
 try:
  require(pin(SOURCE)['sha256']==source_expected,'source changed during run');require(sha((root/'request.json').read_bytes())==request_expected,'request changed during run')
  metadata=json.loads((root/'count-candidate.json').read_text());require(metadata['schema']=='CompleteCountCandidateV3'and metadata['status']=='pickler_returned'and metadata['request_sha256']==request_expected and metadata['source_sha256']==source_expected,'candidate identity mismatch')
  require(metadata['counter']['accepted_bytes']<=request['budget_bytes'],'candidate exceeds budget');require(metadata['serialized_payload_persisted']is False and metadata['deserialization_authorized']is False,'count-only scope changed')
  require({p.name for p in root.iterdir()}==OUTPUT_FILES,'ambiguous pre-seal artifact roster')
  files={name:pin(root/name)for name in OUTPUT_FILES}
  manifest=dict(schema='LocallyGeneratedCompleteCountingManifestV3',status='complete_count_only_no_payload',files=files,request_sha256=request_expected,source_sha256=source_expected,terminal_exit=0,deserialization_authorized=False,trust='trusted local generation and caller-selected source/request expectations; digest is byte identity, not loading authorization')
  global _SEAL_FAULT,_SEAL_FAULT_FIRED
  previous_seal_fault=_SEAL_FAULT;_SEAL_FAULT=request['fault'];_SEAL_FAULT_FIRED=False
  try:atomic_json(root/'complete.json',manifest)
  finally:
   # Keep the denial fault active through the failure handler below.
   if not _SEAL_FAULT_FIRED:_SEAL_FAULT=previous_seal_fault
  return dict(status=manifest['status'],root=str(root),complete=True,manifest_pin=pin(root/'complete.json'),counter=metadata['counter'],terminal=terminal)
 except BaseException as error:
  original_traceback=error.__traceback__
  try:atomic_json(root/'seal-failure.json',dict(error=traceback.format_exc()))
  except BaseException as reporting_error:os.write(2,b'SEAL_FAILURE_UNWRITABLE '+repr(reporting_error).encode()+b'\n')
  finally:_SEAL_FAULT=None
  raise error.with_traceback(original_traceback)

def verify_local_files_only(root,expected_manifest_sha):
 """File-consistency check only; MUST NOT authorize completion by itself."""
 root=Path(root).resolve();require(not (root/'seal-failure.json').exists(),'seal-failure state is never complete')
 require({p.name for p in root.iterdir()}==OUTPUT_FILES|{'complete.json'},'ambiguous/incomplete exact artifact roster')
 raw=(root/'complete.json').read_bytes();require(sha(raw)==expected_manifest_sha,'externally selected manifest SHA required');manifest=json.loads(raw)
 require(manifest['schema']=='LocallyGeneratedCompleteCountingManifestV3'and manifest['status']=='complete_count_only_no_payload'and manifest['terminal_exit']==0 and manifest['deserialization_authorized']is False,'unsupported local manifest')
 require(set(manifest['files'])==OUTPUT_FILES,'exact manifest file roster required')
 for name,expected in manifest['files'].items():
  require(Path(name).name==name,'unsafe manifest path');path=root/name;require(not path.is_symlink()and path.resolve().parent==root,'unsafe file alias');require(pin(path)==expected,'output bytes changed: '+name)
 terminal=json.loads((root/'terminal.json').read_text());require(terminal['schema']=='CountingWorkerTerminalV3'and terminal['returncode']==0 and terminal['cleanup']['verified']and terminal['cleanup']['reaped']and terminal['controller_error']is None,'unverified worker cleanup')
 require(pin(root/'source-snapshot.py')['sha256']==manifest['source_sha256']and sha((root/'request.json').read_bytes())==manifest['request_sha256'],'source/request identity projection')
 require({p.name for p in root.iterdir()}==OUTPUT_FILES|{'complete.json'},'artifact roster changed while checking')
 return manifest

def verify_complete(root,expected_manifest_sha,supervisor_receipt=None,expected_supervisor_sha=None):
 require(supervisor_receipt is not None and expected_supervisor_sha is not None,'external reaped-controller success receipt required; marker/hash alone insufficient')
 require(not Path(supervisor_receipt).is_symlink(),'external receipt symlink unsupported')
 receipt_path=Path(supervisor_receipt).resolve();root=Path(root).resolve();require(not receipt_path.is_relative_to(root),'supervisor evidence must be outside counted output')
 raw=receipt_path.read_bytes();require(sha(raw)==expected_supervisor_sha,'caller-retained supervisor receipt hash required');ack=json.loads(raw)
 require(ack['schema']=='ReapedCountingControllerSuccessV3'and ack['controller_reaped']is True and ack['controller_returncode']==0 and ack['output_root']==str(root)and ack['manifest_sha256']==expected_manifest_sha,'controller success/identity mismatch')
 require({p.name for p in receipt_path.parent.iterdir()}==CONTROL_FILES|{'success.json'},'ambiguous external supervisor artifact roster')
 require(set(ack['supervisor_files'])==CONTROL_FILES,'exact external receipt file roster required')
 for name,expected in ack['supervisor_files'].items():
  path=receipt_path.parent/name;require(not path.is_symlink()and pin(path)==expected,'supervisor evidence changed: '+name)
 reap=json.loads((receipt_path.parent/'terminal.json').read_text());launch=json.loads((receipt_path.parent/'launch.json').read_text());result=json.loads((receipt_path.parent/'controller-result.json').read_text())
 require(reap['controller_reaped']is True and reap['controller_returncode']==0 and reap['controller_pid']==launch['controller_pid']==ack['controller_pid'],'external actual reap/launch identity mismatch')
 require(reap['source_sha256']==launch['source_sha256']==ack['source_sha256']and reap['request_sha256']==launch['request_sha256']==ack['request_sha256'],'external child source/request identity mismatch')
 require(result['complete']is True and result['manifest_pin']['sha256']==expected_manifest_sha,'controller return did not report this completed local manifest')
 manifest=verify_local_files_only(root,expected_manifest_sha)
 require(ack['source_sha256']==manifest['source_sha256']and ack['request_sha256']==manifest['request_sha256'],'supervisor source/request identity mismatch')
 return manifest

def _controller_entry(root,control,request,request_expected,source_expected):
 signal.pthread_sigmask(signal.SIG_SETMASK,set())
 # Separate controller process. Its actual zero exit, not a directory marker,
 # is required by the waiting outer supervisor before an external success ack.
 try:
  with (Path(control)/'controller.log').open('ab',buffering=0)as log:
   os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
   result=run(root,request,request_expected,source_expected)
   atomic_json(Path(control)/'controller-result.json',result)
 except BaseException:
  try:atomic_json(Path(control)/'controller-error.json',dict(error=traceback.format_exc()))
  finally:raise SystemExit(80)

def supervise(root,request,request_expected,source_expected):
 """Trusted-local outer supervisor; returns an ack only after reaped zero exit.

 The returned ack pin must be retained by the caller, never recomputed from a
 potentially ambiguous directory to infer success. Not malicious-rewrite proof.
 """
 require(pin(SOURCE)['sha256']==source_expected,'source SHA mismatch');validate_request(request);require(sha(canonical(request)+b'\n')==request_expected,'request SHA mismatch')
 root=Path(root).resolve();require(not root.exists(),'fresh counted output required');control=root.with_name(root.name+'.supervisor');control.mkdir(exist_ok=False)
 controller=mp.get_context('fork').Process(target=_controller_entry,args=(root,control,request,request_expected,source_expected));started=False;error=None;start=time.monotonic();reason=None
 previous={sig:signal.getsignal(sig)for sig in [signal.SIGINT,signal.SIGTERM]}
 def interrupted(signum,frame):raise ControllerInterrupted(signum)
 for sig in previous:signal.signal(sig,interrupted)
 try:
  old_mask=signal.pthread_sigmask(signal.SIG_BLOCK,set(previous))
  try:controller.start();started=True
  finally:signal.pthread_sigmask(signal.SIG_SETMASK,old_mask)
  # Forked controller must not retain the supervisor's blocked signal mask.
  # _controller_entry restores it before invoking run (see below).
  atomic_json(control/'launch.json',dict(controller_pid=controller.pid,source_sha256=source_expected,request_sha256=request_expected,root=str(root)))
  controller.join(15)
  if controller.is_alive():reason='controller_wall_timeout'
 except BaseException as exc:error=exc;reason='supervisor_interruption_or_exception'
 finally:
  for sig in previous:signal.signal(sig,signal.SIG_IGN)
  try:
   if started and controller.is_alive():
    controller.terminate();controller.join(3)
    if controller.is_alive():controller.kill();controller.join(1)
   if started:controller.join(1)
   terminal=dict(schema='CountingControllerReapV3',controller_pid=controller.pid if started else None,controller_returncode=controller.exitcode if started else None,controller_reaped=started and not controller.is_alive(),reason=reason,wall_seconds=time.monotonic()-start,source_sha256=source_expected,request_sha256=request_expected)
   atomic_json(control/'terminal.json',terminal)
  finally:
   for sig,handler in previous.items():signal.signal(sig,handler)
 if error is not None:raise error
 result_path=control/'controller-result.json'
 if controller.exitcode!=0 or not result_path.exists():
  return dict(status='controller_failed_no_external_success',complete=False,root=str(root),controller_terminal=terminal,terminal=json.loads((root/'terminal.json').read_text())if(root/'terminal.json').exists()else None,supervisor_directory=str(control))
 result=json.loads(result_path.read_text());result['controller_terminal']=terminal;result['supervisor_directory']=str(control)
 if not result['complete']:return result
 local=verify_local_files_only(root,result['manifest_pin']['sha256'])
 require({p.name for p in control.iterdir()}==CONTROL_FILES,'ambiguous supervisor state before success publication')
 ack=dict(schema='ReapedCountingControllerSuccessV3',output_root=str(root),controller_pid=controller.pid,controller_returncode=controller.exitcode,controller_reaped=not controller.is_alive(),manifest_sha256=result['manifest_pin']['sha256'],source_sha256=source_expected,request_sha256=request_expected,controller_result_sha256=pin(result_path)['sha256'],supervisor_files={name:pin(control/name)for name in CONTROL_FILES},deserialization_authorized=False,trust='actual locally waited controller plus caller-retained successful return; not a self-authored digest alone, not coherent-rewrite or power-loss protection')
 receipt=control/'success.json';atomic_json(receipt,ack)
 result['supervisor_receipt']=str(receipt);result['supervisor_receipt_pin']=pin(receipt);return result

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--worker',required=True);parser.add_argument('--request-sha',required=True);parser.add_argument('--source-sha',required=True);a=parser.parse_args();worker(a.worker,a.request_sha,a.source_sha)
