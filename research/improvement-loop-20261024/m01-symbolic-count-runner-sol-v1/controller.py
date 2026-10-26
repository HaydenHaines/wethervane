"""One source-bound child; ephemeral observed acknowledgment, never marker adoption.
Actual mode requires technical source review and fresh resources, not user approval.
"""
import dataclasses,fcntl,hashlib,json,os,signal,subprocess,sys,threading,time,uuid
from pathlib import Path
from types import MappingProxyType
import common
from common import Writer,checked,need,pin,read_regular
HERE=Path(__file__).resolve().parent
def freeze(value):
 if isinstance(value,dict):return MappingProxyType({k:freeze(v)for k,v in value.items()})
 if isinstance(value,list):return tuple(freeze(v)for v in value)
 return value

FILES=('controller.py','worker.py','pipeline.py','common.py','count_core.py')
ACTUAL_CAPS=dict(wall_seconds=120,cpu_soft=90,cpu_hard=95,address_space=6*1024**3,rss_watchdog=2*1024**3,individual_file=1024**2,output_watchdog=16*1024**2,poll_seconds=.25)
TOY_CAPS=dict(wall_seconds=4,cpu_soft=2,cpu_hard=3,address_space=256*1024**2,rss_watchdog=128*1024**2,individual_file=1024**2,output_watchdog=16*1024**2,poll_seconds=.02)

def sources():
 need(Path(common.__file__).resolve()==HERE/'common.py','wrong imported metadata helper')
 return [pin(HERE/name)for name in FILES]
def actual_gate(plan_pin,technical_pin):
 plan=json.loads(checked(plan_pin));technical=json.loads(checked(technical_pin))
 need(plan['schema']=='ActualTXSymbolicCountRunnerPlanV1'and plan['runner_sources']==sources(),'technical plan/source mismatch')
 need(plan['caps']==ACTUAL_CAPS,'fixed actual resource envelope required')
 need(type(plan['count_budget_bytes'])is int and 0<plan['count_budget_bytes']<=512*1024**2,'logical stream cap must be at most512MiB')
 need(technical['schema']=='SingleSymbolicCountTechnicalRecordV1'and technical['plan_sha256']==plan_pin['sha256'],'technical execution record mismatch')
 need(technical['operations']==['symbolic_graph','count_one_protocol5_stream']and technical['numerical_compile']is False and technical['sampler']is False and technical['forecast']is False,'symbolic/count scope only')
 review=json.loads(checked(technical['independent_source_review']))
 need(review['verdict']=='source_semantics_pass_for_symbolic_count_only'and review['runner_sources']==plan['runner_sources']and review['plan_sha256']==plan_pin['sha256'],'independent source/semantics review required')
 observed=json.loads(checked(technical['fresh_resource_record']));age=time.time()-observed['observed_unix']
 need(0<=age<=120 and observed['available_memory_bytes']>=8*1024**3 and observed['available_disk_bytes']>=1024**3,'fresh resource headroom required')
 need(observed['scope_directory']==str(HERE)and observed['cpu_threads_permitted']==1 and observed['nice']==15,'resource scope/priority mismatch')
 need(Path(sys.prefix).resolve()==Path(plan['target_environment']).resolve(),'actual target environment differs')
 for item in plan['source_pins']:checked(item)
 return plan,technical

@dataclasses.dataclass(frozen=True,eq=False)
class AcknowledgedSnapshot:
 caller_pid:int
 nonce:str
 files:object
 observation:object

class Controller:
 def __init__(self,expected_sources):
  self.expected=expected_sources;self.pid=os.getpid();self.issued={};self._check()
 def _check(self):need(os.getpid()==self.pid and sources()==self.expected,'caller/source identity changed')
 def verify_retained(self,token):
  self._check();need(type(token)is AcknowledgedSnapshot and self.issued.get(id(token))is token,'original ephemeral caller acknowledgment required')
  return token
 def fixture(self,root,payload_bytes=1024,fault='none'):
  need(type(payload_bytes)is int and 0<=payload_bytes<=65536,'tiny invented fixture only')
  need(fault in ('none','cap','raise','hold','terminal_fsync'),'unknown invented fault')
  return self._run(root,dict(mode='fixture',payload_bytes=payload_bytes,fault=fault,caps=TOY_CAPS))
 def actual(self,root,plan_pin,technical_pin):
  plan,_=actual_gate(plan_pin,technical_pin)
  root=Path(root).resolve();need(root.is_relative_to(HERE)and root!=HERE,'fresh packet-local actual attempt required')
  return self._run(root,dict(mode='actual',plan=plan_pin,technical=technical_pin,caps=plan['caps']))
 def _run(self,root,request):
  self._check();need(threading.active_count()==1,'start only before threads')
  root=Path(root).resolve();need(not root.exists(),'fresh exclusive attempt required');caps=request['caps']
  fd=os.open(str(root)+'.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600);fcntl.flock(fd,fcntl.LOCK_EX)
  owned=False;child=None;pidfd=None;reason=None;pending=None;peaks=dict(rss=0,output=0,threads=0);start=time.monotonic();old_handlers={}
  try:
   need(not root.exists(),'attempt was created by another cooperating controller')
   root.mkdir();owned=True;(root/'receipts').mkdir();(root/'cache').mkdir();writer=Writer(root,root)
   intent=dict(schema='SymbolicCountIntentV1',controller_pid=self.pid,runner_sources=self.expected,request=request)
   intent_pin=writer.json('intent.json',intent)
   env={'PYTHONDONTWRITEBYTECODE':'1','NUMBA_CACHE_DIR':str(root/'cache/numba'),'MPLCONFIGDIR':str(root/'cache/matplotlib'),'XDG_CACHE_HOME':str(root/'cache/xdg'),
        'PYTENSOR_FLAGS':'cxx=,floatX=float64,numba__fastmath=False,compiledir='+str(root/'cache/pytensor')}
   for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):env[name]='1'
   command=[sys.executable,'-B',str(HERE/'worker.py'),str(root),intent_pin['sha256']]
   def interrupted(signum,frame):raise InterruptedError('controller signal '+str(signum))
   for sig in (signal.SIGTERM,signal.SIGINT):old_handlers[sig]=signal.signal(sig,interrupted)
   try:
    with (root/'process.log').open('xb')as log:
     # Catchable signals cannot land before the new child handle is retained.
     old_mask=signal.pthread_sigmask(signal.SIG_BLOCK,set(old_handlers))
     try:
      child=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,restore_signals=True)
      pidfd=os.pidfd_open(child.pid)
     finally:signal.pthread_sigmask(signal.SIG_SETMASK,old_mask)
     writer.json('launch.json',dict(schema='ObservedCountChildLaunchV1',pid=child.pid,command=command,intent=intent_pin,controller_pid=self.pid))
     while child.poll()is None:
      try:
       values={line.split(':',1)[0]:line.split(':',1)[1].strip()for line in Path(f'/proc/{child.pid}/status').read_text().splitlines()if ':'in line}
       peaks['rss']=max(peaks['rss'],int(values.get('VmRSS','0 kB').split()[0])*1024);peaks['threads']=max(peaks['threads'],int(values.get('Threads','0')))
      except FileNotFoundError:pass
      peaks['output']=max(peaks['output'],sum(p.stat().st_size for p in root.rglob('*')if p.is_file()))
      if time.monotonic()-start>caps['wall_seconds']:reason='wall_watchdog'
      elif peaks['rss']>caps['rss_watchdog']:reason='rss_watchdog'
      elif peaks['output']>caps['output_watchdog']:reason='output_watchdog'
      if reason:break
      time.sleep(caps['poll_seconds'])
   except BaseException as error:pending=error;reason='controller_exception'
   finally:
    for sig in old_handlers:signal.signal(sig,signal.SIG_IGN)
    try:
     if child is not None:
      if child.poll()is None:
       try:signal.pidfd_send_signal(pidfd,signal.SIGTERM)if pidfd is not None else child.terminate()
       except ProcessLookupError:pass
       try:child.wait(timeout=2)
       except subprocess.TimeoutExpired:
        try:signal.pidfd_send_signal(pidfd,signal.SIGKILL)if pidfd is not None else child.kill()
        except ProcessLookupError:pass
        child.wait(timeout=2)
      child.wait(timeout=2)
    finally:
     if pidfd is not None:os.close(pidfd)
     for sig,handler in old_handlers.items():signal.signal(sig,handler)
   terminal=dict(schema='ObservedCountChildTerminalV1',child_pid=child.pid if child else None,returncode=child.returncode if child else None,
     controller_pid=self.pid,reaped=child is not None and child.returncode is not None,reason=reason,peaks=peaks,wall_seconds=time.monotonic()-start,
     scope='direct child only; SIGKILL/power loss of controller and daemon escape not covered',intent_sha256=intent_pin['sha256'])
   Writer(root,root,fail_fsync=request.get('fault')=='terminal_fsync').json('terminal.json',terminal)
   if pending is not None:raise pending
   need(child is not None and child.returncode==0 and reason is None,'count worker failed; no acknowledgment')
   allowed={'intent.json','launch.json','process.log','terminal.json','cache','receipts'}
   need({p.name for p in root.iterdir()}==allowed,'unexpected controller output roster')
   expected_reports={'progress.json','result.json','worker-terminal.json'}
   if request['mode']=='actual':expected_reports|={'stages.json','first-parent-contract.json','factor-inventory.json','coordinate-order.json'}
   need({p.name for p in (root/'receipts').iterdir()}==expected_reports,'unexpected report roster')
   captured={};total=0
   for path in sorted(root.rglob('*')):
    if path.is_dir():continue
    need(path.is_relative_to(root/'cache')is False or path.name not in ('result.json','worker-terminal.json'),'unexpected cache evidence')
    if path.is_relative_to(root/'cache'):continue # not offered as verified scientific output
    need(not path.name.endswith('.pending'),'ambiguous metadata publication')
    size=path.stat().st_size;need(total+size<=8*1024**2,'snapshot budget before capture');raw=read_regular(path);total+=len(raw);captured[str(path.relative_to(root))]=raw
   worker=json.loads(captured['receipts/worker-terminal.json']);result=json.loads(captured['receipts/result.json'])
   need(worker['returncode']==0 and worker['intent_sha256']==result['intent_sha256']==intent_pin['sha256']and worker['runner_sources']==self.expected,'worker/source projection mismatch')
   need(result['pickle_payload_persisted']is False and result['pickle_loaded']is False,'unexpected payload persistence')
   token=AcknowledgedSnapshot(self.pid,uuid.uuid4().hex,MappingProxyType(captured),freeze(terminal));self.issued[id(token)]=token;return token
  except BaseException as error:
   if owned and root.exists():
    try:Writer(root,root).json('controller-failure.json',dict(error=repr(error),child_returncode=child.returncode if child else None,no_acknowledgment=True))
    except BaseException:pass # missing failure evidence never authorizes completion
   raise
  finally:os.close(fd)

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--fixture',action='store_true');parser.add_argument('--payload-bytes',type=int,default=1024);parser.add_argument('--fault',default='none');parser.add_argument('--plan-sha');parser.add_argument('--technical-record');parser.add_argument('--technical-sha');args=parser.parse_args()
 if args.fixture:token=Controller(sources()).fixture(args.output,args.payload_bytes,args.fault)
 else:
  need(args.plan_sha and args.technical_record and args.technical_sha,'technical plan/review/resource record required')
  plan_ref=dict(path=str(HERE/'plan.json'),sha256=args.plan_sha);plan=json.loads(checked(plan_ref))
  token=Controller(plan['runner_sources']).actual(args.output,plan_ref,dict(path=args.technical_record,sha256=args.technical_sha))
 print(json.dumps(dict(status='ephemeral_caller_observed_completion',caller_pid=token.caller_pid,nonce=token.nonce,result=json.loads(token.files['receipts/result.json']),durable_restart_supported=False)))
