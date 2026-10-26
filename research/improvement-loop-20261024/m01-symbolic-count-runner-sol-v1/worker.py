"""Bounded direct child. Scientific imports occur only after actual technical gate."""
import hashlib,json,os,resource,signal,sys,time,traceback
from pathlib import Path
import common
from common import Writer,need,pin,read_regular
HERE=Path(__file__).resolve().parent

def main():
 # Parent blocks catchable signals around spawn/handle assignment. Restore the
 # worker's normal mask immediately; uncatchable deaths remain outside cleanup.
 signal.pthread_sigmask(signal.SIG_SETMASK,set())
 root=Path(sys.argv[1]).resolve();raw=read_regular(root/'intent.json');need(hashlib.sha256(raw).hexdigest()==sys.argv[2],'intent fingerprint changed');intent=json.loads(raw);request=intent['request'];caps=request['caps']
 need(os.getppid()==intent['controller_pid'],'dedicated observed controller required')
 resource.setrlimit(resource.RLIMIT_CPU,(caps['cpu_soft'],caps['cpu_hard']));resource.setrlimit(resource.RLIMIT_AS,(caps['address_space'],caps['address_space']));resource.setrlimit(resource.RLIMIT_FSIZE,(caps['individual_file'],caps['individual_file']));resource.setrlimit(resource.RLIMIT_CORE,(0,0));os.setpriority(os.PRIO_PROCESS,0,15)
 import controller
 need(Path(common.__file__).resolve()==HERE/'common.py'and intent['runner_sources']==controller.sources(),'worker actual module/source identity differs')
 for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):need(os.environ[key]=='1','single-thread controls required')
 writer=Writer(root/'receipts',root);code=1
 try:
  if request['mode']=='actual':
   plan,technical=controller.actual_gate(request['plan'],request['technical']);need(caps==controller.ACTUAL_CAPS,'actual caps differ')
   import pipeline
   pipeline.symbolic(plan,technical,root/'receipts',writer)
   result=json.loads(read_regular(root/'receipts/result.json'));result['intent_sha256']=sys.argv[2];writer.json('result.json',result)
  else:
   need(request['mode']=='fixture'and caps==controller.TOY_CAPS,'invented fixture mode only')
   size=request['payload_bytes'];need(type(size)is int and 0<=size<=65536,'tiny payload only')
   shared=bytearray(i%251 for i in range(size));bundle=dict(original_cost=shared,replaced_cost=shared,rewritten_cost=shared,gradient=[shared],ordered_value_vars=['invented'],E=[[1]],segments=[dict(name='invented',start=0,stop=1)])
   from count_core import count_bundle
   if request['fault']=='hold':time.sleep(10) # test watchdog; not actual graph
   if request['fault']=='raise':raise RuntimeError('invented failure before count')
   counted=count_bundle(bundle,writer,16 if request['fault']=='cap'else 1024**2)
   writer.json('result.json',dict(schema='InventedSymbolicCountFixtureV1',intent_sha256=sys.argv[2],stream_count=counted,pickle_payload_persisted=False,pickle_loaded=False,scientific_qualification=False))
  code=0
 except BaseException:
  try:writer.json('worker-failure.json',dict(error=traceback.format_exc(),intent_sha256=sys.argv[2],scope='failed or unknown; no completion inferred'))
  except BaseException:pass
  raise
 finally:
  try:writer.json('worker-terminal.json',dict(schema='SymbolicCountWorkerTerminalV1',pid=os.getpid(),parent_pid=os.getppid(),returncode=code,intent_sha256=sys.argv[2],runner_sources=intent['runner_sources'],nice=os.getpriority(os.PRIO_PROCESS,0),limits=dict(cpu=resource.getrlimit(resource.RLIMIT_CPU),address_space=resource.getrlimit(resource.RLIMIT_AS),file=resource.getrlimit(resource.RLIMIT_FSIZE)),actual_mode=request['mode']=='actual',numeric_compile=False,sampler=False,forecast=False))
  except BaseException:
   if code==0:raise # metadata uncertainty cannot return successful worker exit

if __name__=='__main__':main()
