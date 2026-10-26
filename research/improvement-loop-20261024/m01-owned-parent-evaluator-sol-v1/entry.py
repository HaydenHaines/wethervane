"""Accidental-entry defense only. Trusted controller/same-user filesystem boundary."""
import ctypes,hashlib,json,os,resource,signal,stat,time
from pathlib import Path
from common import admission,need,read_contract,EXPECTED_CONTRACT

def start_ticks(pid):return int(Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[19])
def accept(arm,out):
 # No file, CLI switch or discovered digest can substitute for the live inherited channel.
 rawfd=os.environ.pop('M01_PROBE_CONTROL_FD',None);expected=os.environ.pop('M01_PROBE_CONTROL_SHA256',None)
 need(rawfd is not None and expected is not None,'live controller channel required before graph imports')
 fd=int(rawfd);need(fd>=3 and stat.S_ISFIFO(os.fstat(fd).st_mode)and os.readlink('/proc/self/fd/'+str(fd)).startswith('pipe:['),'anonymous pipe required')
 # A bounded blocking read; direct execution with an empty inherited pipe cannot hang indefinitely.
 signal.signal(signal.SIGALRM,signal.SIG_DFL);signal.alarm(5)
 data=b''
 try:
  while len(data)<=65536:
   part=os.read(fd,min(8192,65537-len(data)))
   if not part:break
   data+=part
 finally:os.close(fd)
 need(len(data)<=65536 and hashlib.sha256(data).hexdigest()==expected,'controller channel payload invalid');a=json.loads(data)
 need(a['nonce']and len(a['nonce'])==64,'one-use channel nonce missing');need(a['controller_pid']==os.getppid()and a['controller_start_ticks']==start_ticks(os.getppid()),'live controller identity mismatch')
 need(a['arm']==arm and Path(a['out']).resolve()==Path(out).resolve(),'child target mismatch');need(a['contract_sha256']==EXPECTED_CONTRACT,'channel contract mismatch');c=read_contract()
 admission(a['review'],a['review_digest'],a['resource_receipt'],a['resources_digest'],EXPECTED_CONTRACT)
 need(arm in a['review']['allowed_phases']and a['resource_receipt']['phase']==arm,'phase not independently admitted');cap=c['caps'];cpu=a['resource_receipt']['cpu'];need(cpu in os.sched_getaffinity(0),'reviewed CPU unavailable');os.sched_setaffinity(0,{cpu});os.setpriority(os.PRIO_PROCESS,0,max(15,os.getpriority(os.PRIO_PROCESS,0)))
 for key,limits in [(resource.RLIMIT_CPU,(cap['cpu_soft_seconds'],cap['cpu_hard_seconds'])),(resource.RLIMIT_AS,(cap['address_space_bytes'],)*2),(resource.RLIMIT_FSIZE,(cap['file_bytes'],)*2),(resource.RLIMIT_CORE,(0,0))]:
  current=resource.getrlimit(key);need(current[1]==resource.RLIM_INFINITY or current[1]>=limits[1],'inherited hard limit unexpectedly narrower');resource.setrlimit(key,limits)
 # Linux-specific parent-death fail closed; applies only to this owned worker.
 libc=ctypes.CDLL(None,use_errno=True);need(libc.prctl(1,signal.SIGKILL,0,0,0)==0,'parent-death signal unavailable');need(os.getppid()==a['controller_pid'],'controller vanished during entry')
 for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS']:need(os.environ.get(key)=='1','single-thread environment missing '+key)
 signal.alarm(cap['wall_seconds']) # default action terminates even if Python cannot handle callbacks
 return a,c,{'pid':os.getpid(),'controller_pid':a['controller_pid'],'controller_start_ticks':a['controller_start_ticks'],'channel_sha256':expected,'cpu_affinity':list(os.sched_getaffinity(0)),'nice':os.getpriority(os.PRIO_PROCESS,0),'limits':{str(k):resource.getrlimit(k)for k in [resource.RLIMIT_CPU,resource.RLIMIT_AS,resource.RLIMIT_FSIZE,resource.RLIMIT_CORE]},'wall_alarm_seconds':cap['wall_seconds'],'parent_death_signal':'SIGKILL'}
