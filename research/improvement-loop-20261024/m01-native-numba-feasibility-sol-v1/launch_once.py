"""Exclusive-attempt launcher; caps only its own new CPU child, never owner jobs."""
import hashlib,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
marker=HERE/'attempt-created.json'
with marker.open('x')as f:json.dump(dict(schema='ExclusiveActualNumbaAttemptV1',created_unix=time.time()),f)
GiB=1024**3;MiB=1024**2
caps=dict(wall_seconds=600,cpu_soft_seconds=540,cpu_hard_seconds=550,rss_watchdog_bytes=4*GiB,address_space_bytes=8*GiB,individual_file_bytes=64*MiB,aggregate_cache_watchdog_bytes=512*MiB,poll_seconds=.5)
def setup():
 os.nice(15);resource.setrlimit(resource.RLIMIT_CPU,(540,550));resource.setrlimit(resource.RLIMIT_AS,(8*GiB,8*GiB));resource.setrlimit(resource.RLIMIT_FSIZE,(64*MiB,64*MiB));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
cache=HERE/'full-cache';cache.mkdir(exist_ok=False);env=os.environ.copy()
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS']:env[key]='1'
env.update(PYTHONDONTWRITEBYTECODE='1',NUMBA_CACHE_DIR=str(cache/'numba'),PYTENSOR_FLAGS='cxx=,floatX=float64,numba__fastmath=False,compiledir='+str(cache/'pytensor'))
start=time.monotonic();peak_rss=peak_vms=peak_threads=peak_cache=0;reason=None
with (HERE/'full.log').open('wb')as log:
 child=subprocess.Popen([sys.executable,'-B',str(HERE/'full_attempt.py')],cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=setup)
 (HERE/'launch.json').write_text(json.dumps(dict(pid=child.pid,command=child.args,caps=caps,environment={k:env[k]for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS','PYTHONDONTWRITEBYTECODE','NUMBA_CACHE_DIR','PYTENSOR_FLAGS']},source_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()for name in ['preflight.py','full_attempt.py','launch_once.py']}),indent=2)+'\n')
 while child.poll()is None:
  try:
   info={a.split(':',1)[0]:a.split(':',1)[1].strip()for a in Path('/proc/'+str(child.pid)+'/status').read_text().splitlines()if ':'in a}
   peak_rss=max(peak_rss,int(info.get('VmRSS','0 kB').split()[0])*1024);peak_vms=max(peak_vms,int(info.get('VmSize','0 kB').split()[0])*1024);peak_threads=max(peak_threads,int(info.get('Threads','0')))
  except FileNotFoundError:pass
  size=sum(p.stat().st_size for p in cache.rglob('*')if p.is_file());peak_cache=max(peak_cache,size)
  if time.monotonic()-start>caps['wall_seconds']:reason='wall_timeout'
  elif peak_rss>caps['rss_watchdog_bytes']:reason='rss_watchdog_cap'
  elif size>caps['aggregate_cache_watchdog_bytes']:reason='cache_watchdog_cap'
  if reason:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   break
  time.sleep(.5)
 status=child.wait()
terminal=dict(schema='BoundedActualNumbaAttemptTerminalV1',pid=child.pid,returncode=status,termination_reason=reason,wall_seconds=time.monotonic()-start,peak_observed_rss_bytes=peak_rss,peak_observed_vms_bytes=peak_vms,peak_observed_threads=peak_threads,peak_cache_bytes=peak_cache,caps=caps,full_result_exists=(HERE/'full-result.json').exists(),no_retries=True)
(HERE/'terminal.json').write_text(json.dumps(terminal,indent=2)+'\n');print(json.dumps(terminal),flush=True)
