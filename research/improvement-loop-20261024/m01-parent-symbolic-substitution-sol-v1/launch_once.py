"""One bounded actual symbolic child; never compile or touch owner jobs."""
import json,os,resource,signal,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
with (HERE/'attempt-created.json').open('x')as f:json.dump(dict(created_unix=time.time()),f)
MiB=1024**2;GiB=1024**3;cache=HERE/'fresh-cache';cache.mkdir(exist_ok=False)
caps=dict(wall_seconds=120,cpu_seconds=90,rss_watchdog_bytes=2*GiB,address_space_bytes=6*GiB,per_file_bytes=16*MiB,cache_watchdog_bytes=64*MiB,poll_seconds=.25)
def setup():
 os.nice(15);resource.setrlimit(resource.RLIMIT_CPU,(90,95));resource.setrlimit(resource.RLIMIT_AS,(6*GiB,6*GiB));resource.setrlimit(resource.RLIMIT_FSIZE,(16*MiB,16*MiB));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
env=os.environ.copy()
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS']:env[key]='1'
env.update(PYTHONDONTWRITEBYTECODE='1',NUMBA_CACHE_DIR=str(cache/'numba'),PYTENSOR_FLAGS='cxx=,floatX=float64,numba__fastmath=False,compiledir='+str(cache/'pytensor'))
start=time.monotonic();peak_rss=peak_vms=peak_threads=peak_cache=0;reason=None
with (HERE/'symbolic.log').open('wb')as out:
 child=subprocess.Popen([sys.executable,'-B',str(HERE/'symbolic.py')],cwd=HERE,env=env,stdout=out,stderr=subprocess.STDOUT,preexec_fn=setup,start_new_session=True)
 (HERE/'launch.json').write_text(json.dumps(dict(pid=child.pid,command=child.args,caps=caps,environment={k:env[k]for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS','PYTHONDONTWRITEBYTECODE','PYTENSOR_FLAGS','NUMBA_CACHE_DIR']}),indent=2)+'\n')
 while child.poll()is None:
  try:
   status={a.split(':',1)[0]:a.split(':',1)[1].strip()for a in Path('/proc/'+str(child.pid)+'/status').read_text().splitlines()if ':'in a};peak_rss=max(peak_rss,int(status.get('VmRSS','0 kB').split()[0])*1024);peak_vms=max(peak_vms,int(status.get('VmSize','0 kB').split()[0])*1024);peak_threads=max(peak_threads,int(status.get('Threads','0')))
  except FileNotFoundError:pass
  size=sum(p.stat().st_size for p in cache.rglob('*')if p.is_file());peak_cache=max(peak_cache,size)
  if time.monotonic()-start>120:reason='wall_timeout'
  elif peak_rss>2*GiB:reason='rss_watchdog_cap'
  elif size>64*MiB:reason='cache_watchdog_cap'
  if reason:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   break
  time.sleep(.25)
 code=child.wait()
 terminal=dict(schema='BoundedSymbolicOnlyAttemptTerminalV1',pid=child.pid,returncode=code,termination_reason=reason,wall_seconds=time.monotonic()-start,peak_observed_rss_bytes=peak_rss,peak_observed_vms_bytes=peak_vms,peak_observed_threads=peak_threads,peak_cache_bytes=peak_cache,caps=caps,one_attempt=True,no_numerical_compile=True)
 (HERE/'terminal.json').write_text(json.dumps(terminal,indent=2)+'\n');print(json.dumps(terminal))
