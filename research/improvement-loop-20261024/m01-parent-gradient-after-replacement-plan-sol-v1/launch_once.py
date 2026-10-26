"""UNEXECUTED bounded launcher proposal. Gate before scientific imports or output."""
import argparse,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
import experiment
HERE=Path(__file__).resolve().parent

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--execution-receipt',required=True);parser.add_argument('--execution-receipt-sha',required=True);parser.add_argument('--plan-sha',required=True);parser.add_argument('--attempt-directory',required=True);a=parser.parse_args()
 plan,receipt=experiment.gate(HERE/'plan.json',a.plan_sha,a.execution_receipt,a.execution_receipt_sha)
 root=Path(a.attempt_directory).resolve();experiment.need(root.is_relative_to(HERE)and root!=HERE and not root.exists(),'fresh packet-local attempt required');root.mkdir()
 cache=root/'fresh-cache';cache.mkdir();MiB=1024**2;GiB=1024**3
 limits=plan['resource_caps']
 def setup():
  os.nice(15);resource.setrlimit(resource.RLIMIT_CPU,(limits['cpu_soft_seconds'],limits['cpu_hard_seconds']));resource.setrlimit(resource.RLIMIT_AS,(limits['address_space_bytes'],limits['address_space_bytes']));resource.setrlimit(resource.RLIMIT_FSIZE,(limits['individual_file_bytes'],limits['individual_file_bytes']));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 env=os.environ.copy()
 for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS']:env[key]='1'
 env.update(M01_SYMBOLIC_WATCHDOG_PARENT_PID=str(os.getpid()),PYTHONDONTWRITEBYTECODE='1',NUMBA_CACHE_DIR=str(cache/'numba'),PYTENSOR_FLAGS='cxx=,floatX=float64,numba__fastmath=False,compiledir='+str(cache/'pytensor'))
 command=[sys.executable,'-B',str(HERE/'experiment.py'),'--plan',str(HERE/'plan.json'),'--plan-sha',a.plan_sha,'--execution-receipt',str(Path(a.execution_receipt).resolve()),'--execution-receipt-sha',a.execution_receipt_sha,'--output',str(root/'receipts')]
 start=time.monotonic();rss=vms=threads=cache_peak=0;reason=None
 with (root/'process.log').open('xb')as out:
  child=subprocess.Popen(command,env=env,cwd=HERE,stdout=out,stderr=subprocess.STDOUT,preexec_fn=setup,start_new_session=True)
  (root/'launch.json').write_text(json.dumps(dict(pid=child.pid,command=command,limits=limits,environment={k:env[k]for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS','PYTENSOR_FLAGS','NUMBA_CACHE_DIR']},plan_sha256=a.plan_sha),indent=2)+'\n')
  while child.poll()is None:
   try:
    proc={s.split(':',1)[0]:s.split(':',1)[1].strip()for s in Path('/proc/'+str(child.pid)+'/status').read_text().splitlines()if ':'in s};rss=max(rss,int(proc.get('VmRSS','0 kB').split()[0])*1024);vms=max(vms,int(proc.get('VmSize','0 kB').split()[0])*1024);threads=max(threads,int(proc.get('Threads','0')))
   except FileNotFoundError:pass
   size=sum(p.stat().st_size for p in root.rglob('*')if p.is_file());cache_peak=max(cache_peak,size)
   if time.monotonic()-start>limits['wall_seconds']:reason='wall_timeout'
   elif rss>limits['rss_watchdog_bytes']:reason='rss_watchdog_cap'
   elif size>limits['aggregate_output_watchdog_bytes']:reason='output_disk_watchdog_cap'
   if reason:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    break
   time.sleep(.25)
  code=child.wait()
 (root/'terminal.json').write_text(json.dumps(dict(pid=child.pid,returncode=code,termination_reason=reason,wall_seconds=time.monotonic()-start,peak_observed_rss=rss,peak_observed_vms=vms,peak_observed_threads=threads,peak_output_bytes=cache_peak,limits=limits,no_retry=True,numerical_compilation=False),indent=2)+'\n')
if __name__=='__main__':main()
