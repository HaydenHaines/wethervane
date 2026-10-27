"""Prepared sequential two-child controller. No launch without externally pinned gates."""
import argparse,json,os,resource,signal,subprocess,sys,time,traceback,shutil,fcntl,secrets
from entry import start_ticks
from pathlib import Path
from common import *
def stop(child):
 if child is None:return
 if child.poll()is None:
  try:os.killpg(child.pid,signal.SIGTERM)
  except ProcessLookupError:pass
  try:child.wait(timeout=5)
  except subprocess.TimeoutExpired:
   try:os.killpg(child.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   child.wait()
 else:child.wait()
def live_resources(c,permit):
 need(permit['cpu']in os.sched_getaffinity(0),'reviewed CPU unavailable')
 jobs=[]
 for expected in permit['protected_jobs']:
  pid=expected['pid'];tail=Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split();need(int(tail[19])==expected['start_time_ticks'],'protected-job identity changed; fresh assessment required');jobs.append({'pid':pid,'start_time_ticks':int(tail[19]),'state':tail[0]})
 mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()if line.split(':')[0]in ['MemAvailable']};need(mem['MemAvailable']>=permit['minimum_mem_available_bytes'],'insufficient live memory');free=shutil.disk_usage(HERE).free;need(free>=permit['minimum_disk_free_bytes'],'insufficient live disk');return {'mem_available_bytes':mem['MemAvailable'],'disk_free_bytes':free,'protected_jobs':jobs,'cpu':permit['cpu'],'loadavg':os.getloadavg(),'checked_unix':time.time()}
def run_child(arm,out,c,permit):
 cap=c['caps'];cache=out/'cache';cache.mkdir();env=os.environ.copy()
 for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS']:env[key]='1'
 env.update(PYTHONDONTWRITEBYTECODE='1',NUMBA_CACHE_DIR=str(cache/'numba'),PYTENSOR_FLAGS='cxx=,floatX=float64,numba__fastmath=False,compiledir='+str(cache/'pytensor'))
 def setup():
  os.nice(15);os.sched_setaffinity(0,{permit['cpu']});resource.setrlimit(resource.RLIMIT_CPU,(cap['cpu_soft_seconds'],cap['cpu_hard_seconds']));resource.setrlimit(resource.RLIMIT_AS,(cap['address_space_bytes'],)*2);resource.setrlimit(resource.RLIMIT_FSIZE,(cap['file_bytes'],)*2);resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 control_read=control_write=None;child=None;reason=None;error=None;peak_rss=peak_threads=peak_disk=0;start=time.monotonic()
 try:
  live=live_resources(c,permit)
  payload=json.loads((out/'authorization.json').read_text());payload.update(arm=arm,out=str(out.resolve()),controller_start_ticks=start_ticks(os.getpid()),nonce=secrets.token_hex(32));raw=canonical(payload);need(len(raw)<=65536,'bounded controller payload exceeded');control_read,control_write=os.pipe();env['M01_PROBE_CONTROL_FD']=str(control_read);env['M01_PROBE_CONTROL_SHA256']=hashlib.sha256(raw).hexdigest()
  with (out/'child.log').open('xb')as log:
   child=subprocess.Popen([sys.executable,'-B',str(HERE/'child.py'),arm,str(out)],cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=setup,pass_fds=(control_read,))
   os.close(control_read);control_read=None
   remaining=raw
   while remaining:remaining=remaining[os.write(control_write,remaining):]
   os.close(control_write);control_write=None
   atomic_json(out/'launch.json',{'pid':child.pid,'arm':arm,'control_channel_sha256':env['M01_PROBE_CONTROL_SHA256'],'controller_start_ticks':payload['controller_start_ticks'],'command':child.args,'environment':{k:env[k]for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS','PYTENSOR_FLAGS','NUMBA_CACHE_DIR']},'live_resources':live,'caps':cap})
   while child.poll()is None:
    try:
     info={line.split(':',1)[0]:line.split(':',1)[1].strip()for line in Path('/proc/'+str(child.pid)+'/status').read_text().splitlines()if ':'in line};rss=int(info.get('VmRSS','0 kB').split()[0])*1024;threads=int(info.get('Threads','0'));peak_rss=max(peak_rss,rss);peak_threads=max(peak_threads,threads)
    except FileNotFoundError:rss=threads=0
    disk=sum(f.stat().st_size for f in out.rglob('*')if f.is_file());peak_disk=max(peak_disk,disk)
    if time.monotonic()-start>cap['wall_seconds']:reason='wall_timeout'
    elif rss>cap['rss_bytes']:reason='rss_watchdog'
    elif disk>cap['arm_disk_bytes']:reason='arm_disk_watchdog'
    elif threads>1:reason='thread_count_exceeded'
    if reason:break
    time.sleep(.5)
 except BaseException:
  error=traceback.format_exc();reason='controller_exception';raise
 finally:
  for descriptor in [control_read,control_write]:
   if descriptor is not None:
    try:os.close(descriptor)
    except OSError:pass
  stop(child)
  # Best effort receipt after teardown. Its publication can fail; absent/uncertain receipt is failure.
  terminal={'arm':arm,'pid':child.pid if child else None,'returncode':child.returncode if child else None,'reason':reason,'controller_error':error,'seconds':time.monotonic()-start,'peak_rss_bytes':peak_rss,'peak_threads':peak_threads,'peak_disk_bytes':peak_disk,'no_retry':True}
  atomic_json(out/'terminal.json',terminal)
 need(child.returncode==0 and reason is None,'arm did not complete cleanly');need((out/'result.json').is_file(),'missing arm receipt');return terminal

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--parent',type=int,required=True);parser.add_argument('--batch-tool-return',required=True);parser.add_argument('--batch-tool-return-sha256',required=True);parser.add_argument('--review',required=True);parser.add_argument('--review-digest',required=True);parser.add_argument('--resources',required=True);parser.add_argument('--resources-digest',required=True);parser.add_argument('--replay-tool-return');parser.add_argument('--replay-tool-return-sha256');args=parser.parse_args();c=read_contract()
 from selection import selected,reserve,completed,authenticate_predecessors,publish_success,controller_response
 from batch_adapter import select_witness
 selected(c,args.parent);need(Path(sys.prefix).resolve()==Path(c['runtime_prefix']).resolve()and Path(sys.executable).resolve()==Path(c['runtime_executable']).resolve()and sha(sys.executable)==c['runtime_executable_sha256'],'wrong runtime executable');review=json.loads(Path(args.review).read_text());permit=json.loads(Path(args.resources).read_text());admission(review,args.review_digest,permit,args.resources_digest,EXPECTED_CONTRACT);need(permit['phase']=='fixed-from-batch'and permit['selected_parent_ordinal']==args.parent,'selected fixed parent resource mismatch')
 lock=(HERE.parent/'.m01-actual-equivalence-probe.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);authenticate_predecessors(c);witness=select_witness(c,args.parent,args.batch_tool_return,args.batch_tool_return_sha256);parent=HERE/'attempts'/('parent-'+str(args.parent));out=parent/'fixed-from-batch'
 if args.replay_tool_return:
  need(args.replay_tool_return_sha256 is not None,'retained fixed-stage return hash required');completed(out,args.replay_tool_return,args.replay_tool_return_sha256,args.parent,'fixed-from-batch')
 else:
  live_resources(c,permit);out=reserve(HERE/'attempts',c,args.parent,'fixed-from-batch')
  with (out/'batch-selection.json').open('xb')as file:file.write(witness);file.flush();os.fsync(file.fileno())
  atomic_json(out/'authorization.json',{'controller_pid':os.getpid(),'contract_sha256':EXPECTED_CONTRACT,'selected_parent_ordinal':args.parent,'review':review,'review_digest':args.review_digest,'resource_receipt':permit,'resources_digest':args.resources_digest,'batch_tool_return':str(Path(args.batch_tool_return).resolve()),'batch_tool_return_sha256':args.batch_tool_return_sha256,'batch_selection':pin(out/'batch-selection.json')});run_child('fixed-from-batch',out,c,permit)
 read_contract();snapshot=reviewed_sources(review,args.review_digest,EXPECTED_CONTRACT)
 import types
 gate=types.ModuleType('_verified_fixed_batch_gate');gate.__file__=str(HERE/'comparison_gate.py');exec(compile(snapshot['comparison_gate.py'],gate.__file__,'exec'),gate.__dict__);_,execution=gate.execute(parent,c,review,args.review_digest,'fixed-from-batch',args.parent,read_only=bool(args.replay_tool_return))
 if not args.replay_tool_return:atomic_json(out/'comparison-execution.json',execution)
 comparisons={'comparison-execution.json':pin(out/'comparison-execution.json'),'../fixed-from-batch-comparison.json':pin(parent/'fixed-from-batch-comparison.json')};response=publish_success(out,args.parent,'fixed-from-batch',comparisons)if not args.replay_tool_return else dict(controller_response(out,args.parent,'fixed-from-batch'),status='verified_completed_read_only_replay');print(json.dumps(response),flush=True)
if __name__=='__main__':main()
