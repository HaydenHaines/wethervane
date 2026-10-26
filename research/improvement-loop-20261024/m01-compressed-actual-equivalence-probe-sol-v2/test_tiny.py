"""Only invented arrays/receipts and two bounded sleeping children; no model import."""
import ast,copy,json,os,subprocess,sys,tempfile,time,resource
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CPU,(15,20));resource.setrlimit(resource.RLIMIT_AS,(1024**3,)*2);resource.setrlimit(resource.RLIMIT_FSIZE,(4*1024**2,)*2)
import numpy as np
import common as c,compare as compare,launch
HERE=Path(__file__).resolve().parent;passed=[]
def reject(label,fn):
 try:fn()
 except (ValueError,AssertionError):passed.append(label)
 else:raise AssertionError('accepted '+label)
for name in ['common.py','child.py','compare.py','launch.py','test_tiny.py','entry.py']:ast.parse((HERE/name).read_text())
assert c.read_contract()['mode']=='actual_calibration_bank';passed.append('frozen_contract_all_source_pins')
contract='a'*64;review={'schema':'CompressedActualProbeIndependentReviewV1','decision':'allow_one_bounded_attempt','contract_sha256':contract,'packet_pins':{n:c.sha(HERE/n)for n in ['common.py','child.py','launch.py','compare.py','test_tiny.py','contract.json','entry.py']}};rd=c.digest(review);permit={'schema':'CompressedActualProbeFreshResourcesV1','contract_sha256':contract,'checked_unix':1000,'independent_review_sha256':rd,'exclusive_actual_probe_slot':True,'resource_conflicts':[],'overlap_assessment':'no_conflict_for_this_one_cpu_job','cpu':min(os.sched_getaffinity(0)),'protected_jobs':[],'minimum_mem_available_bytes':6*1024**3,'minimum_disk_free_bytes':2*1024**3,'observer':'invented test'}
assert c.admission(review,rd,permit,c.digest(permit),contract,now=1001);passed.append('fresh_receipt')
for label,key,value in [('stale','checked_unix',0),('future','checked_unix',1002),('overlap','resource_conflicts',['shared staging busy']),('slot','exclusive_actual_probe_slot',False),('wrong_contract','contract_sha256','b'*64),('memory_floor','minimum_mem_available_bytes',1)]:
 p=dict(permit);p[key]=value;reject(label,lambda:c.admission(review,rd,p,c.digest(p),contract,now=1001))
reject('retained_digest_mismatch',lambda:c.admission(review,'0'*64,permit,c.digest(permit),contract,now=1001))
a=np.arange(1324,dtype=float).reshape(2,662);assert compare.arrays_equal(a,a.copy())==0;passed.append('complete_vector_equal');b=a.copy();b[-1,-1]+=.1;reject('last_coordinate_diff',lambda:compare.arrays_equal(a,b));reject('missing_coordinate',lambda:compare.arrays_equal(a,a[:,:661]));b=a.copy();b[0,0]=np.nan;reject('nonfinite',lambda:compare.arrays_equal(a,b))
base={'joint_contract':{'composition_identity':'old','scientific_target_id':'oldtarget','parent_mass_used_in_logp':False,'parent_mass':.125},'parent':{'mass':.125},'model_ids':['poll1'],'component_inventory':{'measurement:old':{'implementation':'old','factors':{'poll':['poll1']}}},'composition_record':{'compiler_and_baseline_dependencies':{},'implementations':[{'id':'old'},{'id':'same'}]}}
new=copy.deepcopy(base);new['joint_contract'].update(composition_identity='new',scientific_target_id='newtarget');new['component_inventory']={'measurement:new':{'implementation':'new','factors':{'poll':['poll1']}}};new['composition_record']['implementations'][0]={'id':'new'};assert compare.envelope_equal(base,new)['distinct_target_ids'];passed.append('target_ids_distinct_preserved');bad=copy.deepcopy(new);bad['joint_contract']['parent_mass']=.5;reject('mass_changed',lambda:compare.envelope_equal(base,bad));bad=copy.deepcopy(new);bad['component_inventory']['measurement:new']['factors']={};reject('claim_changed',lambda:compare.envelope_equal(base,bad))
# Real subprocess teardown after post-Popen receipt failure and controller interruption.
actual_popen=launch.subprocess.Popen;actual_write=launch.atomic_json;actual_live=launch.live_resources;children=[];cleanup_receipts=[]
def tiny_popen(*args,**kwargs):
 child=actual_popen([sys.executable,'-c','import time; time.sleep(5)'],**kwargs);children.append(child);return child
cap={'wall_seconds':3,'cpu_soft_seconds':2,'cpu_hard_seconds':3,'address_space_bytes':256*1024**2,'file_bytes':1024**2,'rss_bytes':128*1024**2,'arm_disk_bytes':2*1024**2}
try:
 launch.subprocess.Popen=tiny_popen;launch.live_resources=lambda *a:{}
 with tempfile.TemporaryDirectory(prefix='invented-cleanup-',dir=HERE)as tmp:
  for label,exc in [('receipt_exception',OSError('injected receipt write')),('controller_interruption',KeyboardInterrupt('injected interruption'))]:
   out=Path(tmp)/label;out.mkdir();(out/'authorization.json').write_text(json.dumps({'controller_pid':os.getpid()}))
   def write(path,value):
    if Path(path).name=='launch.json':raise exc
    return actual_write(path,value)
   launch.atomic_json=write
   try:launch.run_child('original',out,{'caps':cap},{'cpu':min(os.sched_getaffinity(0))})
   except (OSError,KeyboardInterrupt):pass
   else:raise AssertionError('missing injected failure')
   assert children[-1].poll()is not None;terminal=json.loads((out/'terminal.json').read_text());cleanup_receipts.append({'test':label,'terminal':terminal});assert terminal['reason']=='controller_exception'and terminal['returncode']is not None;passed.append(label+'_reaped_terminal')
finally:launch.subprocess.Popen=actual_popen;launch.atomic_json=actual_write;launch.live_resources=actual_live
# Direct actual-child entry must reject before imports/build, even with a handmade file.
with tempfile.TemporaryDirectory(prefix='direct-reject-',dir=HERE)as tmp:
 out=Path(tmp);(out/'authorization.json').write_text(json.dumps({'controller_pid':os.getpid(),'contract_sha256':c.EXPECTED_CONTRACT,'arms':['original','compressed']}))
 env=os.environ.copy();env.pop('M01_PROBE_CONTROL_FD',None);env.pop('M01_PROBE_CONTROL_SHA256',None)
 run=subprocess.run([sys.executable,'-B',str(HERE/'child.py'),'original',str(out)],env=env,capture_output=True,timeout=5)
 assert run.returncode!=0 and b'live controller channel required' in run.stderr and not(out/'stage.json').exists();passed.append('direct_actual_child_handmade_file_rejected_before_model_import')
 direct_receipt={'returncode':run.returncode,'stderr':run.stderr.decode(),'stage_exists':(out/'stage.json').exists()}
# Inherited descriptor alone cannot substitute for a valid live channel payload.
with tempfile.TemporaryDirectory(prefix='bad-channel-',dir=HERE)as tmp:
 out=Path(tmp)
 with (out/'ordinary.json').open('w+')as ordinary:
  env=os.environ.copy();env.update(M01_PROBE_CONTROL_FD=str(ordinary.fileno()),M01_PROBE_CONTROL_SHA256='0'*64)
  run=subprocess.run([sys.executable,'-B',str(HERE/'child.py'),'original',str(out)],env=env,pass_fds=(ordinary.fileno(),),capture_output=True,timeout=5);assert run.returncode!=0 and b'anonymous pipe required' in run.stderr and not(out/'stage.json').exists();passed.append('regular_descriptor_rejected')
 readfd,writefd=os.pipe();os.write(writefd,b'{}');os.close(writefd);env=os.environ.copy();env.update(M01_PROBE_CONTROL_FD=str(readfd),M01_PROBE_CONTROL_SHA256='0'*64)
 try:run=subprocess.run([sys.executable,'-B',str(HERE/'child.py'),'original',str(out)],env=env,pass_fds=(readfd,),capture_output=True,timeout=5)
 finally:os.close(readfd)
 assert run.returncode!=0 and b'controller channel payload invalid' in run.stderr and not(out/'stage.json').exists();passed.append('wrong_pipe_digest_rejected')
# Valid invented inherited channel exercises actual worker-side limits and wall alarm.
from entry import start_ticks
entry_receipts=[]
for hanging in [False,True]:
 with tempfile.TemporaryDirectory(prefix='invented-entry-',dir=HERE)as tmp:
  realreview={'schema':'CompressedActualProbeIndependentReviewV1','decision':'allow_one_bounded_attempt','contract_sha256':c.EXPECTED_CONTRACT,'packet_pins':{n:c.sha(HERE/n)for n in ['common.py','child.py','launch.py','compare.py','test_tiny.py','contract.json','entry.py']}};rd=c.digest(realreview)
  realpermit=dict(permit,checked_unix=time.time(),contract_sha256=c.EXPECTED_CONTRACT,independent_review_sha256=rd)
  payload={'controller_pid':os.getpid(),'controller_start_ticks':start_ticks(os.getpid()),'nonce':'a'*64,'arm':'invented','out':tmp,'contract_sha256':c.EXPECTED_CONTRACT,'review':realreview,'review_digest':rd,'resource_receipt':realpermit,'resources_digest':c.digest(realpermit)}
  readfd,writefd=os.pipe();raw=c.canonical(payload);env=os.environ.copy();env.update(M01_PROBE_CONTROL_FD=str(readfd),M01_PROBE_CONTROL_SHA256=__import__('hashlib').sha256(raw).hexdigest());cmd=[sys.executable,'-B',str(HERE/'invented_worker.py'),tmp]+(['hang']if hanging else[])
  child=actual_popen(cmd,env=env,pass_fds=(readfd,),stdout=subprocess.PIPE,stderr=subprocess.PIPE);os.close(readfd)
  try:
   remaining=raw
   while remaining:remaining=remaining[os.write(writefd,remaining):]
  finally:os.close(writefd)
  stdout,stderr=child.communicate(timeout=5);receipt=json.loads(stdout);assert receipt['cpu_affinity']==[realpermit['cpu']]and receipt['wall_alarm_seconds']==2 and receipt['nice']>=15
  assert receipt['limits'][str(resource.RLIMIT_AS)]==[256*1024**2]*2 and receipt['limits'][str(resource.RLIMIT_CPU)]==[2,3]
  assert child.returncode==(-14 if hanging else 0),(child.returncode,stderr)
  entry_receipts.append({'hanging':hanging,'returncode':child.returncode,'receipt':receipt,'stderr':stderr.decode()});passed.append('worker_wall_alarm_terminates'if hanging else'worker_limits_before_invented_work')
result={'schema':'PreparedActualProbeTinyTestsV1','status':'passed','tests':passed,'actual_graph_imported':False,'actual_graph_executed':False,'owned_tiny_children_reaped':len(children),'cleanup_receipts':cleanup_receipts,'direct_entry_receipt':direct_receipt,'worker_entry_receipts':entry_receipts,'source_pins':{n:c.pin(HERE/n)for n in ['common.py','child.py','compare.py','launch.py','test_tiny.py','entry.py']}}
(HERE/'tiny-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'passed','tests':len(passed),'tiny_children':len(children)}))
