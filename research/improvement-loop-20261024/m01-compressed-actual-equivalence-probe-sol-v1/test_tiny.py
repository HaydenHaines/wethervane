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
for name in ['common.py','child.py','compare.py','launch.py','test_tiny.py']:ast.parse((HERE/name).read_text())
assert c.read_contract()['mode']=='actual_calibration_bank';passed.append('frozen_contract_all_source_pins')
contract='a'*64;review={'schema':'CompressedActualProbeIndependentReviewV1','decision':'allow_one_bounded_attempt','contract_sha256':contract,'packet_pins':{n:c.sha(HERE/n)for n in ['common.py','child.py','launch.py','compare.py','test_tiny.py','contract.json']}};rd=c.digest(review);permit={'schema':'CompressedActualProbeFreshResourcesV1','contract_sha256':contract,'checked_unix':1000,'independent_review_sha256':rd,'exclusive_actual_probe_slot':True,'resource_conflicts':[],'overlap_assessment':'no_conflict_for_this_one_cpu_job','cpu':min(os.sched_getaffinity(0)),'protected_jobs':[],'minimum_mem_available_bytes':6*1024**3,'minimum_disk_free_bytes':2*1024**3,'observer':'invented test'}
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
   out=Path(tmp)/label;out.mkdir()
   def write(path,value):
    if Path(path).name=='launch.json':raise exc
    return actual_write(path,value)
   launch.atomic_json=write
   try:launch.run_child('original',out,{'caps':cap},{'cpu':min(os.sched_getaffinity(0))})
   except (OSError,KeyboardInterrupt):pass
   else:raise AssertionError('missing injected failure')
   assert children[-1].poll()is not None;terminal=json.loads((out/'terminal.json').read_text());cleanup_receipts.append({'test':label,'terminal':terminal});assert terminal['reason']=='controller_exception'and terminal['returncode']is not None;passed.append(label+'_reaped_terminal')
finally:launch.subprocess.Popen=actual_popen;launch.atomic_json=actual_write;launch.live_resources=actual_live
result={'schema':'PreparedActualProbeTinyTestsV1','status':'passed','tests':passed,'actual_graph_imported':False,'actual_graph_executed':False,'owned_tiny_children_reaped':len(children),'cleanup_receipts':cleanup_receipts,'source_pins':{n:c.pin(HERE/n)for n in ['common.py','child.py','compare.py','launch.py','test_tiny.py']}}
(HERE/'tiny-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'passed','tests':len(passed),'tiny_children':len(children)}))
