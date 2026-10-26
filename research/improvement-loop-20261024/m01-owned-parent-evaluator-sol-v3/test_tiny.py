"""Invented engines/arrays and bounded sleepers only; actual child never admitted."""
import ast,copy,hashlib,importlib.util,json,os,resource,subprocess,sys,tempfile,time,shutil
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CPU,(20,25));resource.setrlimit(resource.RLIMIT_AS,(1024**3,)*2);resource.setrlimit(resource.RLIMIT_FSIZE,(4*1024**2,)*2)
import numpy as np
import common as c,evaluator as ev,launch,compare
HERE=Path(__file__).resolve().parent;tests=[]
def reject(label,fn):
 try:fn()
 except (ValueError,TypeError,AssertionError):tests.append(label)
 else:raise AssertionError('accepted '+label)
for name in ['common.py','entry.py','evaluator.py','child.py','launch.py','compare.py','test_tiny.py','comparison_gate.py']:ast.parse((HERE/name).read_text())
contract=c.read_contract();spec=importlib.util.spec_from_file_location('fixture_bindings',contract['parent_binding']);bindings=importlib.util.module_from_spec(spec);sys.modules[spec.name]=bindings;spec.loader.exec_module(bindings);records=[r['record']for r in bindings.load()['parents']];tests.append('frozen_source_parent_contract')
layout=[{'name':'invented_x','shape':[4],'size':4,'dtype':'float64'}];values={'invented_x':np.array([.1,-.2,.3,.4],dtype=float)};point=ev.point(layout,values);observed=[]
def invented_engine(E,x):
 observed.append((E.copy(),tuple(v.copy()for v in x)));v=float(E.sum()*1e-8+np.dot(x[0],x[0]));return {'logp':v,'poll_logp':v*.1,'factor_values':[v*.4,v*.6],'rates':{'invented':E.sum(axis=1).tolist()},'physical':{'invented_G':[1,2,3]}}
session=ev._session(bindings,invented_engine,{'execution_scope':'invented fixture, no actual graph','source':'test_tiny.py'});calls=[]
for index in list(range(8))+[0]:
 bound=bindings.bind(records[index]);raw=session.evaluate(bound,point);r=ev.verify(raw,hashlib.sha256(raw).hexdigest(),bindings);assert r['parent_record']==records[index]and r['parent_envelope']==json.loads(bound.envelope_bytes)and r['outer_mass']==.125;expected=float(sum(map(sum,bound.exposure))*1e-8+.3);assert abs(r['output']['logp']-expected)<1e-12;calls.append(r)
assert calls[0]['output']==calls[-1]['output']and all(row['conditional_normalization']=='unverified'for row in calls);tests.append('owned_engine_actual_input_output_all8_A_B_A')
assert observed[0][0].tolist()==[list(row)for row in bindings.bind(records[0]).exposure]and observed[0][1][0].tolist()==values['invented_x'].tolist();tests.append('engine_receives_exact_owned_parent_and_point')
raw=session.evaluate(bindings.bind(records[0]),point);retained=hashlib.sha256(raw).hexdigest();changed=json.loads(raw);changed['output']['logp']+=1;reject('changed_result_retained_digest_rejected',lambda:ev.verify(c.canonical(changed),retained,bindings))
changed=json.loads(raw);changed['parent_envelope']['mass']=.5;reject('coherently_rehashed_mass_mismatch',lambda:ev.verify(c.canonical(changed),hashlib.sha256(c.canonical(changed)).hexdigest(),bindings))
changed=json.loads(raw);changed['point']['values'][0][0]+=.1;reject('point_tamper_digest_rejected',lambda:ev.verify(c.canonical(changed),hashlib.sha256(c.canonical(changed)).hexdigest(),bindings))
reject('point_shape_mismatch',lambda:ev.point(layout,{'invented_x':np.ones(3)}));reject('point_nonfinite',lambda:ev.point(layout,{'invented_x':np.array([1,2,np.inf,4])}));reject('point_dtype_mismatch',lambda:ev.point(layout,{'invented_x':np.ones(4,dtype=np.float32)}))
def mutate_engine(E,x):E[0,0]*=2
reject('engine_input_readonly',lambda:ev._session(bindings,mutate_engine,{'execution_scope':'invented'}).evaluate(bindings.bind(records[0]),point))
def bad_engine(E,x):
 out=invented_engine(E,x);out['factor_values']=[100.];return out
reject('factor_sum_mismatch',lambda:ev._session(bindings,bad_engine,{'execution_scope':'invented'}).evaluate(bindings.bind(records[0]),point))
def nan_engine(E,x):
 out=invented_engine(E,x);out['logp']=float('nan');return out
reject('nonfinite_output',lambda:ev._session(bindings,nan_engine,{'execution_scope':'invented'}).evaluate(bindings.bind(records[0]),point))
# Guard and live cleanup checks use invented subprocess replacement, never actual engine.
review={'schema':'OwnedParentEvaluatorIndependentReviewV1','decision':'allow_one_bounded_attempt','allowed_phases':['dynamic'],'contract_sha256':c.EXPECTED_CONTRACT,'packet_pins':{n:c.sha(HERE/n)for n in ['common.py','entry.py','evaluator.py','child.py','launch.py','compare.py','test_tiny.py','contract.json','comparison_gate.py']}};rd=c.digest(review);permit={'schema':'OwnedParentEvaluatorFreshResourcesV1','contract_sha256':c.EXPECTED_CONTRACT,'checked_unix':time.time(),'independent_review_sha256':rd,'exclusive_actual_probe_slot':True,'resource_conflicts':[],'overlap_assessment':'no_conflict_for_this_one_cpu_job','minimum_mem_available_bytes':6*1024**3,'minimum_disk_free_bytes':2*1024**3,'cpu':min(os.sched_getaffinity(0)),'protected_jobs':[],'observer':'invented','phase':'dynamic'};assert c.admission(review,rd,permit,c.digest(permit),c.EXPECTED_CONTRACT);tests.append('fresh_source_resource_admission');bad=dict(permit,checked_unix=0);reject('stale_admission',lambda:c.admission(review,rd,bad,c.digest(bad),c.EXPECTED_CONTRACT));bad=dict(permit,resource_conflicts=['shared staging']);reject('overlap_conflict',lambda:c.admission(review,rd,bad,c.digest(bad),c.EXPECTED_CONTRACT))
bad=dict(permit,phase='fixed-B');reject('unreviewed_phase_rejected',lambda:c.admission(review,rd,bad,c.digest(bad),c.EXPECTED_CONTRACT))
actual=launch.subprocess.Popen;write=launch.atomic_json;live=launch.live_resources;children=[];terminals=[]
def sleeper(*args,**kwargs):
 p=actual([sys.executable,'-c','import time;time.sleep(5)'],**kwargs);children.append(p);return p
caps={'wall_seconds':3,'cpu_soft_seconds':2,'cpu_hard_seconds':3,'address_space_bytes':256*1024**2,'file_bytes':1024**2,'rss_bytes':128*1024**2,'arm_disk_bytes':2*1024**2}
try:
 launch.subprocess.Popen=sleeper;launch.live_resources=lambda *a:{}
 with tempfile.TemporaryDirectory(dir=HERE,prefix='invented-cleanup-')as tmp:
  for label,exc in [('receipt_write',OSError('injected')),('interruption',KeyboardInterrupt('injected'))]:
   out=Path(tmp)/label;out.mkdir();(out/'authorization.json').write_text(json.dumps({'controller_pid':os.getpid()}))
   def fail(path,value):
    if Path(path).name=='launch.json':raise exc
    return write(path,value)
   launch.atomic_json=fail
   try:launch.run_child('dynamic',out,{'caps':caps},permit)
   except (OSError,KeyboardInterrupt):pass
   else:raise AssertionError('injected failure absent')
   terminal=json.loads((out/'terminal.json').read_text());assert children[-1].poll()is not None and terminal['reason']=='controller_exception';terminals.append(terminal);tests.append(label+'_owned_child_reaped')
finally:launch.subprocess.Popen=actual;launch.atomic_json=write;launch.live_resources=live
with tempfile.TemporaryDirectory(dir=HERE,prefix='direct-child-')as tmp:
 out=Path(tmp);(out/'authorization.json').write_text(json.dumps({'controller_pid':os.getpid()}));env=os.environ.copy();env.pop('M01_PROBE_CONTROL_FD',None);env.pop('M01_PROBE_CONTROL_SHA256',None);run=subprocess.run([sys.executable,'-B',str(HERE/'child.py'),'dynamic',tmp],env=env,capture_output=True,timeout=5);assert run.returncode!=0 and b'live controller channel required' in run.stderr and not(out/'stage.json').exists();tests.append('direct_actual_child_rejected_before_model_import');direct={'returncode':run.returncode,'stderr':run.stderr.decode()}
# Same inherited entry implementation in invented workers, normal and alarm-terminated.
from entry import start_ticks
worker_receipts=[]
for hanging in [False,True]:
 with tempfile.TemporaryDirectory(dir=HERE,prefix='entry-fixture-')as tmp:
  fresh=dict(permit,checked_unix=time.time());payload={'controller_pid':os.getpid(),'controller_start_ticks':start_ticks(os.getpid()),'nonce':'b'*64,'arm':'dynamic','out':tmp,'contract_sha256':c.EXPECTED_CONTRACT,'review':review,'review_digest':rd,'resource_receipt':fresh,'resources_digest':c.digest(fresh)};raw=c.canonical(payload);readfd,writefd=os.pipe();env=os.environ.copy();env.update(M01_PROBE_CONTROL_FD=str(readfd),M01_PROBE_CONTROL_SHA256=hashlib.sha256(raw).hexdigest());child=actual([sys.executable,'-B',str(HERE/'invented_worker.py'),tmp]+(['hang']if hanging else[]),env=env,pass_fds=(readfd,),stdout=subprocess.PIPE,stderr=subprocess.PIPE);os.close(readfd)
  try:
   remaining=raw
   while remaining:remaining=remaining[os.write(writefd,remaining):]
  finally:os.close(writefd)
  stdout,stderr=child.communicate(timeout=5);receipt=json.loads(stdout);assert receipt['limits'][str(resource.RLIMIT_AS)]==[256*1024**2]*2 and receipt['limits'][str(resource.RLIMIT_CPU)]==[2,3]and receipt['cpu_affinity']==[permit['cpu']]and receipt['nice']>=15;assert child.returncode==(-14 if hanging else 0),(child.returncode,stderr);worker_receipts.append({'receipt':receipt,'returncode':child.returncode,'hanging':hanging});tests.append('entry_wall_alarm'if hanging else'entry_owned_worker_limits')
assert compare.close([1,2],[1,2])==0;reject('comparison_shape',lambda:compare.close([1,2],[1]));reject('comparison_numerical_delta',lambda:compare.close([1,2],[1,3]));tests.append('independent_numeric_comparison_helper')
# Farthest-distance selection is deliberately avoided: B is fixed ordinal1 before new values.
assert contract['anchor_index']==1 and contract['missing_fixed_parent_baselines']==[2,3,4,5,6,7];tests.append('predeclared_B_and_missing_six_baselines')
# Exact phase roster and complete point contents are comparator obligations, not producer assumptions.
fixture_points=[]
for offset in [0.,.1,.2]:
 p=ev.point(layout,{'invented_x':values['invented_x']+offset});fixture_points.append({'layout':json.loads(p.layout_bytes),'values':json.loads(p.values_bytes)})
def roster(phase):
 ids=list(range(8))+[0]if phase=='dynamic'else[1];entries=[]
 for i in ids:
  for pi in range(3):
   point=copy.deepcopy(fixture_points[pi]);digest=c.digest(point);entries.append(({'parent_index':i,'point_index':pi,'point_sha256':digest},{'point':point,'point_sha256':digest}))
 return entries
for phase in ['dynamic','fixed-B']:
 original=roster(phase);assert compare.check_roster_and_points(original,fixture_points,phase,1);tests.append(phase+'_exact_roster_points_accept')
 bad=copy.deepcopy(original);bad[1]=copy.deepcopy(bad[0]);reject(phase+'_duplicate_roster_rejected',lambda:compare.check_roster_and_points(bad,fixture_points,phase,1))
 bad=copy.deepcopy(original[:-1]);reject(phase+'_missing_call_rejected',lambda:compare.check_roster_and_points(bad,fixture_points,phase,1))
 bad=copy.deepcopy(original);bad[0],bad[1]=bad[1],bad[0];reject(phase+'_swapped_order_rejected',lambda:compare.check_roster_and_points(bad,fixture_points,phase,1))
 bad=copy.deepcopy(original);bad[0][0]['parent_index']=7;reject(phase+'_wrong_parent_rejected',lambda:compare.check_roster_and_points(bad,fixture_points,phase,1))
 bad=copy.deepcopy(original);bad[0][1]['point']['values'][0][0]+=.77;newdigest=c.digest(bad[0][1]['point']);bad[0][0]['point_sha256']=bad[0][1]['point_sha256']=newdigest;reject(phase+'_coherently_rehashed_wrong_point_rejected',lambda:compare.check_roster_and_points(bad,fixture_points,phase,1))
 bad=copy.deepcopy(original);bad[0][1]['point']=copy.deepcopy(fixture_points[1]);newdigest=c.digest(bad[0][1]['point']);bad[0][0]['point_sha256']=bad[0][1]['point_sha256']=newdigest;reject(phase+'_swapped_point_contents_rejected',lambda:compare.check_roster_and_points(bad,fixture_points,phase,1))
# Reconstruct actual declared point bytes without importing/building a model.
actual_points=compare.expected_points(contract);assert len(actual_points)==3 and sum(r['size']for r in actual_points[0]['layout'])==662;tests.append('complete_saved_and_third_actual_point_reconstruction_only')
for sign in [-1,1]:
 fd=compare.declared_fd_point(actual_points,contract,contract['finite_difference_variables'][0],sign);assert fd!=actual_points[1];tests.append('declared_FD_sign_'+str(sign))
reject('undeclared_FD_variable',lambda:compare.declared_fd_point(actual_points,contract,'unknown',1))
# Invented clean child completes, then comparator drifts: use-time guard must stop it.
drift_receipts=[]
for mode in ['clean','drift']:
 with tempfile.TemporaryDirectory(dir=HERE,prefix='comparator-fixture-')as tmp:
  out=Path(tmp)
  for name in ['common.py','entry.py','child.py','evaluator.py','launch.py','compare.py','test_tiny.py','contract.json','comparison_gate.py']:shutil.copyfile(HERE/name,out/name)
  process=subprocess.run([sys.executable,'-B',str(HERE/'drift_fixture.py'),tmp,mode],capture_output=True,timeout=10);assert process.returncode==0,process.stderr.decode();receipt=json.loads(process.stdout);assert receipt['child_returncode']==0 and not receipt['unreviewed_executed']
  if mode=='drift':assert receipt['rejected']and not receipt['compared']and not receipt['success_published']and 'reviewed packet source changed: compare.py'in receipt['error']
  else:assert not receipt['rejected']and receipt['compared']and receipt['success_published']and receipt['execution']['comparison_source_mode'].startswith('executed verified snapshot')
  drift_receipts.append(receipt);tests.append('child_then_comparator_drift_rejected_no_success'if mode=='drift'else'verified_comparison_snapshot_clean_control')
result={'schema':'OwnedParentEvaluatorInventedTestsV1','status':'passed','tests':tests,'actual_graph_imported':False,'actual_graph_evaluated':False,'conditional_normalization':'unverified','owned_child_terminals':terminals,'direct_child_receipt':direct,'worker_entry_receipts':worker_receipts,'comparison_drift_receipts':drift_receipts,'source_pins':{n:c.pin(HERE/n)for n in ['common.py','entry.py','evaluator.py','child.py','launch.py','compare.py','test_tiny.py','contract.json','comparison_gate.py']}};(HERE/'tiny-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'passed','tests':len(tests)}))
