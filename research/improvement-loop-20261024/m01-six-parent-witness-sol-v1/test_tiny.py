"""Invented all-six engine/receipts and tiny processes; no actual graph."""
import ast,copy,hashlib,importlib.util,json,os,resource,subprocess,sys,tempfile,time,shutil
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CPU,(20,25));resource.setrlimit(resource.RLIMIT_AS,(768*1024**2,)*2);resource.setrlimit(resource.RLIMIT_FSIZE,(4*1024**2,)*2)
import numpy as np
import common as c,evaluator as ev,compare as comp,completion as done,launch,selection
H=Path(__file__).resolve().parent;tests=[]
def reject(label,fn):
 try:fn()
 except (ValueError,FileNotFoundError,KeyError,TypeError):tests.append(label)
 else:raise AssertionError('accepted '+label)
contract=c.read_contract();assert selection.authenticate_predecessors(contract);tests.append('old_successful_caller_ack_chains_and_source_pins')
for path in H.glob('*.py'):ast.parse(path.read_text())
bindings=comp.base.binding(contract);records=[row['record']for row in bindings.load()['parents']];r0=records[0];target={'composition_identity':contract['reference_composition_identity'],'problem_identity':contract['problem_identity'],'parent_identity':c.digest(r0),'parent_id':r0['parent_id'],'upstream_identity':r0['upstream_identity'],'parent_mass':.125,'parent_mass_used_in_logp':False,'inference_target':'population_cut_v1','scientific_target_id':c.digest({'composition_id':contract['reference_composition_identity'],'problem':contract['problem_identity'],'parent':c.digest(r0),'target':'population_cut_v1'})};kernel={'phase':'dynamic-six','parent_roster':list(range(2,8)),'contract_sha256':c.EXPECTED_CONTRACT,'compiler_parent_contract':target,'execution_scope':'invented fixture only, no actual model'}
variables=contract['finite_difference_variables'];layout=[{'name':n,'shape':[],'dtype':'float64','size':1}for n in variables];points=[]
for pair in [(.1,.2),(.3,.4),(.37,.43)]:
 p=ev.point(layout,{n:np.asarray(value,dtype=float)for n,value in zip(variables,pair)});points.append({'layout':json.loads(p.layout_bytes),'values':json.loads(p.values_bytes)})
engine_calls=[]
def engine(E,x):
 engine_calls.append((E.copy(),[float(a)for a in x]));v=float(np.sum(E*np.arange(1,73).reshape(3,24))*1e-9+sum(float(a)**2 for a in x));return {'logp':v,'poll_logp':v/10,'factor_values':[v],'rates':{'fixture':[v+1]},'physical':{'eventG':[1,2,3],'event_intensity':[2,3,4]}}
session=ev._session(bindings,engine,kernel);blobs={};saved={};parents=[]
def store(name,value,raw=False):
 payload=value if raw else c.canonical(value);blobs[name]=payload;return {'file':name,'pin':{'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)}}
for ordinal in range(2,8):
 bound=bindings.bind(records[ordinal]);calls=[]
 for i,point in enumerate(points):
  p=ev.Point(c.canonical(point['layout']),c.canonical(point['values']));raw=session.evaluate(bound,p);item=store('p'+str(ordinal)+'-'+str(i),raw,True);item.update(parent_index=ordinal,point_index=i);calls.append(item);owned=json.loads(raw);oracle=sum((i+1)*float(v)for i,v in enumerate(np.asarray(bound.exposure).reshape(-1)))*1e-9+sum(float(v[0])**2 for v in point['values']);assert abs(owned['output']['logp']-oracle)<1e-10;saved[(ordinal,i)]={'parent_record':records[ordinal],'point':point,'output':owned['output']}
 fd=[]
 for var in variables:
  pair=[]
  for sign in [-1,1]:
   point=comp.base.declared_fd_point(points,contract,var,sign);raw=session.evaluate(bound,ev.Point(c.canonical(point['layout']),c.canonical(point['values'])));item=store('fd'+str(ordinal)+var+str(sign),raw,True);item.update(sign=sign,value=json.loads(raw)['output']['logp']);pair.append(item)
  fd.append({'variable':var,'coordinate':0,'step':contract['fd_step'],'calls':pair,'derivative':(pair[1]['value']-pair[0]['value'])/(2*contract['fd_step'])})
 parent={'schema':'PartialBatchParentCallsV1','status':'numerical_calls_persisted_not_batch_success','ordinal':ordinal,'parent_identity':c.digest(records[ordinal]),'calls':calls,'finite_differences':fd,'conditional_normalization':'unverified'};item=store('parent'+str(ordinal)+'-index.json',parent);item['ordinal']=ordinal;parents.append(item)
result={'schema':'SixParentDynamicWitnessResultV1','status':'all_six_owned_calls_persisted_not_yet_compared','phase':'dynamic-six','contract_sha256':c.EXPECTED_CONTRACT,'parent_roster':list(range(2,8)),'parents':parents}
def loader(mapping):
 def load(item):
  raw=mapping[item['file']];c.need(item['pin']=={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)},'fixture stale call/index pin')
  return json.loads(raw)if item['file'].endswith('-index.json')else ev.verify(raw,item['pin']['sha256'],bindings)
 return load
def verify(r=result,m=blobs):return comp.verify_all(r,contract,loader(m),saved,points,records,kernel)
comparison=verify();assert comparison['call_count']==42 and len(engine_calls)==42;tests.append('normal_all_six_exact_42_owned_calls_and_independent_invented_oracle')
for label,change in [('missing_parent',lambda r:r['parents'].pop()),('reordered_parent',lambda r:r['parents'].reverse()),('duplicate_parent',lambda r:r['parents'].__setitem__(1,copy.deepcopy(r['parents'][0]))),('wrong_roster',lambda r:r['parent_roster'].__setitem__(0,1))]:
 r=copy.deepcopy(result);change(r);reject(label,lambda:verify(r))
# Coherently update fixture pins after logical tampering: content guards must still reject.
def mutate_call(label,change,fd=False):
 r=copy.deepcopy(result);mapping=dict(blobs);pi=r['parents'][0];parent=json.loads(mapping[pi['file']]);item=parent['finite_differences'][0]['calls'][0]if fd else parent['calls'][0];call=json.loads(mapping[item['file']]);change(call,item,parent);raw=c.canonical(call);mapping[item['file']]=raw;item['pin']={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)};raw=c.canonical(parent);mapping[pi['file']]=raw;pi['pin']={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)};reject(label,lambda:verify(r,mapping))
mutate_call('wrong_output_matches_no_saved_reference',lambda call,*_:call['output'].__setitem__('logp',call['output']['logp']+1));mutate_call('wrong_parent_bound_record',lambda call,*_:call.__setitem__('parent_record',records[3]));mutate_call('stale_kernel_source',lambda call,*_:call['provenance'].__setitem__('contract_sha256','wrong'));mutate_call('wrong_FD_complete_point',lambda call,*_:call['point']['values'][0].__setitem__(0,99),True);mutate_call('wrong_FD_value',lambda call,item,*_:item.__setitem__('value',item['value']+1),True)
for count in [0,1,5]:
 r=copy.deepcopy(result);r['parents']=r['parents'][:count];reject('truncated_prefix_'+str(count)+'_never_completes',lambda:verify(r))
mapping=dict(blobs);key=result['parents'][0]['file'];mapping[key]=mapping[key]+b' ';reject('stale_parent_index_bytes_pin',lambda:verify(result,mapping))
# Exact reviewed batch gate, not a partial/single-parent permit.
review={'schema':'OwnedParentEvaluatorIndependentReviewV1','decision':'allow_one_bounded_attempt','allowed_phases':['dynamic-six'],'parent_roster':list(range(2,8)),'contract_sha256':c.EXPECTED_CONTRACT,'packet_pins':{n:c.sha(H/n)for n in ['common.py','child.py','launch.py','compare.py','test_tiny.py','contract.json','entry.py','evaluator.py','comparison_gate.py','baseline_compare.py','selection.py','completion.py']}};rd=c.digest(review);permit={'schema':'OwnedParentEvaluatorFreshResourcesV1','contract_sha256':c.EXPECTED_CONTRACT,'checked_unix':time.time(),'independent_review_sha256':rd,'exclusive_actual_probe_slot':True,'resource_conflicts':[],'overlap_assessment':'no_conflict_for_this_one_cpu_job','minimum_mem_available_bytes':6*1024**3,'minimum_disk_free_bytes':2*1024**3,'phase':'dynamic-six','parent_roster':list(range(2,8)),'cpu':min(os.sched_getaffinity(0)),'protected_jobs':[],'observer':'invented'};assert c.admission(review,rd,permit,c.digest(permit),c.EXPECTED_CONTRACT);tests.append('one_job_all_six_reviewed_resource_gate')
for label,field,value in [('partial_resource_roster','parent_roster',[2]),('stale_resource','checked_unix',0),('owner_conflict','resource_conflicts',['staging'])]:
 bad=dict(permit);bad[field]=value;reject(label,lambda:c.admission(review,rd,bad,c.digest(bad),c.EXPECTED_CONTRACT))
bad=copy.deepcopy(review);bad['packet_pins']['compare.py']='0'*64;reject('stale_comparator_reuse_rejected',lambda:c.reviewed_sources(bad,c.digest(bad),c.EXPECTED_CONTRACT))
# Actual publisher + outer subprocess tool capture is tested separately below.
protocol=[]
with tempfile.TemporaryDirectory(dir=H,prefix='batch-protocol-')as tmp:
 out=Path(tmp)/'dynamic-six';proc=subprocess.run([sys.executable,'-B',str(H/'protocol_fixture.py'),str(out)],capture_output=True,text=True,timeout=10);assert proc.returncode==0,proc.stderr;tool=Path(tmp)/'caller.json';c.atomic_json(tool,{'schema':'RetainedSelectedStageToolReturnV1','write_stdin_return':{'exit_code':proc.returncode,'output':proc.stdout},'scope':'invented protocol payloads'});before={str(p.relative_to(tmp)):p.read_bytes()for p in Path(tmp).rglob('*')if p.is_file()};done.completed(out,tool,c.sha(tool));assert before=={str(p.relative_to(tmp)):p.read_bytes()for p in Path(tmp).rglob('*')if p.is_file()};tests.append('actual_publisher_full_batch_external_capture_readonly_gate');protocol.append({'exit_code':proc.returncode,'stdout':proc.stdout})
 for case in ['partial_comparison_roster','truncated_call_count','failed_child','wrong_parent_roster','missing_artifact']:
  clone=Path(tmp)/case;shutil.copytree(Path(tmp)/'dynamic-six',clone/'dynamic-six');copyout=clone/'dynamic-six';marker=copyout/'controller-success.json';r=json.loads(marker.read_text());compath=clone/'all-six-comparison.json';shutil.copyfile(Path(tmp)/'all-six-comparison.json',compath)
  if case=='failed_child':
   value=json.loads((copyout/'terminal.json').read_text());value['returncode']=1;c.atomic_json(copyout/'terminal.json',value);r['terminal']=c.pin(copyout/'terminal.json')
  elif case=='missing_artifact':r['comparison_artifacts'].pop('../all-six-comparison.json')
  else:
   value=json.loads(compath.read_text())
   if case=='partial_comparison_roster':value['parents'].pop()
   elif case=='truncated_call_count':value['call_count']=41
   else:value['parent_roster'].reverse()
   c.atomic_json(compath,value);r['comparison_artifacts']['../all-six-comparison.json']=c.pin(compath)
  c.atomic_json(marker,r);retained=clone/'caller.json';c.atomic_json(retained,{'schema':'RetainedSelectedStageToolReturnV1','write_stdin_return':{'exit_code':0,'output':json.dumps(done.response(copyout))}});reject('no_batch_success_'+case,lambda:done.completed(copyout,retained,c.sha(retained)))
 r=json.loads((out/'controller-success.json').read_text());r['comparison_artifacts']={};c.atomic_json(out/'controller-success.json',r);c.atomic_json(tool,{'schema':'RetainedSelectedStageToolReturnV1','write_stdin_return':{'exit_code':0,'output':json.dumps(done.response(out))}});reject('empty_artifact_map_even_rehashed_caller_rejected',lambda:done.completed(out,tool,c.sha(tool)))
# Comparator is reauthenticated after a clean tiny child, not just when review is created.
drift=[]
for mode in ['clean','drift']:
 with tempfile.TemporaryDirectory(dir=H,prefix='batch-drift-')as tmp:
  root=Path(tmp)
  for name in ['common.py','child.py','launch.py','compare.py','test_tiny.py','contract.json','entry.py','evaluator.py','comparison_gate.py','baseline_compare.py','selection.py','completion.py']:shutil.copyfile(H/name,root/name)
  observed=subprocess.run([sys.executable,'-B',str(H/'drift_fixture.py'),tmp,mode],capture_output=True,text=True,timeout=10);assert observed.returncode==0,observed.stderr;receipt=json.loads(observed.stdout);assert receipt['child_returncode']==0 and not receipt['bad_import']
  if mode=='drift':assert receipt['rejected']and not receipt['compared']and not receipt['success']
  else:assert not receipt['rejected']and receipt['compared']and receipt['success']
  drift.append(receipt);tests.append('child_then_comparator_drift_no_success'if mode=='drift'else'clean_comparator_snapshot_control')

actual=launch.subprocess.Popen;write=launch.atomic_json;live=launch.live_resources;children=[];terminals=[]
def sleeper(*a,**kw):
 child=actual([sys.executable,'-c','import time;time.sleep(5)'],**kw);children.append(child);return child
caps={'wall_seconds':3,'cpu_soft_seconds':2,'cpu_hard_seconds':3,'address_space_bytes':256*1024**2,'file_bytes':1024**2,'rss_bytes':128*1024**2,'arm_disk_bytes':2*1024**2}
try:
 launch.subprocess.Popen=sleeper;launch.live_resources=lambda *a:{}
 with tempfile.TemporaryDirectory(dir=H,prefix='batch-cleanup-')as tmp:
  for label,error in [('receipt_fault',OSError('injected')),('controller_interruption',KeyboardInterrupt('injected'))]:
   out=Path(tmp)/label;out.mkdir();(out/'authorization.json').write_text(json.dumps({'controller_pid':os.getpid()}))
   def fail(path,value):
    if Path(path).name=='launch.json':raise error
    return write(path,value)
   launch.atomic_json=fail
   try:launch.run_child('dynamic-six',out,{'caps':caps},permit)
   except (OSError,KeyboardInterrupt):pass
   else:raise AssertionError('injected failure absent')
   terminal=json.loads((out/'terminal.json').read_text());assert children[-1].poll()is not None and terminal['reason']=='controller_exception';terminals.append(terminal);tests.append(label+'_owned_child_reaped_no_success')
finally:launch.subprocess.Popen=actual;launch.atomic_json=write;launch.live_resources=live
with tempfile.TemporaryDirectory(dir=H,prefix='batch-direct-')as tmp:
 process=subprocess.run([sys.executable,'-B',str(H/'child.py'),'dynamic-six',tmp],capture_output=True,timeout=5);assert process.returncode!=0 and b'live controller channel required'in process.stderr;tests.append('direct_child_refused_before_actual_graph_import')
r={'schema':'SixParentWitnessTinyTestsV1','status':'passed','tests':tests,'invented_owned_calls':42,'protocol_flow':protocol,'child_failure_terminals':terminals,'comparator_drift_receipts':drift,'actual_graph_imported':False,'actual_graph_evaluated':False,'normalizers_posteriors':'unverified'};(H/'tiny-result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'status':'passed','tests':len(tests),'invented_calls':42}))
