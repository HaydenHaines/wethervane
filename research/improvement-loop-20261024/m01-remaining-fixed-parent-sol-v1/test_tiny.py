"""Metadata/selection/receipts and tiny processes only; no actual graph import."""
import ast,copy,hashlib,importlib.util,json,multiprocessing as mp,os,resource,subprocess,sys,tempfile,time
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CPU,(20,25));resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,)*2);resource.setrlimit(resource.RLIMIT_FSIZE,(4*1024**2,)*2)
import common as c,selection as s,compare as comp,launch
HERE=Path(__file__).resolve().parent;tests=[]
def reject(name,fn):
 try:fn()
 except (ValueError,FileExistsError,FileNotFoundError,KeyError):tests.append(name)
 else:raise AssertionError('accepted '+name)
def crash(root,contract,ordinal):s.reserve(root,contract,ordinal,'dynamic-witness');os._exit(88)
contract=c.read_contract();assert s.authenticate_predecessors(contract);tests.append('frozen_sources_and_all_prior_caller_ack_chains')
for name in ['common.py','selection.py','compare.py','baseline_compare.py','comparison_gate.py','launch.py','entry.py','child.py','evaluator.py','test_tiny.py']:ast.parse((HERE/name).read_text())
for ordinal in range(2,8):assert s.selected(contract,ordinal)['ordinal']==ordinal
for value in [0,1,8,True,[2,3]]:reject('wrong_parent_'+str(value),lambda:s.selected(contract,value))
with tempfile.TemporaryDirectory(dir=HERE,prefix='claim-fixture-')as tmp:
 child=mp.Process(target=crash,args=(tmp,contract,2));child.start();child.join(5);assert child.exitcode==88;reject('post_claim_crash_refuses_repeat',lambda:s.reserve(tmp,contract,2,'dynamic-witness'));reject('partial_claim_not_completed_replay',lambda:s.completed(Path(tmp)/'parent-2/dynamic-witness','0'*64,2,'dynamic-witness'))
# Synthetic completed phase tests enforce retained ack, parent and result pins; no claimed graph.
with tempfile.TemporaryDirectory(dir=HERE,prefix='replay-fixture-')as tmp:
 out=Path(tmp)/'dynamic-witness';out.mkdir();result={'status':'completed_calls_not_normalized_conditionals','selected_parent_ordinal':2,'phase':'dynamic-witness','contract_sha256':c.EXPECTED_CONTRACT};terminal={'arm':'dynamic-witness','returncode':0,'reason':None};c.atomic_json(out/'result.json',result);c.atomic_json(out/'terminal.json',terminal);receipt={'status':'successfully_reaped_selected_phase_and_passed_comparison','contract_sha256':c.EXPECTED_CONTRACT,'selected_parent_ordinal':2,'phase':'dynamic-witness','result':c.pin(out/'result.json'),'terminal':c.pin(out/'terminal.json'),'comparison_artifacts':{}};c.atomic_json(out/'controller-success.json',receipt);ack=c.sha(out/'controller-success.json');before={p.name:p.read_bytes()for p in out.iterdir()};s.completed(out,ack,2,'dynamic-witness');assert before=={p.name:p.read_bytes()for p in out.iterdir()};tests.append('completed_ack_replay_reads_only');reject('wrong_parent_ack_replay',lambda:s.completed(out,ack,3,'dynamic-witness'));reject('wrong_retained_ack',lambda:s.completed(out,'0'*64,2,'dynamic-witness'));(out/'result.json').write_text('{}');reject('stale_result_pin',lambda:s.completed(out,ack,2,'dynamic-witness'))
# Full target formula and all protected identity fields, not just exposure equality.
spec=importlib.util.spec_from_file_location('tiny_bind',contract['parent_binding']);bindings=importlib.util.module_from_spec(spec);sys.modules[spec.name]=bindings;spec.loader.exec_module(bindings);record=bindings.load()['parents'][2]['record'];fixed={'composition_identity':contract['reference_composition_identity'],'problem_identity':contract['problem_identity'],'parent_identity':c.digest(record),'parent_id':record['parent_id'],'upstream_identity':record['upstream_identity'],'parent_mass':.125,'parent_mass_used_in_logp':False,'inference_target':'population_cut_v1','scientific_target_id':comp.target_identity(record,contract)};assert comp.validate_fixed_target(fixed,record,contract);tests.append('full_native_fixed_target_identity')
for key,value in [('scientific_target_id','wrong'),('parent_identity','wrong'),('parent_id','wrong'),('parent_mass',.25),('parent_mass_used_in_logp',True),('inference_target','uncut'),('problem_identity','wrong'),('composition_identity','wrong'),('upstream_identity','wrong')]:
 wrong=dict(fixed);wrong[key]=value;reject('target_'+key,lambda:comp.validate_fixed_target(wrong,record,contract))
# Selected fixed-parent comparator rejects wrong/repeated rows, independently of producer loop.
rows=[{'parent_index':2,'point_index':i}for i in range(3)];assert comp.validate_roster(rows,2);tests.append('exact_selected_three_point_roster')
for label,mutate in [('repeated_point',lambda x:x.__setitem__(1,dict(x[0]))),('wrong_parent',lambda x:x[0].__setitem__('parent_index',3)),('missing_point',lambda x:x.pop()),('swapped_order',lambda x:x.reverse())]:
 wrong=copy.deepcopy(rows);mutate(wrong);reject(label,lambda:comp.validate_roster(wrong,2))
output={'logp':3.,'poll_logp':1.,'factor_values':[1.,2.],'rates':{'invented':[1.,2.]},'physical':{'eventG':[1.],'event_intensity':[2.]}};assert comp.compare_outputs(output,copy.deepcopy(output))=={'logp':0.,'poll_logp':0.,'factors':0.,'rates':0.,'physical':0.};tests.append('invented_all_output_comparator_oracle');wrong=copy.deepcopy(output);wrong['factor_values'][0]+=1;reject('factor_value_mismatch',lambda:comp.compare_outputs(output,wrong))
# Source/result and new review phase/parent gates.
review={'schema':'OwnedParentEvaluatorIndependentReviewV1','decision':'allow_one_bounded_attempt','allowed_phases':['dynamic-witness'],'allowed_parent_ordinals':[2],'contract_sha256':c.EXPECTED_CONTRACT,'packet_pins':{n:c.sha(HERE/n)for n in ['common.py','child.py','launch.py','compare.py','test_tiny.py','contract.json','entry.py','evaluator.py','comparison_gate.py','baseline_compare.py','selection.py']}};rd=c.digest(review);permit={'schema':'OwnedParentEvaluatorFreshResourcesV1','contract_sha256':c.EXPECTED_CONTRACT,'checked_unix':time.time(),'independent_review_sha256':rd,'exclusive_actual_probe_slot':True,'resource_conflicts':[],'overlap_assessment':'no_conflict_for_this_one_cpu_job','minimum_mem_available_bytes':6*1024**3,'minimum_disk_free_bytes':2*1024**3,'phase':'dynamic-witness','selected_parent_ordinal':2,'cpu':min(os.sched_getaffinity(0)),'protected_jobs':[],'observer':'invented'};assert c.admission(review,rd,permit,c.digest(permit),c.EXPECTED_CONTRACT);tests.append('one_reviewed_parent_phase_fresh_admission')
for name,change in [('wrong_resource_parent',{'selected_parent_ordinal':3}),('wrong_phase',{'phase':'fixed-selected'}),('stale_resource',{'checked_unix':0}),('capacity_conflict',{'resource_conflicts':['live owner staging']})]:
 bad=dict(permit,**change);reject(name,lambda:c.admission(review,rd,bad,c.digest(bad),c.EXPECTED_CONTRACT))
bad=copy.deepcopy(review);bad['packet_pins']['compare.py']='0'*64;reject('stale_comparator_source',lambda:c.reviewed_sources(bad,c.digest(bad),c.EXPECTED_CONTRACT));bad=copy.deepcopy(review);bad['allowed_parent_ordinals']=[2,2];reject('repeated_review_parent',lambda:c.reviewed_sources(bad,c.digest(bad),c.EXPECTED_CONTRACT))
# Real bounded sleeping-child teardown on post-Popen controller fault, never actual graph.
actual=launch.subprocess.Popen;write=launch.atomic_json;live=launch.live_resources;children=[];terminals=[]
def sleeper(*args,**kwargs):
 child=actual([sys.executable,'-c','import time;time.sleep(5)'],**kwargs);children.append(child);return child
caps={'wall_seconds':3,'cpu_soft_seconds':2,'cpu_hard_seconds':3,'address_space_bytes':256*1024**2,'file_bytes':1024**2,'rss_bytes':128*1024**2,'arm_disk_bytes':2*1024**2}
try:
 launch.subprocess.Popen=sleeper;launch.live_resources=lambda *a:{}
 with tempfile.TemporaryDirectory(dir=HERE,prefix='cleanup-fixture-')as tmp:
  for label,exc in [('receipt_write',OSError('injected')),('interruption',KeyboardInterrupt('injected'))]:
   out=Path(tmp)/label;out.mkdir();(out/'authorization.json').write_text(json.dumps({'controller_pid':os.getpid()}))
   def fail(path,value):
    if Path(path).name=='launch.json':raise exc
    return write(path,value)
   launch.atomic_json=fail
   try:launch.run_child('dynamic-witness',out,{'caps':caps},permit)
   except (OSError,KeyboardInterrupt):pass
   else:raise AssertionError('missing injected exception')
   terminal=json.loads((out/'terminal.json').read_text());assert children[-1].poll()is not None and terminal['reason']=='controller_exception';terminals.append(terminal);tests.append(label+'_owned_child_reaped')
finally:launch.subprocess.Popen=actual;launch.atomic_json=write;launch.live_resources=live
with tempfile.TemporaryDirectory(dir=HERE,prefix='direct-fixture-')as tmp:
 run=subprocess.run([sys.executable,'-B',str(HERE/'child.py'),'dynamic-witness',tmp],capture_output=True,timeout=5);assert run.returncode!=0 and b'live controller channel required'in run.stderr;tests.append('direct_actual_child_rejected_before_graph_import')
# Saved original numerical receipts are verified, not executed or altered.
saved,_=comp.base.read_phase(contract['baseline_attempt_root'],'dynamic',contract);assert len(saved)==27;tests.append('saved_27_owned_calls_exact_parent_point_roster')
r={'schema':'RemainingParentVerifierTinyV1','status':'passed','tests':tests,'owned_child_terminals':terminals,'actual_graph_imported':False,'actual_graph_evaluated':False,'normalizers_posteriors':'unverified'};(HERE/'tiny-result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'status':'passed','tests':len(tests)}))
