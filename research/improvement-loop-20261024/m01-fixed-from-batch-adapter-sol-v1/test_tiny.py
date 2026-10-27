"""Adapter evidence/provenance fixtures only; no actual graph/worker import."""
import ast,copy,hashlib,json,os,resource,subprocess,sys,tempfile
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CPU,(25,30));resource.setrlimit(resource.RLIMIT_AS,(768*1024**2,)*2);resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**2,)*2)
import common as c,batch_adapter as adapter,compare as comp
from fixture_batch import build,fixed_fixture
H=Path(__file__).resolve().parent;tests=[]
def reject(name,fn):
 try:fn()
 except (ValueError,KeyError,FileNotFoundError,TypeError):tests.append(name)
 else:raise AssertionError('accepted '+name)
contract=c.read_contract()
for path in H.glob('*.py'):ast.parse(path.read_text())
original_common=sys.modules['common'];original_compare=sys.modules['compare'];original_aliases={name:sys.modules.get(name)for name in [*adapter.MODULES,'compare_parent_binding']}
with tempfile.TemporaryDirectory(dir=H,prefix='full-batch-fixture-')as tmp:
 cfg,tool,toolsha=build(Path(tmp)/'batch',contract);root=Path(cfg['batch_attempt_root']);before={str(p.relative_to(root)):p.read_bytes()for p in root.rglob('*')if p.is_file()};selected=[]
 for ordinal in range(2,8):
  raw=adapter.select_witness(cfg,ordinal,tool,toolsha);w=json.loads(raw);assert w['ordinal']==ordinal and len(w['calls'])==7 and w['parent']['ordinal']==ordinal and w['all_six_comparison']['call_count']==42;selected.append(raw)
 assert before=={str(p.relative_to(root)):p.read_bytes()for p in root.rglob('*')if p.is_file()};assert sys.modules['common']is original_common and sys.modules['compare']is original_compare;tests.append('all_six_positive_select_only_seven_immutable_calls_and_readonly_namespace_restore')
 for value in [0,1,8,True,[2,3]]:reject('wrong_parent_'+str(value),lambda:adapter.select_witness(cfg,value,tool,toolsha))
 reject('missing_tool_return',lambda:adapter.select_witness(cfg,2,root/'missing.json','0'*64));reject('stale_tool_return_hash',lambda:adapter.select_witness(cfg,2,tool,'0'*64))
 retained=json.loads(tool.read_text());retained['write_stdin_return']['exit_code']=1;c.atomic_json(tool,retained);reject('failed_external_tool_return_even_matching_hash',lambda:adapter.select_witness(cfg,2,tool,c.sha(tool)));tool.write_bytes(before['fixture-caller-return.json'])
 rpath=root/'dynamic-six/result.json';original=rpath.read_bytes();result=json.loads(original);result['parents'].pop();c.atomic_json(rpath,result);reject('partial_batch_result_not_selectable',lambda:adapter.select_witness(cfg,2,tool,toolsha));rpath.write_bytes(original)
 # Coherently rewrite summary/marker/caller hashes; full re-comparison must catch the numeric lie.
 summary=root/'all-six-comparison.json';oldsummary=summary.read_bytes();marker=root/'dynamic-six/controller-success.json';oldmarker=marker.read_bytes();oldtool=tool.read_bytes();changed=json.loads(oldsummary);changed['parents'][0]['errors'][0]['logp']=1e100;c.atomic_json(summary,changed)
 with adapter.frozen_batch(cfg)as b:
  receipt=json.loads(marker.read_text());receipt['comparison_artifacts']['../all-six-comparison.json']=b['common'].pin(summary);b['common'].atomic_json(marker,receipt);ack=b['completion'].response(root/'dynamic-six');b['common'].atomic_json(tool,{'schema':'RetainedSelectedStageToolReturnV1','write_stdin_return':{'exit_code':0,'output':json.dumps(ack)}})
 reject('coherently_rehashed_false_summary_requires_actual42recomparison',lambda:adapter.select_witness(cfg,2,tool,c.sha(tool)));summary.write_bytes(oldsummary);marker.write_bytes(oldmarker);tool.write_bytes(oldtool)
 # Any selected or unselected call stale byte invalidates full batch before parent selection.
 r=json.loads(rpath.read_text());pindex=root/'dynamic-six'/r['parents'][-1]['file'];parent=json.loads(pindex.read_text());call=root/'dynamic-six'/parent['calls'][0]['file'];oldcall=call.read_bytes();call.write_bytes(oldcall+b' ');reject('stale_unselected_parent_call_rejects_selected_parent',lambda:adapter.select_witness(cfg,2,tool,toolsha));call.write_bytes(oldcall)
 # Partial proof projection cannot be treated as selected completed witness.
 w=json.loads(selected[0]);bad=dict(w['all_six_comparison']);bad['parents']=bad['parents'][:-1]
 reject('partial_all_six_projection_rejected',lambda:adapter.project(2,w['batch_success'],w['batch_tool_return'],bad,{2:w['parent']},w['calls']))
 # Fixed-native target and numerical mismatch tests remain exact.
 record=w['calls'][w['parent']['calls'][0]['file']]['parent_record'];native={'composition_identity':cfg['reference_composition_identity'],'problem_identity':cfg['problem_identity'],'parent_identity':c.digest(record),'parent_id':record['parent_id'],'upstream_identity':record['upstream_identity'],'parent_mass':.125,'parent_mass_used_in_logp':False,'inference_target':'population_cut_v1','scientific_target_id':comp.target_identity(record,cfg)};assert comp.validate_fixed_target(native,record,cfg);tests.append('full_fixed_target_identity_preserved')
 bad=dict(native,parent_mass=.25);reject('wrong_fixed_target_mass',lambda:comp.validate_fixed_target(bad,record,cfg));output=w['calls'][w['parent']['calls'][0]['file']]['output'];changed=copy.deepcopy(output);changed['logp']+=10;reject('fixed_numerical_comparison_mismatch',lambda:comp.compare_outputs(output,changed))
 # Full fixed/completed-batch comparator wiring on invented receipts; no graph is built.
 fixedroot=Path(tmp)/'fixed-fixture';fixedout=fixed_fixture(fixedroot,cfg,2,tool,toolsha,selected[0]);comparison=comp.compare_fixed_from_batch(fixedroot,cfg,2);assert comparison['status']=='selected_fixed_parent_matches_completed_batch_points_and_FD'and len(comparison['errors'])==3 and all(len(row['output_errors'])==2 for row in comparison['FD_errors']);tests.append('complete_fixed_to_batch_point_and_full_FD_output_integration_fixture')
 fixedresult=json.loads((fixedout/'result.json').read_text());row=fixedresult['calls'][0];callpath=fixedout/row['file'];oldbytes=callpath.read_bytes();call=json.loads(oldbytes);call['output']['logp']+=10;callpath.write_bytes(c.canonical(call));row['pin']=c.pin(callpath);c.atomic_json(fixedout/'result.json',fixedresult);reject('coherently_rehashed_fixed_result_numerical_mismatch',lambda:comp.compare_fixed_from_batch(fixedroot,cfg,2))
 # Independent review identity cannot be silently exchanged even if same source code is named.
 before_failure_aliases={name:sys.modules.get(name)for name in [*adapter.MODULES,'compare_parent_binding']};badcfg=dict(cfg,batch_admission_review_pin={'sha256':'0'*64,'bytes':cfg['batch_admission_review_pin']['bytes']});reject('wrong_independent_batch_review_identity',lambda:adapter.select_witness(badcfg,2,tool,toolsha))
 assert all(sys.modules.get(name)is value for name,value in before_failure_aliases.items());tests.append('failure_restores_contract_module_namespace')
# Source drift and resource phase/ordinal gates without a model.
review={'schema':'OwnedParentEvaluatorIndependentReviewV1','decision':'allow_one_bounded_attempt','allowed_phases':['fixed-from-batch'],'allowed_parent_ordinals':[2],'contract_sha256':c.EXPECTED_CONTRACT,'packet_pins':{n:c.sha(H/n)for n in ['common.py','child.py','launch.py','compare.py','test_tiny.py','contract.json','entry.py','evaluator.py','comparison_gate.py','baseline_compare.py','selection.py','batch_adapter.py']}};bad=copy.deepcopy(review);bad['packet_pins']['batch_adapter.py']='0'*64;reject('stale_adapter_source_review_rejected',lambda:c.reviewed_sources(bad,c.digest(bad),c.EXPECTED_CONTRACT))
with tempfile.TemporaryDirectory(dir=H,prefix='fixed-direct-')as tmp:
 run=subprocess.run([sys.executable,'-B',str(H/'child.py'),'fixed-from-batch',tmp],capture_output=True,timeout=5);assert run.returncode!=0 and b'live controller channel required'in run.stderr;tests.append('direct_fixed_child_rejected_before_graph')
r={'schema':'FixedFromBatchAdapterTinyV1','status':'passed','tests':tests,'fixtures':'invented complete42-call protocol using copied saved base outputs/invented FD; never actual graph authority','actual_graph_imported':False,'actual_graph_evaluated':False,'normalizers_posteriors':'unverified'};(H/'tiny-result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'status':'passed','tests':len(tests)}))
