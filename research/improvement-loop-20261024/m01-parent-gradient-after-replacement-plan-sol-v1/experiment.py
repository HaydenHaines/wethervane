"""UNEXECUTED prospective symbolic experiment. No numerical compiler or sampler.

Requires a separately selected execution receipt AND fresh resource recheck.
Only standard-library modules load until all preparation/execution gates pass.
"""
import argparse,hashlib,json,os,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent

def need(ok,message):
 if not ok:raise ValueError(message)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def read_pin(item):
 path=Path(item['path']);raw=path.read_bytes();need(digest(raw)==item['sha256'],'pinned bytes changed: '+str(path));return raw
def gate(plan_path,plan_sha,receipt_path,receipt_sha):
 raw=Path(plan_path).read_bytes();need(digest(raw)==plan_sha,'externally selected plan SHA required');plan=json.loads(raw)
 raw=Path(receipt_path).read_bytes();need(digest(raw)==receipt_sha,'externally selected execution receipt SHA required');receipt=json.loads(raw)
 need(receipt['schema']=='AuthorizeSingleSymbolicDerivationV1'and receipt['plan_sha256']==plan_sha,'unsupported execution receipt')
 need(receipt['allow_symbolic_graph_build']is True and all(receipt[k]is False for k in ['allow_numerical_compile','allow_sampler','allow_forecast']),'only symbolic stage permitted')
 need(receipt['independent_failure_review']==plan['independent_failure_review'],'independent review must be exact selected bytes')
 review=json.loads(read_pin(receipt['independent_failure_review']));need(review['verdict']==plan['review_verdict'],'review verdict changed')
 check=json.loads(read_pin(receipt['resource_recheck']));age=time.time()-check['observed_unix']
 need(0<=age<=120,'resource observation must be fresh, not future dated')
 need(check['available_memory_bytes']>=plan['resource_requirements']['min_available_memory_bytes']and check['available_disk_bytes']>=plan['resource_requirements']['min_available_disk_bytes'],'insufficient fresh headroom')
 need(check['scope_directory']==str(HERE.resolve()),'resource receipt must cover this exact directory')
 need(check['cpu_threads_permitted']==1 and check['nice']==15,'one nice15 CPU required')
 for item in plan['source_pins']:read_pin(item)
 return plan,receipt

def symbolic(plan,receipt,output):
 # No scientific/native imports appear above the execution gate.
 import numpy as np
 import pytensor
 import pytensor.tensor as pt
 from pytensor.graph.replace import graph_replace
 from pytensor.graph.traversal import ancestors,graph_inputs
 from pytensor.raise_op import CheckAndRaise
 from pymc.pytensorf import rewrite_pregrad,gradient
 out=Path(output);need(not out.exists(),'new exclusive output directory required');out.mkdir()
 records=[]
 def persist(name,detail):
  records.append(dict(stage=name,detail=detail,unix=time.time()))
  raw=json.dumps(dict(schema='ProspectiveDerivationStageReceiptV1',records=records),indent=2)+'\n'
  temp=out/'stages.tmp';temp.write_text(raw)
  with temp.open('rb')as f:os.fsync(f.fileno())
  os.replace(temp,out/'stages.json')
  print(name,flush=True)
 try:
  persist('execution_gate_passed',dict(plan_scope=plan['scope'],numerical_compile=False))
  old=Path(plan['frozen_provider_directory']);sys.path.append(str(old/'schema-overlay'));sys.path.insert(0,str(old));import provider as p
  s=p.selection();problem=json.loads(Path(s['problem_path']).read_text());parent=s['parents'][0];p.validate(problem,parent,s)
  roles_path=p.OWNER/'research/senate-stack-integration-20261020/candidate-role-source-sol-v1/prepared-v2/candidate-role-manifest.json';rm=json.loads(roles_path.read_text());original=json.loads((p.OWNER/rm['bound_prepared_problem']['path']).read_text());refs=sys.modules['m01_successor_native.references']
  roles=refs.roles.load_role_binding(original,roles_path,'a1d73f7dece21a4cb5b104c69caec5157e4dd106b411ce17833f7f5a41bbeb3d',admission_cutoff_utc=problem['cutoff_utc'],root=p.OWNER);roles=refs.roles.rebind_role_binding(original,problem,roles)
  registry=p.components.standard_registry();successor=p.M01Measurement('actual_calibration_bank');registry.register(successor);composition=p.components.baseline_composition(role_hierarchy=True);composition=p.graph.Composition(tuple((r,successor.implementation_id if r=='measurement'else n)for r,n in composition.selections),composition.standardized,composition.person_coordinates)
  model=p.graph.compile_model(registry,composition,problem,parent,roles=roles)
  contract=p.dict_plain(model.joint_contract);inventory=p.dict_plain(model.component_inventory);frozen_inventory=json.loads(read_pin(plan['frozen_factor_inventory']))
  need(inventory==frozen_inventory,'factor/parameter/token ownership changed')
  need(contract['parent_identity']==p.graph.digest(parent)and contract['parent_mass']==.125 and contract['parent_mass_used_in_logp']is False and contract['inference_target']=='population_cut_v1','parent/cut contract changed')
  need(contract['scientific_target_id']==p.graph.digest(dict(composition_id=contract['composition_identity'],problem=contract['problem_identity'],parent=p.graph.digest(parent),target='population_cut_v1')),'scientific target mismatch')
  (out/'first-parent-contract.json').write_text(json.dumps(contract,indent=2)+'\n');(out/'factor-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
  persist('original_graph_and_first_parent_authenticated',dict(contract=contract,inventory_sha256=digest(json.dumps(inventory,sort_keys=True,separators=(',',':')).encode())))
  ordered=list(model.continuous_value_vars);ordered_set=set(ordered)
  need(len(ordered)==len(model.value_vars),'discrete or omitted coordinates unsupported')
  need([rv.name for rv in model.free_RVs]==contract['required_free_rvs'],'original free-RV order changed')
  witness=json.loads(read_pin(plan['frozen_shape_witness']));shapes=next(c for c in witness['cases']if c['case']=='actual_TX_role_graph_only')['modular']['parameter_shapes']
  need({v.name for v in ordered}==set(shapes),'full transformed variable names changed')
  segments=[];offset=0
  for v in ordered:
   shape=shapes[v.name];need(v.dtype=='float64'and v.ndim==len(shape),'coordinate type/rank mismatch')
   need(all(n is None or n==m for n,m in zip(v.type.shape,shape)),'static coordinate shape mismatch')
   size=int(np.prod(shape,dtype=int))if shape else 1;segments.append(dict(name=v.name,shape=shape,dtype=v.dtype,start=offset,stop=offset+size));offset+=size
  need(offset==662,'full coordinate dimension changed');(out/'coordinate-order.json').write_text(json.dumps(segments,indent=2)+'\n')
  persist('exact_native_continuous_value_order_saved',dict(dimensions=offset,variables=len(ordered),order='model.continuous_value_vars; native gradient helper flatten/concat'))
  old_port=model.typed_ports['exposure'];need(hasattr(old_port,'data')and np.array_equal(old_port.data,np.asarray(parent['exposure'])),'exact original exposure constant required')
  cost=model.logp(jacobian=True);before=set(ancestors([cost]));need(old_port in before,'exact exposure identity absent from original complete logp')
  E=pt.dmatrix('authenticated_whole_parent_exposure');replacement=pt.specify_shape(E,(3,24));replacement=CheckAndRaise(ValueError,'positive finite whole-parent exposure required')(replacement,pt.all((replacement>0)&~pt.isnan(replacement)&~pt.isinf(replacement)))
  need(E not in ordered_set,'conditioning input cannot become a sampled coordinate')
  replaced=graph_replace(cost,{old_port:replacement},strict=True)
  original_inputs={v for v in graph_inputs([cost])if v in ordered_set}
  def structure(label,expression):
   nodes=set(ancestors([expression]));inputs=set(graph_inputs([expression]));need(E in inputs,label+': explicit E identity/dependence lost');need(old_port not in nodes,label+': original exposure node retained')
   stale=[]
   for node in nodes:
    if hasattr(node,'data'):
     data=np.asarray(node.data)
     if data.shape==(3,24)and any(np.array_equal(data,np.asarray(row['exposure']))for row in s['parents']):stale.append(str(node))
   need(not stale,label+': stale whole-parent constant detected')
   return dict(expression=label,ancestor_variables=len(nodes),explicit_E_input=True,old_port_absent=True,stale_whole_parent_arrays=0,value_input_names=sorted(v.name for v in inputs if v in ordered_set))
  pre=structure('complete_replaced_logp',replaced);need({v for v in graph_inputs([replaced])if v in ordered_set}==original_inputs,'original value inputs changed during replacement');persist('strict_logp_replacement_passed',pre)
  # Preserve the actual native Model.dlogp path: rewrite_pregrad, THEN native gradient helper.
  rewritten=rewrite_pregrad(replaced)
  post=structure('rewrite_pregrad_replaced_logp',rewritten);need({v for v in graph_inputs([rewritten])if v in ordered_set}==original_inputs,'value input set changed under pregradient rewrite');persist('native_pregradient_rewrite_checked',post)
  derived=pt.specify_shape(gradient(rewritten,ordered),(662,))
  final=structure('native_order_derived_gradient',derived);need(derived.ndim==1,'gradient must be one vector');persist('gradient_after_replacement_checked',final)
  need(p.dict_plain(model.component_inventory)==inventory,'symbolic work mutated factors or ownership')
  # Save symbolic artifacts only; trusted-pickle bytes must be SHA verified before any future load.
  import pickle
  with (out/'symbolic-graphs.pkl').open('xb')as f:pickle.dump(dict(original_cost=cost,replaced_cost=replaced,rewritten_cost=rewritten,gradient=derived,ordered_value_vars=ordered,E=E,segments=segments),f,protocol=5)
  result=dict(schema='GradientAfterStrictExposureReplacementSymbolicV1',status='symbolic_only_hypothesis_passed_not_numeric',first_parent_contract=contract,checks=[pre,post,final],coordinate_segments=segments,dimensions=662,gradient_path='model.logp(jacobian=True) -> strict identity exposure replace -> pinned pymc.rewrite_pregrad -> pinned pymc.gradient(cost,model.continuous_value_vars)',mass_argument=False,numerical_compilation=False,numerical_point_saved=False,numerical_gradient_saved=False,scientific_qualification=False,notes=['Exposure remains conditioning input only.','No second-parent target evaluated.','Absence of equal whole exposure constants is a diagnostic, not a full algebraic independence proof.','Complete future numeric persistence contract is in plan.json.'])
  (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');persist('completed_symbolic_only',dict(status=result['status']))
 except BaseException:
  (out/'failure.log').write_text(traceback.format_exc());persist('failed_closed',dict(error=traceback.format_exc()));raise

def main():
 args=argparse.ArgumentParser();args.add_argument('--plan',required=True);args.add_argument('--plan-sha',required=True);args.add_argument('--execution-receipt',required=True);args.add_argument('--execution-receipt-sha',required=True);args.add_argument('--output',required=True);a=args.parse_args()
 plan,receipt=gate(a.plan,a.plan_sha,a.execution_receipt,a.execution_receipt_sha)
 import resource
 limits=plan['resource_caps']
 for kind,bound in [(resource.RLIMIT_CPU,limits['cpu_hard_seconds']),(resource.RLIMIT_AS,limits['address_space_bytes']),(resource.RLIMIT_FSIZE,limits['individual_file_bytes'])]:
  value=resource.getrlimit(kind)[1];need(value!=resource.RLIM_INFINITY and value<=bound,'bounded launcher hard limits required')
 need(os.getppid()==int(os.environ.get('M01_SYMBOLIC_WATCHDOG_PARENT_PID','-1')),'live dedicated watchdog launcher required')
 symbolic(plan,receipt,a.output)
if __name__=='__main__':main()
