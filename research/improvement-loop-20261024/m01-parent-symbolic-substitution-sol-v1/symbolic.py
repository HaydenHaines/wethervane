"""One actual symbolic-only replacement probe; NO numerical graph compilation."""
import hashlib,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pytensor.graph.replace import graph_replace
from pytensor.graph.traversal import ancestors,graph_inputs
from pytensor.raise_op import CheckAndRaise
HERE=Path(__file__).resolve().parent;LAB=HERE.parent;OLD=LAB/'m01-native-successor-provider-sol-v1'
start=time.monotonic()
def stage(name):
 (HERE/'stage.json').write_text(json.dumps(dict(stage=name,pid=os.getpid(),elapsed=time.monotonic()-start))+'\n');print(name,flush=True)
def pin(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest(),bytes=Path(p).stat().st_size)
try:
 stage('authenticate_frozen_sources')
 expected={'provider.py':'6154ecb3d66ad38b5ba5fa94719d1da865f3e32bdf749ef1bb4ed4250606d853','selection.json':'03edb47049384b662cdd645053fc6083bde8fb41d972b1a07da8b71ecf6283b5'}
 for name,sha in expected.items():assert pin(OLD/name)['sha256']==sha
 sys.path.append(str(OLD/'schema-overlay'));sys.path.insert(0,str(OLD));import provider as p
 s=p.selection();problem=json.loads(Path(s['problem_path']).read_text());parent=s['parents'][0];p.validate(problem,parent,s)
 rolepath=p.OWNER/'research/senate-stack-integration-20261020/candidate-role-source-sol-v1/prepared-v2/candidate-role-manifest.json';rm=json.loads(rolepath.read_text());original=json.loads((p.OWNER/rm['bound_prepared_problem']['path']).read_text());refs=sys.modules['m01_successor_native.references']
 roles=refs.roles.load_role_binding(original,rolepath,'a1d73f7dece21a4cb5b104c69caec5157e4dd106b411ce17833f7f5a41bbeb3d',admission_cutoff_utc=problem['cutoff_utc'],root=p.OWNER);roles=refs.roles.rebind_role_binding(original,problem,roles)
 reg=p.components.standard_registry();measurement=p.M01Measurement('actual_calibration_bank');reg.register(measurement);comp=p.components.baseline_composition(role_hierarchy=True);comp=p.graph.Composition(tuple((role,measurement.implementation_id if role=='measurement'else name)for role,name in comp.selections),comp.standardized,comp.person_coordinates)
 stage('build_original_first_parent_graph');model=p.graph.compile_model(reg,comp,problem,parent,roles=roles)
 contract=p.dict_plain(model.joint_contract);parent_id=p.graph.digest(parent)
 assert contract['parent_identity']==parent_id and contract['parent_mass']==.125 and contract['parent_mass_used_in_logp']is False and contract['inference_target']=='population_cut_v1'
 assert contract['scientific_target_id']==p.graph.digest(dict(composition_id=contract['composition_identity'],problem=contract['problem_identity'],parent=parent_id,target='population_cut_v1'))
 port=model.typed_ports['exposure'];raw=np.asarray(parent['exposure']);assert raw.shape==(3,24)
 # The numerical source array itself must exactly match the typed constant.
 assert hasattr(port,'data')and np.array_equal(port.data,raw)
 stage('construct_complete_original_jacobian_logp_and_dlogp');logp=model.logp(jacobian=True);gradient=model.dlogp(jacobian=True)
 E=pt.dmatrix('explicit_whole_parent_exposure');replacement=pt.specify_shape(E,(3,24));replacement=CheckAndRaise(ValueError,'positive finite authenticated whole-parent exposure required')(replacement,pt.all((replacement>0)&~pt.isnan(replacement)&~pt.isinf(replacement)))
 outcomes=[]
 for name,expression in [('complete_logp',logp),('complete_dlogp',gradient)]:
  stage('strict_replace_'+name)
  before=set(ancestors([expression]));before_inputs=set(graph_inputs([expression]));assert port in before
  try:
   changed=graph_replace(expression,{port:replacement},strict=True)
   after=set(ancestors([changed]));after_inputs=set(graph_inputs([changed]));assert E in after_inputs and port not in after
   # Ordinary transformed value inputs retained; no new stochastic RV or prior.
   transformed_before={v.name for v in model.value_vars if v in before_inputs};transformed_after={v.name for v in model.value_vars if v in after_inputs};assert transformed_before==transformed_after
   outcomes.append(dict(expression=name,status='strict_identity_replacement_succeeded',original_port_in_ancestors=True,original_port_removed=True,explicit_exposure_is_input=True,original_value_input_names=sorted(transformed_before),new_value_input_names=sorted(transformed_after),ancestor_nodes_before=len(before),ancestor_nodes_after=len(after)))
  except Exception as error:
   (HERE/(name+'-failure.log')).write_text(traceback.format_exc());outcomes.append(dict(expression=name,status='failed',error=repr(error)))
 # Parent target family is metadata here; no second graph or callable is built.
 envelopes=[]
 for item in s['parents']:
  p.validate(problem,item,s)
  identity=p.graph.digest(item);target=p.graph.digest(dict(composition_id=contract['composition_identity'],problem=contract['problem_identity'],parent=identity,target='population_cut_v1'))
  envelopes.append(dict(parent_id=item['parent_id'],parent_identity=identity,mass=item['mass'],target_if_same_composition_and_parent=target,graph_built=item==parent))
 assert len({e['parent_identity']for e in envelopes})==len({e['target_if_same_composition_and_parent']for e in envelopes})==8
 stage('completed_without_numerical_compilation')
 result=dict(schema='ActualTXStrictExposureSymbolicSubstitutionV1',status='symbolic_replacement_only_no_numerical_reuse_proven',pid=os.getpid(),seconds=time.monotonic()-start,checks=outcomes,original_contract=contract,parent_target_envelopes=envelopes,exposure_port=dict(type=type(port).__module__+'.'+type(port).__name__,shape=[3,24],dtype=str(port.dtype),exact_array_equal=True),conditioning_input=dict(name=E.name,dtype=E.dtype,shape_gate=[3,24],positive_finite_gate=True,stochastic=False,mass_input=False),pins={name:pin(OLD/name)for name in expected},source_dependency_pins=s['pins'],runtime=dict(executable=sys.executable,pytensor=pytensor.__version__,pymc=pm.__version__,numpy=np.__version__,cxx=pytensor.config.cxx),scope='Actual first-parent graph construction and symbolic replacement only. No function/linker compile, full gradient evaluation, other-parent graph, normalization, sampler or forecast. Existing first-parent contract is unchanged and cannot label another parent evaluation.')
 (HERE/'result.json').write_text(json.dumps(result,indent=2)+'\n')
except BaseException:
 (HERE/'failure.log').write_text(traceback.format_exc());raise
