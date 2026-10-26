"""Gated actual evaluator. No graph imports until live inherited entry admission."""
import importlib.util,json,os,sys,time,traceback,warnings
from pathlib import Path
from common import *
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def main():
 phase=sys.argv[1];out=Path(sys.argv[2]).resolve()
 from entry import accept
 auth,c,entry=accept(phase,out);atomic_json(out/'entry.json',entry);need(phase in ['dynamic','fixed-B'],'unsupported phase')
 start=time.monotonic()
 def stage(name):atomic_json(out/'stage.json',{'stage':name,'pid':os.getpid(),'seconds':time.monotonic()-start});print(name,flush=True)
 stage('entry_authenticated_before_graph_import')
 import numpy as np,pymc as pm,pytensor,pytensor.tensor as pt,numba,llvmlite,arviz
 from pytensor.graph.replace import graph_replace
 from pytensor.graph.traversal import ancestors,graph_inputs
 from pytensor.raise_op import CheckAndRaise
 import evaluator as ev
 need(pytensor.config.cxx==''and pytensor.config.numba__fastmath is False,'runtime/compiler law mismatch')
 
 # Expose only the existing lab schema overlay, never owner site-packages.
 # Provider installs this overlay; import schema packages after provider below if unavailable.
 q=load('owned_frozen_compressed_provider',c['compressed_provider']);p=q.p
 import pydantic,pydantic_core
 modules={'numpy':np,'pymc':pm,'pytensor':pytensor,'numba':numba,'llvmlite':llvmlite,'arviz':arviz,'pydantic':pydantic,'pydantic_core':pydantic_core}
 for name,expected in c['runtime_expected'].items():need(modules[name].__version__==expected['version']and Path(modules[name].__file__).resolve()==Path(expected['path']).resolve(),'runtime substitution '+name)
 bindings=load('owned_frozen_parent_binding',c['parent_binding']);m=bindings.load();records=[row['record']for row in m['parents']];s=p.selection();problem=json.loads(Path(s['problem_path']).read_text());parent=records[0]if phase=='dynamic'else records[c['anchor_index']]
 p.validate(problem,parent,s);need(records==s['parents']and [r['mass']for r in records]==[.125]*8,'parent bank changed')
 refs=sys.modules['m01_successor_native.references'];rolepath=Path(c['role_manifest']);rolemanifest=json.loads(rolepath.read_text());old=json.loads((p.OWNER/rolemanifest['bound_prepared_problem']['path']).read_text());roles=refs.roles.load_role_binding(old,rolepath,c['role_manifest_sha256'],admission_cutoff_utc=problem['cutoff_utc'],root=p.OWNER);roles=refs.roles.rebind_role_binding(old,problem,roles)
 reg=p.components.standard_registry();provider=q.CompressedM01('actual_calibration_bank');reg.register(provider);comp=p.components.baseline_composition(role_hierarchy=True);comp=p.graph.Composition(tuple((r,provider.implementation_id if r=='measurement'else n)for r,n in comp.selections),comp.standardized,comp.person_coordinates)
 stage('building_one_complete_native_'+phase+'_graph');model=p.graph.compile_model(reg,comp,problem,parent,roles=roles);initial=model.initial_point();ordered=list(model.continuous_value_vars);need([v.name for v in ordered]==[v.name for v in model.value_vars],'unsupported coordinate roster');layout=[{'name':v.name,'shape':list(initial[v.name].shape),'size':int(initial[v.name].size),'dtype':str(initial[v.name].dtype)}for v in ordered];need(sum(row['size']for row in layout)==662,'full coordinate count changed');need(layout==json.loads(Path(c['saved_layout']).read_text()),'saved coordinate identity/order mismatch')
 contract=p.dict_plain(model.joint_contract);need(contract['parent_identity']==p.graph.digest(parent)and contract['parent_mass']==.125 and contract['parent_mass_used_in_logp']is False and contract['inference_target']=='population_cut_v1','protected contract mismatch')
 need(contract['composition_identity']==c['reference_composition_identity'],'static provider composition changed');need(contract['scientific_target_id']==p.graph.digest({'composition_id':contract['composition_identity'],'problem':contract['problem_identity'],'parent':p.graph.digest(parent),'target':'population_cut_v1'}),'native target identity mismatch')
 inventory=p.dict_plain(model.component_inventory);need(inventory==json.loads(Path(c['saved_model_envelope']).read_text())['component_inventory'],'evidence ownership changed')
 with np.load(c['saved_points'],allow_pickle=False)as data:points=[{v.name:data[str(i)+'__'+v.name].copy()for v in ordered}for i in range(2)]
 points.append({k:v.copy()for k,v in points[1].items()})
 for name,delta in c['third_point_deltas'].items():need(name in points[2],'missing third-point coordinate');points[2][name].flat[0]+=delta
 frozen_points=[ev.point(layout,point)for point in points]
 factor_exprs=model.logp(jacobian=True,sum=False);cost=model.logp(jacobian=True);poll=model.logp(vars=[model['M01_complete_poll_vector']],jacobian=True);contests=[row['id']for row in problem['contests']];rates=[model.replace_rvs_by_values([model['category_intensity_'+name]])[0]for name in contests];physical_names=['eventG','event_intensity'];physical=[model.replace_rvs_by_values([model[name]])[0]for name in physical_names];expressions=[cost,poll,*factor_exprs,*rates,*physical];E=None
 if phase=='dynamic':
  port=model.typed_ports['exposure'];need(hasattr(port,'data')and np.array_equal(port.data,np.asarray(parent['exposure']))and port in set(ancestors([cost])),'exact exposure port missing');E=pt.dmatrix('owned_whole_parent_exposure');replacement=CheckAndRaise(ValueError,'whole positive finite exposure required')(pt.specify_shape(E,(3,24)),pt.all((E>0)&~pt.isnan(E)&~pt.isinf(E)));expressions=graph_replace(expressions,{port:replacement},strict=True);nodes=set(ancestors(expressions));need(port not in nodes and E in set(graph_inputs(expressions)),'strict exposure replacement identity failed');need(E not in ordered,'parent exposure cannot be sampled');need(E in set(graph_inputs([expressions[0]])),'complete logp lost conditioning input');need({v for v in graph_inputs(expressions)if v in set(ordered)}==set(ordered),'original coordinate dependence changed')
  stale=[]
  for node in nodes:
   if hasattr(node,'data'):
    array=np.asarray(node.data)
    if array.shape==(3,24)and any(np.array_equal(array,np.asarray(row['exposure']))for row in records):stale.append(str(node))
  need(not stale,'stale whole-parent constant remains in dynamic outputs')
  atomic_json(out/'replacement-structure.json',{'strict_identity_replaced':True,'all_outputs_old_port_absent':True,'complete_logp_E_input':True,'stale_whole_parent_constants':0,'value_coordinates':662})
 need(json.loads(bindings.bind(parent).envelope_bytes)['reference_fixed_parent_target_id']==contract['scientific_target_id'],'compiled parent target differs from authenticated formula reference')
 provenance={'contract_sha256':EXPECTED_CONTRACT,'phase':phase,'pid':os.getpid(),'execution_scope':'actual bounded graph execution only if this gated child later runs','compiler_parent_contract':contract,'model_inventory_sha256':digest(inventory),'coordinate_layout':layout,'parent_source_manifest_sha256':bindings.sha(Path(c['parent_binding']).parent/'parents-manifest.json'),'point_source':pin(c['saved_points']),'source_bundle_sha256':digest(c['pins']),'gradient':'not_constructed_logp_only','conditional_normalization':'unverified','runtime':{n:{'version':mod.__version__,'path':mod.__file__}for n,mod in modules.items()}}
 atomic_json(out/'kernel.json',provenance);atomic_json(out/'model-envelope.json',{'contract':contract,'inventory':inventory,'layout':layout,'factor_count':len(factor_exprs),'contests':contests,'physical':physical_names});stage('single_NUMBA_compile_logp_all_factors_rates_physical')
 with warnings.catch_warnings(record=True)as notes:
  warnings.simplefilter('always');fn=model.compile_fn(expressions,inputs=[*ordered,*([E]if E is not None else[])],point_fn=False,mode='NUMBA');factor_count=len(factor_exprs)
  def engine(exposure,values):
   if phase=='fixed-B':need(np.array_equal(exposure,np.asarray(parent['exposure'])),'fixed graph parent input mismatch')
   values=fn(*values,*([exposure]if E is not None else[]));offset=2+factor_count;return {'logp':float(values[0]),'poll_logp':float(values[1]),'factor_values':[np.asarray(v).tolist()for v in values[2:offset]],'rates':{name:np.asarray(v).tolist()for name,v in zip(contests,values[offset:offset+len(contests)])},'physical':{name:np.asarray(v).tolist()for name,v in zip(physical_names,values[offset+len(contests):])}}
  session=ev._session(bindings,engine,provenance);rows=[];indices=list(range(8))+[0]if phase=='dynamic'else[c['anchor_index']]
  for sequence,index in enumerate(indices):
   bound=bindings.bind(records[index])
   for pi,point in enumerate(frozen_points):
    stage('owned_call_parent_'+str(index)+'_point_'+str(pi));raw=session.evaluate(bound,point);name='call-'+str(sequence)+'-'+str(index)+'-'+str(pi)+'.json';path=out/name
    with path.open('xb')as file:file.write(raw);file.flush();os.fsync(file.fileno())
    rows.append({'parent_index':index,'point_index':pi,'file':name,'pin':pin(path),'point_sha256':json.loads(raw)['point_sha256'],'parent_identity':json.loads(raw)['parent_envelope']['parent_identity']})
  # Full-logp finite differences at the same B point; persist both complete calls.
  fd=[];bound=bindings.bind(records[c['anchor_index']])
  for variable in c['finite_difference_variables']:
   pair=[]
   for sign in [-1,1]:
    changed={key:value.copy()for key,value in points[1].items()};changed[variable].flat[0]+=sign*c['fd_step'];raw=session.evaluate(bound,ev.point(layout,changed));name='fd-'+variable+'-'+str(sign)+'.json'
    with (out/name).open('xb')as file:file.write(raw);file.flush();os.fsync(file.fileno())
    pair.append({'file':name,'pin':pin(out/name),'value':json.loads(raw)['output']['logp']})
   fd.append({'variable':variable,'coordinate':0,'step':c['fd_step'],'derivative':(pair[1]['value']-pair[0]['value'])/(2*c['fd_step']),'calls':pair})
  atomic_json(out/'finite-differences.json',fd)
  # Persist every owned call before later comparison gates; a call list is not completion.
  atomic_json(out/'call-index.json',rows)
 read_contract();bindings.load();result={'schema':'OwnedParentEvaluatorPhaseV1','status':'completed_calls_not_normalized_conditionals','phase':phase,'contract_sha256':EXPECTED_CONTRACT,'anchor_index':c['anchor_index'],'calls':rows,'artifacts':{n:pin(out/n)for n in ['entry.json','kernel.json','model-envelope.json','call-index.json','finite-differences.json',*(['replacement-structure.json']if phase=='dynamic'else[])]},'warnings':[str(w.message)for w in notes],'seconds':time.monotonic()-start,'all_other_fixed_parent_baselines':'missing_except_saved_parent0_and_separate_B_phase','sampling':False,'conditional_normalization':'unverified'};atomic_json(out/'result.json',result);stage('completed_owned_calls')
if __name__=='__main__':
 try:main()
 except BaseException:
  if len(sys.argv)>2:Path(sys.argv[2],'failure.log').write_text(traceback.format_exc())
  raise
