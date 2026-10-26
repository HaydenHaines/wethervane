"""Actual graph child: unreachable in preparation tests; controller-only invocation."""
import json,os,sys,time,traceback,warnings,importlib.util
from pathlib import Path
from common import *
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def main():
 arm=sys.argv[1];out=Path(sys.argv[2]).resolve();authorization=json.loads((out/'authorization.json').read_text());c=read_contract()
 need(authorization['contract_sha256']==EXPECTED_CONTRACT and authorization['arms']==['original','compressed'],'controller binding missing')
 need(arm in authorization['arms']and out.name==arm,'arm mismatch');need(os.getppid()==authorization['controller_pid'],'controller identity changed')
 start=time.monotonic()
 def stage(name):atomic_json(out/'stage.json',{'stage':name,'pid':os.getpid(),'seconds':time.monotonic()-start});print(name,flush=True)
 stage('authenticated_before_model_import')
 import numpy as np,pytensor,pymc as pm
 need(pytensor.config.cxx==''and pytensor.config.numba__fastmath is False,'compiler law changed')
 q=load('equivalence_compressed_provider',c['compressed_provider'])
 import numba,llvmlite,arviz,pydantic,pydantic_core
 runtime_modules={'numpy':np,'pytensor':pytensor,'pymc':pm,'numba':numba,'llvmlite':llvmlite,'arviz':arviz,'pydantic':pydantic,'pydantic_core':pydantic_core}
 for name,expected in c['runtime_expected'].items():need(runtime_modules[name].__version__==expected['version']and str(Path(runtime_modules[name].__file__).resolve())==str(Path(expected['path']).resolve()),'runtime substitution '+name)
 p=q.p;s=p.selection();problem=json.loads(Path(s['problem_path']).read_text());parent=s['parents'][0]
 need(p.graph.digest(problem)==c['problem_identity']and digest(parent)==c['parent_digest'],'problem/parent mismatch')
 need([row['mass']for row in s['parents']]==[.125]*8,'protected masses changed');p.validate(problem,parent,s)
 refs=sys.modules['m01_successor_native.references'];rolepath=Path(c['role_manifest']);rm=json.loads(rolepath.read_text());old=json.loads((p.OWNER/rm['bound_prepared_problem']['path']).read_text())
 roles=refs.roles.load_role_binding(old,rolepath,c['role_manifest_sha256'],admission_cutoff_utc=problem['cutoff_utc'],root=p.OWNER);roles=refs.roles.rebind_role_binding(old,problem,roles)
 provider=p.M01Measurement(c['mode'])if arm=='original'else q.CompressedM01(c['mode']);reg=p.components.standard_registry();reg.register(provider);composition=p.components.baseline_composition(role_hierarchy=True);composition=p.graph.Composition(tuple((r,provider.implementation_id if r=='measurement'else n)for r,n in composition.selections),composition.standardized,composition.person_coordinates)
 stage('building_complete_actual_graph');model=p.graph.compile_model(reg,composition,problem,parent,roles=roles);initial=model.initial_point()
 ordered=model.continuous_value_vars;need([v.name for v in ordered]==[v.name for v in model.value_vars],'noncontinuous or reordered coordinates unsupported')
 layout=[{'name':v.name,'shape':list(initial[v.name].shape),'size':int(initial[v.name].size),'dtype':str(initial[v.name].dtype)}for v in ordered];need(sum(x['size']for x in layout)==662,'full coordinate count changed')
 # Original child persists exact shared states first; compressed child requires exact initial equality.
 points_path=out.parent/'original'/'points.npz';layout_path=out.parent/'original'/'layout.json'
 if arm=='original':
  points=[{v.name:np.array(initial[v.name],copy=True)for v in ordered}for _ in range(2)]
  for name,delta in c['perturbations'].items():need(name in initial,'missing perturbation coordinate '+name);points[1][name].flat[0]+=delta
  np.savez(out/'points.npz',**{str(i)+'__'+k:v for i,point in enumerate(points)for k,v in point.items()});atomic_json(out/'layout.json',layout)
 else:
  receipt=json.loads((out.parent/'original'/'result.json').read_text());need(pin(points_path)==receipt['artifacts']['points.npz']and pin(layout_path)==receipt['artifacts']['layout.json'],'original states not authenticated');need(layout==json.loads(layout_path.read_text()),'coordinate identity/order changed')
  with np.load(points_path,allow_pickle=False)as data:points=[{v.name:data[str(i)+'__'+v.name].copy()for v in ordered}for i in range(2)]
  need(all(np.array_equal(initial[k],points[0][k])for k in initial),'initial point changed');np.savez(out/'points.npz',**{str(i)+'__'+k:v for i,point in enumerate(points)for k,v in point.items()});atomic_json(out/'layout.json',layout)
 stage('constructing_Jacobian_inclusive_ordered_full_gradient');logp=model.logp(jacobian=True);gradient=model.dlogp(jacobian=True);factors=model.logp(jacobian=True,sum=False);poll=model.logp(vars=[model['M01_complete_poll_vector']],jacobian=True);contest=problem['polls'][0]['rows'][0]['contest_id'];rate=model.replace_rvs_by_values([model['category_intensity_'+contest]])[0]
 with warnings.catch_warnings(record=True)as notes:
  warnings.simplefilter('always');stage('one_NUMBA_compile_full_logp_gradient_factors_rates');fn=model.compile_fn([logp,gradient,poll,rate,*factors],inputs=model.value_vars,point_fn=True,mode='NUMBA')
  values=[];gradients=[];polls=[];rates=[];factor_values=[]
  for i,point in enumerate(points):
   stage('evaluating_complete_point_'+str(i));res=fn(point);value=float(res[0]);g=np.asarray(res[1]);need(g.shape==(662,)and np.isfinite(g).all()and np.isfinite(value),'nonfinite/incomplete gradient');values.append(value);gradients.append(g);polls.append(float(res[2]));rates.append(np.asarray(res[3]));factor_values.append([float(np.sum(a))for a in res[4:]]);need(abs(sum(factor_values[-1])-value)<1e-6,'Jacobian-inclusive factor sum differs')
  # Preserve complete numerical evaluations before later finite-difference gates can fail.
  np.savez(out/'evaluated-before-fd.npz',logp=values,gradient=np.stack(gradients),poll_logp=polls,rates=np.stack(rates),factor_logp=np.array(factor_values))
  atomic_json(out/'evaluation-receipt.json',{'artifact':pin(out/'evaluated-before-fd.npz'),'points':pin(out/'points.npz'),'layout':pin(out/'layout.json'),'status':'numerical_values_recorded_not_completion'})
  # Full-joint finite differences at the shared perturbed state, not poll-only checks.
  checks=[];offsets={};offset=0
  for row in layout:offsets[row['name']]=offset;offset+=row['size']
  for name in c['perturbations']:
   plus={k:v.copy()for k,v in points[1].items()};minus={k:v.copy()for k,v in points[1].items()};h=c['fd_step'];plus[name].flat[0]+=h;minus[name].flat[0]-=h;fd=float((fn(plus)[0]-fn(minus)[0])/(2*h));analytic=float(gradients[1][offsets[name]]);tol=c['fd_atol']+c['fd_rtol']*abs(fd);need(abs(fd-analytic)<=tol,'full-joint finite difference mismatch '+name);checks.append({'name':name,'flat_index':offsets[name],'step':h,'numeric':fd,'analytic':analytic,'error':abs(fd-analytic),'tolerance':tol})
 np.savez(out/'numerical.npz',logp=values,gradient=np.stack(gradients),poll_logp=polls,rates=np.stack(rates),factor_logp=np.array(factor_values))
 atomic_json(out/'model-envelope.json',{'composition_record':p.dict_plain(model.composition_record),'component_inventory':p.dict_plain(model.component_inventory),'joint_contract':p.dict_plain(model.joint_contract),'parent':parent,'model_ids':s['model_ids'],'provider_spec':p.dict_plain(provider.spec.to_dict())if hasattr(provider.spec,'to_dict')else {'implementation_id':provider.implementation_id,'configuration':provider.spec.configuration,'dependency_pins':list(provider.spec.dependency_pins)}})
 import numba,llvmlite,arviz,pydantic,pydantic_core
 result={'schema':'ActualCompressedEquivalenceArmV1','status':'completed','arm':arm,'mode':c['mode'],'parent_digest':c['parent_digest'],'contract_sha256':EXPECTED_CONTRACT,'pid':os.getpid(),'seconds':time.monotonic()-start,'coordinates':662,'finite_difference_checks':checks,'warnings':[str(w.message)for w in notes],'artifacts':{n:pin(out/n)for n in ['points.npz','layout.json','numerical.npz','model-envelope.json','evaluated-before-fd.npz','evaluation-receipt.json']},'runtime':{n:{'version':m.__version__,'path':m.__file__}for n,m in [('numpy',np),('pytensor',pytensor),('pymc',pm),('numba',numba),('llvmlite',llvmlite),('arviz',arviz),('pydantic',pydantic),('pydantic_core',pydantic_core)]}}
 read_contract();atomic_json(out/'result.json',result);stage('completed')
if __name__=='__main__':
 try:main()
 except BaseException:
  if len(sys.argv)>2:
   path=Path(sys.argv[2])/'failure.log';path.write_text(traceback.format_exc())
  raise
