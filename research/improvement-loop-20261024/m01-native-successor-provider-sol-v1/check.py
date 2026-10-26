"""Actual complete graph construction and supplied-state factor/gradient checks. No fit."""
import copy,json,hashlib,time
from pathlib import Path
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
import sys
sys.path.append(str(Path(__file__).resolve().parent/'schema-overlay'))
import provider as p
HERE=Path(__file__).resolve().parent
s=p.selection();problem=json.loads(Path(s['problem_path']).read_text());parent=s['parents'][0]
refs=p.native.__dict__.get('references')or __import__('sys').modules['m01_successor_native.references']
rolepath=p.OWNER/'research/senate-stack-integration-20261020/candidate-role-source-sol-v1/prepared-v2/candidate-role-manifest.json'
rm=json.loads(rolepath.read_text());old=json.loads((p.OWNER/rm['bound_prepared_problem']['path']).read_text())
roles=refs.roles.load_role_binding(old,rolepath,'a1d73f7dece21a4cb5b104c69caec5157e4dd106b411ce17833f7f5a41bbeb3d',admission_cutoff_utc=problem['cutoff_utc'],root=p.OWNER)
roles=refs.roles.rebind_role_binding(old,problem,roles)
def build(mode):
 reg=p.components.standard_registry();provider=p.M01Measurement(mode);reg.register(provider)
 comp=p.components.baseline_composition(role_hierarchy=True)
 comp=p.graph.Composition(tuple((r,provider.implementation_id if r=='measurement'else n)for r,n in comp.selections),comp.standardized,comp.person_coordinates)
 return p.graph.compile_model(reg,comp,problem,parent,roles=roles)
start=time.monotonic();print('building actual reduction graph',flush=True);model=build('incumbent_reduction');print('built',len(model.free_RVs),'free RVs',flush=True)
# Only target factor and its actual upstream rate dependencies are numerically compiled.
# Full graph construction nevertheless executes all original providers and ownership gates.
point=model.initial_point();factor=model['M01_complete_poll_vector']
fn=model.compile_logp(vars=[factor])
value=float(fn(point));print('compiled actual factor',value,flush=True)
# Retrieve actual symbolic pair means by deterministic category intensity output.
contest=problem['polls'][0]['rows'][0]['contest_id'];rate=model['category_intensity_'+contest]
ratefn=model.compile_fn(model.replace_rvs_by_values([rate])[0],inputs=model.value_vars,on_unused_input='ignore',point_fn=True);rv=np.asarray(ratefn(point))
names=[c['id']for c in next(c for c in problem['contests']if c['id']==contest)['candidates']]
means=np.array([np.log(rv[names.index(r['first_candidate_id'])])-np.log(rv[names.index(r['second_candidate_id'])])for r in problem['polls'][0]['rows']]);y=np.array([r['logit_value']for r in problem['polls'][0]['rows']]);reference=multivariate_normal.logpdf(y,mean=means,cov=s['incumbent_covariance']);assert abs(value-reference)<1e-8,(value,reference)
# Full 4000-component actual bank oracle and pure symbolic gradient at actual graph means.
mixture=p.mixture('actual_calibration_bank')
with np.load(HERE/'reference-mixture.npz',allow_pickle=False)as reference_arrays:
 assert np.array_equal(mixture.covariances,reference_arrays['covariances'])
 assert np.array_equal(mixture.weights,reference_arrays['weights'])
mu=pt.dvector('supplied_actual_mean');logp=mixture.symbolic_logpdf(y,mu);f=pytensor.function([mu],[logp,pt.grad(logp,mu)]);multi,grad=f(means)
oracle=logsumexp([np.log(w)+multivariate_normal.logpdf(y,mean=means,cov=c)for w,c in zip(mixture.weights,mixture.covariances)]);assert abs(multi-oracle)<1e-9
fd=np.array([(f(means+np.eye(6)[k]*1e-6)[0]-f(means-np.eye(6)[k]*1e-6)[0])/(2e-6)for k in range(6)]);assert np.max(abs(fd-grad))<1e-6
print('building actual bank graph',flush=True);multi_model=build('actual_calibration_bank');print('built actual bank graph',flush=True)
multi_fn=multi_model.compile_logp(vars=[multi_model['M01_complete_poll_vector']]);multi_actual=float(multi_fn(multi_model.initial_point()));assert abs(multi_actual-oracle)<1e-8
negative=[]
def rejection(name,fun):
 try:fun()
 except (ValueError,AssertionError)as e:negative.append(dict(case=name,error=str(e)));return
 raise AssertionError('accepted '+name)
for name in ['split_blocks','duplicate_poll','candidate_reorder','covariance_double','poll_order','source_value']:
 bad=copy.deepcopy(problem)
 if name=='split_blocks':bad['polls'].append(copy.deepcopy(bad['polls'][0]))
 elif name=='duplicate_poll':bad['polls'][0]['rows'][1]=copy.deepcopy(bad['polls'][0]['rows'][0])
 elif name=='candidate_reorder':bad['contests'][-1]['candidates'].reverse()
 elif name=='covariance_double':bad['polls'][0]['covariance']=(2*np.array(bad['polls'][0]['covariance'])).tolist()
 elif name=='poll_order':bad['polls'][0]['rows'].reverse()
 else:bad['polls'][0]['rows'][0]['logit_value']+=.01
 rejection(name,lambda bad=bad:p.validate(bad,parent,s))
bad=copy.deepcopy(parent);bad['mass']=.25;rejection('protected_parent_mass',lambda:p.validate(problem,bad,s))
bad=copy.deepcopy(s);bad['variance_owner']='extra_sampling';rejection('second_variance_owner',lambda:p.validate(problem,parent,bad))
rejection('coherent_selection_rewrite',lambda:p.parse_selection(p.SELECTION.read_bytes()+b' '))
# Complete native witness must own exactly the original 4304 atoms.
def receipt(m):
 inv=m.component_inventory
 return p.dict_plain(inv)
print('compiling complete actual logp/gradient',flush=True)
full_reduction=model.compile_logp(sum=False);full_multi=multi_model.compile_logp(sum=False);total_reduction=float(sum(np.sum(v)for v in full_reduction(point)));total_multi=float(sum(np.sum(v)for v in full_multi(multi_model.initial_point())))
assert np.isfinite([total_reduction,total_multi]).all()
assert abs((total_multi-total_reduction)-(multi_actual-value))<1e-6
full_fd=[]
for name in ['choice_intercept','turnout_intercept']:
 estimates=[]
 for step in [1e-5,5e-6]:
  plus=copy.deepcopy(multi_model.initial_point());minus=copy.deepcopy(plus);plus[name]=plus[name]+step;minus[name]=minus[name]-step
  estimates.append(float((sum(np.sum(v)for v in full_multi(plus))-sum(np.sum(v)for v in full_multi(minus)))/(2*step)))
 full_fd.append(dict(variable=name,central_differences=estimates,step_consistency_error=abs(estimates[0]-estimates[1])))
assert max(x['step_consistency_error']for x in full_fd)<.01
# Exact unchanged nonpoll ownership receipts, and one complete six-token poll owner.
a=p.dict_plain(model.component_inventory);b=p.dict_plain(multi_model.component_inventory)
assert a==b
witness=json.loads((p.NATIVE/'graph-inventory-v1.json').read_text());incumbent=next(x for x in witness['cases']if x['case']=='actual_TX_role_graph_only')['ownership']
for stage, row in b.items():
 oldrow=copy.deepcopy(incumbent[stage]);newrow=copy.deepcopy(row)
 if stage=='measurement:observations':
  oldpoll='poll_likelihood_'+problem['polls'][0]['id']
  oldrow['factors'].pop(oldpoll);newrow['factors'].pop('M01_complete_poll_vector')
  oldrow['observed'].remove(oldpoll);newrow['potentials'].remove('M01_complete_poll_vector')
  oldrow.pop('implementation');newrow.pop('implementation')
 assert oldrow==newrow,stage
alltokens=[token for stage in b.values()for factor in stage['factors'].values()for token in factor['tokens']]
assert len(alltokens)==len(set(alltokens))==4304
(HERE/'compiler-inventory.json').write_text(json.dumps(b,indent=2)+'\n')
print('complete factor-list logp passed; full gradient unavailable',flush=True)
result=dict(schema='M01ActualSuccessorSuppliedStateV1',status='partial_computational_pass_full_gradient_unavailable',seconds=time.monotonic()-start,selection_sha256=p.sha(p.SELECTION),provider_sha256=p.sha(Path(p.__file__)),reduction=dict(native=value,direct=reference,error=abs(value-reference)),multi_component=dict(components=4000,native=multi_actual,symbolic=multi,oracle=oracle,gradient_max_error=float(np.max(abs(fd-grad)))),full_joint_fixed_state=dict(reduction_logp=total_reduction,multi_logp=total_multi,poll_delta_error=abs((total_multi-total_reduction)-(multi_actual-value)),full_gradient='unavailable_no_C_runtime_large_Add_failure',transformed_dimensions=662,finite_difference_checks=full_fd),parent_masses=[a['mass']for a in s['parents']],negative_checks=negative,array_parity='bitwise equal all4000 covariances and weights to authenticated frozen helper',runtime={name:dict(version=__import__(name).__version__,path=__import__(name).__file__)for name in ['numpy','pymc','pytensor','arviz','pydantic','pydantic_core']},contract=p.dict_plain(multi_model.joint_contract),scope='Actual complete compiler graph built for first protected parent; poll factor evaluated at initial state. No full joint normalization, sampler or scientific qualification.')
(HERE/'result.json').write_text(json.dumps(p.dict_plain(result),indent=2)+'\n');print(json.dumps(dict(status=result['status'],seconds=result['seconds'],negative_checks=len(negative))),flush=True)
