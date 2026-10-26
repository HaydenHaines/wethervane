"""Tiny supplied-rate graphs only; not the actual TX graph."""
import resource,time,json,hashlib,dataclasses,copy
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CPU,(45,50));resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3));resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**2,8*1024**2))
import provider as q
import numpy as np,pymc as pm,pytensor,pytensor.tensor as pt
H=Path(__file__).resolve().parent
@dataclasses.dataclass
class Context:
 data:dict
 protected:dict
 contest_map:dict
class Scope:
 def __init__(self,rates):self.inputs={'category_rates':rates};self.claims=[]
 def factor(self,name,tokens):self.claims.append((name,tokens))
s=q.p.selection();problem=json.loads(Path(next(p for p in s['pins'] if p.endswith('problem.json'))).read_text());parent=s['parents'][0];cm={c['id']:c for c in problem['contests']};context=Context(problem,parent,cm)
block=q.S['poll_block'];contest=block['rows'][0]['contest_id'];names=[r['id']for r in cm[contest]['candidates']];first=names.index(block['rows'][0]['first_candidate_id']);second=names.index(block['rows'][0]['second_candidate_id'])
forward=[]
def delegated(self,c,scope):
 assert not c.data['polls'];assert {k:v for k,v in c.data.items()if k!='polls'}=={k:v for k,v in problem.items()if k!='polls'};assert c.protected==parent;forward.append(True);return {}
original=q.p.components.Measurement.observations;q.p.components.Measurement.observations=delegated
results=[];start=time.monotonic()
try:
 for mode in q.S['modes']:
  funcs=[];claims=[]
  for cls in [q.p.M01Measurement,q.CompressedM01]:
   with pm.Model()as model:
    x=pm.Flat('supplied_log_rates',shape=len(names));scope=Scope({contest:pt.exp(x)});obj=cls(mode);obj.observations(context,scope);f=model.compile_logp();g=model.compile_dlogp();funcs.append((f,g));claims.append(scope.claims)
   assert obj.spec.configuration['mode']==mode and obj.spec.configuration['selection_sha256']==q.p.EXPECTED_SELECTION_SHA256
  assert claims[0]==claims[1] and len(claims[0])==1 and len(claims[0][0][1])==6
  for mean in [-8.,-1.,0.,.6,2.,8.]:
   point=np.zeros(len(names));point[first]=mean/2;point[second]=-mean/2;point={'supplied_log_rates':point};old,new=[(float(f(point)),g(point))for f,g in funcs];manual=q.compact(mode).evaluate(mean)
   assert abs(old[0]-new[0])<1e-8;assert np.max(np.abs(old[1]-new[1]))<1e-8;assert abs(new[0]-manual[0])<1e-8
   results.append({'mode':mode,'mean':mean,'logp_error':abs(old[0]-new[0]),'gradient_max_error':float(np.max(np.abs(old[1]-new[1])))})
finally:q.p.components.Measurement.observations=original
# Independent exact native prefix mapping checks names rather than assumed positions.
x=pt.dvector('invented_positive_rates');mapping=pytensor.function([x],q.map_means(context,{'category_rates':{contest:x}},block));rates=np.arange(1,len(names)+1,dtype=float);assert np.allclose(mapping(rates),np.log(rates[first]/rates[second]))
negative=[]
def reject(label,fn):
 try:fn()
 except (ValueError,AssertionError):negative.append(label)
 else:raise AssertionError('accepted '+label)
a=np.repeat(np.eye(6)[None],1,axis=0);a[0,0,1]=5e-13;reject('subtolerance_asymmetry',lambda:q.exact_symmetric(a));reject('unsupported_mode',lambda:q.CompressedM01('unknown'))
for label,mutate in [('row_order',lambda p:p['polls'][0]['rows'].reverse()),('duplicate',lambda p:p['polls'][0]['rows'].__setitem__(1,p['polls'][0]['rows'][0])),('candidate_reorder',lambda p:p['contests'][0]['candidates'].reverse()),('observation_value',lambda p:p['polls'][0]['rows'][0].__setitem__('logit_value',99.))]:
 changed=copy.deepcopy(problem);mutate(changed);reject(label,lambda:q.p.validate(changed,parent,s))
changed=copy.deepcopy(parent);changed['mass']=.5;reject('parent_mass',lambda:q.p.validate(problem,changed,s))
# Frozen coefficients independently recomputed from original provider bank arrays.
bank=q.p.mixture('actual_calibration_bank')
with np.load(q.S['reference_bank'],allow_pickle=False)as archive:
 assert np.array_equal(bank.covariances,archive['covariances']) and np.array_equal(bank.weights,archive['weights'])
y=np.array([r['logit_value']for r in block['rows']]);recomputed=q.compression.AffineScalarMixture(bank.covariances,bank.weights,y,np.ones(6),np.zeros(6));stored=q.compact('actual_calibration_bank');coefficient_errors={n:float(np.max(np.abs(getattr(recomputed,n)-getattr(stored,n))))for n in ['precision','center','base']};assert max(coefficient_errors.values())==0
expected=q.EXPECTED_SELECTION
try:
 q.EXPECTED_SELECTION='0'*64;reject('frozen_selection_tamper',q.selected)
finally:q.EXPECTED_SELECTION=expected
wrong=copy.deepcopy(cm);wrong[contest]['candidates'].reverse();reject('contest_rate_reference_mismatch',lambda:q.CompressedM01().observations(Context(problem,parent,wrong),Scope({})))
result={'schema':'CompressedM01SuppliedStateCheckV1','status':'passed','elapsed_seconds':time.monotonic()-start,'comparisons':results,'negatives':negative,'coefficient_max_errors':coefficient_errors,'all_4000_original_bank_rows_weights_exact':True,'forwarded_nonpoll_calls':len(forward),'mapping_ast_sha256':q.S['mapping_ast_sha256'],'candidate_ids':names,'pair_indices':[first,second],'scope':'Only tiny supplied-log-rate graphs; native nonpoll delegate intercepted and context checked; no actual complete graph.','source_pins':{str(H/n):q.sha(H/n)for n in ['provider.py','selection.json','check.py','native-mapping-source.txt']}}
(H/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'passed','comparisons':len(results),'negatives':negative}))
