"""Source support inventory and tiny semantic probes; no full numerical graph compile."""
import collections,hashlib,importlib.util,json,os,sys,time,traceback,warnings
from pathlib import Path
import numpy as np
import pytensor
import pytensor.tensor as pt
import pymc as pm
from pytensor.raise_op import CheckAndRaise
from pytensor.graph.basic import io_toposort
from pytensor.link.numba.dispatch import numba_funcify
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'m01-native-successor-provider-sol-v1'
sys.path.append(str(OLD/'schema-overlay'));sys.path.insert(0,str(OLD));import provider as p
# Source and actual-likelihood identity pinned by frozen prior evidence, never repin silently.
evidence=json.loads((OLD/'evidence.json').read_text())
for name in ['provider.py','selection.json','result.json']:
 assert hashlib.sha256((OLD/name).read_bytes()).hexdigest()==evidence['files'][name]['sha256']
s=p.selection();problem=json.loads(Path(s['problem_path']).read_text());parent=s['parents'][0]
refs=sys.modules['m01_successor_native.references'];rolepath=p.OWNER/'research/senate-stack-integration-20261020/candidate-role-source-sol-v1/prepared-v2/candidate-role-manifest.json';rm=json.loads(rolepath.read_text());old=json.loads((p.OWNER/rm['bound_prepared_problem']['path']).read_text());roles=refs.roles.load_role_binding(old,rolepath,'a1d73f7dece21a4cb5b104c69caec5157e4dd106b411ce17833f7f5a41bbeb3d',admission_cutoff_utc=problem['cutoff_utc'],root=p.OWNER);roles=refs.roles.rebind_role_binding(old,problem,roles)
def build():
 reg=p.components.standard_registry();successor=p.M01Measurement('actual_calibration_bank');reg.register(successor);comp=p.components.baseline_composition(role_hierarchy=True);comp=p.graph.Composition(tuple((r,successor.implementation_id if r=='measurement'else n)for r,n in comp.selections),comp.standardized,comp.person_coordinates);return p.graph.compile_model(reg,comp,problem,parent,roles=roles)
def run():
 start=time.monotonic();tests=[]
 # Check exceptions are preserved. A linker silently ignoring safeguards is unsuitable.
 x=pt.dscalar('tiny_positive');safe=CheckAndRaise(ValueError,'positive required')(x,x>0)
 with warnings.catch_warnings(record=True)as notes:
  warnings.simplefilter('always');f=pytensor.function([x],[safe**2,pt.grad(safe**2,x)],mode='NUMBA')
  assert np.allclose(f(.5),[.25,1.]);invalid_rejected=False
  try:f(-.5)
  except ValueError:invalid_rejected=True
  tests.append(dict(test='CheckAndRaise preserved',passed=invalid_rejected,warnings=[str(w.message)for w in notes]))
 # Special functions used by count likelihood and derivatives, including NB shape score.
 x=pt.dscalar('tiny_shape');expr=pt.gammaln(x+3)-pt.gammaln(x)+x*pt.log(x/(x+2))
 f=pytensor.function([x],[expr,pt.grad(expr,x)],mode='NUMBA');v,g=f(2.5)
 from scipy.special import gammaln,digamma
 oracle=gammaln(5.5)-gammaln(2.5)+2.5*np.log(2.5/4.5);dg=digamma(5.5)-digamma(2.5)+np.log(2.5/4.5)+1-2.5/4.5
 tests.append(dict(test='gammaln digamma count shape',passed=bool(abs(v-oracle)<1e-10 and abs(g-dg)<1e-10),value_error=float(abs(v-oracle)),gradient_error=float(abs(g-dg))))
 print('tiny semantic probes',tests,flush=True)
 model=build();point=model.initial_point();grad=model.dlogp();nodes=io_toposort(model.value_vars,[grad]);ops=collections.Counter(type(n.op).__module__+'.'+type(n.op).__name__ for n in nodes)
 dispatch=[];seen=set()
 for n in nodes:
  typ=type(n.op)
  if typ in seen:continue
  seen.add(typ);impl=numba_funcify.dispatch(typ)
  dispatch.append(dict(op=typ.__module__+'.'+typ.__name__,implementation=impl.__module__+'.'+impl.__name__,generic_fallback=impl is numba_funcify.dispatch(object)))
 result=dict(schema='ActualM01NumbaPreflightV1',tests=tests,full_attempt_permitted=all(t['passed']for t in tests),source_pins={name:evidence['files'][name]for name in ['provider.py','selection.json','result.json']},actual_gradient=dict(transformed_dimensions=sum(v.size for v in point.values()),raw_apply_nodes=len(nodes),op_counts=dict(ops),dispatch=dispatch,scope='Symbolic unoptimized gradient only; dispatch may change under NUMBA optimizations, Elemwise scalar composites need deeper lowering. No full numerical compile here.'),seconds=time.monotonic()-start,runtime=dict(executable=sys.executable,pytensor=pytensor.__version__,numpy=np.__version__,cxx=pytensor.config.cxx))
 (HERE/'preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(preflight_seconds=result['seconds'],full_attempt_permitted=result['full_attempt_permitted'],raw_nodes=len(nodes),ops=len(ops))),flush=True)
if __name__=='__main__':
 try:run()
 except BaseException:
  (HERE/'preflight-failure.log').write_text(traceback.format_exc());raise
