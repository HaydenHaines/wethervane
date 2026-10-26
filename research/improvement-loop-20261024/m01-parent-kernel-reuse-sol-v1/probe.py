"""Tiny synthetic exposure-input kernel; no actual graph build or full-model compile."""
import hashlib,json,sys,traceback
from pathlib import Path
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pytensor.graph.replace import graph_replace
from pytensor.graph.traversal import ancestors
HERE=Path(__file__).resolve().parent
A=np.array([[10.,20.,30.],[5.,8.,9.]])
B=np.array([[15.,17.,35.],[7.,10.,8.]])
def build(exposure):
 with pm.Model()as m:
  port=pt.specify_shape(pt.as_tensor_variable(exposure),(2,3))
  beta=pm.Normal('beta',0,1,shape=3)
  k=pm.LogNormal('kappa',np.log(4),.7)
  t=pm.Flat('total_coordinate',shape=2,initval=np.zeros(2))
  base=pt.sum(port*pm.math.sigmoid(beta)[None,:],axis=1)
  center=np.log(np.array([20.,9.]));scale=np.array([.1,.15]);log_g=center+scale*t-pt.log(base);g=pt.exp(log_g)
  pm.Potential('gamma_coordinate_and_jacobian',pt.sum(pm.logp(pm.Gamma.dist(alpha=k,beta=k),g)+log_g+np.log(scale)))
  pm.Poisson('observed_counts',mu=base*g,observed=np.array([20,9]))
 return m,port
m,port=build(A);logp=m.logp(jacobian=True);grad=m.dlogp(jacobian=True)
E=pt.dmatrix('whole_parent_exposure');checked=pt.specify_shape(E,(2,3))
from pytensor.raise_op import CheckAndRaise
checked=CheckAndRaise(ValueError,'positive finite whole parent required')(checked,pt.all((checked>0)&~pt.isinf(checked)&~pt.isnan(checked)))
attempts=[]
# Exact typed-port identity may have been cloned by model.logp; never match constants by numeric equality.
targets=[('typed_port',port)]
if port.owner is not None:targets.append(('exact_tensor_constant_leaf',port.owner.inputs[0]))
for name,target in targets:
 try:
  repl_logp=graph_replace(logp,{target:checked},strict=True);repl_grad=graph_replace(grad,{target:checked},strict=True)
  attempts.append(dict(target=name,status='exact_node_present_replaced'))
  break
 except Exception as error:attempts.append(dict(target=name,status='rejected',error=repr(error)))
else:raise AssertionError('No exact original port identity survived; symbolic API needed')
inputs=[*m.value_vars,E];f=pytensor.function(inputs,[repl_logp,repl_grad],mode='NUMBA')
point=m.initial_point();point['beta']=np.array([.2,-.3,.1]);point['total_coordinate']=np.array([.1,-.1]);args=[point[v.name]for v in m.value_vars]
checks=[]
for label,parent in [('A',A),('B',B),('A_again',A)]:
 fixed,_=build(parent);fixedfn=fixed.compile_fn([fixed.logp(jacobian=True),fixed.dlogp(jacobian=True)],inputs=fixed.value_vars,point_fn=True,mode='FAST_COMPILE')
 direct,derivative=fixedfn(point);v,g=f(*args,parent)
 assert abs(v-direct)<1e-10 and np.max(np.abs(g-derivative))<1e-10
 checks.append(dict(parent=label,logp=float(v),fixed_density_error=float(abs(v-direct)),gradient_max_error=float(np.max(abs(g-derivative)))))
assert abs(checks[0]['logp']-checks[1]['logp'])>1e-5
negative=[]
for name,parent in [('wrong_shape',A[:1]),('negative',-A),('nonfinite',A*np.nan)]:
 try:f(*args,parent)
 except Exception as error:negative.append(dict(case=name,error_type=type(error).__name__));continue
 raise AssertionError('accepted '+name)
# Fixed mixture masses remain outside parent-specific likelihoods. This is not normalization proof.
masses=np.array([.3,.7]);saved=masses.copy();kernels=np.array([checks[0]['logp'],checks[1]['logp']]);assert np.array_equal(masses,saved)
# Execute only the original native compiler's exposure guard AST on tiny values.
import ast
from types import SimpleNamespace
native_source=Path('/home/hayden/projects/wethervane-redesign-20261006/integration/research/senate-stack-integration-20261020/modular-count-graph-astra-v1/graph.py')
tree=ast.parse(native_source.read_text());compiler=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='compile_model')
guard=next(n for n in ast.walk(compiler)if isinstance(n,ast.If)and ast.unparse(n.test)=="port.name == 'exposure'")
guard_code=compile(ast.Module(body=[guard],type_ignores=[]),str(native_source),'exec');native_guard=[]
for name,value in [('exact_original_array',A),('different_parent_array',B),('symbolic_input',E)]:
 try:exec(guard_code,dict(np=np,port=SimpleNamespace(name='exposure'),value=value,ctx=SimpleNamespace(protected={'exposure':A})))
 except ValueError as error:native_guard.append(dict(case=name,status='rejected',message=str(error)))
 else:native_guard.append(dict(case=name,status='accepted'))
assert [r['status']for r in native_guard]==['accepted','rejected','rejected']
source_files=[Path(__file__),native_source,Path(pm.__file__).parent/'model/core.py',Path(pytensor.__file__).parent/'graph/replace.py']
result=dict(schema='TinyParentExposureKernelReuseV1',scope='Synthetic2event3cell/6coordinate model only. No actual TX graph build/compile, fits, parent normalization or scientific equivalence certificate.',native_exposure_guard=native_guard,exact_identity_substitution=attempts,checks=checks,negative_checks=negative,masses=masses.tolist(),mass_used_in_conditional_logp=False,normalization_established=False,runtime=dict(pymc=pm.__version__,pytensor=pytensor.__version__,numpy=np.__version__,fastmath=pytensor.config.numba__fastmath,cxx=pytensor.config.cxx),pins={str(p):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)for p in source_files})
(HERE/'probe-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
