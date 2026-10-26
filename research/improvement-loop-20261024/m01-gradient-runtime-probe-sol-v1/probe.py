"""Tiny 40-contribution analytic gradient fixture. No actual graph/C compiler."""
import hashlib,importlib.util,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
import pytensor
import pytensor.tensor as pt
HERE=Path(__file__).resolve().parent
# One scalar controls 40 distinct quadratic contributions. Known exact derivative.
x=pt.dscalar('x');offsets=np.arange(40,dtype=float)/10
terms=[(x-float(a))**2 for a in offsets];loss=pt.add(*terms);gradient=pt.grad(loss,x)
point=.37;expected_loss=float(np.sum((point-offsets)**2));expected_grad=float(np.sum(2*(point-offsets)))
results=[]
for mode in ['FAST_COMPILE','NUMBA']:
 start=time.monotonic()
 try:
  f=pytensor.function([x],[loss,gradient],mode=mode)
  value,derivative=f(point)
  assert abs(value-expected_loss)<1e-10 and abs(derivative-expected_grad)<1e-10
  results.append(dict(mode=mode,status='passed',loss=float(value),gradient=float(derivative),seconds=time.monotonic()-start))
 except Exception as error:
  trace=traceback.format_exc();(HERE/(mode.lower()+'-failure.log')).write_text(trace)
  results.append(dict(mode=mode,status='failed',error=repr(error),seconds=time.monotonic()-start,failure_log=mode.lower()+'-failure.log'))
result=dict(schema='TinyGradientRuntimeProbeV1',scope='40-contribution scalar fixture only. No actual M01 graph/model build or full662 gradient claim.',fixture=dict(contributions=40,point=point,expected_loss=expected_loss,expected_gradient=expected_grad),results=results,runtime=dict(executable=sys.executable,numpy=dict(version=np.__version__,path=np.__file__),pytensor=dict(version=pytensor.__version__,path=pytensor.__file__),numba_available=importlib.util.find_spec('numba') is not None,cxx=pytensor.config.cxx,threads={k:os.environ.get(k)for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS']}),probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
if result['runtime']['numba_available']:
 import numba,llvmlite
 result['runtime']['numba']=dict(version=numba.__version__,path=numba.__file__);result['runtime']['llvmlite']=dict(version=llvmlite.__version__,path=llvmlite.__file__)
(HERE/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
