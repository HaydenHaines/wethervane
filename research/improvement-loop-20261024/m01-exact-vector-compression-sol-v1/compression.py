"""Exact finite Gaussian-mixture restriction to a declared affine scalar mean.

No model fit, sampler, source admission or graph registration. Component-specific
normalizers/residual evidence remain; original mixing weights are not reset.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp

def need(ok,message):
 if not ok:raise ValueError(message)
def frozen(a):
 a=np.ascontiguousarray(a,dtype=np.float64);return np.frombuffer(a.tobytes(),dtype=np.float64).reshape(a.shape)
@dataclass(frozen=True,init=False)
class AffineScalarMixture:
 precision:np.ndarray
 center:np.ndarray
 base:np.ndarray
 def __init__(self,covariances,weights,y,direction,offset):
  cov=np.asarray(covariances,float);y=np.asarray(y,float);t=np.asarray(direction,float);d=np.asarray(offset,float);w=np.asarray(weights,float)
  need(cov.ndim==3 and cov.shape[1]==cov.shape[2]and len(cov)>0,'component covariance bank required');n=cov.shape[1]
  need(y.shape==t.shape==d.shape==(n,)and np.isfinite([y,t,d]).all(),'complete finite aligned affine vector required')
  need(w.shape==(len(cov),)and np.isfinite(w).all()and(w>=0).all()and abs(w.sum()-1)<1e-12,'original normalized nonnegative weights required')
  need(np.isfinite(cov).all()and np.allclose(cov,cov.transpose(0,2,1),rtol=0,atol=1e-12),'finite symmetric covariance required')
  a=[];m=[];b=[];k=[];norm=[]
  for C in cov:
   L=np.linalg.cholesky(C);u=np.linalg.solve(L,t);v=np.linalg.solve(L,y-d);need(np.isfinite(u).all()and np.isfinite(v).all(),'unsupported whitening range');A=float(u@u);need(np.isfinite(A)and A>0,'nonzero identifiable mean direction required');M=float(u@v/A)
   # Centered whitened residual avoids subtracting two large close quadratics.
   residual=v-u*M;K=float(residual@residual);N=-.5*(n*np.log(2*np.pi)+2*np.log(np.diag(L)).sum());need(np.isfinite([M,K,N]).all(),'unsupported centered component range')
   a.append(A);m.append(M);k.append(K);norm.append(N)
  with np.errstate(divide='ignore'):base=np.log(w)+np.array(norm)-.5*np.array(k)
  need(not np.isnan(base).any()and not np.isposinf(base).any()and np.isfinite(base).any(),'unsupported component offset range')
  for key,value in [('precision',a),('center',m),('base',base)]:object.__setattr__(self,key,frozen(value))
 def evaluate(self,mean):
  need(np.ndim(mean)==0 and np.isfinite(mean),'finite scalar mean required')
  with np.errstate(over='ignore',invalid='ignore'):
   delta=float(mean)-self.center;quadratic=self.precision*delta**2
  need(np.isfinite(quadratic).all(),'unsupported quadratic floating range')
  terms=self.base-.5*quadratic;value=float(logsumexp(terms));r=np.exp(terms-value);score=-self.precision*delta;gradient=float(r@score)
  need(np.isfinite(value)and np.isfinite(gradient),'unsupported likelihood/gradient range');return value,gradient,r
 def symbolic_logpdf(self,mean):
  import pytensor.tensor as pt
  from pytensor.raise_op import CheckAndRaise
  need(mean.ndim==0 and mean.dtype=='float64','float64 symbolic scalar required')
  mean=CheckAndRaise(ValueError,'finite scalar mean required')(mean,~pt.isnan(mean)&~pt.isinf(mean));quadratic=pt.as_tensor_variable(self.precision)*(mean-pt.as_tensor_variable(self.center))**2
  quadratic=CheckAndRaise(ValueError,'unsupported quadratic range')(quadratic,pt.all(~pt.isnan(quadratic)&~pt.isinf(quadratic)))
  terms=pt.as_tensor_variable(self.base)-.5*quadratic
  return pt.logsumexp(terms)

def pair_direction(block):
 rows=block['rows'];need(len(rows)>0 and len({r['observation_id']for r in rows})==len(rows),'unique complete observation IDs required')
 first=rows[0];pair=(first['first_candidate_id'],first['second_candidate_id']);need(pair[0]!=pair[1],'distinct pair required');signs=[]
 for row in rows:
  need(row['contest_id']==first['contest_id'],'one shared contest mean required')
  actual=(row['first_candidate_id'],row['second_candidate_id']);need(actual in [pair,pair[::-1]],'all rows must share the same unordered candidate pair')
  signs.append(1. if actual==pair else -1.)
 return np.array(signs)
