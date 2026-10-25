"""Independent supplied-only proper score and mixture-weight derivative oracles."""
from pathlib import Path
import sys,json
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import logit
from scipy.stats import norm
HERE=Path(__file__).resolve().parent;P=HERE.parent/'momentum-forecast-scoring-sol-v1'
sys.path[:0]=[str(P),str(P/'snapshots')]
import score_math as s
x=np.array([.1,.2,.2,.4,.7,.9]).reshape(2,3)
y=.56
point,inf=s.empirical_crps_and_influence(x,y)
A=abs(x.ravel()-y);B=abs(x.ravel()[:,None]-x.ravel()[None,:]).mean(axis=1)
assert abs(point-(A.mean()-.5*B.mean()))<1e-15
assert np.max(abs(inf.ravel()-((A-B)-(A-B).mean())))<1e-15
m=np.array([-.5,.1,.4,.2,.6,-.2]).reshape(2,3);sd=np.array([.3,.5,.4,.6,.3,.7]).reshape(2,3);truth=.61
r=s.gaussian_race(m,sd,truth,win_tail_qualified=True,lose_tail_qualified=True)
mu=m.ravel();sig=sd.ravel();base=np.full(6,1/6)
levels=(.025,.1,.25,.5,.75,.9,.975)
def scores(w):
 def cdf(t):
  if t==0:return 0.
  if t==1:return 1.
  return float(w@norm.cdf((logit(t)-mu)/sig))
 q={p:brentq(lambda t:cdf(t)-p,1e-12,1-1e-12,xtol=1e-14) for p in levels}
 crps=quad(lambda t:cdf(t)**2,0,truth,epsabs=1e-12)[0]+quad(lambda t:(1-cdf(t))**2,truth,1,epsabs=1e-12)[0]
 wis=.5*abs(truth-q[.5])
 for alpha in (.5,.2,.05):
  lo,hi=q[alpha/2],q[1-alpha/2]
  wis+=alpha/2*(hi-lo+2/alpha*max(lo-truth,0)+2/alpha*max(truth-hi,0))
 p=float(w@norm.cdf(mu/sig))
 return dict(crps=crps,wis=wis/3.5,brier=(1-p)**2,negative_log=-np.log(p))
expected=scores(base);point_errors={k:abs(expected[k]-r['metrics'][k]['value']) for k in expected}
assert max(point_errors.values())<2e-9
errors={k:0. for k in expected};eps=1e-4
for i in range(6):
 direction=np.eye(6)[i]-base
 high,low=scores(base+eps*direction),scores(base-eps*direction)
 for k in expected:
  obj=r['influences'][k];actual=np.exp(obj.log_scale)*obj.values.ravel()[i]
  derivative=(high[k]-low[k])/(2*eps)
  errors[k]=max(errors[k],abs(actual-derivative))
assert max(errors.values())<2e-6
# Saturated large probabilities still have nonconstant score influences.
z=np.array([9.,9.2,9.4,9.6,9.8,10.]).reshape(2,3)
small=norm.sf(z);delta=small-small.mean()
a=s.gaussian_race(z,np.ones_like(z),.6,win_tail_qualified=True,lose_tail_qualified=True)
b=s.gaussian_race(z,np.ones_like(z),.4,win_tail_qualified=True,lose_tail_qualified=True)
ni=a['influences']['negative_log'];bi=b['influences']['brier']
assert np.ptp(ni.values)>0 and np.ptp(bi.values)>0
assert np.max(abs(np.exp(ni.log_scale)*ni.values-delta))<1e-32
assert np.max(abs(np.exp(bi.log_scale)*bi.values+2*delta))<2e-32
out=dict(schema='IndependentSyntheticForecastScoreOraclesSolV1',synthetic_only=True,empirical_pairwise_crps_verified=True,gaussian_score_point_errors=point_errors,finite_mixture_weight_influence_errors=errors,saturated_true_side_NLL_and_surprising_outcome_Brier_variation_preserved=True,actual_target_labels=False,actual_forecasts=False)
(HERE/'math-checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
