from pathlib import Path
import sys,json,hashlib
import numpy as np
from scipy.special import logsumexp,log_ndtr
src=Path('/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/momentum-marginal-math-root-v1')
assert hashlib.sha256((src/'artifact-manifest.json').read_bytes()).hexdigest()=='ad58a67d3e204abc6fd02fa23e85205a44a01a681cf12f29f5adda8223993b94'
sys.path.insert(0,str(src))
from marginal import GaussianLogitMixture
raw=np.array([1e-300,1e300]); mu=np.array([-100.,100.]); sd=np.ones(2)
m=GaussianLogitMixture(mu,sd,raw)
win,lose=m.log_win_lose()
logw=np.log(raw)-logsumexp(np.log(raw))
expected=float(logsumexp(logw+log_ndtr(-mu/sd)))
print(json.dumps(dict(supplied_only=True,normalized_weights=m.weights.tolist(),actual_log_lose=lose,oracle_log_lose=expected,absolute_log_error=abs(lose-expected))))
