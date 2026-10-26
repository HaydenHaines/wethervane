"""Invented exact-arithmetic warning, not a CPS estimator or design validation."""
from fractions import Fraction as F
import json
out=[]
for scale in (1,100):
    weights=[scale,scale]
    estimate=F(weights[1],sum(weights))
    fake_variance=estimate*(1-estimate)/sum(weights)
    out.append({'weightScale':scale,'estimate':str(estimate),'incorrectWeightSumBinomialVariance':str(fake_variance)})
assert out[0]['estimate']==out[1]['estimate']
assert F(out[0]['incorrectWeightSumBinomialVariance'])==100*F(out[1]['incorrectWeightSumBinomialVariance'])
print(json.dumps({'scope':'invented weight-normalization counterexample only','results':out}))
