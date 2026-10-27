"""Independent invented checks for normalized scenario moments."""
import json, math, sys
from fractions import Fraction
sys.path.insert(0,"/home/hayden/projects/wethervane-redesign-20261006/integration/research/coordination-frontend-20261026/participation-simplex-correction-candidate-v1")
from participation_extrapolation_prototype import Cell,Design,Event,Scenario,mixture_moments

def scen(w,intercept,states):
    return Scenario(w,intercept,states,{2020:0})
design=Design(["X","Y"],[2020],"X",2020)
# Accepted weights just below/above unit normalize exactly as a simplex.
for n in (1_000_000,2**53):
    event=Event("e","X",2020,[Cell("c",n,0)],False)
    for weights in ((.5,.5-0.9e-12),(.5,.5+0.9e-12)):
        result=mixture_moments(design,[event],[scen(weights[0],1000,{"X":0,"Y":0}),
                                               scen(weights[1],-1000,{"X":0,"Y":0})])
        exact_w=[Fraction.from_float(x) for x in weights]; total=sum(exact_w)
        exact_mean=Fraction(n)*exact_w[0]/total
        exact_var=Fraction(n*n)*exact_w[0]*exact_w[1]/(total*total)
        assert 0<=result["means"][0]<=n
        assert math.isclose(result["means"][0],float(exact_mean),rel_tol=2e-16,abs_tol=1)
        assert math.isclose(result["covariance"][0][0],float(exact_var),rel_tol=2e-15)
# Splitting an identical scenario is invariant, including the permitted near-unit sum.
for n in (1_000_000,2**53):
    e=Event("e","X",2020,[Cell("c",n,0)],False)
    base=mixture_moments(design,[e],[scen(1,0,{"X":0,"Y":0})])
    for weights in ((.5,.5000000000005),(.2,.3,.5000000000005),(.1,)*10):
        split=mixture_moments(design,[e],[scen(w,0,{"X":0,"Y":0}) for w in weights])
        assert split==base
# Exact positive and negative shared covariance at maximum-scale support.
N=2**53
events=[Event("x","X",2020,[Cell("x",N,0)],False),
        Event("y","Y",2020,[Cell("y",N-1,0)],False)]
weights=(.5,.5000000000005)
w=[Fraction.from_float(x) for x in weights]; w=[x/sum(w) for x in w]
# shared latent scenarios make x/y perfectly positively coupled
positive=mixture_moments(design,events,[scen(weights[0],-1000,{"X":0,"Y":0}),
                                      scen(weights[1],1000,{"X":0,"Y":0})])
want_pos=float(w[0]*w[1]*N*(N-1))
assert math.isclose(positive["covariance"][0][1],want_pos,rel_tol=1e-15)
# change Y effect so it moves opposite to X; covariance should be negative
negative=mixture_moments(design,events,[scen(weights[0],-1000,{"X":0,"Y":2000}),
                                      scen(weights[1],1000,{"X":0,"Y":-2000})])
want_neg=-want_pos
assert math.isclose(negative["covariance"][0][1],want_neg,rel_tol=1e-15)
assert negative["covariance"][0][1]==negative["covariance"][1][0]
# covariance remains PSD on these rank-one boundary mixtures
for r in (positive,negative):
    c=r["covariance"]
    assert c[0][0]>=0 and c[1][1]>=0
    assert c[0][0]*c[1][1]-c[0][1]*c[1][0] >= -1e-12*max(1,c[0][0]*c[1][1])
# Inputs just beyond the documented tolerance are rejected.
for weights_bad in ((.5,.50000000000101),(.5,.49999999999899)):
    try:
        mixture_moments(design,[events[0]],[scen(weights_bad[0],0,{"X":0,"Y":0}),
                                            scen(weights_bad[1],1,{"X":0,"Y":0})])
    except ValueError: pass
    else: raise AssertionError(f"out-of-tolerance weights accepted: {weights_bad}")
print(json.dumps({"status":"PASS","tested_population_sizes":[1000000,2**53],
 "near_unit_weights":"normalized against exact Fractions",
 "identical_scenario_splits":"same means/covariance",
 "shared_covariance":{"positive":positive["covariance"][0][1],
                      "negative":negative["covariance"][0][1],
                      "both_symmetric_and_PSD_within_roundoff":True},
 "outside_tolerance":"rejected"},indent=2))
