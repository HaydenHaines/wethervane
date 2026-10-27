"""Independent tiny algebra checks; invented values only; no model/data imports."""
import json, math, sys
from fractions import Fraction
sys.path.insert(0, "/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/candidate-fixed-profile-oracle-sol-v1")
from oracle import allocate, softmax

# A profile's softmax allocation with proportional utilities. Response and
# common event scale multiply all named categories equally, so cancel in D/R.
single=[]
for k in (0,1,4,12):
    x=allocate([(1000,[math.log(3),0]+[math.log(.25)]*k)],.8)
    expected=Fraction(800)*Fraction(k,4)/(4+Fraction(k,4))
    assert math.isclose(x["major_ratio"],3.0)
    assert math.isclose(x["minor_marks"],float(expected))
    assert math.isclose(sum(x["named"])+sum(x["noncandidate"]),1000)
    single.append({"minor_candidates":k,"minor_expected":x["minor_marks"],"major_ratio":x["major_ratio"]})

# Exact ratio-of-sums counterexample. With same profile totals/response,
# major odds are 9 and 1. Adding the same minor utility changes mixture weights.
without=allocate([(100,[math.log(9),0]),(100,[0,0])],.8)
with_minor=allocate([(100,[math.log(9),0,math.log(10)]),(100,[0,0,math.log(10)])],.8)
assert math.isclose(without["major_ratio"],float(Fraction(7,3)))
assert math.isclose(with_minor["major_ratio"],4.0)
assert math.isclose(with_minor["profiles"][0]["major_ratio"],9)
assert math.isclose(with_minor["profiles"][1]["major_ratio"],1)
# Conversely, if every profile has odds ratio 3, changing the profile-specific
# softmax normalizers cannot change aggregate D/R.
invariant=allocate([(100,[math.log(9),math.log(3),math.log(10)]),(200,[math.log(6),math.log(2),0])],.7)
assert math.isclose(invariant["major_ratio"],3.0)

# Sensitivity omitted by the scalar-response oracle API: actual source response
# is a profile tensor. Unequal response changes aggregate profile weights.
def direct_ratio(rows):
    d=r=0.0
    for total,response,utilities in rows:
        p=softmax(utilities)
        d += total*response*p[0]
        r += total*response*p[1]
    return d/r
heterogeneous=direct_ratio([(100,.9,[math.log(9),0]),(100,.1,[0,0])])
common_response=allocate([(100,[math.log(9),0]),(100,[0,0])],.5)["major_ratio"]
assert math.isclose(heterogeneous, float(Fraction(43,7))) # (81+5)/(9+5)
assert math.isclose(common_response,float(Fraction(7,3)))
# Float softmax can underflow a mathematically positive category to zero.
underflow=softmax([0,-1000])
assert underflow[1]==0.0
print(json.dumps({"status":"PASS","single_profile":single,
 "ratio_of_sums":{"without_minor":without["major_ratio"],"with_minor":with_minor["major_ratio"],
                  "within_profile_with_minor":[x["major_ratio"] for x in with_minor["profiles"]],
                  "same_odds_aggregate_invariant":invariant["major_ratio"]},
 "heterogeneous_response_counterexample":{"variable_response_ratio":heterogeneous,"scalar_response_ratio":common_response},
 "finite_precision_underflow_softmax":[1.0,0.0]},indent=2))
