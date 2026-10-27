"""Independent exact-rational check of the invented V2 algebra."""
import json, math, sys
from fractions import Fraction as F
sys.path.insert(0, "/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/candidate-fixed-profile-oracle-sol-v2")
from oracle import allocate
profiles=[(100,[math.log(9),0,math.log(10)]),(100,[0,0,math.log(10)])]
rates=[.9,.1]
r=allocate(profiles,rates,(.25,.75))
weights=[[9,1,10],[1,1,10]]
r_exact=[]
for k in range(3):
    r_exact.append(sum(F(100)*F(str(rate))*F(w[k],sum(w)) for rate,w in zip([F(9,10),F(1,10)],weights)))
assert r_exact==[F(124,3),F(16,3),F(160,3)]
assert all(math.isclose(a,float(b)) for a,b in zip(r['named'],r_exact))
assert r['noncandidate']==(25.0,75.0)
assert math.isclose(sum(r['named'])+sum(r['noncandidate']),200.0)
assert math.isclose(r['major_ratio'],F(31,4))
scalar=allocate(profiles,.5,(.25,.75))
assert math.isclose(scalar['major_ratio'],4.0)
assert math.isclose(sum(scalar['named']),100.0)
assert math.isclose(sum(scalar['noncandidate']),100.0)
# Adding a candidate only reallocates named marks at fixed participant/intensity
# and response; total named and noncandidate marks are conserved.
no_minor=allocate([(200,[math.log(9),0])],.9)
with_minor=allocate([(200,[math.log(9),0,math.log(10)])],.9)
assert math.isclose(sum(no_minor['named']),sum(with_minor['named']))
assert math.isclose(sum(no_minor['noncandidate']),sum(with_minor['noncandidate']))
print(json.dumps({"status":"PASS","rates":[.9,.1],"exact_named":[str(x) for x in r_exact],
 "computed_named":r['named'],"computed_noncandidate":r['noncandidate'],
 "aggregate_D_over_R":r['major_ratio'],"scalar_0.5_D_over_R":scalar['major_ratio'],
 "named_conservation_at_fixed_inputs":sum(no_minor['named'])},indent=2))
