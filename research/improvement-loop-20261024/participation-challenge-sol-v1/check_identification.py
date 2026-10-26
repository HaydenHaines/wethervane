"""Invented exact-rational identification counterexample; no model imports/data."""
from fractions import Fraction as F
from math import comb
import json
pairs=[(F(3,4),F(4,5)),(F(4,5),F(3,4))]
q=[r*v for r,v in pairs]
assert q[0]==q[1]==F(3,5)
def law(p): return [comb(10,k)*p**k*(1-p)**(10-k) for k in range(11)]
assert law(q[0])==law(q[1])
assert pairs[0][0]!=pairs[1][0]
print(json.dumps({'scope':'invented binomial thinning counterexample, not an actual fitted graph','marginalVotingProbability':str(q[0]),'all11BinomialMassesExactlyEqual':True,'distinctRegistrationStages':True}))
