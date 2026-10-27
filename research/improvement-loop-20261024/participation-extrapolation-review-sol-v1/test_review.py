import itertools,math,unittest
from fractions import Fraction as F
from participation_extrapolation_prototype import *

def binomial(n,p):return [F(math.comb(n,k))*p**k*(1-p)**(n-k)for k in range(n+1)]

class ReviewTests(unittest.TestCase):
 def test_joint_fraction_enumeration_matches_mixture_covariance(self):
  d=Design(('A','B'),(1,),'A',1);events=[Event('x','B',1,(Cell('c',2,0),),True),Event('y','B',1,(Cell('c',3,0),),False)]
  scenarios=[Scenario(.5,0,{'A':0,'B':s*math.log(3)},{1:0})for s in [-1,1]]
  joint={(i,j):sum(F(1,2)*binomial(2,p)[i]*binomial(3,p)[j]for p in [F(1,4),F(3,4)])for i in range(3)for j in range(4)}
  mx=sum(i*w for(i,j),w in joint.items());my=sum(j*w for(i,j),w in joint.items());cov=sum((i-mx)*(j-my)*w for(i,j),w in joint.items())
  r=mixture_moments(d,events,scenarios);self.assertAlmostEqual(r['means'][0],float(mx));self.assertAlmostEqual(r['means'][1],float(my));self.assertAlmostEqual(r['covariance'][0][1],float(cov));self.assertEqual(cov,F(3,8))
 def test_pmf_against_independent_person_enumeration(self):
  d=Design(('A',),(1,),'A',1);event=Event('e','A',1,(Cell('x',2,-math.log(3)),Cell('y',1,math.log(3))),True);s=Scenario(1,0,{'A':0},{1:0});probabilities=[F(1,4),F(1,4),F(3,4)];pmf=[F(0)]*4
  for bits in itertools.product([0,1],repeat=3):pmf[sum(bits)]+=math.prod(p if bit else 1-p for p,bit in zip(probabilities,bits))
  for got,want in zip(conditional_pmf(d,event,s),pmf):self.assertAlmostEqual(got,float(want))
 def test_near_unit_weights_can_break_mixture_support(self):
  d=Design(('A',),(1,),'A',1)
  ss=[Scenario(w,1000,{'A':0},{1:0})for w in [.5,.5000000000005]]
  for n in [1_000_000,10_000_000,100_000_000,2**53]:
   event=Event('e','A',1,(Cell('c',n,0),),False);r=mixture_moments(d,[event],ss)
   self.assertGreater(r['means'][0],n);self.assertGreater(r['covariance'][0][0],0)
   print('NORMALIZATION_COUNTEREXAMPLE',{'eligible':n,'weight_sum':sum(s.weight for s in ss),'mean':r['means'][0],'excess':r['means'][0]-n,'variance':r['covariance'][0][0]})
 def test_design_rank_not_numeric_information(self):
  d=Design(('A',),(1,),'A',1);e=Event('e','A',1,(Cell('c',2,1000),),True);s=Scenario(1,0,{'A':0},{1:0})
  self.assertTrue(structural_identifiability(d,[e])['full_column_rank']);self.assertEqual(conditional_moments(d,e,s)['variance'],0)

if __name__=='__main__':unittest.main(verbosity=2)
