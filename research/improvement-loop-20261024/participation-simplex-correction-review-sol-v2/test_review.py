import itertools,math,unittest
from fractions import Fraction as F
from participation_extrapolation_prototype import *

class ReviewTests(unittest.TestCase):
 def setUp(self):self.d=Design(('A','B'),(1,),'A',1)
 def s(self,w,x,y=0):return Scenario(w,x,{'A':0,'B':y},{1:0})
 def e(self,n,state='A',id='e'):return Event(id,state,1,(Cell('c',n,0),),False)
 def test_all_permutations_rare_anchor_and_subnormal_weight(self):
  ws=[9.555591182889165e-19,.2294746878335557,.7556416181139466,.014883694052497535]
  ss=[self.s(ws[0],1000)]+[self.s(w,-1000)for w in ws[1:]];n=2**53;p=F(ws[0])/sum(map(F,ws));want=(float(p*n),float(p*(1-p)*n*n))
  for order in itertools.permutations(ss):
   r=mixture_moments(self.d,[self.e(n)],list(order));self.assertEqual((r['means'][0],r['covariance'][0][0]),want)
  tiny=math.ulp(0.);r=mixture_moments(self.d,[self.e(n)],[self.s(tiny,1000),self.s(1,-1000)]);p=F(tiny)/(1+F(tiny))
  self.assertEqual(r['means'][0],float(p*n));self.assertEqual(r['covariance'][0][0],float(p*(1-p)*n*n))
 def test_normalized_near_unit_both_sides_at_support(self):
  for n in [0,1,1_000_000,2**53]:
   for ws in [[.5,.5000000000005],[.5,.4999999999995]]:
    for logit,expected in [(1000,float(n)),(-1000,0.)]:
     r=mixture_moments(self.d,[self.e(n)],[self.s(w,logit)for w in ws]);self.assertEqual(r['means'],[expected]);self.assertEqual(r['covariance'],[[0.]])
 def test_multiple_events_exact_fraction_covariance_and_split(self):
  # Shared cases: A/B probabilities (0,1) and (1,0), conditionally deterministic.
  ws=[.25,.7500000000005];ss=[self.s(ws[0],-1000,2000),self.s(ws[1],1000,-2000)];events=[self.e(7,'A','a'),self.e(11,'B','b')]
  p=F(ws[0])/sum(map(F,ws));r=mixture_moments(self.d,events,ss)
  self.assertEqual(r['means'],[float((1-p)*7),float(p*11)])
  self.assertEqual(r['covariance'],[[float(p*(1-p)*49),float(-p*(1-p)*77)],[float(-p*(1-p)*77),float(p*(1-p)*121)]])
  split=[self.s(ws[0]/2,-1000,2000),self.s(ws[0]/2,-1000,2000),ss[1]];self.assertEqual(r,mixture_moments(self.d,events,split))
 def test_conditional_binomial_plus_shared_variance_reference(self):
  n=101;ws=[.5,.5000000000005];p=F(ws[0])/sum(map(F,ws));mu=p*F(n,2)+(1-p)*n
  var=p*F(n,4)+p*(F(n,2)-mu)**2+(1-p)*(n-mu)**2
  r=mixture_moments(self.d,[self.e(n)],[self.s(ws[0],0),self.s(ws[1],1000)])
  self.assertEqual(r['means'],[float(mu)]);self.assertEqual(r['covariance'],[[float(var)]])
 def test_invalid_and_zero_weight_scenario(self):
  e=self.e(3)
  for ws in [[.5,.51],[0,0],[-.1,1.1],[float('nan'),1]]:
   with self.assertRaises(ValueError):mixture_moments(self.d,[e],[self.s(w,0)for w in ws])
  self.assertEqual(mixture_moments(self.d,[e],[self.s(0,1000),self.s(1,0)]),mixture_moments(self.d,[e],[self.s(1,0)]))
if __name__=='__main__':unittest.main(verbosity=2)
