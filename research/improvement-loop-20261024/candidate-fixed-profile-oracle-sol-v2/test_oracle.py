import math,unittest
from fractions import Fraction
from oracle import allocate,softmax

class OracleTests(unittest.TestCase):
 def check_conservation(self,result):
  self.assertAlmostEqual(sum(result['named'])+sum(result['noncandidate']),result['participants'])
  for row in result['profiles']:self.assertAlmostEqual(sum(row['named'])+sum(row['noncandidate']),row['participants'])
 def test_single_profile_major_ratio_and_increasing_minor_mass(self):
  previous=-1
  for k in [0,1,4,12]:
   r=allocate([(1000,[math.log(3),0]+[math.log(.25)]*k)],.8,(.5,.25,.25));self.check_conservation(r)
   self.assertAlmostEqual(r['major_ratio'],3);self.assertAlmostEqual(sum(r['named']),800);self.assertAlmostEqual(sum(r['noncandidate']),200)
   expected=Fraction(800)*Fraction(k,4)/(4+Fraction(k,4));self.assertAlmostEqual(r['minor_marks'],float(expected));self.assertGreater(r['minor_marks'],previous);previous=r['minor_marks']
 def test_multiple_profiles_ratio_of_sums_changes(self):
  for minor,expected in [(False,Fraction(7,3)),(True,Fraction(4))]:
   tail=[math.log(10)]if minor else[]
   r=allocate([(100,[math.log(9),0]+tail),(100,[0,0]+tail)],.8)
   self.check_conservation(r);self.assertAlmostEqual(r['major_ratio'],float(expected));self.assertAlmostEqual(r['profiles'][0]['major_ratio'],9);self.assertAlmostEqual(r['profiles'][1]['major_ratio'],1);self.assertAlmostEqual(sum(r['named']),160)
 def test_same_profile_major_ratio_gives_aggregate_invariance(self):
  for tail in [[],[-20],[10,0,2]]:
   r=allocate([(100,[math.log(3),0]+tail),(200,[math.log(6),math.log(2)]+tail)],.7)
   self.assertAlmostEqual(r['major_ratio'],3);self.check_conservation(r)
 def test_independent_fraction_oracle_for_multiple_profiles(self):
  weights=[[9,1,10],[1,1,10]];r=allocate([(100,list(map(math.log,w)))for w in weights],.8,(.25,.75))
  expected=[sum(Fraction(80)*Fraction(w[i],sum(w))for w in weights)for i in range(3)]
  for actual,want in zip(r['named'],expected):self.assertAlmostEqual(actual,float(want))
 def test_common_shift_extreme_finite_utilities(self):
  a=softmax([2,1,-3]);b=softmax([10002,10001,9997]);self.assertEqual(a,b)
  p=softmax([10000,-10000,0]);self.assertTrue(all(math.isfinite(x)for x in p));self.assertEqual(sum(p),1)
 def test_zero_response_or_zero_participants_no_ratio_claim(self):
  for profiles,response in [([(100,[0,0])],0),([(0,[0,0])],1)]:
   r=allocate(profiles,response);self.check_conservation(r);self.assertIsNone(r['major_ratio']);self.assertIsNone(r['major_pair_share'])
 def test_heterogeneous_response_fraction_oracle(self):
  profiles=[(100,[math.log(9),0,math.log(10)]),(100,[0,0,math.log(10)])]
  rates=[Fraction(9,10),Fraction(1,10)]
  r=allocate(profiles,list(map(float,rates)),(.25,.75));self.check_conservation(r)
  weights=[[9,1,10],[1,1,10]]
  expected=[sum(Fraction(100)*rate*Fraction(w[i],sum(w))for rate,w in zip(rates,weights))for i in range(3)]
  for got,want in zip(r['named'],expected):self.assertAlmostEqual(got,float(want))
  non=sum(Fraction(100)*(1-rate)for rate in rates)
  self.assertAlmostEqual(r['noncandidate'][0],float(non/4));self.assertAlmostEqual(r['noncandidate'][1],float(non*3/4))
  self.assertAlmostEqual(r['major_ratio'],float(expected[0]/expected[1]))
  self.assertNotAlmostEqual(r['major_ratio'],allocate(profiles,.5)['major_ratio'])
 def test_scalar_is_identical_special_case(self):
  profiles=[(13,[1,0,-2]),(27,[2,1,-3])]
  self.assertEqual(allocate(profiles,.7,(.2,.3,.5)),allocate(profiles,[.7,.7],(.2,.3,.5)))
 def test_response_zero_one_and_empty_noncandidate_mass(self):
  r=allocate([(10,[0,0]),(20,[0,0])],[0,1],(0,1));self.check_conservation(r)
  self.assertEqual(r['named'],(10,10));self.assertEqual(r['noncandidate'],(0,10))
  self.assertEqual(sum(r['profiles'][0]['named']),0);self.assertEqual(sum(r['profiles'][1]['noncandidate']),0)
 def test_bad_response_axes_and_simplex(self):
  profiles=[(1,[0,0]),(2,[0,0])]
  for response in [[.5],[.5,.5,.5],[float('nan'),.2],[True,.2],[0,1.1],None]:
   with self.assertRaises(ValueError):allocate(profiles,response)
  for simplex in [(),(-.1,1.1),(float('nan'),1),(.4,.5),(True,0)]:
   with self.assertRaises(ValueError):allocate(profiles,.5,simplex)
 def test_unsupported_inputs(self):
  for args in [([],1),([(1,[0,float('nan')])],1),([(-1,[0,0])],1),([(1,[0,0])],1.1),([(1,[0,0]),(1,[0,0,0])],1)]:
   with self.assertRaises(ValueError):allocate(*args)
  with self.assertRaises(ValueError):allocate([(1,[0,0])],1,(.3,.3))
if __name__=='__main__':unittest.main(verbosity=2)
