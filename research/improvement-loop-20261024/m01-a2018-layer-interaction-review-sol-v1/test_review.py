import contextlib,io,json,math,unittest
from decimal import Decimal as D,localcontext
from unittest.mock import patch
import check

def oracle(weights,pairs,other):
 with localcontext()as c:
  c.prec=60;num=den=D(0)
  for weight,(a,b)in zip(weights,pairs,strict=True):
   x,y,z=D(str(a)).exp(),D(str(b)).exp(),D(str(other)).exp();total=x+y+z
   num+=D(str(weight))*x/total;den+=D(str(weight))*y/total
  return (num/den).ln()
class ReviewTests(unittest.TestCase):
 def test_decimal_independent_fixture_and_interaction(self):
  pairs=((1.4,0.),(0.,1.2));deltas=[]
  for weights in [(1.,1.),(9.,1.)]:
   before=oracle(weights,pairs,-1);after=oracle(weights,pairs,1);deltas.append(after-before)
   for other,want in [(-1,before),(1,after)]:self.assertAlmostEqual(check.ordinary_poll_logit(weights,pairs,other,.72,1.5),float(want),places=14)
  self.assertGreater(abs(float(deltas[1]-deltas[0])),.001)
 def test_positive_scalar_cancellation_and_constant_contrast(self):
  for weights in [(1.,1.),(9.,1.)]:
   for other in [-1.,1.]:
    base=check.ordinary_poll_logit(weights,((1.4,0),(0,1.2)),other,.72,1.5)
    self.assertAlmostEqual(base,check.ordinary_poll_logit(weights,((1.4,0),(0,1.2)),other,.19,6.4),places=14)
  for other in [-1.,1.]:self.assertAlmostEqual(check.ordinary_poll_logit((1,9),((1,0),(2,1)),other,.72,1.5),1.,places=14)
 def test_native_centered_major_contrasts_also_interact(self):
  # Respects selected source's D=eta/2,R=-eta/2 necessary contrast pattern.
  pairs=((.7,-.7),(-.6,.6));delta=[]
  for weights in [(1.,1.),(9.,1.)]:delta.append(oracle(weights,pairs,1)-oracle(weights,pairs,-1))
  self.assertGreater(abs(float(delta[1]-delta[0])),.001)
  print('INVENTED_CENTERED_CONTRAST_INTERACTION',float(delta[1]-delta[0]))
 def test_normal_runtime_pin_mismatch_rejects(self):
  with patch.dict(check.EXPECTED,{check.SOURCE:'0'*64}):
   with self.assertRaises(AssertionError):check.main()
 def test_helpers_are_not_general_numeric_or_shape_validator(self):
  with self.assertRaises(OverflowError):check.softmax3(1000,999,998)
  a=check.ordinary_poll_logit((1,),((1,0),(0,1)),0,.5,1)
  b=check.ordinary_poll_logit((1,),((1,0),),0,.5,1)
  self.assertEqual(a,b)
if __name__=='__main__':unittest.main(verbosity=2)
