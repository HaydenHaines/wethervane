import unittest,math
from dataclasses import replace
from gvf import *
from check import check
class Gate(unittest.TestCase):
 def test_four_official_examples(self):self.assertEqual(check()['passed'],4)
 def test_invalid_variance_not_clipped(self):
  for args in [(10,-1,0),(1e308,1e308,1)]:
   with self.assertRaises(ValueError):total_se(*args)
 def test_negative_or_nonfinite_inputs(self):
  for x in [-1,float('nan'),float('inf'),True]:
   with self.assertRaises(ValueError):total_se(x,-.000031,2947)
 def test_percent_and_base_domain(self):
  for p,base,b in [(-1,100,1),(101,100,1),(50,0,1),(50,-1,1),(50,float('nan'),1),(50,100,-1)]:
   with self.assertRaises(ValueError):percentage_se(p,base,b)
 def test_reject_mapping_changes(self):
  m=MAPPINGS['2018-percent']
  for bad in [replace(m,year=2020),replace(m,b=5411),replace(m,geography='NY'),replace(m,parameter_table=7),replace(m,unit='proportion'),replace(m,characteristic='registration')]:
   with self.assertRaises(ValueError):checked_se('2018-percent',bad,p=49.1,base=248937000)
 def test_unsupported_state_not_inferred(self):
  with self.assertRaises(ValueError):checked_se('2018-NY',MAPPINGS['2018-percent'],p=49.1,base=248937000)
 def test_incompatible_formula_inputs(self):
  with self.assertRaises(ValueError):checked_se('2018-total',MAPPINGS['2018-total'],p=49.1,base=248937000)
  with self.assertRaises(ValueError):checked_se('2018-percent',MAPPINGS['2018-percent'],x=100)
 def test_units_are_not_implicitly_converted(self):
  # A fraction supplied as p gives a different, wrong estimand; arithmetic cannot infer intent.
  a=percentage_se(49.1,248937000,3496);b=percentage_se(.491,248937000,3496)
  self.assertNotAlmostEqual(a,b,places=2)
  self.assertAlmostEqual(percentage_se(50,1000,4),math.sqrt(10))
 def test_negative_a_can_be_valid(self):self.assertGreater(total_se(2941000,-.000031,2947),0)
if __name__=='__main__':unittest.main()
