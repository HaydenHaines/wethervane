import unittest
from analyze import idle_model
class ArithmeticTests(unittest.TestCase):
 def test_independent_small_enumeration(self):
  for n in range(17):
   pairs=sum(1 for i in range(n)for j in range(i+1,n));self.assertEqual(idle_model(n)['containment_comparisons_per_idle_tick'],2*n*pairs)
 def test_thousand_is_projection_not_workload(self):self.assertEqual(idle_model(1000)['containment_comparisons_per_idle_tick'],999000000)
 def test_zero_and_bytes(self):self.assertEqual(idle_model(0,10)['logical_executable_bytes_per_idle_tick'],0);self.assertEqual(idle_model(3,7)['logical_executable_bytes_per_idle_tick'],126)
 def test_reject_bad_dimensions(self):
  for x in [-1,True,1.5,'8']:
   with self.assertRaises(ValueError):idle_model(x)
  with self.assertRaises(ValueError):idle_model(1,-1)
if __name__=='__main__':unittest.main()
