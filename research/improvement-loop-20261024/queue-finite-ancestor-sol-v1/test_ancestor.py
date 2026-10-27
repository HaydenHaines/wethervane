import itertools,random,tempfile,unittest
from pathlib import Path
import execution as ex

class AncestorTests(unittest.TestCase):
 def oracle(self,paths):
  return any(ex.overlap(a,b) for i,a in enumerate(paths) for b in paths[i+1:])
 def check(self,paths):self.assertEqual(ex.overlapping_outputs(paths),self.oracle(paths),paths)
 def test_exhaustive_order_duplicates_roots_and_lexical_paths(self):
  paths=['/','/a','/a-b','/a/b','/b','a','a/b','.','/a//b','/a/./b','/a/../b']
  for n in range(4):
   for case in itertools.product(paths,repeat=n):self.check(case)
 def test_random_property(self):
  r=random.Random(6027)
  for _ in range(2000):
   paths=['/'+ '/'.join(r.choices(['a','a-b','b','c'],k=r.randrange(0,7)))for i in range(r.randrange(0,20))]
   self.check(paths)
 def test_all_prefix_counterexample_permutations(self):
  for case in itertools.permutations(['/a','/a-b','/a/b']):self.assertTrue(ex.overlapping_outputs(case))
 def test_symlink_canonical_validation_unchanged(self):
  with tempfile.TemporaryDirectory()as name:
   root=Path(name).resolve();(root/'target').mkdir();(root/'alias').symlink_to(root/'target',target_is_directory=True)
   self.check([str(root/'alias'),str(root/'target')])
   with self.assertRaises(Exception):ex.absolute(str(root/'alias'))
   self.assertEqual(ex.absolute(str(root/'target')),root/'target')
 def test_noncanonical_validation_unchanged(self):
  for value in ['/tmp/a/../b','/tmp//b','relative']:
   with self.assertRaises(Exception):ex.absolute(value)

if __name__=='__main__':unittest.main(verbosity=2)
