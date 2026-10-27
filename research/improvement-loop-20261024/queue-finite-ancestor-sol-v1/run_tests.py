"""Bounded actual-fixture regression runner with authenticated cleanup verification."""
import json,time,unittest
from pathlib import Path
import execution as ex
import test_experiment_queue as inherited

observed=[]
original=inherited.QueueTests.cleanup

def cleanup(self):
 identities=[]
 for store in [self.root/'executions',self.base/'controllers']:
  for name in ['supervisor.json','child.json']:
   for path in store.glob('attempts/*/'+name):
    identities.append(json.loads(path.read_text())['identity'])
 original(self)
 deadline=time.monotonic()+3
 while any(ex.observe(i)['state']=='alive'for i in identities)and time.monotonic()<deadline:time.sleep(.01)
 self.assertFalse(any(ex.observe(i)['state']=='alive'for i in identities),'owned fixture survived cleanup')
 observed.extend(identities)

inherited.QueueTests.cleanup=cleanup
suite=unittest.defaultTestLoader.loadTestsFromNames(['test_compatibility.SuccessorQueueTests','test_finite.FiniteTests','test_ownership.OwnershipTests','test_integration.IntegrationTests'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
Path('cleanup-verification.json').write_text(json.dumps({'fixture_identity_observations':len(observed),'identities':observed,'scope':'only this test runner inherited temporary roots; direct-executor suite separately asserts own cleanup; no identity alive at completion of each cleanup'},indent=2)+'\n')
raise SystemExit(not result.wasSuccessful())
