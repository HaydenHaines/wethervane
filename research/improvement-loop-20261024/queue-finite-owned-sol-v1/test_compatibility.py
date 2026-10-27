"""Three explicit finite-intake API substitutions; all other original tests inherited."""
import os
from pathlib import Path
import execution as ex
import experiment_queue as q
import finite_intake as f
from test_experiment_queue import QueueTests

class SuccessorQueueTests(QueueTests):
 def test_empty_scaffold_admission_and_failed_terminal_drift_are_explicit(self):
  # Unlike pre-lifecycle V3, unknown filesystem scaffolds cannot be adopted.
  (self.root/'queues'/'scaffold'/'jobs').mkdir(parents=True)
  with self.assertRaisesRegex(ValueError,'durable intent'):self.admit([self.job('failure','nonzero')],'scaffold')
  self.assertEqual(f.info(self.root)['intents'],0);self.assertFalse((self.root/'executions').exists())
  # Preserve the original failed-terminal mutation assertion in a clean generation.
  self.root=self.base/'legitimate';q.initialize(self.root,2);self.admit([self.job('failure','nonzero')],'legitimate');self.run_queue();terminal=self.result('failure','legitimate')['executor_terminal'];Path(terminal['path']).write_text('changed failed terminal evidence');state=q.tick(self.root);self.assertFalse(state['all_jobs_recorded']);self.assertEqual(state['reserved'],1);self.assertEqual(state['queues']['legitimate']['failure']['outcome'],'quarantined_integrity')
 def test_malformed_admission_preserves_unknown_reservation_count(self):
  # ACCEPTED DB pins make corruption an early exception, not an observation API success.
  self.admit([self.job('live','hold')]);q.tick(self.root);self.wait(lambda:(self.root/'executions'/'attempts'/'live'/'child.json').exists());reservation=self.root/'queues'/'q0'/'jobs'/'live'/'reservation.json';pin=ex.pin(reservation);(self.root/'queues'/'q0'/'receipt.json').write_text('malformed')
  with self.assertRaises(ValueError):q.tick(self.root)
  with self.assertRaises(ValueError):q.finalize(self.root)
  self.assertEqual(ex.pin(reservation),pin);self.assertFalse((reservation.parent/'release.json').exists());self.assertFalse((self.root/'final-run.json').exists());self.assertEqual(len(list(self.root.glob('queues/*/jobs/*/reservation.json'))),1)
 def test_missing_admission_with_reservation_freezes_other_queue_dispatch(self):
  self.admit([self.job('live','hold')],'first');q.tick(self.root);self.wait(lambda:(self.root/'executions'/'attempts'/'live'/'child.json').exists());reservation=self.root/'queues'/'first'/'jobs'/'live'/'reservation.json';pin=ex.pin(reservation);receipt=self.root/'queues'/'first'/'receipt.json';receipt.rename(receipt.with_suffix('.preserved'))
  with self.assertRaises(ValueError):self.admit([self.job('must-wait','hold')],'second')
  with self.assertRaises(ValueError):q.tick(self.root)
  self.assertEqual(ex.pin(reservation),pin);self.assertEqual(f.info(self.root)['intents'],1);self.assertFalse((self.root/'queues'/'second').exists());self.assertFalse((self.root/'executions'/'attempts'/'must-wait').exists());self.assertFalse((reservation.parent/'release.json').exists())
