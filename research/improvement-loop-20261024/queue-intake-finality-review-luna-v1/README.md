# Queue intake-finality gap review

The concrete remaining obstacle is the lack of an atomic boundary between “all jobs currently admitted are recorded” and “no more jobs can be admitted to this run.” In the current queue-v3 path, `work()` consumes a snapshot from `tick()`, exits when `all_jobs_recorded` is true, and publishes `finished`. The dispatcher lock protects each tick and enqueue separately; once tick returns, a cooperating enqueue can land before the worker receipt is written. No epoch is sealed and enqueue does not reject late work.

This is a process-lifecycle gap, not an error in the snapshot counter: the snapshot can be true for work admitted so far. It prevents `finished` from serving as a reliable finite-batch or shutdown signal. The newer nonempty lifecycle API improves empty/failure semantics but explicitly remains a snapshot; it is not integrated into queue-v3 intake, and it does not close the race.

The smallest correct route is either an always-running supervisor that reports idle and waits for more work, or a finite batch API with a durable OPEN/SEALED/FINALIZED epoch and atomic enqueue/seal ordering. See `review.json` for crash cases. Finalization remains workflow bookkeeping, never scientific qualification. No test or queue/model job was run.
