# Independent review: partial queue admission recovery

## Verdict

The isolated patch supports merging for the identified partial-plan ownership fault, subject to documenting the malformed-plan operator recovery path. I found no identity escape in the inspected change. It preserves exclusive dispatch/admission locking and checks ownership of durable partial plans before admitting another plan. It permits unrelated admissions, rejects a colliding attempt or output directory, and lets the exact immutable plan seal its interrupted admission. A hard exit after publishing the admission receipt also retries idempotently; `tick()` creates missing job directories and launches once.

## Fault and patch

The original `enqueue()` writes `queues/<queue_id>/jobs`, then the durable `plan.json`, then the `receipt.json`. If the process dies after the plan write, `entries()` ignores that unsealed queue. A different queue ID can therefore be admitted using the same `attempt_id`/`output_directory`, execute, and claim those identities. Retrying the original exact plan then fails with “attempt/output already belongs to another queue plan.” The preserved probe records that one contender launches, the original cannot be repaired, and the original job remains unrecorded.

The patch scans every other unsealed queue directory while holding the existing dispatcher flock. It validates the partial plan and queue ID, and rejects only incoming jobs that reuse a partial plan's attempt ID or owner output directory. An unresolved empty scaffold without `plan.json` blocks unrelated admissions until its own exact repair; this is conservative because the scaffold has no durable claim list to distinguish a safe unrelated admission from unknown prior ownership.

## Independent verification

I ran all 24 original `QueueTests` with Python import precedence forcing the frozen `patched/experiment_queue.py`; all passed (13.5 s). This includes dependency readiness, exact producer receipt/output pins, duplicate/unbound dependency validation, capacity-lock contention, malformed admissions, partial retry, prelaunch and launch crash recovery, the accepted-result-before-release crash boundary, result corruption quarantine, and output/input integrity cases.

I separately ran the three Sol recovery tests; all passed (7.15 s). In fresh `TemporaryDirectory` stores, my additional checks showed:

- An empty `queues/scaffold/jobs` scaffold blocks an unrelated admission; admitting its exact plan seals it, after which an unrelated plan is admitted successfully.
- A child was hard-killed immediately after the patched `enqueue()` published `receipt.json`, before creating the per-job directory. Exact retry returned the same receipt pin; `tick()` materialized the missing directory, reached a recorded accepted execution, launched once, and released capacity to zero.
- A malformed durable partial plan fails unrelated admission closed (`unsupported queue plan`). A different plan cannot replace it (`queue ID reused with changed plan`). This avoids treating unknown bytes as a valid ownership claim but has no built-in recovery command.

The patch changes only `experiment_queue.py`; the copied `execution.py`, `ledger.py`, and `run_receipts.py` match the current root files byte-for-byte. The diff adds the pending-plan ownership scan inside the already-held dispatcher lock; it does not alter receipt, result, release, launch, or dependency-binding logic.

## Limits / merge condition

A malformed or truncated `plan.json`, or an unrecognized queue-store entry, can freeze new enqueue operations until an operator preserves and quarantines/repairs that entry. That is a safe failure mode, but should be documented with a preservation-first recovery procedure; there is no safe automatic repair for malformed plan bytes because the original identity claims cannot be reconstructed. Do not delete a partial plan casually. This patch is not a defense against a malicious process that rewrites store files outside the cooperating API; the queue remains a local trusted-filesystem design.

This is an infrastructure-level recovery fix only. Queue qualification remains `unassessed`; it does not validate any scientific results or authorize promotion/publication.
