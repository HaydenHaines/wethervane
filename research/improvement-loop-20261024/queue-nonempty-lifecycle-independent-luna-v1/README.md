# Independent review: nonempty lifecycle consumer contract

All 11 declared source pins match. I parsed old/new dispatcher source and independently hashed ASTs: exactly `initialize`, `opened`, `locked`, and `completion` changed; the other 16 dispatch functions match the source-delta hashes. I replayed the ten tiny lifecycle tests in a fresh temporary copy (10 passed, 5.555 seconds). They exercise empty, admitted, active after reservation, unknown supervisor exit, terminal blocked descendants, failed-only work, successful work and admission reopening, plus the unchanged crash/closure transitions.

The new schema prevents the prior empty-store shortcut: the empty state is `empty_no_admitted_work`, and both new nonempty completion flags are false. A nonempty failure/blocked terminal state is all-terminal but not accepted. A nonempty accepted terminal state is explicitly marked “unqualified,” not scientifically qualified. New admissions reopen the snapshot. The old ambiguous booleans are omitted and the consumer rejects those legacy fields.

One boundary remains: `consumer.py` validates internal shape/state invariants, not artifact provenance. A fully self-consistent fabricated success snapshot is accepted. That is consistent with the code's fixture-only documentation and the trusted-local snapshot boundary; do not use it to authenticate arbitrary serialized JSON. If an actual cross-process or persisted consumer is added, it needs a trusted source/digest/signature or direct authenticated database read.

No lifecycle defect was found in the failure/crash paths in the replay. Process-death tests remain distinct from power-loss durability; full descendant closure remains one transaction that may scale with DAG size; no service intake seal, migration, large scale test, or production integration is supplied. The review and replay were read-only outside this separate lab packet.

Hashes and replay receipt are in `review.json` and `evidence.json`.
