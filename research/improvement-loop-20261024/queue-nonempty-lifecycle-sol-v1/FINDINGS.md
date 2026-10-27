Consumer-safe nonempty lifecycle API — opt-in research successor

The independent prior review found no new false-launch/crash transition defect, but identified a consumer ambiguity: the frozen prior all_terminal predicate was vacuously true for an empty store. Its documented state='empty' and success=false made the intended semantics distinguishable, yet a caller reading that boolean alone could mistake no work for completed work. This version changes the completion contract explicitly; all prior source, reports and queues remain unchanged.

New machine contract

completion(db) now returns schema NonemptyCurrentAdmissionsLifecycleV1 and never returns legacy all_terminal or all_executions_accepted_unqualified fields. There are no compatibility aliases. Both new fields require total>0:

- nonempty_current_admissions_all_terminal: every currently admitted job is execution-terminal or terminal-blocked, with no pending/active jobs, and at least one admitted job.
- nonempty_current_admissions_all_accepted_unqualified: the preceding condition plus no failed executions and no terminal-blocked jobs. This is not scientific qualification.

An empty store returns state=empty_no_admitted_work and both flags false. It cannot report terminal/completed work through this API. Other explicit states are nonempty_current_admissions_active, nonempty_current_admissions_runnable, nonempty_current_admissions_waiting_unresolved, nonempty_current_admissions_terminal_with_failures_or_blocks and nonempty_current_admissions_terminal_all_accepted_unqualified. A nonempty terminal failure remains terminal but never accepted success. New admission reopens the current-admission snapshot. This is still not a work-intake seal or service shutdown predicate.

consumer.py is a direct machine-consumer fixture: it requires the new schema, rejects legacy ambiguous fields, checks integer partition invariants and strict boolean semantics, and accepts only the nonempty all-accepted-unqualified state. It returns false for empty, admitted, active, waiting, failed or blocked work, and rejects malformed or legacy payloads. It is a local snapshot-consistency consumer, not an authority for execution, model promotion, publication or unattended service shutdown. No API can stop a caller from deliberately interpreting failure as success; the explicit names and fixture eliminate the old empty/vacuous boolean shortcut under the supported contract.

Preserved implementation and source evidence

This is a fresh config/database version; old domains/receipts are not adopted. source-delta.json hashes and compares function ASTs against the frozen lifecycle predecessor. Only initialize/opened/locked version identity gates and completion change. Admission, reservation, selected dependency authentication, release, transitive descendant blocking, first-cause idempotence, counters, tick and structural audit function ASTs remain exactly unchanged. All original source bytes are preserved. The new consumer helper is separate and source-pinned in evidence; it is not secretly integrated into production.

The full prior lifecycle tests remain with the explicit new field/state assertions. Ten final tests passed in6.070seconds (tests.log). An earlier nine-test run is preserved as tests-first.log. Tests use real tiny executor commands and actual SQLite client process death/replay at admission, reservation, release and propagation boundaries; cover multiroot/deep diamond failure without descendant launch or duplicate logical blocking, cooperating concurrent operations, unknown exit holding capacity, positive-count waiting corruption, balanced false-zero counter guards, successful chain, new admission reopening, source/domain refusal and a real failed-only job.

The direct machine-consumer test receives actual database snapshots: empty and admitted refuse; a crash immediately after reservation exposes active and refuses; failed producer with terminal-blocked child refuses; a separate successful tiny-job domain accepts; a legacy field and a fabricated true empty flag reject; another admission reopens and refuses again. A failed standalone command with blocked_terminal=0 is independently terminal but refused by the same consumer. Thus failure-without-block and failure-with-block are both distinguished from accepted execution.

Tests run nice15, numerical thread controls1, CPU soft40/hard45seconds, address space512MiB, file16MiB and wall60seconds. Temporary roots are removed; only fixture-owned process identities/sessions are stopped. Aggregate suite CPU, transient disk and process-tree RSS were not separately instrumented. No model fits, scientific outputs, owner queue changes or production integration occurred.

Scope and remaining limits

This corrects one API ambiguity. It does not establish service durability, unattended operation, thousands-cycle throughput, a protected history head or work-intake finality. SQLite/counters and cooperating writers remain trusted; coherent raw SQL/filesystem rewrites or database rollback can defeat operational projections. Bounded presence guards catch specific inconsistencies, not arbitrary malicious tampering. Explicit structural audit remains O(history+graph) and is not a full historical external-input/executor audit.

Descendant propagation is still one write transaction over the entire reachable closure; large graphs can hold the lock/use significant journal/memory, and independent failing roots can revisit shared graph nodes without rewriting logical blocked records. No new scale benchmark is claimed. Failed bootstrap repair, counter/frontier repair, portable restore, cancellation/unblock, same-plan resume, cross-plan dependency authority and durable daemon operation remain absent. Power-loss/disk-full/fsync uncertainty and inherited executor limitations remain; os._exit tests do not establish physical power-loss durability.

Nonempty all-terminal failure is an explicit terminal workflow outcome, not successful execution or a scientific rejection. Even all accepted results remain unqualified. A new admission may arrive immediately after any snapshot; a separately designed intake/seal/service protocol is still needed before automated shutdown or long-running supervision. Independent review is required before any integration.
