# Authenticated within-plan dependency dispatch research

This fresh opt-in prototype runs real producer→consumer jobs through the original executor. It extends the independently reviewed admission and independent-job research, with original fixed within-plan dependency declarations, capacity one, exported immutable-through-API bindings/results/releases, and synchronous authentication of the selected producer chain. It does not replace the original queue or modify older prototypes, owner jobs, scientific plans or forecasts.

## Supported scope and identities

A fresh dedicated root has config schema `DependencyDispatchResearchV1`. Old independent-dispatch configs and old queue receipts are refused. The database retains V2 admission semantics and adds local reservation/release tables; direct V2 database mutation or direct executor launch bypasses this cooperating API and is unsupported. Initialization requires a fresh root and does not repair interrupted bootstrap. Config/source pins cover this dispatcher, V2 admission, and original queue/executor/ledger/receipt sources. All public dispatch/admission operations serialize under one domain flock; capacity is fixed at one.

`enqueue(root, plan, expected_sha256)` admits a fixed original-shape plan through V2. Dependencies must use jobs in that same plan, authenticated native receipt roles, exact fixed absolute path arguments and explicitly bound SHA sentinels. The original validator rejects unknown producer/role, cycles, duplicate edge/argument bindings and unbound substitutions; V2 rejects duplicate attempts and equal/nested canonical outputs. Cross-plan producer selection and new-plan prelaunch resumption are not implemented. Scientific content and approval of each plan remain external decisions.

`tick(root)` considers pending indexed owners, skipping known not-ready dependencies rather than relying on job insertion order. A failed producer leaves its consumer pending with a not-ready observation; it does not create a false consumer result or launch. This is not an all-work-completed worker API. An integrity failure in pending selection blocks that tick, conservatively including unrelated work. An unresolved active reservation always retains capacity.

## Binding and producer authentication

Selection rechecks the canonical current admission bytes/digest, queue and job identity, attempt/output ownership and canonical path, current plan/executable/input pins, and any retry predecessor receipt. The admitted base spec remains unchanged.

For each declared dependency, the dispatcher requires a **committed SQLite producer release**, not an output or release file that merely exists. It authenticates the row's canonical bytes and digest, the producer reservation and frozen binding, recursively selected ancestor bindings, exact registered effective executor specification, original executor terminal/evidence pins, known process return, verified cleanup and absent process/session/job lock. It re-derives the outcome and native receipt role/size gates. A nonzero, timeout or invalid native result is not eligible. An unknown exit has no release at all and retains producer capacity even after explicit existing-executor recovery establishes cleanup.

The producer result is an external `DependencyJobResultV1` document, with exact attempt, reservation digest, terminal pin, outcome, selected role pins and `unassessed` qualification. The separate `DependencyCapacityReleaseV1` document binds that result pin, terminal pin and reservation digest. SQLite stores a `DependencyReleaseRecordV1` referencing exact paths/hashes of both documents. Consumers verify all those identities and recomputed contents, not just a success string or bare row. Exported file names are fixed under the current domain's `evidence/ATTEMPT/`; switched paths are rejected.

A consumer's external `DependencyExecutionBindingV1` stores its frozen base job/admission digest, full effective spec and each declared edge's exact producer attempt/result/release/terminal/output pins. Only declared SHA argument slots change; fixed path arguments remain original. Result, release, terminal and selected output pins enter the effective executor's `input_pins`, with conflicting duplicate paths rejected. Thus original executor preparation and launch independently check their exact bytes. The dispatcher also reconstructs/rechecks the selected producer chain on every active reconciliation and again after preparation immediately before launch. Caller-supplied arbitrary files are not promoted to dependency roles.

An explicit raw file referenced inside a receipt remains the owner command's validation responsibility. The tiny consumer fixture validates its producer receipt's data pin itself. This layer authenticates selected native receipt bytes; it does not recursively discover arbitrary transitive scientific references or respondent identities inside them.

## Publication, reservation and crash recovery

A binding is immutably published/fsynced before its SQLite reservation commits. A caller death after binding publication but before reservation is recoverable only if the existing binding exactly equals the newly authenticated selection and there is no executor attempt/result/release scaffold. Existing external attempts are not adopted. Unknown evidence scaffold files fail closed. A crash during publication can leave unknown temporary files and therefore require explicit investigation rather than automatic deletion.

A committed reservation pins that binding and precedes executor preparation. Existing exact, registered, never-launched preparation may continue within that same reservation; partial registration retains capacity. Once a launch intent exists, the dispatcher never automatically relaunches it. Missing/orphan/live/unknown/corrupt state stays reserved. Dispatcher code does not call stop, recover or signal processes.

After terminal/cleanup verification, result and release files are immutably published, in that order, while the reservation stays active. Only a later FULL-synchronized SQLite transaction committing the release record frees capacity and authorizes a dependent. On a crash after either publication or before database commit, the next tick re-derives the executor facts, requires exact equality of any existing files, and commits the record. Changed published files are not overwritten or rebound. A lost response after commit does not recreate a reservation or launch. There are no guessed exits, PID-only cleanup rules or existence-only dependency success.

## Mutable-storage trust contract

Operational observations say `trusted_mutable_index_no_historical_audit`. SQLite/config/source records are trusted local state under cooperating writers. Installed triggers enforce API immutability; a filesystem owner can drop triggers, coherently rewrite/rollback the database or replace the local config. Hashes and SQLite version strings do not prevent that. No protected historical head or stronger write-once storage is supplied.

Current active and selected dependency-chain bytes are checked synchronously, including released ancestors actually needed by a selected consumer. Unreferenced released history is **not** audited on every tick. Forging a consistent trusted release row may invalidate global capacity outside this trust boundary. The old queue's immediate arbitrary historical-file tamper detection is not preserved or claimed. No full dispatch structural audit/migration is implemented. Local SQLite locking/fsync, original Linux pidfd/session/cooperating-lock semantics and V2 path-alias limitations apply. A noncooperating writer can still race between checks; launch input pins narrow but do not eliminate that filesystem threat.

## Repeats and scientific authorization

A new explicit plan/job/attempt/output may repeat an earlier successful or failed executor job, provided the original executor accepts the exact pinned prior terminal/recovery lineage and nonempty reason. Successful predecessor repeats are allowed and tested; there is no failure-only or scientific-necessity gate. Old dependent bindings remain attached to their original producer. A new producer version cannot silently replace a pinned producer in an existing binding. No source admission, scientific retry approval, hypothesis choice, fit, prediction score or promotion follows from dispatch eligibility.

## Reproducer and bounded evidence

Run from this directory:

```sh
nice -n 15 python3 -B -m unittest test_dependency -v
```

Fixtures use temporary directories and original real detached supervisors/child commands. Every source edit, intentional file/database corruption, explicit test-only stop/recover and cleanup signal targets these synthetic fixture identities. No heavy jobs or owner domains are touched.

Tests exercise consumer-before-producer ordering, real SHA argument and input pin chains, a transitive three-job chain, nonzero/native-invalid/unknown producers, output/result/release/terminal/database-release tampering, consumer binding tamper, mutation after preparation before launch, caller death at binding publication, reservation, preparation, launch, result publication, release publication, release insertion before commit and release commit, explicit successful predecessor repeat, unknown cross-plan edge rejection and refusal of old domains. Unknown-exit testing waits for the actual child receipt, then kills the fixture supervisor and uses original explicit stop/recover; the resulting unknown remains reserved. Process death uses `os._exit(79)`, not exceptions. Exact final results and source pins are in `tests.log` and `evidence.json`.

## Remaining gaps

This is not an unattended service, production migration, full historical audit, protected archive, scientific controller or thousands-cycle demonstration. It has no cross-plan result authorization, admission-time prior-output authority, durable blocked-result records, cancellation/prelaunch-deferral release, new-plan resumption, automatic orphan recovery, portable restore, independent storage anchor or comprehensive concurrent-client/power-loss testing. Pending selection scans indexed candidates and selected plan validation reads sibling jobs; recursive ancestry verification can revisit shared predecessors and has no scale bound or measured throughput claim. Multiple-parent joins, many-role graphs, every crash instruction/fsync boundary and malicious coherent state rewrites require further targeted tests. Capacity-one unknown/corrupt reservations may block indefinitely by design.
