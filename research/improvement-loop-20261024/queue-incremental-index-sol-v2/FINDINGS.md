# Opt-in incremental admission research core V2

This is a runnable admission/ownership core, **not a dispatcher or a completed adaptive queue**. Existing `experiment_queue.py`, stores, receipts, and production jobs are unchanged. `core.py` accepts the existing fixed plan shape after its existing validator checks current executable/input pins, but uses a separate `IncrementalAdmissionResearchV2` SQLite store. There is no migration, queue receipt compatibility, launch API, capacity claim, model fitting, or scientific qualification. Adaptive callers may supply a new approved plan after an earlier result; this core does not approve or generate that plan.

## Correction from the preserved failed V1

V1 had a blocking ownership defect: checking only neighboring lexically sorted paths missed `owned` versus `owned/child` when `owned-other` sorted between them. Its structural audit also falsely accepted this invalid ownership projection. V1 and its original evidence remain unchanged; `../queue-incremental-index-sol-v1-failure-preserved/` holds byte-for-byte copies, the independent rejecting review, and a runnable reproducer with observed false admission/audit success. V1 is unsuitable for adoption.

V2 checks every output's complete parent chain against the entire batch set; duplicates and ancestor pairs fail regardless of order. Structural audit independently applies the same ownership invariant to both reconstructed evidence and indexed rows, so matching invalid projections are rejected. Regressions cover all six permutations of the three-path counterexample, ordinary and reversed ancestor pairs, exact duplicates, harmless sibling prefixes, cross-plan admission in both orders, and an injected preexisting internally consistent nested projection.

V2 uses a distinct `IncrementalAdmissionResearchV2` schema and explicitly refuses V1 stores. There is no migration or implicit upgrade; a regression verifies that V1 metadata is rejected by open, admission and audit and cannot be replaced by initialize. This is still research-only and not production integration.

## Trust and status

The operational contract is explicitly `trusted_index_not_historical_audit`. The SQLite file is authoritative admission evidence and ownership index, under trusted local storage and cooperating writers. Evidence and ownership are append-only through the supplied API and SQL triggers. These are logical immutability rules, not write protection against the filesystem owner. Arbitrary malicious SQL can drop triggers; a writer can replace or roll back the entire database. Neither ordinary admission nor operational status detects arbitrary historical modification. In particular the test deliberately drops an ownership trigger and deletes claims: operational status reads the modified count, while explicit structural audit rejects the broken projection. A consistent malicious rewrite/rollback needs an independently protected anchor and is not detected by this prototype.

`audit()` scans every canonical plan body and digest and reconstructs the exact ownership projection in one read transaction, plus SQLite integrity checks. It does **not** verify historic mutable executable/input bytes, historical schema semantics, execution receipts, or an independently recorded head. It is called a structural audit for that reason. Original queue full-history integrity/status semantics remain separate and stronger.

## Transaction and ownership behavior

`initialize(path)` requires a fresh file and a caller-provisioned durable parent directory. It uses O_EXCL, SQLite FULL synchronization with the default rollback journal, then fsyncs the parent. An interrupted bootstrap is a visible unsupported partial database: no auto-delete or silent adoption. Use a local filesystem with SQLite's documented locking/sync semantics; network filesystems and broken disk sync are outside this experiment.

`admit(path, plan, expected_sha256)` freezes a canonical logical snapshot of the caller's plan, authenticates the supplied digest, obtains `BEGIN IMMEDIATE`, and freshly invokes the original plan validator on that snapshot. Thus all declared executable/input bytes are checked on every admission, including exact retry. This does not prevent an external writer changing inputs after validation: launch must recheck through the original executor gates. Canonical plan bytes/digest and all attempt/output claims commit in **one transaction**. There are no separately published evidence files, executor scaffolds, reservations, results, or release records to reconcile. A rollback-journal recovery discards uncommitted evidence and claims together; a lost reply after commit is an exact idempotent retry.

Attempt and canonical absolute output ownership use unique SQLite indexes. Output ancestors are checked by exact indexed queries; descendants use an indexed binary prefix range. Exact, ancestor, descendant, normalized alias and currently resolvable symlink conflicts fail closed. Future externally created symlinks, bind mounts and filesystem aliases are outside this cooperative ownership boundary. Fresh output paths must not already exist; even an exact retry fails if its output now exists. The core owns no executor substore and cannot know about attempts registered elsewhere. Dispatch integration must retain original external attempt/preparation gates.

Competing writers serialize before checking ownership; evidence and ownership are never independently visible as committed state. An exact retry checks the selected plan and its complete ownership rows without scanning history. All resumption proofs are rejected, including valid original queue proofs. No ownership is ever released or reused. This is intentionally conservative until exact prelaunch lineage, competing active claims and original executor cleanup can be integrated and independently tested. A launched retry needs a new attempt/output and independent original scientific authorization; this core's admission is not that authorization.

## Reproducer and evidence

Run from this directory:

```sh
nice -n 15 python3 -B -m unittest test_core -v
```

Tests use temporary directories, invented plan/input files and no executor launches. They verify fresh input drift and digest checks, changed admission identity, exact retry, rejected resumption, duplicate attempt/output claims, nested/alias outputs, concurrent admission, enforced API immutability, explicit stale/malicious-index status limitations, and process death after evidence insertion, ownership insertion and commit. The death probes use `os._exit` from separate processes, not merely Python exceptions. They establish atomic recovery at those boundaries, not all SQLite kernel/fsync fault interleavings or physical power-loss durability. The bounded scale case admits 1,000 synthetic plans serially and audits all 1,000; it is not 1,000 executed adaptive scientific cycles and provides no comparative performance claim. See `tests.log` for the exact final run.

## Required work before production

There is no dispatch frontier, active reservation reconciliation, dependency binding, result/release authority, worker controller, cross-plan output authorization, external execution detection, original prelaunch resumption, stronger sealed storage, protected audit head, migration or CLI. Integrating these remains necessary before replacing any current queue behavior. In particular, an incremental admission index alone cannot repair the original tick's repeated historical verification cost. A production opt-in queue must explicitly choose and independently review its narrower audit timing/storage contract, authenticate all active reservation and launch dependency bytes synchronously, and test actual executor crash boundaries. No production adoption is proposed here.
