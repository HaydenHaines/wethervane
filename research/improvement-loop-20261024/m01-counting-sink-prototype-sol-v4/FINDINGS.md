# V4: retained caller acknowledgment and verified byte snapshots

Six invented-object tests pass. V4 improves the uncertain-publication and verification/use boundaries through a deliberately narrower contract: an **ephemeral acknowledgment retained by the original caller process**, and an immutable snapshot of all checked artifact bytes. It does not turn visible marker files into durable approval.

`engine.py` is a private copy of the frozen V3 transport, with the same source SHA. Its raw file verifier retains V3's limitation; it is used explicitly as a counterexample in the tests. V4 verification is `Session.verify`, not a call to that lower-layer verifier or a later reopen of verified paths. V1–V3 are unchanged.

## Caller acknowledgment

The caller selects expected engine and session source SHAs. `Session.execute` holds the output's exclusive cooperating lock, runs the known-source controller/supervisor chain, waits for its successful returned completion, verifies the captured evidence, and **only then** issues a frozen acknowledgment object. It binds source versions, original caller PID, issuer/nonce, output/control directory identities, lock identity, manifest/receipt digests and a session MAC. The original object is also retained in a private issuance registry. A dictionary, recomputed file hash, cloned object, new Session or forked process cannot reconstruct/adopt it.

In the actual injected regression, external `success.json` is renamed and its directory-fsync call raises EIO. The file remains readable, and the copied V3 verifier accepts it if its hash is recomputed. V4 issues **zero acknowledgments** and rejects an attempted rehashed-metadata substitute. Thus the corrected API makes the missing successful caller acknowledgment operationally significant rather than a documentation instruction alone.

This is not hostile in-process security: trusted Python code with access to private memory can subvert the registry/key or bypass Session entirely. The MAC is not an independent approval signature. The capability is intentionally process/session-local and volatile. There is no acknowledgment loader, restart recovery or durable cross-process adoption. A process crash loses acknowledgment state; surviving files cannot recreate it. No claim of arbitrary malicious-filesystem protection or power-loss durability follows.

## Snapshot and concurrency boundary

Cooperating readers take a shared flock; writers and initial execution take an exclusive flock on the same separately identified lock file. Directory identities are retained, exact rosters are checked, and regular files are opened through directory descriptors with no symlink following. Per-file captures compare stat identities/sizes/times, bound each file to 1 MiB, and bound the combined snapshot to 8 MiB. Every digest and cross-file controller/source/request projection is checked from **the captured bytes**. Parsed data never comes from a later reopen. Returned byte mappings and nested metadata views are immutable.

The concurrent test holds two reader locks simultaneously, verifies that an exclusive cooperating writer is blocked, then releases readers. Both captured snapshots retain the original bytes after the writer changes metadata. A subsequent verification rejects the altered file. Thread coordination and join timeouts bound the test; no actual scientific process is involved.

The noncooperating tests deliberately ignore the flock:

- Changing bytes before capture makes snapshot verification reject.
- Changing the on-disk candidate immediately after its bytes are captured leaves a valid original-byte snapshot while the live directory is already changed. The next verification rejects the changed directory.

The latter is an intentional counterexample to a **live-directory** claim. A verified snapshot authenticates the exact returned byte image, not whatever a pathname contains later. Consumers must use `VerifiedSnapshot.files` and its derived views, never reopen paths and call that use verified. A hostile writer can ignore advisory locks; no flock forces arbitrary filesystem owners to cooperate. Stat checks and hashes are tripwires, not a proof against every hostile interleaving, coherent rewrite or rollback. Under noncooperation, captured files may come from different instants; they must still match the one caller-acknowledged manifest and all cross-projections, but the API does not attest global simultaneous on-disk existence.

Lock acquisition has no built-in timeout or recovery protocol in this prototype. A cooperating writer retaining its lock can block verification; no production liveness guarantee is claimed. Initial controller execution is allowed only before reader/writer threads are created, avoiding a threaded-fork claim. The multithreaded tests perform tiny bounded filesystem coordination, not parallel model computation.

## Scope and preserved evidence

The copied worker still accepts only fixed invented objects, has CPU 2/3-second, address-space 256 MiB and per-file 1 MiB hard limits, and the existing bounded local watchdogs. Tests run nice15. The new snapshot logic stores only small artifact bytes; no serialized object payload, loader, actual graph, sampler, partial pickle or owner file is touched. Per-Pickler-write two-fsync progress overhead remains unchanged and unmeasured for real graphs. Chunk reconstruction remains absent.

Both V4 synthetic runs are preserved. Final tests cover normal acknowledgment/snapshot success; copy/restart/fork rejection; uncertain receipt publication; two readers versus a cooperating writer; and mutations before/after capture by an uncooperative writer. `evidence.json` pins source, test records and every retained fixture, and verifies V1–V3 packet files unchanged. This is a bounded research approach and explicit counterexample set, not production readiness or scientific qualification.
