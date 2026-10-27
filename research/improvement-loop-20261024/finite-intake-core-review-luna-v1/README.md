# Independent review: finite intake core V1

Reviewed the frozen `finite-intake-core-sol-v1` source and ran its 13 tests in a temporary fixture context. The basic OPEN → SEALED → FINALIZED protocol behaves as specified for the tested cooperating processes: enqueue/seal serialize, a committed seal rejects new IDs, pending jobs block finalization, and final receipt recovery preserves committed bytes. Empty and failure dispositions are explicit. The packet is correctly scoped as bookkeeping only; it does not integrate queue-v3, validate real executor receipts, or establish scientific success or thousands-cycle capability.

One concrete recovery blocker remains. `project()` names its exclusive temporary file using only run ID and PID. An ordinary exception after temp fsync but before publication leaves the temp. Retrying `finalize()` in the same process then fails at `open(..., "xb")` with `FileExistsError`, despite the authoritative final row already being committed. I reproduced this with `reproduce_stale_tmp.py`. A fresh random/exclusive per-attempt temp name is the smallest correction; final link publication can remain exclusive and exact-byte checked.

The exact source/test/evidence pins, commands, observed output, crash-scope distinctions, and limits are in `review.json`. No frozen packet was modified.
