# Independent review: same-UID history authority reference

## Summary

I reviewed the frozen reference as a correctness prototype under its declared `trusted_mutable_same_uid_reference` boundary. The 16 supplied tests passed in a fresh scratch copy. I found no contradiction in its stated core model: event, evidence object, head, idempotency response and derived projection updates are committed in one SQLite transaction; replay reconstructs projections from the event chain; one active slot is enforced transactionally; idempotent retries return historical receipts; and cleanup/release claims remain explicitly unauthenticated.

This does not establish a protected authority or production queue. The reference has no worker launcher, OS identity/cleanup verifier, protected storage, independent monotonic head, network authentication, bootstrap repair, or scale evidence. Use it as a contract/correctness oracle only.

## Exact inputs

Frozen source packet: `/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/queue-history-authority-reference-sol-v1`.

- `reference.py`: SHA-256 `dbfb14d9b9365fcf0a9feaaebe24b11f80e874273b601ea55606a8a89027f1f2`.
- `test_reference.py`: SHA-256 `dd80817397690fe67013adf5e43b7931614f8cda34a89449bb9194d7d4690e8b`.
- `evidence.json`: SHA-256 `59cce4cfe9622b85364947329c9d585b40453c158f0d6823a48ae16510d41078`.
- `README.md`: SHA-256 `906770f403dd8103ae0fea05cb09bc5320957899e8d9b1c5dc105a9961025a49`.
- `source-pins.json`: SHA-256 `1fc7a49225868d31a388916826273be579509ac19b164dab2c8ca837bbd06b8f`.
- `test-result.json`: SHA-256 `8bedc500c5bfaddfccd94b74d691fc4bf5e1b31ef4b6a9ed26e68af821f9e6b9`.
- `tests.log`: SHA-256 `87da41e395b4072237e5228d099025ff1daddf46d2870fb6fb25799e877f4d5f`.

I rehashed the packet after review; these bytes were unchanged. The source-pins file references the architectural design and earlier memo review; the four design files and two memo-context files matched those pins on inspection. Those sources themselves describe the protected authority as future design, not implemented capability.

## Reproduction and targeted probes

Copied only `reference.py` and `test_reference.py` to a fresh `/tmp` directory, then ran `nice -n 15 python3 -B -m unittest test_reference -v`. All 16 tests passed in 0.711 seconds. The suite includes eight real child-process `os._exit` boundaries, concurrent capacity-one reservation, rollback/fork with and without a caller floor, response-loss retry, rebuild, fake cleanup, projection/object tamper, and empty/failed/accepted distinctions. These are local SQLite/process-death fixtures, not kernel or power-loss simulation.

I also ran the preserved scratch probe `targeted_probe.py` (SHA-256 `2606d6284f8be988c051b565e00052d899a399656635484568c257f4d113ff7f`); output is `targeted-probe-result.json` (SHA-256 `1df5cad05fdf737e8eadda100db52ba69a15d19a269e639cc26e3f700c3c94ff`). It verifies:

- Exact retry returns the original historical receipt after both later event-head and lease advancement, without appending or authorizing a launch.
- Reusing that idempotency key for a different action is refused as `idempotency_body_conflict`.
- A coherently fabricated cleanup object matching active job/worker/reservation fields is accepted by this synthetic API, with `runtime_worker_evidence_authenticated=false`. This is an explicit reference limitation, not evidence of cleanup.
- `check_response` accepts a forged later head with arbitrary hash when domain, nonce and echoed floor match. It returns `consistent_reply_claim=true` and `authenticated=false`; it does not cryptographically prove that the later head descends from the supplied floor. This is consistent with the disclosed unauthenticated trust boundary, but downstream code must not use this helper as ancestry authentication.

Run the targeted probe with `python3 -B targeted_probe.py` from this review directory. It checks the frozen reference hash, copies it to temporary storage and uses invented data only.

## Source-level assessment

### Atomicity and idempotence

`_mutate` starts `BEGIN IMMEDIATE`, verifies canonical history and current projection, verifies the expected head/lease for a new transition, stores bounded admission bytes as content-addressed rows, applies the transition, appends the hash-linked event, updates the canonical head and projections, inserts the exact idempotency response, and commits once. All these records are in the same SQLite database transaction. A crash before commit rolls back; after-commit response loss is resolved by reading the exact original response. Retry lookup precedes new-transition stale-head/lease checks but still requires the caller's expected head to be a valid historical prefix. This matches the stated historical-replay semantics; the result explicitly has `historical_commit_only=true` and `live_launch_authorization=false`.

`_read` recomputes the canonical event chain, applies every event, verifies referenced evidence bytes, checks the canonical head and response table, and compares derived job/ready/active/frontier projections. Corrupt projections refuse normal operations; explicit rebuild derives them from canonical history and is transactionally repeatable. This is an O(history) oracle on every operation, not an incremental performance implementation.

### Ownership, reservations and cleanup

Output ownership is canonical logical POSIX path-component overlap (including either ancestor direction), not physical filesystem identity; symlink and actual output materialization are explicitly absent. Dependencies must name earlier unique jobs, so this API cannot create forward edges or cycles through a later edit. `BEGIN IMMEDIATE` plus the unique active slot and head/lease conditions make the tested concurrent reserve have one winner. Lease advancement fences stale transition requests but never releases active capacity. Release checks the current active job, worker token and reservation hash and requires exited/cleanup flags plus well-formed digest fields; it then accepts only unqualified accepted or failed outcomes and propagates failure to descendants. Those cleanup fields are caller assertions. No process, terminal artifact, output bytes, or OS state is authenticated here.

### Head, rollback, fork and nonce

The source verifies the caller-supplied floor as an exact prefix in the current chain. It correctly refuses the tested rollback and same-sequence fork when the caller retains the old exact floor. It also openly reports no rollback protection when no floor is supplied. A floor only protects its recorded prefix: it is not an independently hosted monotonic head, and a coherent same-UID rewrite/fork after the last retained checkpoint is not prevented by local hashes. `head_response` echoes nonce and floor; `check_response` is unauthenticated and, as the targeted probe shows, checks an echoed floor and sequence comparison rather than a proof that the returned later head extends that floor. No signature, key custody, transport or remote authority exists.

### Crash/rebuild/concurrency boundaries

Tests exercise process termination at event, projection, before/after commit and rebuild boundaries, plus two concurrent SQLite clients racing for the single reservation. They do not demonstrate power-loss, filesystem/journal corruption, disk-full/fsync uncertainty, external object CAS, partial bootstrap recovery, or independent backup/restore protection. Initialization has no repair path for partially created state and fails closed. This is suitable for the declared small correctness-reference scope, not a durability or recovery guarantee for production.

## Findings and recommendation

No blocking defect was found within the explicitly unprotected same-UID reference contract. Preserve the forged-cleanup and unauthenticated-response probes as scope evidence. The necessary next step for any stronger authority claim is external to this packet: an independently protected append/head/object authority and bootstrap/rollback policy, plus a reviewed adapter that authenticates executor terminal, output and cleanup evidence. Keep the reference's status/receipts unqualified and do not connect them to live launching or scientific completion.

No production integration, executor/model worker, or owner file was changed.
