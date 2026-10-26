# Mutable-history counterexamples and a V1/V2 release invariant failure

Five bounded adversarial tests exercised the unchanged actual V1, frozen V2 and source-bound incremental derivative in separate fresh temporary domains. **A substantive V1/V2 capacity failure was reproduced:** a malformed release file without a result caused a live capacity-one reservation to stop counting, allowing a second real worker. No implementation was edited. This packet is frozen negative evidence, not a fix or readiness claim.

## Confirmed V1/V2 failure

Each implementation first admits two independent tiny fixture jobs under capacity one. Job A launches and writes its real native receipt, proving it passed the executor gate. Job B has not prepared. A remains alive behind a bounded fixture gate, with its recorded child/supervisor and held job lock. There is no queue result or legitimate capacity release for A.

An explicit owner-writable adversarial mutation inserts `{"forged":true}` as A's queue `release.json`. No legitimate queue publisher generated that file. The next actual tick launches B. Independent original-executor birth/session/lock observations find **two live attempts**, while the queue reports **reserved=1**. This occurred in both V1 and V2. Their exact source paths/hashes remain unchanged; V2 source identity checks therefore do not prevent this state-validation failure.

The source cause is `verified_result()`: it returns `None` immediately when `result.json` is absent, before considering the existing release file. Tick's release-exists branch accepts this non-result as verification and skips the reservation; status likewise sees no error and treats it as released. The readiness pass revisits the already-launched A without adding it back to the active list, then has apparent capacity for B. A full historical traversal is insufficient if the verifier fails to enforce release/result completeness.

This is **not** a demonstrated normal crash transition. The legitimate publisher writes result before release. The test deliberately crosses the owner-writable store boundary with an unauthorized malformed file. It establishes a fail-closed integrity/capacity defect under that corruption state, not a claim that ordinary caller death spontaneously creates this file. Accidental selective deletion/restore could be a related state, but was not simulated or attributed here.

A separately versioned correction should require any release to have a complete valid result bound to the reservation/binding and, where required, executor terminal/cleanup evidence; missing or malformed evidence must retain/count the reservation in both reconciliation and status. No fix is supplied here, and neither V1 nor V2 was modified or migrated. Further edge cases, including orphan release files before reservation and blocked-never-launched results, need explicit tests rather than inference from this one case.

## Incremental trusted-index capacity counterexample

The source-bound incremental derivative similarly starts A live and B pending. A malicious raw SQLite writer inserts a canonical/digested `DependencyReleaseRecordV1` row for A, without result/release files and without verified cleanup. No trigger was removed: append-only triggers do not stop an unauthorized INSERT of a release. Its active-selection join trusts row existence. B launches, creating two actual live attempts while the operational observation reports one reservation.

This is the explicitly documented **trusted mutable-index** boundary, not a newly discovered violation inside that premise. It demonstrates concretely why faster ownership/frontier SQL cannot substitute for an authenticated release authority. The selected-only verifier never authenticates A's alleged release when dispatching unrelated B. Canonical digests do not authorize the state transition. The original V1/V2 malformed-file failure above is distinct: it is an incomplete verification invariant despite traversing the reservation.

## Unvisited historical bytes and ineffective same-owner “sealing”

Both comparison cases complete A, then mark its native receipt `0444`. The owner writes a replacement file in the writable parent and atomically replaces the read-only pathname, appending one whitespace byte. This is an intentional owner-writable replacement, not an accidental bit flip. File read-only permission alone is not a historical immutability boundary against that owner.

With a new unrelated B pending:

- V2's next tick pins the changed A receipt twice, reports A quarantined, retains capacity and does not launch B.
- The incremental derivative's next tick never pins A's receipt and launches B. Its narrower selected-history trust contract explicitly permits this observation.

The cached hash of the expected A pin is unchanged while current A bytes differ. A cached digest/root authenticates an expected identity, not the current unvisited mutable file. These actual observations instantiate the information limit: if the operation reads only unchanged cached/selected inputs, it cannot distinguish the original store from a store with an unvisited changed file. Preserving V2's immediate arbitrary historical drift detection requires revisiting those bytes or a materially stronger storage/write boundary. Metadata watches, chmod, triggers and expected-hash chains are not established replacements by this packet.

## Cleanup and resource evidence

The five tests passed in **4.986 seconds**; measured wrapper wall time was 4.989 seconds and reaped-child CPU time 5.313 seconds. These are bounded local observations, not physical I/O or a throughput benchmark. Nine tiny owner commands ran across the five temporary domains. At most two intentionally overlapping fixture workers existed in the three capacity counterexamples; they slept at a short polling gate with a five-second maximum, not scientific computation.

After each assertion the fixture release gate was opened. Every recorded child terminal reported return code zero, no timeout and verified cleanup. Original process-identity/session/job-lock checks then found **zero live owned attempts**, all supervisors were reaped, zero fallback signals were required, and every temporary domain was removed. Terminal pins and observed exit/cleanup fields are retained in `negative-results.json`. For intentionally corrupted historical receipt cases, this is process cleanup evidence; it is not a claim that the corrupted terminal's native evidence chain now verifies.

End-state bytes before removal range from about 13 KB to 73 KB per domain. Peak transient filesystem bytes, hardware I/O and machine-wide CPU contention were not measured. Everything ran under `nice -n 15`. No owner jobs, production stores, scientific plans, real model results or public registry were read/changed/adopted/signaled.

## Reproducer and remaining work

```sh
nice -n 15 python3 -B test_counterexamples.py > negative-results.json 2> tests.log
```

Each test starts a separate process to avoid cross-version Python import ambiguity. `probe.py CASE` independently runs a selected case. The only instrumentation wraps the actual pin function to count historical receipt reads and returns actual results; no executor outcome, release, child or cleanup result is mocked. Forged local metadata is explicitly the adversarial input.

The sources and evidence are frozen and pinned for independent review. A protected append-only/immutable storage boundary with separately authorized writers, or an explicitly narrower durable audit-timing contract, remains necessary before incremental historical verification can replace V2 behavior. Transactional runnable/active frontiers may reduce SQL scans, but must be tied to that release authority and recovery model. Neither the earlier incremental derivative nor this counterexample supplies a protected head, service, production migration, exhaustive corruption/power-loss proof or thousands of scientific cycles.
