# Frozen V2 scaling assessment and source-bound incremental research

This packet measures the frozen real queue V2 and tests a separately versioned derivative of the existing incremental dependency prototype. **Nothing is integrated into queue-v2 or any production store.** Source pins, measured operation counts and bounded crash/concurrency evidence are preserved. No thousands of model cycles, scientific inference, performance improvement or unattended service is claimed.

## Real V2: measured bottleneck

`measure.py` runs under `nice -n 15`, with 64 pending single-job admissions and eight real tiny commands in another capacity-one store. All stores are temporary and removed. Instrumentation wraps the actual pin/read/history/directory functions and returns their actual results. Timings include instrumentation, fixture plan/spec construction and filesystem/cache/load effects; they are one bounded observation, not comparative throughput evidence.

| Operation | Observed work |
|---|---|
| Admission with no history | 2 directory listings; 1 history scan; 15 pin calls; 19.82 ms |
| Admission with 63 prior plans | 2 listings visiting 126 directory entries; 1 history scan; 63 authenticated plan reads; 78 pin calls; 34.67 ms |
| First / last eight admission median | 20.76 / 33.73 ms |
| Status of 64 pending plans | 1 history scan; 64 authenticated plan reads; 72 pin calls; 25.90 ms |
| Idle tick after 1 completed command | 2 history scans; 4 authenticated document reads; 25 pin calls; 3.42 ms |
| Idle tick after 8 completed commands | 2 history scans; 32 authenticated document reads; 144 pin calls; 21.27 ms |

Strict V2 source identity adds **fixed** work: eight source pin calls and 234,336 source bytes authenticated per measured admission/status/tick. It is not the history-growth driver. Plan construction plus admission also repeatedly authenticates the Python executable: pending-admission pin instrumentation observed about 1.044 GB of **logical bytes hashed** across 64 calls, predominantly this fixed executable cost. Checked historical-plan bytes total about 2.095 MB. These are software read counts, not physical disk I/O: the executable is repeatedly cacheable. This experiment does not measure physical disk contention or a cold-cache benchmark.

The whole bounded probe took 5.04 wall seconds and 3.97 controller CPU seconds, with 1.60 reported reaped-child CPU seconds. Controller max RSS was about 30 MB. At the end, temporary stores contained 185,626 bytes across 318 files; all were removed. That is end-state disk occupancy, not peak transient publication occupancy or cumulative writes. At most one fixture worker ran at a time. The separate nine-test incremental suite took 10.07 seconds; its process-wide CPU/peak disk use was not instrumented. It ran nine tiny owner commands plus fault/concurrency clients, all in temporary domains. These resource observations do not assert zero contention with other jobs; low priority and strict bounds limited the work.

## Asymptotic cause survives V2

Let H be historical admitted jobs/plan bytes, P partial-plan jobs, J incoming-plan jobs and E dependency edges. V2 correctly scans partial admission directories and validates their plans to protect interrupted ownership; that adds another directory traversal and partial-plan validation. It then materializes all sealed entries, authenticates every historical plan and linearly searches ownership references for each incoming job. Ordinary one-job serial admissions remain O(H), accumulating roughly O(N²) history work across N cycles; partial scans add their own P-dependent work. Strict source hashing is fixed in history size and must remain authenticated.

Ordinary tick still authenticates all admissions, reconciles every historical reservation/release/result/binding/output, traverses readiness, then calls status and repeats history verification. Completed history therefore increases idle polling cost even when no historical state may legitimately change. Dependency and role lookup uses repeated linear searches and repeated whole-plan reads; large plans/DAGs can add worse constants and repeated J/E work. The original 160-admission/16-command probe identified this shape; the smaller V2 probe reproduces the operation-count growth without extrapolating timing to thousands of real runs.

## Incremental ideas that remain valid

A **new opt-in store** can commit canonical admission evidence and unique ownership rows together in one SQLite transaction. This avoids a separately published-plan-before-receipt ownership window. A crash before COMMIT rolls back both evidence and claims; a crash after COMMIT is an exact retry with ownership retained. This is a distinct atomic admission contract: uncommitted caller intent is not promised to survive, so callers must retain and retry their exact supplied plan. It is not a migration or a claim that V2's visible partial-plan files can be discarded. Imported V2 validators still authenticate the complete new plan and current pins.

Durable reservation and committed release gates, exact immutable dependency bindings and selected producer result/release/terminal/output pin chains survive. Active jobs and every selected dependency ancestor must still be checked synchronously through the original executor, including cleanup and unknown-return retention. Source identity must cover actual loaded modules and exact current paths/hashes, rather than a cached version label or adjacent-file assumption.

An incremental index **cannot preserve V2's immediate arbitrary historical-mutable-file tamper gate** without visiting those bytes or changing the storage boundary. Hash chains alone do not authenticate unvisited current file contents. A protected immutable archive or an explicitly narrower operational/audit-timing contract is required. No full-history promise is silently transferred to this research store.

## Tested source-bound derivative

`dispatch.py` derives from the previously independently reviewed `queue-dependency-research-sol-v1/dispatch.py`. Changes are limited to actual V2 validator/executor imports, exact source roster/module-path validation before initialization/dispatch mutation, a distinct `SourceBoundIncrementalDependencyResearchV1` config, and an admission fault-injection seam. It uses the unchanged V2 admission-index core and real V2 executor. Old independent/dependency/V2 queue receipts are not adopted. Current sources include this dispatcher, the exact admission core and all actual V2 queue/executor/ledger/receipt modules; changed valid-but-different source pins and mixed imports are rejected before creating a root or lock.

The existing dependency layer's capacity-one reservation/binding/result/release logic remains. Consumers require a committed producer release, exact result/release/terminal/output pins, current accepted outcome, verified cleanup and absent owned processes/session/lock. Their effective executor specs carry those pins and substitute only admitted SHA arguments. Producer nonzero or unknown exits do not produce readiness. Selected chains are re-derived after preparation immediately before launch. All remaining trusted-mutable-storage, non-daemonizing-process and source/receipt-transitive-validation limitations of that reviewed prototype remain.

`python3 -B test_source_bound.py` explicitly runs nine selected tests: four new source/admission/concurrency tests and five existing dependency/fault tests. All passed in 10.067 seconds. Evidence covers real SQLite process death after evidence/claim insertions and after commit; exact postcommit retry and competing claim rejection; two competing admissions with one retained owner/one actual launch; changed source roster and four mixed-module imports; refusal of an actual V2 queue domain; real consumer preparation/launch recheck; all eight binding/reservation/executor/result/release caller-crash boundaries; nonzero/unknown producer retention; and selected output/result/release/terminal/database tampering. No fake successful outcomes or model outputs are used. The copied full prior test module is retained for context, but its unselected tests were not rerun and are not counted.

The final-state source-bound tests also clarify partial recovery: an uncommitted admission transaction leaves neither evidence nor claim; an acknowledged/postcommit admission retains both. Published binding/result/release scaffolds are governed by the existing exact-equality recovery rules; unknown bootstrap/scaffold files still require investigation. No cancellation/new-plan prelaunch resumption or automatic repair was added.

## Remaining SQL and graph bottlenecks: not solved

The derivative does **not** implement a durable runnable/active frontier. `query_plans.py` ran EXPLAIN QUERY PLAN in one fresh empty store (zero executed jobs). SQLite reports `SCAN r` for active-reservation selection and `SCAN o` for pending owners, with indexed joins into release/reservation identities. Thus even though old external plan/receipt bytes need not all be rehashed, selection still scans historical database rows. Empty-store optimizer output is not a measured large-table latency benchmark.

Recursive producer-chain authentication can revisit shared ancestors per path, potentially multiplying work in branching DAGs, while V2 validation can re-read all sibling jobs. The prior independent diamond established small-graph correctness, not scalability. Transactional active/pending/frontier tables, dependency-count updates, bounded/cached verification only under a valid immutable-storage/audit contract, and crash-safe frontier replay remain needed. No constant-time or thousands-cycle scheduling claim is justified.

## Explicit trust and next decision

The derivative's observation says `trusted_mutable_index_no_historical_audit`. SQLite/config and cooperating local writers remain trusted. Triggers and local digests are logical immutability checks, not protection against a filesystem owner dropping triggers, coherently rewriting/rolling back state or racing external files. Current active/selected chains are authenticated; arbitrary unreferenced released history is not checked every tick. The V2 public default/full-history implementation remains separate and unchanged.

Neither prototype supplies a protected archive head, full dispatch audit, interrupted bootstrap repair, cross-plan scientific result authority, portable restore, cancellation, new-plan prelaunch lineage, unattended controller or a scientific promotion service. Successful-predecessor repeats remain explicit new attempt/output plus pinned predecessor/reason; no best-performance or scientific-necessity gate is invented. Scientific qualification/promotion/publication are separate.

All source and test files in this packet are frozen for independent review. Further implementation or adoption must resolve these gaps under its own explicit trust contract, not replace V2 based on this bounded probe.
