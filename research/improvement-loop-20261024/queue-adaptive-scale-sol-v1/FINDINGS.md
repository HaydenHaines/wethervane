# Serial queue scale probe

This probe leaves the queue core unchanged. `result.json` contains all per-operation observations and source pins; `probe.py` is the bounded reproducer. `pending-store/` retains 160 synthetic pending admissions and no executor attempts. `incremental-store/` retains 16 completed synthetic subprocess attempts, all accepted, zero reservations. These are infrastructure fixtures, not scientific comparisons or qualifications. No owner worktree was used.

The run used one low-priority Python controller (`nice -n 15`) and at most one tiny fixture command at a time. The incremental loop had a 45-second deadline. Instrumentation wraps `entries` and executor `pin`, so timings include mock call-recording overhead and filesystem/cache/load effects. This is one bounded run, not a benchmark of production or thousands of cycles.

| Observation | Result |
| --- | --- |
| 160 ordinary admissions, total | 5.539 seconds |
| Admission median, existing history 0–19 | 19.78 ms |
| Admission median, existing history 70–89 | 31.26 ms |
| Admission median, existing history 140–159 | 53.46 ms |
| Status of 160 pending plans | 71.49 ms; 1 history scan; 164 pin calls |
| Idle tick after 1 completed command | 4.00 ms; 2 scans; 21 pin calls |
| Idle tick after 8 completed commands | 27.50 ms; 2 scans; 140 pin calls |
| Idle tick after 16 completed commands | 68.72 ms; 2 scans; 276 pin calls |
| Dispatch with 15 historical completions + new job | 147.99 ms; 2 scans; 281 pin calls |

Appending one whitespace byte to the oldest admitted plan caused real `tick` to report admission quarantine and incomplete workflow. Original bytes were restored in `finally`; clean status returned. No new attempt was pending during this tamper probe. This verifies this historical-plan gate only, not comprehensive crash/integrity behavior.

## Actual bottleneck

`enqueue` materializes `list(entries(root))`, authenticating every historical admission/plan, then linearly searches all historical jobs for attempt/output ownership. A one-job new plan therefore costs O(H) in historical jobs/plan bytes; N serial admissions accumulate O(N²) work. For a large new plan with J jobs, ownership matching is O(JH). The fixed fresh-batch utility amortizes admission of already known plans, but cannot authorize plans whose scientific content depends on future outcomes.

`tick` authenticates all history, traverses every job to reconcile reservations, verifies every released historical job, traverses all jobs again for readiness, then calls `status`, which scans and verifies history again. The instrumentation observes two `entries` scans per ordinary tick; terminal receipt/binding/output pin checks dominate the growing completed-history workload. Polls repeat this work even when no historical state can legitimately change. Thus admission batching alone cannot repair serial adaptive throughput. Constant-frequency ticks over N serial cycles also accumulate roughly O(N²) historical work, with a larger constant for completed jobs. Dependency resolution within a plan uses repeated linear `next(...)` lookups and can introduce additional large-DAG cost; the probe does not quantify that.

## Safe incremental design to implement separately

1. Preserve the current strict store and receipt meanings. Build a versioned opt-in dispatcher/store format, with migration implemented as a read-only full verification followed by new pinned migration evidence. Do not quietly add an index trusted by old stores.
2. Under the existing sole dispatcher lock, use a transactional durable index for attempt/output ownership, admitted plan identity, runnable dependency counts, active reservations, and unresolved/incomplete admissions. A unique constraint alone is insufficient for resumed prelaunch attempts: preserve the existing verified resumption lineage and exclusive active claim rules. Keep exact immutable plan, binding, reservation, result and release records authoritative.
3. Journal each intended state transition before publication; fsync immutable evidence; update the index and committed journal position transactionally. Never release capacity solely because an index row says complete. A result-before-release or reservation-before-executor crash stays active until the unchanged executor cleanup proof is re-derived. Recover by replaying the uncommitted suffix, not by deleting evidence or synthesizing outcomes. Unknown scaffolds remain fail-closed.
4. Per ordinary incremental tick, inspect active reservations and changed dependency frontier only. Enqueue checks indexed ownership plus current plan/source pins, and durably registers the new plan before dispatch. Future scientific plans are supplied and independently approved after prior results; the scheduler does not invent them. Cross-plan inputs must be pinned to earlier authenticated output/result/terminal evidence with explicit authorization, rather than abusing within-plan dependency IDs or predeclaring unknown jobs.
5. Keep an explicit full audit operation and a separately named compact operational status. The existing full status and historical integrity claim cannot simply become a cached answer. Test unknown admissions, duplicate attempts/outputs, release corruption, tampered dependency outputs, killed clients at every journal/publication boundary, competing admissions and prelaunch resumption before any production adoption.

There is an important integrity constraint: on a mutable local filesystem, an incremental index cannot detect arbitrary edits to an unvisited historical file at the next tick. A hash chain or Merkle root alone does not prove unchanged unvisited bytes. Retaining the current immediate historical-tamper detection requires a full scan, or a materially stronger storage boundary that prevents writes to sealed historical bytes and authenticates new writes. Any amortized archival audit design must explicitly narrow the detection timing contract and remain opt-in; it cannot be sold as preserving the present gate. Active reservation and launch dependency bytes must always be checked synchronously. This trust/storage decision precedes a production implementation.

No claim is made about full crash-at-scale durability, external modification isolation, scientific throughput, automatic hypothesis generation, model quality, or a completed thousand-cycle adaptive loop. The findings identify the infrastructure gap and a reviewable next design, not a delivered durable adaptive controller.
