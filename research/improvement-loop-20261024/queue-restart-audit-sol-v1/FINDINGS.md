# Existing queue restart/concurrency audit and partial-ownership repair

The unchanged existing `experiment_queue.py` can run and resume a small bounded sequence of real synthetic subprocesses, but it has a reproducible interrupted-admission ownership defect. This packet preserves that failure and supplies a narrow **unadopted isolated source copy** with regression evidence. It does not establish a production adaptive controller, thousands of executed cycles or scientific promotion authority.

## Actual existing implementation: verified behavior and defect

`baseline_serial.py` uses the unchanged lab queue/executor, fresh temporary stores and tiny original fixture commands. Twelve plans are constructed serially after the preceding result is authenticated; each later spec explicitly pins that prior result. Exact repeated admissions return the same receipt. Two real subprocess clients per cycle call the original tick and then exit with code 83. Resumed parent ticks record all twelve jobs, twelve executor launch intents and twelve releases, with zero retained reservations. This exercises 24 competing clients and repeated restart/reconciliation against the real code, not a mocked execution interface. These are invented infrastructure tasks, not scientific adaptive hypotheses or fitted experiments. See `baseline-serial.json` for exact observations and elapsed time.

`probe.py` injects a real caller death with `os._exit(82)` immediately **after** the original publisher durably publishes `queues/original/plan.json`, before publishing its admission receipt. The only interception calls the actual original publish first; plan/execution/cleanup operations are real. A subsequent original enqueue under a different queue ID with the **same attempt/output** is accepted, because `entries()` deliberately ignores partial admissions and admission ownership checks inspect only sealed entries. Retrying the original exact interrupted admission then fails with “attempt/output already belongs to another queue plan.” The contender runs once, but the original partial remains; `all_jobs_recorded=false` persists even with zero reservations.

`original-repro.json` preserves this observed failure. It is a visible incomplete workflow, **not silent success, duplicate process execution or an already-observed conflicting promotion**. The defect prevents the promised exact interrupted admission from completing after ownership is stolen. It cannot be explained away as source drift or an external malicious writer: both clients use normal queue admission APIs after the injected caller crash.

## Narrow isolated repair

`patched/experiment_queue.py` is a copy of the original with one admission-only guard. Before checking completed entries, it inspects other incomplete admission directories. A partial plan must exist, decode, pass the current plan validator and name its own queue directory. Its attempt/output identities are reserved against new admissions even though it has no sealed receipt. Exact current admission repair remains allowed; unrelated plans can still be admitted. Missing or corrupt unresolved scaffold claims fail closed for new admission. Dispatch, current full-history checks, reservation/result/release behavior, source pin semantics and scientific qualification are unchanged.

This closes both attempt-only and output-only claim collisions, including the original same-attempt-and-output reproducer. The patched repro rejects the contender, repairs the original exact plan, executes it exactly once and records a complete workflow with no partials/reservations.

The first conservative repair blocked **every** admission while any partial existed. Its 24-test regression run exposed an existing supported behavior: unrelated admission may be recorded while a missing-receipt reservation freezes dispatch. That overly broad patch and failed log are preserved under `failed-blanket-guard/`; the final guard instead checks partial ownership. No failed evidence is retroactively described as passing.

## Recovery and compatibility checks

Final focused tests cover the original failure versus repaired behavior at the actual caller-death boundary; separate attempt/output collision rejection; unrelated admission followed by exact partial repair; missing/corrupt scaffold refusal without creating a new admission; and a durable receipt-before-job-directory caller crash. In that latter case exact admission replay returns the original receipt, and original tick creates the missing job directory and launches once. An existing empty same-ID scaffold remains repairable under the original rules (covered by the original regression suite).

The focused suite additionally runs twelve serial patched plans with 24 competing exiting processes, repeated exact admission and one launch/release each. Historical whitespace mutation to the oldest plan triggers quarantine and restoring original bytes recovers clean status. The repair therefore does not introduce the research index's narrower cached-history contract: original full-history authentication remains.

All **24 original queue tests** pass against the isolated copy, including missing-admission reservation accounting, dependencies, result-before-release crash, source drift, unknown/live cleanup, retries and prelaunch resumption. Exact final run summaries are in `tests.log` and `regression.log`. The helper executor/ledger/receipt copies under `patched/` are byte-identical to original sources; they exist to make initializer source pin paths available for isolated execution. No production source was edited or adopted, and no existing source-pinned store was migrated. This packet is not authorization to substitute code under an old config: production adoption must separately review source/config binding and any store transition.

## Promotion and stale evidence boundary

The queue never promotes a model. Its results remain `unassessed`; a successful process/native receipt cannot authorize scientific qualification, journal advancement, selection or publication. The journal can record a “promote” recommendation without activating anything.

The separate unchanged `scoped_selection.py` has a transaction/CAS registry path. Two existing real-process tests were independently rerun: concurrent candidates at one scoped predecessor accept exactly one and retain both candidates' evidence; a response-output failure after commit replays the exact committed event even after journal/registry heads advance. Both pass (`promotion-concurrency.log`). These tests use the existing explicitly synthetic native/policy fixtures, not real comparative evidence or an actual promotion. One initial runner used the wrong test-class name and failed before executing tests; that runner error is retained separately.

This supports a bounded software property of scoped selection, not automatic end-to-end coordination between the queue and registry. Scientific necessity, independently approved plans, matched scores, task/scope identity, public-default changes and publication remain separate. Cross-scope selection is intentionally not one global winner lock.

## Reproduction

From this directory, with `nice -n 15`:

```sh
python3 -B probe.py
python3 -B baseline_serial.py
python3 -B -m unittest test_recovery -v
```

The existing original regression suite can be run with its module-level `queue` bound to `test_recovery.patched`; no test/authoritative source file needs editing. The evidence manifest pins all checked source copies, packet scripts, prior findings/checkpoints and logs.

All subprocesses, outputs, intentional source edits and cleanup signals target fresh temporary fixture domains. Stores are removed after assertions; executable reproductions and compact logs/results are retained. No heavy/scientific jobs, owner stores or promotion registries are touched. Existing active jobs are neither adopted nor signaled.

## Remaining gaps

The prior scale finding still applies: original enqueue/tick work grows with historical plan/receipt/output verification, with repeated scans and roughly quadratic accumulated work over serial cycles. This repair adds correct partial ownership, not an incremental scheduler or a throughput solution. Twelve real synthetic cycles and earlier thousand-admission/index/journal probes are not thousands of scientific runs or an unattended service.

The latest dependency research supplies a separate narrower trusted-mutable-index prototype and was independently reviewed; it is not silently substituted here. No protected storage head, tamper-proof local filesystem, machine/power-loss test, daemonized-process containment, comprehensive malicious concurrent writer test, durable source archival service or automated scientific hypothesis/qualification/promotion loop is supplied by this packet. Current source/input pins detect changed bytes only within their declared/transitive owner validation boundaries; they do not prove original historical publication or statistical independence. Full production readiness remains unsupported.
