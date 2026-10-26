# Source-bound symbolic-only count runner — actual TX unexecuted

This separate packet implements the reviewed measurement route but **has not executed an actual TX graph**. Only invented-object subprocesses and source/AST tests ran. No actual graph build, scientific module import, numerical compilation, sampler, forecast, old partial-pickle access or owner change occurred. Existing attempts and counting V1–V4 remain untouched.

Actual mode requires a source-bound independent technical review and a fresh resource record. These are technical scope/resource checks under the standing authorized research task, **not a new user-permission gate**. No passing actual review/resource record is supplied. No automatic retry, resumption or actual execution is enabled by a marker or by this README.

## Scientific pipeline and count boundary

`pipeline.py` is derived from the exact prior `experiment.py` source pinned in `plan.json`. The original parent, source, role, complete native inventory and scientific-target guards remain. The source test compares every scientific `need(...)` call, and compares the exact seven-root bundle AST to the old serializer argument. The old output-root setup moved to the controller; report writes became bounded durable metadata writes. Graph-object file persistence alone is replaced by `count_bundle`.

The preserved route is complete `model.logp(jacobian=True)` → strict exact exposure-node replacement → pinned `rewrite_pregrad` → pinned native `gradient(rewritten, model.continuous_value_vars)` with its ordered flatten/concatenate semantics. All transformed-name/shape/dtype/segment and 662-coordinate checks, explicit E input/old-port/stale-whole-array checks, (3,24) positive finite whole-parent conditioning and unchanged factor/parameter/token ownership checks remain. The first parent keeps its original identity and mass 0.125 outside `population_cut_v1` conditional logp; it is not relabeled as another parent or a normalized all-parent result.

The one bundle retains `original_cost`, `replaced_cost`, `rewritten_cost`, `gradient`, `ordered_value_vars`, `E`, `segments`, in that order. One protocol-5 Pickler/memo sends bytes into the count/hash/discard sink; there is no out-of-band callback, independent-root stream, compression, pickle output or deserializer. The original reference cost legitimately retains its exposure constant; modified-expression guards apply as originally written.

The source-level `p.graph.compile_model` call constructs the symbolic model, as in the reviewed original route. It is not a new numerical logp compiler. This packet calls no numerical `pytensor.function`, logp evaluation or sampler. AST parity does not prove runtime semantic/numerical correctness: independent review of the generated source is required, and actual mode remains untested.

## Technical gate and actual bounds

`controller.actual(root, plan_pin, technical_pin)` checks exact runner sources, fixed resource envelope, plan identity and a separately pinned review with verdict `source_semantics_pass_for_symbolic_count_only`. The review must bind the exact plan SHA and runner source list. The technical record permits only symbolic graph plus one protocol-5 stream count and explicitly forbids numerical compile/sampling/forecast. It also pins a resource observation scoped to this packet, at most 120 seconds old and not future-dated, reporting at least 8 GiB available memory, 1 GiB disk, one permitted CPU and nice15. The controller must run from the pinned target environment. All twelve inherited source pins are rechecked; the child rechecks technical/source/resource evidence before importing the scientific pipeline.

No such passing records are created by tests. `source_review_record` and `fresh_resource_record` remain null in the prepared plan. Caller-selected hashes identify bytes within this trusted local technical workflow; they do not independently prove scientific authority.

Actual child bounds: CPU soft/hard 90/95 seconds, address-space hard limit 6 GiB, individual-file hard limit 1 MiB, core dumps disabled, nice15 and five thread controls fixed at one. A 120-second wall watchdog, 2-GiB observed-RSS watchdog and 16-MiB output/cache watchdog are sampled every 0.25 seconds. CPU/AS/file bounds are rlimits; wall/RSS/aggregate output are sampled watchdogs and can overshoot. Native/runtime cache writes are not synchronously intercepted. Bounded metadata writes enforce 16-MiB aggregate headroom before each temporary write, including existing temporary files; the child file rlimit also applies to its inherited log descriptor.

The logical stream budget is a positive integer at most 512 MiB, checked before accepting each write. SHA updates use at most 64-KiB views. Pickler may allocate a write buffer before the sink sees it; counting therefore does not make memory feasibility automatic. A cap produces prefix/attempt information and nonzero failure, not a complete-size or restoration claim. Wall time includes graph construction, counting and per-write progress I/O. Each progress publication has file fsync, rename and directory fsync; actual-graph write count and fsync overhead remain unknown.

## Controller, terminal and failure behavior

There is exactly one direct worker child. Source/intent and locally observed Popen PID/command are recorded in a durable launch receipt. Catchable signals are deferred around handle assignment; the worker immediately restores its normal signal mask and applies hard limits before scientific imports. The controller samples its own child and uses its retained pidfd for bounded TERM/KILL and reaping. Handler/pidfd restoration is in a cleanup finally block. A normal exit is not fabricated from PID disappearance.

The controller writes an observed terminal with actual child return code, reaping, reason, sampled RSS/output/thread peaks, wall duration and intent identity. Worker terminal/failure receipts are best effort under output failure; absence or uncertainty never yields success. Serialization/logical cap/worker errors preserve nonzero outcomes and failure metadata. A controller signal/watchdog reaps its direct toy child in tests. Uncatchable controller death, power loss, uninterruptible I/O and daemonized/escaped descendants are not solved; no generic process-tree containment or guaranteed terminal under those failures is claimed. If teardown itself cannot reap, completion remains failed/unknown.

Freshness is rechecked under the cooperating exclusive attempt lock. Only the controller that created the root may write failure metadata there, preventing a losing concurrent creator from modifying a winner. Existing attempts are not overwritten or resumed. A deterministic two-process test forces both creators past the initial absent-root check: one obtains acknowledgment, the other rejects without adding a failure record to the winner.

## Completion boundary and V3/V4 limits

No marker-file adoption or acknowledgment loader exists. `Controller` issues a private, original-object/PID-bound **ephemeral** `AcknowledgedSnapshot` only after actual zero child exit/reaping, successful terminal publication/fsync, exact metadata rosters, bounded capture and worker/source/intent/result checks. A copied token, dictionary/recomputed hash or new controller cannot reconstruct it. Hostile in-process Python can subvert private state and is outside this contract; this is not a signature or durable authorization service.

If terminal rename succeeds but directory fsync fails, the terminal file may remain visible and a controller-failure record may be written, but no acknowledgment is issued. Tests deliberately reproduce this uncertain publication. Do not recompute a hash after failed/uncertain return and treat the file as acknowledged success. Process restart loses the ephemeral issuance registry; this runner supplies no durable restart/adoption path.

The exclusive cooperating lock covers execution and final capture. The worker is the delegated writer in that interval. Returned file bytes and nested observation metadata are immutable; consumers use those bytes, never reopen live paths and call that use verified. No public live-directory verifier is supplied. Capture opens with NOFOLLOW/NONBLOCK and checks regular file identity/size/times; a tiny test shows FIFO replacement rejects rather than blocking at open (the V4 review's new liveness counterexample). Snapshot byte budget is checked before each capture. Caches are excluded from offered snapshot bytes and are not advertised as verified scientific output.

A noncooperating filesystem owner can ignore locks or coherently rewrite metadata. Capture may reflect different instants; this local protocol is not malicious-writer isolation, immutable storage, a trusted timestamp, power-loss proof or a complete runtime supply-chain attestation. The original V3 late-verify and recomputed-success states are not silently adopted as actual scientific completion. Immutable returned bytes solve verification/use reopening within this narrow API; they do not make mutable pathnames immutable.

## Tested scope and preserved failures

Run only the bounded toy/source suite for this review:

```sh
nice -n 15 python3 -B test_runner.py
```

Tests cover exact scientific-guard/seven-root AST parity; actual-mode refusal without technical evidence and no created attempt; tiny normal count with hard-limit/priority receipts; cap and raised-error nonzero failures; uncertain terminal fsync with no token; immutable snapshot/copy/new-controller rejection; FIFO rejection; real controller SIGTERM and wall-watchdog direct-child reaping; and deterministic competing fresh-root creators. No actual graph object is provided to the count core. Toy payloads are at most 64 KiB; CPU 2/3 seconds, AS 256 MiB, RSS watchdog 128 MiB, wall watchdog four seconds and one thread bound them. The wall case sleeps, rather than running scientific computation. All temporary test roots are removed after receipt assertions and child cleanup.

The initial seven-test run had one test failure: its forbidden-pickle AST predicate mistakenly included `json.loads/dumps`. That failed test source/log/result are retained as `first-*`; the predicate was corrected to qualify the pickle module. The scientific implementation was not changed to bypass that test. Final exact source/test/result pins and timing are in `evidence.json`.

## Independent review and remaining scientific work

Review the current source and prepared plan, the source-derived scientific statements against the original, the earlier unexecuted size-design packet, and Luna's V3/V4 boundary findings. Do not run actual mode, generate a passing actual technical record or reuse a current resource observation as part of the toy review. Appropriate later execution remains within standing authorization only after independent technical/source review and fresh resource headroom in the shared-job context.

Actual symbolic import compatibility, size, memory, duration, graph Pickler behavior and feasibility are untested. Counting cannot establish numerical logp/gradient parity, all-parent normalization/lineage, restored-graph identity, sampling qualification, empirical improvement or a forecast. Full numerical point/vector persistence, complete conditional tests and any future serializer/loader need separate source-bound experiments. No scientific or durable-restart readiness is claimed here.
