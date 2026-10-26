# Persistent experiment command queue

`queue-v3/experiment_queue.py` executes fixed dependency graphs through its pinned
`execution.py`. It adds persistent dispatch and capacity accounting for future
explicitly selected commands. It does not generate hypotheses, fit scientific
models by itself, assess forecast quality, promote models or publish results.
Tests use invented subprocesses; no existing owner job or completed scientific
comparison is adopted, restarted or repeated.

**V1/V2 integrity warning:** an inserted release record with no result can make
a live reservation appear released. A bounded adversarial test launched two
workers under capacity one. Do not rely on V1 or V2 capacity accounting after
inconsistent record changes. Existing source-pinned stores have not been migrated.

The original top-level `experiment_queue.py` remains unchanged for stores that
pin it. It has a known interrupted-admission ownership defect and does not have
the protection described below. V2 repairs the partial-admission ownership defect
but has the release-record defect above. V3 is for **new stores**: store receipts
pin exact source paths and hashes, so switching an existing store to these files
is not a supported migration. Keep old sources and receipts available; do not
edit store pins or silently rewrite an existing plan to use the new code.
V3 uses a distinct store schema, compares store pins with its source files and
checks actual imported helper-module paths, and validates the store before creating
a worker output. An older V1 client may still create a missing `queues` directory
in an incomplete incompatible store, or an empty requested worker-output
directory, before rejecting it; do not use V1 or V2 clients on V3 stores.

## Execution domain

`initialize(store, capacity)` creates an immutable local store with capacity one
or two. Public APIs and CLI commands canonicalize the store root, so relative and
absolute paths to the same store share its admission and receipt identities.
All cooperating queue clients use one dispatcher lock and the dedicated
`store/executions` executor substore. **Direct external `execution.launch()` into
that substore is not an admitted path.** Other executor stores and existing owner
jobs are outside this capacity claim. Select capacity with their resource use in
mind. Controllers run outside the worker substore and its capacity accounting.

`enqueue(store, plan, expected_sha256)` records a fixed plan. The expected digest
is SHA256 of `ledger.canonical(plan).encode()`. Repeating the identical admission
returns its original receipt; changing an existing queue ID is rejected. A caller
crash after directory/plan creation is visible as a partial admission, and the
same exact plan can finish that admission. Unknown scaffold files fail closed.

The plan is an object with exactly:

```json
{"schema":"LocalExperimentQueuePlanV1","queue_id":"comparison-v1","jobs":[]}
```

Supply a nonempty `jobs` list. Each job has these fields:

| Field | Contract |
| --- | --- |
| `id` | Unique job ID within this plan |
| `execution_spec` | Exact `LocalExecutionSpecV1`, including source/plan pins, minimal environment, attempt identity, owner output path, bounds and native metadata gates |
| `dependencies` | Explicit predecessor receipt roles and consumer argument bindings |
| `outputs` | Explicit receipt roles available to dependents |
| `resume_prelaunch` | `null` normally; narrowly pinned resumption proof described below |

An output declaration is `{role, path, minimum_bytes}`. `path` must be an exact
JSON receipt path already declared in that job's `execution_spec.native_receipts`.
The executor authenticates its bytes and selected metadata. Arbitrary extra output
files cannot become accepted dependency roles. The owner command remains
responsible for validating raw prediction banks and other transitive scientific
references inside its receipt.

Each dependency is:

```json
{"job_id":"producer","role":"native-receipt","path_arguments":[4],"sha256_arguments":[5]}
```

Every path argument must already equal the producer's exact absolute selected
receipt path. Each SHA argument must be the literal `<dependency-sha256>` in the
frozen base spec. Both lists must be nonempty, indices must be unique, and every
sentinel must have one declared binding. Dispatch replaces only those digest
arguments with the authenticated producer receipt SHA. There is no general
templating, interpolation, evaluation, shell construction or environment/path
substitution. Duplicate attempts/output paths, unknown roles and dependency
cycles are rejected.

## Dispatch and dependency admission

`tick(store)` reconciles all admitted plans under the store-wide dispatcher lock,
then launches ready work up to capacity. It records these immutable steps:

1. A binding containing the exact effective execution spec, producer result and
   executor-terminal pins, selected receipt pins and explicit SHA arguments.
2. A durable capacity reservation before executor preparation or launch.
3. A queue dispatch intent and the executor's existing gated launch lifecycle.
4. A job result and a separately bound reservation release after verified cleanup.

The reservation outlives its caller and worker. Competing clients serialize and
count every unreleased reservation, even if preparation or terminal inspection
is corrupt. A process's elapsed time, PID disappearance or caller death does not
free capacity. A launched reservation releases only after executor inspection
finds a terminal, verified cleanup, no held job lock, no live owned session and no
live recorded process. A result-before-release crash is incomplete; the next tick
re-derives its outcome from the executor before accepting that cached result.

Successful dependency admission requires return code zero, no timeout, verified
native receipt metadata, declared output gates and released capacity. The result,
release, reservation, effective spec and executor terminal must bind to each
other. Downstream specs include the exact producer result, terminal and selected
receipt as input pins. Changed bytes are quarantined/deferred; they are never
silently rebound to a producer retry or a file that merely exists.

The following remain separate:

- Observed process return, including an explicitly unknown exit.
- Native receipt metadata and declared workflow output gates.
- Scientific qualification, always `unassessed` in this queue.

Nonzero exit, timeout, invalid/missing native receipts and declared-output failure
have distinct failure outcomes. A recovered unknown exit is deferred, and its
dependents are blocked. An orphan that is still live keeps its reservation;
use the executor's explicit `stop`/`recover` operations for that owned attempt.
The queue never invokes those operations automatically or synthesizes an exit.

Integrity failures retain/count reservations and expose `quarantined_integrity`.
Malformed admissions quarantine store dispatch because their capacity history
cannot safely be trusted. A partial admission containing a reservation also
freezes dispatch; an empty pre-admission scaffold consumes no capacity.
Other per-job source/input failures do not prevent
unrelated eligible work from using the remaining safe capacity. Inspection checks
compact receipt links; owner scientific verification remains independent.

V3 rejects a release record whose result is missing and retains any existing
reservation as quarantined. A release without a reservation is quarantined before
launch, even if no result is present. These checks address the tested inconsistent
records; they do not make the store safe against arbitrary outside modification
or restore evidence that has been completely removed.

An admission can be interrupted after its plan is durable but before its receipt
is published. That partial plan still owns every declared attempt ID and output
directory. While holding the dispatcher lock, `enqueue` validates other partial
plans and rejects collisions; unrelated plans can be admitted. Retrying the exact
original plan seals its receipt. If death occurs after receipt publication but
before job directories are created, an exact retry retains the receipt and a
subsequent tick creates those directories. A scaffold without a readable plan,
a malformed plan, or an unknown queue-store entry instead blocks new admissions:
its claims cannot be safely inferred. This is a fail-closed availability limit,
not automatic repair.

For a malformed partial admission, first stop dispatch clients and controller
workers without killing or modifying any separately owned scientific process.
Record `status`, the queue path, process identities, reservations and attempted
inputs; make a consistent preservation copy of the store and hash the damaged
bytes. Look for the exact original canonical plan and its independently recorded
digest in the admission request or a verified backup. Only if those establish
the full original identity, and the attempt/reservation history is reconciled,
may an operator preserve the damaged entry outside the queue directory, restore
the exact verified plan at its original path, and retry that same plan with its
original digest. Keep both the damaged bytes and a dated recovery record. If the
original claims or process state cannot be established, leave admission blocked
and investigate; do not delete the partial entry, invent a plan, reassign its
output, or silently reset the store. There is currently no automated quarantine
or repair command.

## Prelaunch deferral and explicit retries

If pinned bytes drift before any executor launch intent, dispatch records a
queue-level `deferred_prelaunch_integrity` and releases capacity as
`never_launched`, under the sole dispatcher lock. It does not invent an executor
terminal or call `recover` on an unlaunched attempt. A registered/prepared executor
claim may remain preserved.

To resume a completely bound, never-launched attempt after restoring its original
pinned inputs, admit a **new immutable queue plan/job** with this proof:

```json
{
  "queue_id":"old-plan",
  "job_id":"old-job",
  "result":{"path":"...","sha256":"...","size":0},
  "release":{"path":"...","sha256":"...","size":0},
  "binding":{"path":"...","sha256":"...","size":0},
  "reason":"Original pinned input bytes restored; explicit resumption"
}
```

Use actual `execution.pin()` values, not the illustrative sizes above. Set this
object as `resume_prelaunch`, copy the old binding's **entire effective execution
spec** without changing its canonical bytes, and preserve its native roles.
The new job has an empty dependency list because the earlier exact dependency
pins are already materialized in that spec. Its queue binding explicitly preserves
the resumption evidence/reason. It reuses the same executor attempt and fresh
output path; it never overwrites an output directory.

Admission verifies every earlier queue reference to that attempt/output is a
sealed, released prelaunch deferral; the selected result/release/binding pins
match; no reference is active or quarantined; and no partial/malformed admission
is ignored. Existing executor preparation must be registered, byte-identical and
observed unused, with no launch intent, processes, session or held job lock.
Competing resumption plans are rejected under the dispatcher lock. Changing the
spec, adopting a preexisting external attempt, or resuming an attempted scientific
run through this route is prohibited. Unregistered partial executor preparation
fails closed and remains an explicit investigation case.

A job that actually launched requires a new immutable plan/job/attempt and fresh
output path to retry. Its execution spec must contain the exact predecessor
terminal/recovery pin and a nonempty retry reason. Existing native scientific
retry authorization remains required. Earlier downstream bindings stay attached
to their original producer; a new producer version requires a new explicit plan.

## Worker and CLI

`status(store)` reports job outcomes, active reservations, incomplete admissions
and integrity quarantine. `all_jobs_recorded` requires valid recorded outcomes
and **zero** retained capacity reservations; it does not mean every job succeeded.

`work(store, output, poll_seconds, max_runtime)` repeatedly ticks until all jobs
are recorded or its explicit deadline arrives, then writes
`LocalExperimentQueueWorkerReceiptV1`. Its `finished` status means workflow records
are complete, including recorded failures. A deadline with pending work leaves
the queue resumable. Neither status establishes scientific qualification.

```sh
python3 -B research/improvement-loop-20261024/queue-v3/experiment_queue.py --root STORE initialize --capacity 1
python3 -B research/improvement-loop-20261024/queue-v3/experiment_queue.py --root STORE enqueue --plan PLAN.json --sha256 CANONICAL_SHA
python3 -B research/improvement-loop-20261024/queue-v3/experiment_queue.py --root STORE tick
python3 -B research/improvement-loop-20261024/queue-v3/experiment_queue.py --root STORE status
python3 -B research/improvement-loop-20261024/queue-v3/experiment_queue.py --root STORE work --output FRESH_CONTROLLER_OUTPUT --poll-seconds 1 --max-runtime 3600
```

For continued dispatch after the invoking caller exits, launch this exact pinned
worker command through `execution.py` in a **separate controller executor store**.
Declare its sources, queue-store receipt and worker native receipt in that
controller spec. This reuses detached supervision instead of inventing a second
process manager. Controller loss does not release worker reservations; another
explicit controller or `tick` client resumes them. This is a bounded local worker,
not an installed unattended service or monitoring system.

Source/config/plan records and owner output files remain locally pinned.
Portable queue-store archive inspection and executable restore/rebinding are
separate work; copying lock files does not create a valid resumed execution domain.
The current House queue's PID adoption and existence-only dependencies are not
converted or changed by this implementation.

## Verification

```sh
python3 -B -m unittest discover -s research/improvement-loop-20261024/queue-v3 -p test_experiment_queue.py -v
```

Invented owner commands exercise a real dependent receipt/SHA chain, independent
jobs, concurrent killed clients, separate detached controller execution, reservation
and admission crash boundaries, result/release integrity, output drift, orphan
recovery, process/native failures, explicit producer retry and exact prelaunch
resumption. No scientific fitting, registry update or publication occurs.
