# Selection commands through the durable queue

`selection_queue.py` compiles two separately admitted scientific workflow phases into the existing `experiment_queue.py` / `execution.py` command contracts. It emits immutable plan data; it does not create a queue, launch a command, inspect native results, install a scheduler, write a selection registry, or decide which model wins. `scoped_selection.py` remains the sole native/journal/selection admission boundary.

Preparation is one job. A successful process can report either `blocked_requires_computation` or `review_required`; both remain an unselected preparation with empirical status `unassessed` and publication false. There is no dependency edge that automatically advances a prepared candidate into application. An independent review and exact source-bound root decision must exist before a separate application recipe can be compiled. The actual TX2026 count request remains pending; no positive scientific selection has been demonstrated here.

## Recipe and admission

The JSON request has exactly these fields:

```json
{
  "schema": "ScopedSelectionQueueRecipeV1",
  "phase": "prepare",
  "queueId": "count-preparation-v1",
  "cycleId": "explicit-research-cycle",
  "jobKey": "native-preparation",
  "attemptId": "native-preparation-attempt-v1",
  "nativeSpec": {"path": "/absolute/native-spec.json", "sha256": "EXACT_SHA", "size": 1},
  "pythonExecutable": "/usr/bin/python3",
  "outputDirectory": "/absolute/fresh-native-output",
  "timeoutSeconds": 180,
  "application": null
}
```

Use real file pins; the size/SHA above are illustrative. For `phase="apply"`, `application` must instead be:

```json
{
  "registryRoot": "/absolute/reviewed-existing-research-registry",
  "preparation": {"path": "/absolute/preparation/receipt.json", "sha256": "EXACT_SHA", "size": 1},
  "decision": {"path": "/absolute/reviewed-decision.json", "sha256": "EXACT_SHA", "size": 1},
  "expectedRevision": -1,
  "expectedPredecessor": null,
  "commandId": "stable-payload-bound-selection-command"
}
```

The application request's `cycleId` must match the decision's original journal cycle. Exact preparation/native-spec/candidate/task/decision/CAS identities are checked when compiling. Pending preparation cannot produce an application plan. These are byte/identity checks, not independent scientific review or proof that journal authority remains current. The unchanged selection CLI freshly reconstructs native readiness, full review/evaluation links, journal checkpoint and source objects, and registry CAS before any mutation. A prepared file cannot supply serialized authority.

From this research directory:

```sh
python3 -B selection_queue.py build --request REQUEST.json --sha256 RAW_REQUEST_SHA --output FRESH_RECIPE_DIRECTORY
```

The output `receipt.json` pins `plan.json` and gives `canonicalPlanSha256`. The input SHA above authenticates raw request bytes. The queue digest is SHA-256 of `ledger.canonical(plan).encode()`, as given by that receipt; it is not necessarily the raw plan-file SHA.

Admit and run through the existing queue commands:

```sh
python3 -B experiment_queue.py --root FRESH_QUEUE_STORE initialize --capacity 1
python3 -B experiment_queue.py --root FRESH_QUEUE_STORE enqueue --plan RECIPE/plan.json --sha256 CANONICAL_PLAN_SHA
python3 -B experiment_queue.py --root FRESH_QUEUE_STORE work --output FRESH_WORKER_OUTPUT --poll-seconds 1 --max-runtime 240
python3 -B experiment_queue.py --root FRESH_QUEUE_STORE status
```

Capacity applies to cooperating dispatchers in that store's dedicated executor substore. Direct executor launches into it are not admitted. Controller supervision, caller-loss recovery, explicit stop/recovery, reservations and process identity are inherited from the existing queue/executor. For durable controller execution, use the already documented separate controller executor domain; this compiler creates no process manager.

Each plan contains one fixed job and explicit argv with the pinned interpreter, canonical working directory and minimal `PYTHONDONTWRITEBYTECODE=1` environment. It pins the request, native spec, preparation/decision where applicable, and all seven local orchestration sources: compiler, scoped selection, native selection, queue, executor, ledger and receipt decoder. Native owner/source/result pins inside the spec and review are authenticated by their existing trusted validators, not reinterpreted by this compiler. Mutable SQLite DB bytes are deliberately not treated as immutable executor input files: the selection transaction checks their exact decision/revision/binding instead.

The executor claims but does not create the owner output directory. The existing selection CLI creates that fresh directory and writes `receipt.json`. Preparation receipt gates preserve empirical unassessed, unselected and publication false; application gates schema, domain and publication false. Application status is deliberately flexible: `selected`, `retained`, or historical `already_applied` is interpreted by the selection contract. Queue `accepted_execution` and worker `finished` are process/workflow observations; qualification remains `unassessed` in every queue/executor receipt.

## Explicit launched retry

After a launched attempt has a verified terminal and cleanup, an explicit retry copies its entire prior command and payload:

```sh
python3 -B selection_queue.py retry \
  --previous-plan ORIGINAL_PLAN.json --previous-plan-sha256 RAW_PLAN_SHA \
  --terminal EXACT_EXECUTOR_TERMINAL.json --terminal-sha256 TERMINAL_SHA \
  --queue-id NEW_QUEUE_ID --attempt-id NEW_ATTEMPT_ID \
  --owner-output FRESH_OWNER_OUTPUT --reason 'Specific reviewed reason' \
  --output FRESH_RETRY_RECIPE_DIRECTORY
```

This verifies the previous terminal/specification/evidence link and copies the exact spec, changing only queue ID, executor attempt, owner output, predecessor terminal and explicit reason. It preserves the original preparation/spec/decision pins, registry/CAS expectation, scientific journal cycle and stable selection command ID. It adds the original plan/terminal as explicit inputs. If the launched attempt originally resumed a prelaunch deferral, its old `resume_prelaunch` proof is cleared: the new attempt follows terminal retry lineage. Same attempt/output reuse, absent reason and cross-bound prior plans are rejected. Enqueue into the **same queue store**; the executor checks live process/lock/session absence and exact latest job lineage before accepting a successor. A supplied terminal does not authorize external job adoption.

A crash after selection commit but before application receipt can therefore produce a failed process receipt while the selection event exists. A new attempt in a fresh directory with the same selection command payload returns `already_applied`; it does not append another event. Earlier partial outputs, process logs and plans remain untouched. A timeout or unknown outcome is not permission to spawn again while ownership remains live. No automatic retry is added.

## Explicit prelaunch resumption

Source drift before launch follows the existing queue's sealed prelaunch-deferral contract. It is distinct from a launched retry: there may be no executor terminal, including when executor preparation already registered an unused attempt. Restore the exact original input bytes, then copy the existing proof into a new plan:

```sh
python3 -B selection_queue.py resume \
  --previous-plan ORIGINAL_PLAN.json --previous-plan-sha256 RAW_PLAN_SHA \
  --binding SEALED_JOB/binding.json --binding-sha256 BINDING_SHA \
  --result SEALED_JOB/result.json --result-sha256 RESULT_SHA \
  --release SEALED_JOB/release.json --release-sha256 RELEASE_SHA \
  --queue-id NEW_QUEUE_ID --reason 'Exact original source bytes restored before launch' \
  --output FRESH_RESUME_RECIPE_DIRECTORY
```

The compiler authenticates the prior plan/binding and sealed never-launched result/release links, preserves the entire execution spec and native output roles, and adds the existing queue's `resume_prelaunch` proof. It reuses the same unused attempt and fresh owner output path. Enqueue into the **same queue store**. Existing queue admission checks every earlier reference under its dispatcher lock, rejects active/competing/quarantined references, and verifies any existing executor claim is registered, identical and unused. Compilation alone does not establish current ownership. Never fabricate a terminal or put a new attempt behind an unlaunched retained claim.

## Focused verification and actual boundary

`python3 -B -m unittest -v test_selection_queue` exercises actual compiler/queue/worker/native-command subprocesses, not mocked subprocess return codes. Only a temporary, explicitly named synthetic interpreter fixture replaces native inspection; it then executes the unchanged `scoped_selection.py` CLI. This fixture is confined to the test file and temporary files. The public compiler has no native-inspector injection option. Native adapter behavior and scientific authorization remain independently reviewed boundaries.

The six cases cover pending preparation with no selection or journal creation, separately authorized synthetic baseline application with empirical status still unassessed, two competing CAS payloads admitted before selection with the stale queued competitor rejected without a second event, source drift deferred before launch, registered executor preparation followed by drift and exact proof-based resumption, subsequent terminal retry shedding the consumed prelaunch proof, process exit after actual temporary-registry COMMIT but before receipt publication followed by a new queue/attempt returning historical acknowledgment exactly once, and cross-bound recipe rejection. All positive results are invented software-mechanics evidence. Tests do not touch the actual scientific journal or launch owner models.

The author packet in `selection-queue-sol-v1` includes a concrete actual **preparation-only** request and compiled plan, bound to the existing TX2026 native spec. Compilation only reads compact pinned files and does not invoke native preparation or inspect native results. Root owns any subsequent queued pending check and before/after proof. There is no actual application recipe, registry initialization, decision mutation, result qualification or public activation in that packet. Do not reuse an earlier completed preparation output; every actual queued attempt must use a fresh directory.
