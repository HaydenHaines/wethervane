# Fresh fixed-plan batch admission V1

`fresh_batch.py` is a separate optional admission tool. Existing queue core files,
store sources, per-plan receipt schemas, dispatch, capacity, dependency binding,
and reconciliation are unchanged. It never calls `tick`, `work`, or a model.

Use only for a bounded list of already known independent plans (at most 1,000
plans). DAG dependencies remain local to each existing V1 plan. This can avoid
repeated history scans during admission; it does not demonstrate thousands of
adaptive sequential scientific cycles, scheduling throughput, or qualification.
Read-only status and dispatch still scan history as before.

The caller supplies an independently trusted SHA256 of the manifest's exact
file bytes. Do not substitute a digest freshly computed from an untrusted file.
The manifest is `FreshQueueBatchPlanV1` with exactly `schema`, `batch_id`,
`store` (existing store.json pin), `sources` (the exact ordered list returned by
`fresh_batch.sources()`), and `plans` (ordered ordinary plan-file pins). Every
pin uses existing `execution.pin` fields: absolute `path`, `sha256`, and `size`.
Approve the actual tool and core sources before trusting this manifest. Existing
stores do not automatically trust the new tool through their core source list.

Invocation: `python3 -B fresh_batch.py --root ABS_STORE --manifest ABS_MANIFEST
--sha256 TRUSTED_MANIFEST_BYTE_SHA256`.

The utility takes the existing `dispatcher.lock`, authenticates existing admitted
history once, and builds attempt/output ownership sets. It validates the complete
new roster before writing admission evidence and rejects conflicts with history
or other batch plans. Existing queue paths, partial/quarantined admission records,
prelaunch resumption, explicit retries, preexisting executor attempts and owner
outputs are rejected. Files and directory ancestors used for admission must not
be symlinks. Candidate plans retain all existing queue/executor validation.
Pinned manifest, sources and current plan plus filesystem freshness are checked
again before each plan is published. The unchanged dispatcher rechecks execution
inputs at launch as usual.

Durable evidence lives in `STORE/batch-admissions/BATCH_ID/`: `intent.json`
authenticates the complete fixed manifest and records its content before any
plan admission; `receipt.json` is published only after every per-plan admission
and job directory is present. Both use the existing fsynced immutable publication
primitive. Qualification remains `unassessed`. Batch evidence does not modify
the meaning or authority of individual `LocalExperimentQueueAdmissionV1` receipts.

Admission is a committed prefix, not an atomic transaction. After any crash, keep
the batch intent and every existing byte. Use **ordinary `experiment_queue.enqueue`
for each original plan with its original canonical plan digest**. Existing
complete admissions remain idempotent, and ordinary partial-admission recovery
applies. The fresh batch utility deliberately refuses reruns against existing
batch or queue paths. An interrupted batch never gains a synthetic success receipt;
the ordinary receipts and retained incomplete intent describe what happened.
No rollback, deletion, automatic launch or resume is performed. A crash after a
per-plan receipt but before job directories follows the unchanged queue core's
recovery behavior. Unknown files require investigation, not cleanup by this tool.

The lock protects cooperating clients only. This is not isolation against an
adversarial process changing files outside that lock. Checks detect encountered
drift but cannot make filesystem observation and use atomic against such a writer.

Tests use isolated temporary stores, admission-only fixtures, bounded counts,
and explicit crash/tamper injection. Run `PYTHONDONTWRITEBYTECODE=1 python3 -m
unittest discover -s research/improvement-loop-20261024/queue-fresh-batch-sol-v1
-p test_fresh_batch.py -v` from the lab root.
