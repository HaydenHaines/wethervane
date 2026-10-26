# Actual-TX symbolic stream-size measurement: unexecuted design

This is a **source-bound design**, not an actual-TX experiment or a graph artifact. No TX graph was built, no scientific module imported, no numerical compiler/sampler run, and no pickle loaded. The old incomplete pickle was not opened, hashed, modified, retried or adopted. Only standard-library source/AST/metadata inspection and tiny invented-object tests ran.

The standing research task already authorizes appropriate bounded work. The future execution boundary described here is a technical source/review/resource gate, **not a request for new user permission**. Actual execution remains disabled in this packet because this task explicitly restricts it to source analysis and tiny tests.

## Exact source and scientific boundary

`design.py` verifies all 12 pinned sources in the prior gradient-after-replacement plan, parses that runner without importing it, and records its exact statement order and seven-key serialization bundle. `design.json` retains the exact source plan/runner hashes, factor inventory and coordinate witness pins. PyMC 5.28.0, PyTensor 2.38.2 and other target environment versions are read from pinned distribution METADATA files, not imported. This is metadata identity, not a full runtime dependency-closure attestation; a future runner must verify actual imported module paths/bytes and its Python executable/environment as well.

A future measurement must preserve the existing actual pipeline:

1. Authenticate the selected Problem/provider/role/source bindings, complete factor/parameter/token inventory, exact first parent/upstream identity, original mass 0.125, `population_cut_v1`, no parent mass in conditional logp, composition/problem/scientific-target identities, and the original eight parent records/masses. The measured first-parent graph is not an all-parent evaluation.
2. Form complete `model.logp(jacobian=True)`. Strictly replace only the exact original exposure-port node with the positive finite float64 whole-parent E of shape (3,24). Keep E a conditioning input, outside sampled coordinates.
3. Preserve installed pinned `rewrite_pregrad(replaced)` and native `gradient(rewritten, ordered)`, where `ordered` is exactly `model.continuous_value_vars`. Preserve the 64 transformed names/shapes/dtypes, contiguous offsets and complete 662-coordinate length against the frozen witness. Assert E identity/dependence, old-port absence and unchanged original value inputs before/after rewrite and after differentiation; retain the conservative whole-array stale-constant tripwire and unchanged native inventory.
4. Only after those guards pass, count the **same single bundle**: `original_cost`, `replaced_cost`, `rewritten_cost`, `gradient`, `ordered_value_vars`, `E`, `segments`. Use one protocol-5 Pickler and one memo, no out-of-band callback, no independently serialized roots and no silent compression. The original reference cost intentionally contains its original exposure; old-port/stale-array rejection applies to the modified expressions, not indiscriminately to the whole reference bundle.

The future runner must be separately source-pinned and reviewed at the exact replacement of the old persistence boundary. No executable actual-TX runner was generated here. `design.py` is not that runner: its AST checks confirm literal structure/order only and cannot certify the full scientific meaning or runtime execution of the source.

## What counting could establish

Use a streaming counting/hash sink that discards serialized bytes and updates SHA256 in at most 64-KiB views. Normal Pickler return would give the exact byte length/digest of that one attempted in-memory bundle under that source/runtime and state. It would answer whether this stream exceeded 64 MiB and whether it fits a separately proposed persistence budget, **without retaining or loading a pickle**.

A pre-write logical cap failure records accepted-prefix bytes/hash and the next attempted endpoint. It does not establish complete size, a usable/restorable pickle, or even that later reducers would have succeeded. The sink prevents accepting an over-budget write; it cannot prevent Pickler/reducer allocation of that write buffer before the callback, so memory limits remain necessary. Serialization itself may invoke trusted reducer code and is not presumed pure or safe merely because no bytes are saved.

The tiny fixture tests verify one-memo aliasing against `pickle.dumps` of invented objects entirely in memory. They never call a loader. Separate-root streams are larger in that fixture; no conclusion about actual-TX compression ratio or total size follows.

## Proposed bounds and resource check

The prospective envelope keeps 120 seconds wall **including graph construction and progress I/O**, 90/95 seconds CPU soft/hard, 2 GiB observed-RSS watchdog, 6 GiB address-space hard limit, nice15 and one thread. The logical stream cap is 512 MiB. Reports plus fresh caches have a combined 16-MiB cap and each metadata file a 1-MiB hard cap; there is no data-payload output. Every metadata write must enforce aggregate limits before writing, including temporary files; polling alone is insufficient. The watchdog interval is at most 0.25 seconds, so RSS/wall observations can overshoot between samples. No fresh available-memory/disk observation was made and no feasibility assertion follows from these prospective numbers.

Before later execution, make a new source-bound technical execution record for the exact runner/plan and fresh attempt directory; verify a resource observation no more than 120 seconds old, not future-dated, with at least 8 GiB available memory, 1 GiB free disk and permission for one nice15 CPU in the current shared-job context. Preserve separate structural/poll/source identities. Do not reuse the old attempt, cache, partial pickle or stale resource receipt. The exact policy remains subject to independent technical review and contemporaneous resource limits, not an automatic retry or user-approval prompt.

## Per-write durability cost: observed only on tiny objects

The conservative baseline is one durable progress update after every accepted Pickler write: temporary metadata write, file flush/fsync, atomic rename, directory fsync. This is **two fsync calls and one rename per write**, plus stage/failure/final records. Capped-write progress must distinguish attempted from accepted bytes. If fsync/publication fails, no successful count acknowledgment is returned; the last durable prefix may lag the in-memory accepted counter.

Final tiny observations: 111-byte and 1,135-byte streams each used one write/two fsyncs. A 65,656-byte stream used three writes/six fsyncs; its durable version took 5.41 ms versus 2.46 ms without durable progress, with 3.19 ms measured inside progress I/O. The 1,135-byte durable case took 9.55 ms, illustrating local variability. These are one-run observations on tiny invented objects, not actual graph write counts or fsync latency predictions. The seven-test suite passed in 0.070 seconds.

If an actual stream produced thousands of small writes, this baseline could consume much of the wall budget. A hypothetical 8,192 updates would require 16,384 fsyncs; **8,192 is an illustrative arithmetic scenario, not an estimated TX write count**. Do not silently throttle progress: throttling changes recoverable-prefix semantics and needs a separately reviewed plan. A count-only run may fail its wall bound because of durability overhead and must report that failure rather than raising caps automatically.

## Completion and provenance: Luna V3 findings are not assumed solved

Luna's pinned counting-sink V3 review reproduced (a) late mutation after a file was pinned but before verification returned, and (b) a success receipt left visible after its directory fsync failed, later accepted when the caller recomputed the receipt hash. These are **unverified/uncertain states**, not admissible actual-TX completion evidence.

A future controller must locally observe its own selected child identity, start/exit/return code and reaping, limit signals, exact source/request/plan projection and terminal resource measurements. The tiny test records actual `Popen.pid`, actual waited return code zero and reaping of one direct child; it is locally observed provenance, not a signature, scientific approval or proof about an unobserved remote controller. Tiny workers have no descendants. Actual graph cleanup/orphan containment still needs independent controller tests; SIGKILL/power loss cannot be assumed catchable.

For the proposed local completion protocol:

- Hold an explicit cooperating exclusive-writer contract through mutation and sealing. Verification and consumption must share a stable immutable snapshot or a lock held through **both** verification and use, with all cooperating writers excluded. A post-hash roster check alone does not close the reviewed race. A same-UID malicious writer remains outside a mere flock/chmod contract; no stronger storage boundary is implemented here.
- Require actual observed worker/controller completion and cleanup, exact source/request/parent/factor/coordinate projections, and successful durable seal/publication operations. Do not accept a marker/candidate or a locally consistent directory by itself.
- The successful caller return must be retained in a separate trusted caller record. Never recover success by hashing a visible success file after an exception, uncertain fsync, failed caller acknowledgment or unknown exit. Failure to persist that acknowledged caller record leaves the operation unknown, even if files look consistent. This local provenance rule is not an unforgeable external authority.
- Freshly verify opened identities/bytes in that stable-consumption boundary; any change before consumption invalidates the result. No verifier, stable snapshot/lease, external caller-record protocol or actual-TX supervisor is implemented by this design. Independent review must accept and test that boundary before adaptation.

`tiny_count.py` deliberately publishes **no public completion receipt** and has no actual-object or file input API. Its `failure.json` is only an unsealed tiny diagnostic, not a durable/external failure attestation. The future controller, not this helper, must provide the required observed failure evidence. This separation avoids presenting toy file consistency as resolved actual-TX authority.

## Required failure receipts and separately testable limits

Every later stage/failure record should bind plan/runner/runtime identities, attempt and observed process identities, parent/upstream/target/factor/witness pins, completed symbolic stage and coordinate ledger, caps, last durable prefix versus current in-memory counter, attempted endpoint, error/signal, resource peaks, output roster and actual cleanup/reap observations. Distinguish symbolic guard failure, serialization error, logical cap, CPU/wall/RSS/address-space bound, metadata budget, progress fsync failure, unknown exit/cleanup, snapshot drift and uncertain completion/caller publication. A missing failure write never establishes success; an outer observed terminal must retain unknown/error if the output filesystem fails.

Separate later checks are needed for numerical complete logp/gradient parity, all ordered vectors and perturbed points, exact exposure lineage beyond the conservative tripwire, all-parent conditional normalization and protected masses, complete persistence/restoration and imported runtime closure. Counting does not compile/evaluate a density, qualify sampling, prove scientific normalization/independence, or establish predictive improvement.

## Safe independent-review path

Review `design.json`, exact prior source/receipt pins, Luna V3's findings and this source-only AST analysis. Run only:

```sh
nice -n 15 python3 -B design.py
nice -n 15 python3 -B test_design.py
```

The tests accept at most 64-KiB invented payloads, use temporary directories, persist only small progress/failure metadata and reap their one toy direct child. They exercise source drift rejection, exact source/bundle order, one-memo byte/hash oracle, pre-write cap, fsync failure and locally observed provenance. They do not import scientific packages or read any old pickle. The evidence manifest pins the frozen files and target package METADATA. A subsequent actual experiment requires its own technical source/resource gate and reviewable execution version; this packet's explicit `allow_actual_tx_execution=false` cannot be promoted by changing a status string.
