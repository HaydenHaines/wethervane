# Symbolic checks passed; persistence hit its per-file cap

Read-only verification supports every recorded stage through `gradient_after_replacement_checked`: exact exposure substitution in the complete Jacobian-inclusive logp, the native pre-gradient rewrite, preserved explicit E input, removal of the old port, the stale-whole-array tripwire, unchanged 64 value-input names and the saved contiguous 662-coordinate segment order. The native factor inventory matches the frozen incumbent successor, and the first-parent contract keeps mass 0.125 outside conditional logp. These are **symbolic structure checks**, not evaluated density/gradient parity or a proof of all exposure lineage/conditional normalization.

The attempt then failed in `pickle.dump(..., protocol=5)` with `OSError: [Errno 27] File too large`. PID 1947544 exited 1 after 24.799 seconds, without a watchdog termination. Peak observed RSS was 637,960,192 bytes (608.406 MiB), virtual memory 956,334,080 bytes and observed threads one. The original limits were 120 seconds wall, 90/95 seconds CPU, 2 GiB RSS watchdog, 6 GiB address space, **67,108,864 bytes per file** and 268,435,456 bytes aggregate output watchdog. The per-file limit, rather than aggregate disk, wall or RSS, was the demonstrated blocker.

The preserved partial `symbolic-graphs.pkl` is exactly **67,108,864 bytes**, SHA256 `7214811bb5497fe59cbca93d0824762af71bb7427ab111c137f8383015729006`. It was statted and hashed once using 1 MiB reads; it was never unpickled, parsed, loaded, resumed, altered or deleted. Its complete required size and restorability remain unknown. A file reaching its cap does not show whether the full bundle needs 65 MiB, hundreds of MiB or more. The internal cause of its size was not diagnosed from this partial artifact.

| Accounting item | Bytes |
|---|---:|
| Partial pickle | 67,108,864 |
| Other final attempt files | 347,914 |
| Final logical attempt total | 67,456,778 |
| Sampled live output peak | 67,454,233 |
| Final minus sampled peak | 2,545 |

The final stat sum includes terminal/failure/log updates after the live sampling window, so it need not equal the sampled peak. `audit-result.json` lists every file's logical and allocated bytes and current hashes. Source-plan, execution-receipt, source dependencies and selected review/resource pins were checked; original attempt files were not changed. No `result.json` or completed restorable graph artifact was produced. Durable stage receipts retain the successful symbolic checkpoints despite the later failure.

# Non-destructive next design, unexecuted

Do not retry, append to or load the current partial pickle. Keep it and the failed terminal immutable. A new experiment requires a new directory, source-bound plan, independent review and fresh resource observation.

First measure serialization demand without writing a bulk pickle: after one separately authorized symbolic construction, use the **same single bundle, protocol and one Pickler memo** with a bounded streaming counting/hash sink. Count accepted bytes and the next attempted write, update SHA in bounded slices, and discard serialized bytes. Cap logical stream work at 512 MiB, retain the same 120-second wall/90-second CPU/2-GiB RSS/6-GiB address-space bounds, and allow only 16 MiB of reports/cache on disk. A cap yields an explicit lower bound, not a complete-size claim. If serialization returns normally, record exact stream bytes and digest for that one bundle, serialization time and peak memory. This measurement-only next experiment has **not** run.

If a later separately reviewed persistence design is justified, use a single logical pickle stream split into at most 32 MiB chunks. This avoids a 64 MiB physical-file cap without changing the Pickler memo or logical byte stream. Do not serialize roots separately: shared node/input identities across logp, rewritten cost, gradient and ordered input objects may matter. Chunking changes storage layout, not total serialization demand. Do not assume compression or protocol-5 out-of-band buffers reduce size; either would be a distinct format and restore experiment.

A concrete prospective persistence budget is:

- Data payload at most 384 MiB (402,653,184 bytes).
- Metadata/logs at most 16 MiB (16,777,216 bytes).
- Fresh cache at most 32 MiB (33,554,432 bytes).
- Planned payload plus overhead at most 432 MiB (452,984,832 bytes).
- A synchronous writer budget of 448 MiB (469,762,048 bytes), leaving 16 MiB over the planned sum.
- An aggregate watchdog at 512 MiB (536,870,912 bytes), with a further 64 MiB observation margin; per-file hard limit remains 64 MiB, while data chunks stay at or below 32 MiB.

The writer must enforce byte budgets **before each write**; polling alone is insufficient. All `.partial` chunks count toward the budget. Same-filesystem rename avoids duplicate complete copies, but hash verification adds read I/O. Count the preserved old 64 MiB artifact in filesystem headroom even though it is outside the new attempt's output budget. If the counting result exceeds the payload budget, refuse persistence under that design rather than quietly raising limits.

Write each new chunk with exclusive creation and keep a manifest of sequence, exact length, digest and cumulative bytes. Only after the entire single stream returns, every chunk is flushed/fsynced and all lengths/hashes agree should an atomic, fsynced completion manifest publish success. Failure leaves partial chunks and an explicit failed receipt; no completion marker and no loader access. A later restore test must first verify complete manifests, every chunk, runtime/source identity and the original parent/target envelope. **No restore test is proposed on the current partial file, and none was run.** A complete stream would still need independent restore/structure and numerical parity checks.

Future numerical testing must separately preserve exact complete point arrays, parent exposure, coordinate order and full gradient vector before comparison assertions. Successful symbolic structure or successful persistence does not establish numerical equivalence, cross-parent reuse, normalized conditionals, sampler readiness, scientific qualification or a 2026 forecast.

This packet executed only bounded read-only receipt/source/hash accounting. No new actual graph, numerical compile, pickle load, sampler or owner write occurred.
