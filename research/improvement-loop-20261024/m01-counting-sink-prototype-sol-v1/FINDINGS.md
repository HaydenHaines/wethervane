# Invented-object counting sink: eight tests pass

`counting.py` counts and hashes the logical bytes emitted by one protocol-5 Pickler into a bounded sink. It stores **no serialized payload**. The only accepted input is a deterministic invented fixture described by exact JSON fields, with at most 64 KiB of payload and a 1 MiB logical counting budget. It has no actual graph input, pickle loader, compiler or sampler path. This prototype cannot be used as an actual-TX counting runner without a separately reviewed source/scope change.

Eight tests passed in 3.670 seconds. All fixture attempt directories are retained under the path in `test-result.json`, including failures and deliberate metadata tampering. No existing attempt or incomplete pickle was opened or modified.

The sink checks the next write against its logical budget **before** hashing/accepting any of it, hashes in 64 KiB slices, and records accepted-prefix bytes/digest plus the next attempted end. A budget failure does not claim a complete-stream size/hash. The tiny success matches independently generated `pickle.dumps` bytes exactly in length and SHA; neither implementation deserializes them. Tests also verify accepted-prefix behavior after interruption, source/request digest rejection and refusal to reuse an existing output directory.

# Complete metadata and failure publication

The caller supplies expected source and canonical request SHAs. Source bytes are checked before launch and again inside the child; a source snapshot and request are preserved. After Pickler returns, the worker writes a **candidate** count record. The controller publishes `complete.json` only after child exit zero, fresh source/request checks, and candidate identity/budget/scope checks. It pins request, source snapshot, process log, progress, candidate and terminal bytes. Verification requires an externally retained expected manifest SHA and checks every listed file/path, including symlink rejection.

This authenticates byte identity under trusted local generation/cooperating storage. It is not a signature, protection against a coherent malicious rewrite, proof that arbitrary caller-selected source is authorized, or authorization to deserialize anything. Both candidate and completion records say `deserialization_authorized=false`; no payload exists to restore. Chunk reconstruction is **not implemented or tested** here. No hash or count result changes that requirement.

Observed failure behavior:

- A logical budget violation exits 1, preserving progress and traceback without a completion marker.
- An invented exception after a persisted write also exits 1 and preserves the accepted-prefix counter; no completion marker is published.
- Actual child death with exit 73 after candidate publication leaves that candidate but no completion marker.
- The CPU-busy invented fixture is killed by kernel SIGXCPU (return −24). The terminal labels this `hard_cpu_soft_limit_signal`: specifically it is the configured **2-CPU-second soft limit**; the 3-second hard ceiling was not reached. No Python failure receipt survives that signal, so the retained progress and controller terminal are authoritative.
- Deliberately changing candidate metadata after a successful run makes verification against the saved complete-manifest SHA reject. This altered test directory remains preserved and must not be mistaken for a currently valid complete bundle.

# Exact resource boundary

Each child uses nice15, five one-thread environment controls, hard address-space limit 268,435,456 bytes, hard per-file limit 1,048,576 bytes, CPU soft/hard limits 2/3 seconds and no core dump. Controller watchdogs allow 10 seconds wall, 134,217,728 bytes observed RSS and 4,194,304 bytes aggregate output, sampled every 0.05 seconds. The effective CPU/file limits and nice value are checked in the successful fixture. The actual CPU-limit termination is tested; address-space and file-limit exhaustion are not tested.

All observed children used one thread. Sampled peak RSS across the tests was 30,699,520 bytes. Short-lived peaks can be missed by polling; address-space/per-file/CPU limits are kernel limits, while RSS/output/wall thresholds are watchdogs. The CPU-cap fixture took 3.123 seconds wall under scheduling; this is not a breach of a 2-second CPU-time limit. Signals target only the newly created toy process group. No real owner job was inspected or signaled.

# Progress durability cost and limits

The current implementation persists progress on **every Pickler write**. Each callback writes a temporary JSON file, flushes and fsyncs it, renames it, then fsyncs its directory: **two fsync calls plus create/write/rename per write**. Successful toy fixtures use one Pickler write, so they do not measure scaling to a real graph. A large graph could make many writes; progress I/O could materially increase runtime and disk traffic. No actual-graph feasibility or throughput claim follows.

A future throttled design must state its crash contract. One possible design keeps exact counters/hashes and pre-write budgets in memory, checkpoints at declared byte/time boundaries, and treats a crash checkpoint as a durable **lower-bound prefix**, never an exact resume point. The maximum uncheckpointed bytes depends on the checkpoint threshold and largest accepted write; time checks made only between writes do not guarantee a wall-time freshness bound during a long write. A successful completion must still persist the exact final counter before publication. No throttling was implemented or tested here, and no counting run may resume from a prefix digest.

This is source-bound invented-object research. It establishes neither actual TX graph serialization size, full graph counting performance, complete graph persistence, restored identity, numerical parity, conditional normalization, source admission nor scientific qualification.

Reproduce with `nice -n 15 python3 -B .../test_counting.py`. It creates a new UUID fixture directory and preserves it; it never adopts or deletes old attempts. `evidence.json` freezes source, tests and every retained fixture artifact from this run.
