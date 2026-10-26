# V3 rejects ambiguous seals and requires an external reaped-controller receipt

V1 and V2 are preserved unchanged. V3's final suite passes **15 invented-object tests in 4.313 seconds**. It fixes the reviewed case where `complete.json` was renamed but directory fsync failed: marker presence and matching file hashes no longer qualify completion.

The counted output has an exact roster of seven files—request, source snapshot, worker log, progress, candidate, launch and worker terminal—plus `complete.json`. Unknown files/directories, partial files, symlinked listed files and `seal-failure.json` reject. The manifest must name exactly the seven allowed files and bind their bytes. Candidate/terminal/manifest schemas are V3. A local consistency helper remains explicitly insufficient to authenticate completion.

# Both injected failures are preserved

The tests run the actual count, actual marker rename, then inject an EIO at the directory-fsync call site. This is an injected filesystem failure, not a physical power-loss test.

- With writable negative evidence, the marker remains and `seal-failure.json` is written. The separately supervised controller exits **80** and is reaped; no external success receipt is created. Local and public verification reject.
- With an additional injected denial of the negative-evidence write, the marker remains, `seal-failure.json` is absent, and the exact local roster/file-hash check **passes**. The controller still exits **80** and is reaped; no external success receipt exists. Public verification rejects. The test deliberately establishes that a normal-looking directory cannot by itself prove successful sealing.

The final run's controller PIDs were 1961390 and 1961388 respectively. Their count workers had already finished; the failure is controller sealing, not counting or worker cleanup. Full logs, errors, worker terminals, outer reaping terminals and ambiguous marker bytes remain in their isolated fixture directories. Neither ambiguous output is repaired, adopted or deleted.

# External success boundary

`supervise` runs a separate known-source controller process and waits for its actual exit. Only a reaped zero exit, a returned successful controller result, and successful exact local-file validation permit publication of `success.json` in a **separate sibling supervisor directory**. That receipt binds output-root identity, controller PID/return/reap status, caller-selected source/request SHAs, the count manifest, and exact supervisor launch/log/result/reap files. The supervisor directory also has an exact roster. Verification checks all pinned bytes and cross-projections, and rejects unknown supervisor artifacts.

The public verifier requires both the count-manifest SHA and the **caller-retained successful supervisor receipt/path/hash**. Reading a marker and recomputing a digest after an uncertain failure is not a substitute for this successful return. No inference is made from a self-authored hash alone: the local evidence includes a real separately waited controller and the caller's selected source/request identity.

This remains a trusted-local process/storage contract. The verifier cannot prove where an arbitrary caller obtained its expected hashes. A caller fabricating a coherent receipt/history is outside the contract. A successful fsync return is not a universal power-loss guarantee. If external success-receipt publication itself fails or its outcome is uncertain, no successful caller acknowledgment is returned; the operation must stay unknown, even if a receipt file is visible. Do not manufacture a new acknowledgment by hashing that file. No malicious coherent-rewrite, uncatchable SIGKILL, kernel/disk failure or power-loss protection is claimed.

# Tests and remaining limits

The final suite retains normal count/hash oracle, logical-budget failure, persisted-prefix interruption, worker death after candidate metadata, real inner-controller SIGTERM/owned-worker cleanup, launcher receipt-write exception, precise soft CPU-limit labeling, source/request/duplicate rejection, metadata drift and version refusal. New tests exercise both seal failures plus unknown counted-output and supervisor artifacts. Both V3 test runs (the earlier 14-test run and final 15-test run) and all deliberately invalid fixtures are retained; source snapshots distinguish their code versions. Only the final suite qualifies the final frozen source.

Counting worker hard limits remain CPU 2/3 seconds, 256 MiB address space and 1 MiB per file, with nice15 and one-thread settings. Its watchdog thresholds are 10 seconds, 128 MiB observed RSS and 4 MiB observed output. The added supervisor waits at most 15 seconds for its controller, then uses bounded termination/reap. It does not add an independent hard CPU/address-space limit to the metadata-only controller. The tests are run under nice15. The inner worker never spawns descendants; this is not a generic process-tree controller. Uncatchable controller/supervisor death can prevent teardown or acknowledgment; no universal orphan-prevention promise follows.

Progress still performs **two fsyncs, a temporary write and rename per Pickler write**. Tiny successful fixtures make one write, so full-graph progress overhead and any throttled crash semantics remain untested. No payload is persisted or deserialized, no chunk reconstruction is implemented, and no TX graph, sampler, incomplete pickle or owner job/file is accessed.

`evidence.json` freezes final/earlier source-test records and every fixture artifact from both runs, and verifies V1/V2 packet-file hashes remain unchanged. This is completion-state research for invented objects only, not actual graph serialization feasibility, production readiness or scientific qualification.
