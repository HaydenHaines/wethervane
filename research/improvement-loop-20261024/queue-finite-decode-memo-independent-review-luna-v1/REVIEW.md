# Independent review: finite queue decode memo

Reviewed `/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/queue-finite-decode-memo-sol-v1` without changing its files or launching owner jobs. Reproductions used fresh temporary copies only. The target's `evidence.json` pins and all entries in `source-pins.json` matched at review time; the additional exact inputs used for this review are in `input-pins.json`.

## Frozen source and diff

The reviewed target has these key pins:

- `execution.py`: `6cb2a3e94be5326e94447fe25df40333498d405978b20705500c96a64ac233f0`
- `experiment_queue.py`: `76435714a5be8cb5dac72705b25d6301cda6cab029d3eded4544cfb31bea87e2`
- `source-pins.json`: `3598226c419cf7d50593aafcc31fbcdb3fb4df43d955878d9cb10b26788a7f7f`
- `evidence.json`: `308befb31d59edd9e97a98afef6828570bac69b63ec08e87b3617baebc2544d6`

I compared all common Python modules in the target against the exact immediate ancestor `queue-finite-ancestor-sol-v1`: 13 common files, with only `execution.py` and `experiment_queue.py` differing. The three additional target modules are the measurement scripts and new memo tests. The runtime diff matches the packet: `execution.decode` delegates to the existing `run_receipts.decode` outside a scope; inside it uses only the raw-input memo. `status`, `tick`, and `finalize` enter a scope after acquiring dispatcher D and exit it before closing D. `enqueue`, `seal`, direct executor operations, and all other runtime logic are unchanged. I checked the full source diff and `source-comparison.json`; no broader ownership, pin, read, validation, or dispatch cache was introduced.

## Exact parse-key, freshness, aliasing, errors, scope, and bounds

The memo key is `(type(data), full_data)` for exact built-in `bytes` and `str`; bytes/text do not alias and no digest, filename, mtime, inode, or size substitutes for byte equality. Other input types bypass. For every cached source document, the caller still reads bytes and runs the existing pin/hash/inode/namespace/live checks before parsing; a key hit only skips the JSON parser. The included same-size/same-mtime source edit test passed and confirms historical input drift remains visible. The ownership-lock contention test also passed with a populated memo.

Misses return the base parser's fresh object and keep a separate `deepcopy`; hits return another `deepcopy`. Tests mutate first and subsequent results and verify no alias between calls. Only successful parses are inserted. Duplicate keys, non-finite JSON constants, malformed JSON, and bad encoding retain the original exception classes; errors do not become cache hits. Python's existing `1e400` overflow-to-infinity parse behavior is deliberately retained, with finite-value checks left to downstream validators.

Each operation has a new FIFO memo, cleared in `finally` on normal return and on `BaseException`, including `KeyboardInterrupt` and `SystemExit`. Nested scopes reject. Fork-PID mismatch bypasses the inherited memo. Limits are 64 entries, 256 KiB aggregate raw key charge, and 64 KiB per item; smaller limits and zero bypass are tested. These are **key-charge limits, not an RSS cap**: dict/object overhead and deep copies are not charged. Deep-copy memory/recursion failure bypasses the memo and falls back to the original parser behavior.

The 19 targeted tests cover these behaviors plus invalid bounds, unicode/surrogates, deep nesting fallback, abrupt process exit/restart, public operation scope exits, and validation after memo population. They passed in a fresh scratch copy in 2.098 seconds.

I also ran a separate actual `os.fork` probe from copied sources. In a child forked during an active operation, `decode()` called the base parser once and registered no memo hit, as intended by the PID check. The child still inherits the ContextVar reference, so trying to open a new decode operation before clearing that context fails as nested. This is narrower than cache reuse, and agrees with the documented synchronous-only boundary. The complementary independent ContextVar review found fresh ordinary threads start empty, while deliberately copied contexts and asyncio tasks can retain/repopulate the mutable memo after owner-scope exit. Current queue entry points are synchronous with no `await`; do not reuse this API from copied contexts, concurrent copied threads, or async services without generation invalidation and synchronization.

## Fresh-scratch inherited regression and layout reconciliation

I reproduced the README's layout-sensitive tests in a fresh temporary tree with the actual parent-level original dispatcher/runtime sources at `improvement-loop-20261024/`, the packet under that directory, and `queue-v2/` beside it. Under that exact source layout, targeted tests passed 19/19 and `run_tests.py` passed 73/73 in 76.821 seconds. The cleanup wrapper recorded 123 fixture process identities. I independently checked boot ID, PID start ticks, process group, and session afterward; zero matching owned fixture processes remained live. These were invented test jobs, not owner/model jobs.

I preserve two earlier harness attempts separately rather than folding them into the passing result:

1. My first copy placed the immediate ancestor at the parent level and put `queue-v2` at the wrong relative directory. Its six source-version failures were harness setup errors.
2. A later layout put `queue-v2` correctly but still used the finite-owned immediate ancestor as the parent baseline. It ran 73 tests with 72 passing; the single failure was `test_store_versions_cannot_dispatch_each_others_plans`, which expected `unsupported or moved queue store` but received `queue store source identity differs from running dispatcher`. The immediate ancestor and memo successor share the finite-owned schema, so the test took the source-pin mismatch branch. I confirmed the correct parent-level source is the older root dispatcher with `LocalExperimentQueueStoreV1`; with that pinned source layout, all 73 passed.

The older owner regression log says 73 tests passed, but its evidence does not pin the parent-old-source bytes used by that prior run. My passing result above is independently reproduced against the exact layout and sources pinned in `input-pins.json`; it does not retroactively establish the older run's source provenance.

## Tiny measurement and performance scope

I reran `measure_pair.py` in a fresh scratch tree. It uses one accepted synthetic queue job and two full ownership audits per phase. Prototype off-before/on/off-after phases had 46 `execution.decode` calls each; base parser calls were 46/7/46, with 39 memo hits, seven entries, and 4,905 peak raw key bytes in the on phase. Within the prototype store, snapshots, pin sequences, and checked sequences were identical across phases. Pin/checked calls and byte counts were unchanged while parser calls dropped.

My instrumented elapsed observations were 18.760/18.791/19.520 ms for prototype off/on/off. The author's recorded measurements are similarly close and noisy. The separate original source store has different source/input bytes and pin-reported totals, so these are not byte-identical matched histories. This is one tiny, warm-order, instrumented observation; it establishes neither a speedup nor throughput, scale, RSS bound, or queue-goal solution. The packet's explicit no-speedup/no-large-benchmark wording is correct. The measurement helper itself uses ordinary Python `assert` checks and was run without `-O`; those are test/instrumentation checks, not runtime validation gates. The new memo's production checks use explicit `require` exceptions, not Python assertions.

## Assessment

I found no parse equivalence, stale-byte, aliasing, bounds, operation cleanup, or current synchronous queue-path blocker. The implementation remains a narrow parsing optimization: fresh file reads, full audit/ownership checks, and repeated hashing stay in place. The operation-scope and PID behavior is bounded as documented, with the copied-context/async limitation above. The evidence supports fewer JSON parser calls on this tiny fixture only; it supports no speedup or production adoption claim. The exact queue-packet source pins remain unchanged after review.
