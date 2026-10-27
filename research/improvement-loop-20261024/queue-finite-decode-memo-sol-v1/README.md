# Operation-scoped exact-input JSON memo — research successor

New source-bound derivative of queue-finite-ancestor-sol-v1; original files/stores/evidence remain unchanged. Runtime changes are confined to execution.py and experiment_queue.py. This is a pure parsing optimization, NOT cached ownership success, current-file state, drift, expected ownership mapping, process/terminal/dependency evidence, source admission or scientific qualification. Old stores cannot be adopted under changed source hashes. No production adoption/model/owner job/large benchmark.

## Exact semantics and lifetime

execution.decode delegates to the original run_receipts.decode unless a synchronous operation scope is active. status/tick/finalize create scope AFTER dispatcherD acquisition and clear it in finally BEFORE closingD. enqueue/seal/direct standalone executor paths remain uncached. Every audit's existing namespace/source/claim/inode checks, bounded ownershipO lock, file reads/hashes, expectedmapping and livevalidate(spec) run unchanged. Finalize still executes its full original final audit; only repeated pure JSON parsing may hit this memo.

Key is(type(data), FULL data) for exact built-in bytes/str, not hash/filename/mtime/size. String charge is UTF8withsurrogatepass; bytes and text never alias. Mutable bytearrays/other types bypass. Only successful original decodes are stored. Duplicate-key/NaN/Infinity/invalidJSON/encoding exceptions retain their original classes and are not cached. Existing exponent-overflow parsing behavior(1e400) is intentionally preserved, not covertly hardened; downstream validators retain finite-value responsibility.

Each miss returns the original fresh parse while storing an independent deepcopy; each hit returns a new deepcopy. Caller mutation cannot change another result. Memory/recursion failure in optional deepcopy bypasses caching/falls back to the original parser; deeply nested valid data are not rejected merely because copying would recurse. Invalid parser results/errors are not memoized as success. A failed source/spec/claim validation is never stored here as an ownership result.

Fixed maximum64entries,256KiB aggregate rawkey charge,64KiB individual item. Smaller limits/zero bypass are testable; larger/negative/bool limits reject. FIFO eviction preserves bytebudget. These are raw-key/entry limits, NOT a hard RSS cap: Python key/value/dict overhead, parsed objects and caller deepcopies add memory and were not profiled. No persistence/file/cache directory exists. Success, ordinary exception, KeyboardInterrupt and SystemExit clear all values and reset the ContextVar; abrupt process death drops memory. A changed PID bypasses an inherited parent memo. Nested operation scopes reject; ordinary independent threads have ContextVar isolation. This is the current synchronous dispatcher contract, not a reviewed arbitrary copied-context/async service API.

Prepare/crash/repair/mutations do not need a structural epoch for pure decoding: every next call first performs the existing fresh read and gives new bytes to decode. Different bytes miss; identical bytes decode to the same value but all source/ownership/live checks still execute. Deleted files/read errors/symlink/inode changes happen outside the memo and remain visible. Do not broaden this to any path/state/audit caching.

## Verification

19targeted tests pass2.017s:14memo/operation cases plus5ancestor property tests. Cases cover exact bytes/type, separate mutable returns, successful-only errors, bounded limits/bypass/Unicode/surrogates, nested/deep-copy fallback, all exit classes, PIDchange, abrupt childexit93/restart coldscope, actual tiny released-history input change preserving length/mtime, namespace corruption, samebytes/different job-lock inode, ownership-lock contention after cache population, public operation exits and D release on errors. Changed fixture script bytes are never executed. No fake historical rows are inserted.

The original73-test real tiny executor regression passes76.579s in isolated scratch with the correct older-version relative source layout. It includes partial preparation crashes/replay, bounded O/D contention, concurrent managers/capacity, duplicate/path/source-version and ownership-tamper checks. Full runner's authenticated owned-fixture cleanup verification is retained. No runtime implementation edits occurred after that regression; final targeted fixture additions leave runtime bytes unchanged. First targeted passing log is preserved before final crash/inode/contention additions; no failing overall run occurred in this packet.

## Measured tiny histories, no speedup claim

measure_pair.py runs original and prototype SERIAL in separate fresh source-bound scratch stores,1actual tiny accepted job each. It then measures2full audits per phase; prototype has off-before/on/off-after phases within the SAME store whileD stays held. All snapshot and execution.pin/checked invocation sequences are identical within that store.

|Prototype phase|decode calls|base parser calls|pin calls|pin-reported bytes|checked calls/bytes|
|---|---:|---:|---:|---:|---:|
|off-before|46|46|58|16,478,512|6/5,112|
|on|46|7|58|16,478,512|6/5,112|
|off-after|46|46|58|16,478,512|6/5,112|

On-phase has39hits,7entries and4,905peak rawkey bytes, all cleared on exit. Decode-input bytes remain35,520 across prototype phases; actual base-parser input shrinks to4,905. The separate original has46parser calls and16,464,724pin-reported bytes; source/path/version bytes differ and are NOT byte-identical matched workloads. Earlier independent measurements also had different byte totals with cause unestablished; this packet does not reinterpret them as exact expected byte budgets.

Instrumented prototype elapsed times were19.53/19.38/20.16ms for off/on/off. These SINGLE warm-order, cache-uncontrolled, instrumented observations do not establish a speedup, throughput, scalability or hard resource envelope. Instrumentation counts execution.pin and checked, not all reads/hashes/systemcalls. Full repeated history-wide hashing remains, including potential quadratic aggregate tick work. Saving39JSON parses in this tiny case does not solve the durable queue scaling goal; deepcopy/memo overhead can erase or reverse gains on other workloads.

## Reproduction and remaining gates

Run new tests in a fresh local copy of runtime*.py and tests: nice -n15 python3 -B -m unittest test_decode_memo test_ancestor -v. Inherited run_tests.py additionally needs copied actual older modules at parent-level and queue-v2 for its version-isolation tests; never point it at owner jobs/stores. measure_pair.py creates/deletes only its own scratch stores and calls the unmodified logical queue; no fake records. Output receipts/logs/source diff and exact pins retained. Independent review is required before any integration; no service durability, thousands-cycle, scientific calibration or publication claim.
