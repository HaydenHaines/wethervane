# Independent review: protected-history queue architecture

I verified all 20 source entries in the frozen source manifest and all four packet self-pins. The source-pin check is preserved in `check_pins.py`/`pin-check.json`. This was a source/design review only; the architecture packet correctly reports no tests or implementation, and I ran no service, system changes, queue jobs, or scientific jobs.

## Assessment

The central boundary is well formed: an independent evidence authority owns canonical admission/result/release records, immutable objects and a monotonic head; the coordinator's SQLite frontier and local file views are disposable projections; the executor owns operational attempt/terminal/cleanup evidence. The design explicitly says a digest, hash chain, SQLite transaction or chmod is not protection when the same mutable principal can rewrite the database and its head. It also correctly separates successful command execution from scientific admission, fit/calibration, qualification and publication.

The rollback protocol requires an expected head/generation on transitions and a separately retained monotonic checkpoint or authority-side floor. That is the right distinction: an internally consistent older database+head cannot prove freshness. The packet also marks authority backup restore, root/host power-loss and same-UID attacks as outside claims unless independent infrastructure is actually supplied.

Crash handling is directionally coherent: write/fsync protected objects before references; commit reservation/ownership/frontier atomically; lost responses resolve by idempotency lookup; unknown active processes retain capacity; failed descendants are explicit; local projection rebuilds from committed history. Selected objects and live executor identities are checked on use; unvisited archived-object corruption is an availability/scrub issue rather than a false claim of immediate scans.

## Gaps before a concrete implementation claim

1. Add an explicit split-brain/fork test: two restored/cloned authority copies produce different heads for the same domain and sequence. A fresh nonce/head response and independently retained checkpoint must detect the conflict and stop dispatch; never choose the longer/local branch by itself.
2. Test nonce and generation replay directly, including a correctly signed but stale response. Bind response to the request nonce, current commit ID, generation and status as the protocol says.
3. Test projection rebuild after every crash position around authority commit and projection commit, including repeated replay, deleted projection, duplicate release and partially propagated failed descendants. Check exact active capacity/frontier without relaunching.
4. Bind release acceptance to an actual executor-owned receipt chain; a caller-supplied `accepted` status or a hash-consistent fabricated terminal must remain unknown/blocked.
5. Freeze the durability contract for the reference implementation (SQLite journal/synchronous mode, fsync ordering, filesystem assumptions, recovery path). Process-death tests do not prove storage power-loss durability, which the packet already disclaims.

These are test/design refinements, not evidence that the current system already has protected history. The proposed reference implementation remains same-UID mutable research and cannot demonstrate antirollback, immutable object permissions, or protected service availability. A minimal safe next experiment is the packet's disposable capacity-one SQLite+CAS reference, labeled `trusted_mutable_same_uid_reference`, with synthetic commands and crash/idempotency/frontier tests. Do not attach production or scientific-authority claims until separate service/UID/key custody, monotonic rollback anchoring, object permissions, storage durability and resource/lease ownership are established.
