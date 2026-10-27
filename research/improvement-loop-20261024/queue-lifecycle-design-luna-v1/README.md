# Queue lifecycle durability: source review and tiny crash counterexample

This packet reads the frozen `queue-durable-frontier-sol-v1` findings first and checks the durability claims against SQLite's own transaction, locking, atomic-commit, and PRAGMA documentation. It makes no production edits and runs no scientific job.

## Source-specific findings

The dispatcher writes admission/frontier changes in `BEGIN IMMEDIATE` transactions. The pinned admission connection enables foreign keys and `PRAGMA synchronous=FULL`; it deliberately leaves SQLite in the default rollback-journal mode. SQLite documents that `BEGIN IMMEDIATE` obtains the write transaction up front and can fail `SQLITE_BUSY` if another writer is active. SQLite permits only one simultaneous writer, but that lock is database-engine coordination, distinct from the dispatcher `flock` (which only cooperating queue clients use).

Within one SQLite transaction the frontier's release record, dependent-ready/block projection, and active-slot deletion have the right atomic boundary. An application-process exit before commit leaves these database changes uncommitted and recoverable by SQLite; a committed transition can be read on retry. SQLite documentation explicitly distinguishes application crashes from OS/power failures. For rollback journal mode, `FULL` syncs data but is not necessarily power-loss durable on all filesystems; `EXTRA` also syncs the directory after unlinking the rollback journal, reducing the chance that the last commit disappears after a nearby power loss. Neither setting can make an arbitrary JSON file and the database one transaction.

The executor's `publish` helper writes and fsyncs a temporary file, hard-links it to the final path, and fsyncs the containing directory. That supports publication of one immutable file under the host's filesystem contract. It does not atomically publish `result.json`, then `release.json`, then SQLite release/frontier changes. These are separate recovery boundaries. The frontier findings correctly retain the active reservation until the release transaction commits and describe replay at the file/transaction boundary; they do not claim power-cut durability.

The exact two-domain counterexample in `counterexample.py` publishes and syncs an external release file, then exits a child process inside a SQLite release transaction. On restart, the JSON file exists but SQLite has no release row and has not decremented the consumer's unresolved count. This is expected, not corruption. Safe recovery must authenticate that same output/terminal evidence and idempotently complete the DB transition; the file alone cannot make the consumer runnable. The complementary bad design would update readiness first and publish release evidence later: a crash then exposes a runnable consumer without the producer's required artifact. Keep the dependency-count update and release ledger in the same SQLite commit, and require the external record to be validated before that commit.

Failure handling is a separate lifecycle issue from database atomicity. The frontier packet explicitly shows direct consumers can be blocked while grandchildren remain unresolved and pending. A zero-ready count, like `reserved=0`, is not queue completion. Completion requires an explicit definition that accounts for active, runnable, blocked, unresolved, failed, cancelled, and retryable work; failure propagation or operator intervention needs an explicit policy.

## Concrete test implications

The existing real-process `os._exit` tests are good evidence for application-process crash recovery in the tested rollback-journal/FULL environment. They are not evidence for kernel crash, power failure, broken `fsync`, disk-full/fsync errors, storage corruption, or cross-file atomicity. A bounded lifecycle test matrix should include:

1. Death after output/release file publication but before the release transaction; confirm active capacity remains occupied, DB release is absent, dependents stay nonready, and exact replay completes once.
2. Death after SQLite commit but before caller acknowledgment; confirm retry reads the committed release and never repeats the decrement or execution.
3. Injected failure on each file fsync, directory fsync, DB commit and post-commit response; classify state as committed/uncommitted/uncertain by reopening and authenticating, never by the thrown exception alone.
4. A failure chain producer → consumer → grandchild; assert no runnable descendants and that status remains incomplete/blocked until a named propagation/cancel/retry policy resolves it.
5. A same-database competing writer and a held reader at commit; handle documented `SQLITE_BUSY` as contention, not as a failed job or as evidence of commit. Existing queue-level flock tests establish cooperating dispatcher behavior but do not exercise all SQLite lock/error cases.
6. If power-loss durability is a requirement, test the actual deployed VFS/filesystem/storage contract (or select an SQLite sync mode whose documented guarantee matches the target). Process-kill tests cannot establish this. `EXTRA` addresses the rollback-journal directory-entry detail but not arbitrary external files or dishonest/failed storage flushes.

The tiny script tests only the first scenario's core split and the empty-frontier counterexample on a temporary SQLite store. It is an invented state-machine demonstration, not an additional production-queue test. No actual power loss is simulated.

## Official primary documentation checked (2026-10-26)

- SQLite [Transaction](https://www.sqlite.org/lang_transaction.html), sections 2.1–2.2 and 3: one writer; `BEGIN IMMEDIATE`; `SQLITE_BUSY`; a failed `COMMIT` due to a reader leaves the transaction active; some errors such as `SQLITE_FULL` and `SQLITE_IOERR` can cause automatic rollback.
- SQLite [PRAGMA synchronous](https://www.sqlite.org/pragma.html#pragma_synchronous), lines 807–844: `EXTRA` and `FULL` meanings; rollback-mode `FULL` may not be durable across power loss, while `EXTRA` syncs the directory after journal unlink; application-crash durability distinction.
- SQLite [Atomic Commit](https://www.sqlite.org/atomiccommit.html), sections 2, 3, 5, and 9.2–9.4: SQLite's hardware/filesystem assumptions, its `fsync` assumptions, atomicity scope for one database and SQLite `ATTACH` multi-database transactions, and what happens when flush/deletion assumptions fail. Arbitrary application JSON files are outside a SQLite transaction.
- SQLite [File Locking And Concurrency](https://www.sqlite.org/lockingv3.html), sections 2–4: pager and rollback-journal locking, hot-journal recovery after process/OS/power failure. This is relevant to the default rollback journal used by the pinned connection helper.

## Limits

This is not an adoption recommendation. It identifies what the current frozen packet demonstrates and what a later service lifecycle contract/test plan should explicitly cover. It does not establish that the current filesystem is unsafe; it establishes that SQLite's own documented `FULL` rollback-mode guarantee is conditional on that filesystem. A new lifecycle derivative must be reviewed against its own frozen bytes; this packet does not review a future version.
