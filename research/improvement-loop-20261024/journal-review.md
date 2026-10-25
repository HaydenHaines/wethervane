# Initial journal review

Sol implemented `ledger.py` and its standard-library tests. Root reviewed the code and requested timestamp, record-only promotion and crash-durability changes. Luna independently inspected the implementation and reproduced a history-integrity flaw in a temporary database. Root then reran the final checks and exercised competing processes.

## Findings and repairs

- Initial decision values omitted a positive promotion recommendation. `promote` is now a record-only value; there is no active registry, model qualification or deployment operation.
- Checkpoints now carry UTC recording timestamps. Exact command retries preserve the original event and timestamp. These dates do not claim when an experiment ran.
- Copied objects are flushed before registration, and directory entries are flushed for both new and existing objects. Root identified that skipping the existing-object branch could register a file before a prior failed or concurrent directory flush. Temporary files are cleaned after write/flush failures.
- Luna reproduced SQLite `INSERT OR REPLACE` bypassing ordinary UPDATE/DELETE immutability triggers. Duplicate-primary-key insertion guards now reject replacements on every immutable table, including through an independently opened SQLite connection. These are application-level safeguards, not protection from an owner deliberately dropping triggers or editing files.

## Final verification

All **12 tests passed** in root's fresh final run. They cover transaction rollback, reopening, immutable artifact copies, command idempotence/conflict, stale writes, corrupted references, failure retention, negative decisions, record-only promotion, SQL replacement guards, flush failure/retry, CLI access and **1,000 synthetic cycle records retained after reopening**. [Test log](evidence/ledger-tests-root-v1.log).

A separate root check started two processes at a shared barrier, both requesting revision zero of one invented cycle. Exactly one write succeeded, the other was rejected as stale, and the exported history contained only the creation plus one checkpoint. [Source-pinned receipt](evidence/ledger-concurrency-root-v1.json).

Ruff check and format checks pass for both Python files. These checks validate bookkeeping behavior; they are not model fitting, historical forecast comparisons or load tests of a scientific compute service.

## Remaining limits

- The journal validates syntax and artifact identity, not scientific completeness or honesty. A phase or recommendation alone cannot qualify a result.
- Artifacts are local copies and checked when read. External tampering is detected, not prevented. Ordinary local filesystem/SQLite durability is assumed; this is not a distributed storage or backup system.
- An object copied before a failed database transaction may remain unreferenced. It is deliberately retained; there is no deletion/garbage-collection API.
- Large artifacts are currently read into memory. Prefer compact manifests for large prediction banks, with the underlying immutable banks retained in owner storage. Streaming and remote durable storage are future integrations if required.
- There is no autonomous experiment scheduler, process supervisor, scientific runner, active-model pointer or publication adapter. Those remain work under the standing goal and require actual owner result contracts.
