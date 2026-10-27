# Durable runnable/active frontier research

New opt-in store only; frozen queues and model jobs unchanged. Atomic SQLite admission/reservation/release transitions maintain pending counts, a partial ready index and capacity-one active slot. Ordinary idle selection avoids released-history scans, while selected dependency/executor authentication remains.

Eight real crash/replay/concurrency tests pass. SQL-only fixtures at4096 synthetic historical rows show fixed selection VM counts; they are not thousands of authenticated dispatch cycles. Failure blocks direct consumers; grandchildren wait. Empty runnable frontier never means queue completion.

Read FINDINGS.md, tests.log, measurements.json and evidence.json. Independent review is required before adoption. No model run or production integration.
