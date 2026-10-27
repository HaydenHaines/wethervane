# Frontier descendant lifecycle research

New isolated queue domain. Failed producer release atomically records the complete transitive blocked-descendant closure and lifecycle counters. Completion distinguishes success, failure/blocked, active, runnable and unresolved waiting; a new admission reopens the snapshot.

Real tiny executor/process-death/concurrency tests cover branching failures, idempotence, counter/ownership partitions and unknown exits. No model run or production change. Read FINDINGS.md, tests-final.log and evidence.json. Independent review is required; trusted mutable storage, bootstrap/repair and large-graph limits remain.
