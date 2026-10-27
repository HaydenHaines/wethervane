# Independent review: disposable cgroup emptiness fixtures

## Scope and disposition

Static source/receipt review only. I did not execute either disposable systemd fixture, launch a bank, or modify the House worktree. The newer nested fixture provides concrete evidence of one successful manager-directed teardown in one disposable delegated systemd unit. It is materially stronger than the earlier non-nested fixture for demonstrating recursive cgroup emptiness, but it does not establish a race-safe invocation-bound finalizer or authorize a real run.

## Frozen inputs

The inspected packet is `/home/hayden/projects/ballotmancy-house-benchmark/research/house-legacy-pair-bounded-20261026-v1`.

- `scope_empty_fixture.py`: SHA-256 `0e0c5c1c6bf6259174758b4c83034826663cfc2ebf363c7b81b541d7fb3743d0`.
- `scope-empty-fixture-v1.json`: SHA-256 `6dd3fc2938110c4d2377c2c58d74afe9282dcfaf91da55efddca14a48016eecf`.
- `scope_nested_fixture.py`: SHA-256 `56e6db9d3e809414ff459f39658e9222ebbc65249eb5cbe7c3c97e66c7287097`.
- `scope-nested-fixture-v1.json`: SHA-256 `af42bbd42d0d6a21a24fc4a687a4f9b9a902b98a9bf17828b6ce22045552e8cd`.
- `README.md` at review: SHA-256 `4a910342118fddbfbd5e74b7bf78bd3eb5613947e7d168bb6bfe96455619e3d1`.

I re-hashed all five files after review; the hashes were unchanged. Exact values are also recorded in `evidence.json`.

## What the nested fixture establishes

The source creates a unique transient user service with `KillMode=control-group`, `Restart=no`, `Delegate=yes`, `RemainAfterExit=yes`, `MemoryMax=64M`, and `MemorySwapMax=0` (lines 81–92). It verifies those effective properties through `systemctl show` (100–107). The worker creates one nested cgroup and moves a SIGTERM-resistant child into it; it confirms the child’s PID/start ticks, separate process group, and `/proc/<pid>/cgroup` membership (48–72).

After the main service process exits successfully, the fixture confirms the parent has no direct processes while the child remains in the nested group, both parent and nested `cgroup.events` report `populated 1`, and the unit invocation has not changed (116–125). It then requests manager-directed SIGKILL for all unit processes (126–127), waits for that PID/start-time identity to cease being live, and observes parent `cgroup.events` contain `populated 0` before requesting stop (128–145). Before that empty observation it rechecks the unit’s invocation ID, control-group path and active/exited state, and compares the parent directory’s device/inode with its initial observation (136–143). The captured JSON reports those observations and pins the fixture source hash (receipt above).

This supports the narrow statement: in that recorded disposable run, a SIGKILL request through the user service manager was followed by the recorded child no longer being live and by a recursive parent-scope `populated 0` observation before the stop request, with matching sampled unit invocation/path and parent device/inode.

The earlier `scope_empty_fixture.py` independently records the same broad sequence without a nested child. It confirms invocation/path and `populated 0`, but its own README correctly says it neither creates a nested scope nor checks effective memory/swap limits. The nested fixture supersedes that gap for one nested delegated example, not for finalizer safety.

## Remaining gaps and races

1. **No atomic invocation-bound manager operation.** `systemctl kill ... <unit>` addresses a unit name. The fixture checks invocation ID/path before and after the kill, but the manager action is not conditional on that invocation ID. A stop/restart or competing manager operation between observations could target a replacement invocation; endpoint checks can miss an intervening change.
2. **Path observations are not a held object identity.** Device/inode equality is useful evidence, but `stat()` and subsequent `events` reads are path lookups, not operations on a pinned directory file descriptor. It does not rule out transient replacement or reuse between samples. The receipt itself acknowledges that two inode observations do not prove exclusive control or race-safe acceptance.
3. **Narrow hierarchy and configuration.** It tests one nested child cgroup with `Delegate=yes`; the proposed real service is nondelegated. It does not establish behavior for arbitrary nested descendants, process migration, multiple children, cgroup deletion/recreation, or the production service's exact unit configuration.
4. **Snapshot, not durable finalization.** `populated 0` is a point-in-time observation. There is no crash/restart recovery, stop failure, manager restart, unit-name reuse test, durable final receipt, or independent retained tool-return chain. The output is captured stdout, not a protected acceptance record.
5. **Cleanup is best effort.** The `finally` block requests unit stop and then may signal the original process group if the recorded child still appears live (162–168). It does not assert teardown success, recheck recursive emptiness after cleanup, or establish what happens if the fixture/controller itself is killed.
6. **Scope boundary.** The fixture has no forecast/bank and is explicitly `publication_qualified: false`. It demonstrates a lifecycle mechanism in a disposable setup; resource admission, held-lock identity, supervisor-death handling, and real-run authorization remain separate requirements.

## Smallest useful next proof

A finalizer should retain the exact transient unit invocation identity and cgroup directory object, perform manager teardown under an explicit cooperating lifecycle lock or equivalent serialized manager protocol, and make acceptance conditional on the same invocation still owning that object. It should record child identities and recursive emptiness, fail closed on restart/path/inode/manager ambiguity, and durably bind the full observation and successful process return in the final receipt. Exercise replacement between every read/action boundary, manager/controller death, stop errors, and nested descendant cases with disposable units. This is a design recommendation, not a claim that systemd offers an atomic unit-name-plus-invocation kill primitive.
