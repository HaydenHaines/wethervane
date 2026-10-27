# Independent review: disposable scope success and teardown fixture

Reviewed exact packet files in `/home/hayden/projects/ballotmancy-house-benchmark/research/house-legacy-pair-bounded-20261026-v1`. This was a static review only; I did not run or reproduce the fixture because the owner files already contain a run and receipt, and the request directed against reproduction when that evidence exists. No bank/model/workload was launched and no owner file was modified.

## Source and receipt pins

Before review, `scope_success_fixture.py` was 6,792 bytes with SHA-256 `92fcdfd6776d61d3dd474f161e179d807ef240947b286d177c21c6cefa6e2d0d`. The receipt's `fixture_sha256` exactly matches this source digest.

The existing `scope-success-fixture-v1.json` was 768 bytes with SHA-256 `da6b6192b05ce1451c55f9481c7dabda21d1174d0a6e9a895a5f1751708ed6d9`. The fixture prints this aggregate receipt to stdout; it does not write a new receipt file itself. Its existing receipt says `scope_events_after_stop: null`, `child_not_live_after_stop: true`, and `publication_qualified: false`.

## What the fixture mechanism demonstrates

The source requests a fresh user systemd service with `MemoryMax=64M`, `MemorySwapMax=0`, `KillMode=control-group`, `Restart=no`, `Delegate=no`, and `RemainAfterExit=yes`. The worker launches a SIGTERM-resistant child with `start_new_session=True`, so it leaves the worker's process group while remaining in the same cgroup. The parent publishes child PID plus start-time identity and waits until the observer creates a finish marker.

The observer verifies the service is active and has an invocation ID, checks the child's cgroup equals the service cgroup, observes `cgroup.events` as populated and the original child as live, then lets the worker exit. It waits for `SubState=exited` and verifies the invocation ID is unchanged, the service result is success with main exit status zero, while the escaped child remains live and the cgroup remains populated. It then explicitly calls `systemctl --user stop` and waits until that original PID/start-time identity is no longer live. This is a concrete disposable demonstration that the configured service manager can clean an escaped process-group child after successful main-process exit.

## What it does not demonstrate

This is not an invocation-bound finalizer or an empty-scope receipt. The fixture does not write a final receipt after the stop; its JSON is emitted as output. The recorded `scope_events_after_stop` is `null`, which the source sets when the cgroup events file no longer exists. Therefore the evidence shows the original child was no longer live after service stop and the cgroup path had disappeared; it does not show a post-stop `populated 0` observation. Nor does it establish that a production runner is bound to that exact unit/invocation, waits for scope teardown before issuing any future receipt, handles nested cgroups or unit replacement, or is robust to crash/finalizer races. The receipt itself accurately names several of these limits, including that scope disappearance alone is not proof of invocation identity or manager teardown.

## Hash stability

I repeated SHA-256 checks after the review: source remains `92fcdfd6776d61d3dd474f161e179d807ef240947b286d177c21c6cefa6e2d0d`; receipt remains `da6b6192b05ce1451c55f9481c7dabda21d1174d0a6e9a895a5f1751708ed6d9`. No source-version inference from cached bytecode was used; this review read the source text directly and did not execute it.
