# Independent review: legacy House pair runner lifecycle

Target packet: `/home/hayden/projects/ballotmancy-house-benchmark/research/house-legacy-pair-bounded-20261026-v1`.

This was a read-only review of the packet and a disposable-process lifecycle probe. No real bank/model was launched, and no owner source, jobs, or site files were changed. Test/probe artifacts were kept temporary and removed; the escaped test child was killed in cleanup.

## Source pins reviewed

- `README.md`: `9435ed29f150280c65f8bd9b36e3fd6b5fec0d0b601019110865b4bf26b69ff5`
- `runner.py`: `66924c1faf1641384ff8c35f3774e3e3dc68dea8d95929565444b6db216629f1`
- `test_runner.py`: `cb1ea15a91e3dfafa97f8d7dd553eca16011b1e2bb0097dfb4eeb5f76fa0f8b3`
- `scope_lifecycle_fixture.py`: `30f6fcc6696c78bebe183d06ee58d2b1ababfd1d37955d7eeff5a2f867344b0f`
- `scope-lifecycle-fixture-v1.json`: `2a6e1a69c0986d0d95e7c1d1feaafc10bd60e2c625247d18264f8804bc9114c0`
- Imported supervisor: `ae702cb59fc5d5ec15c9ddead96973a143c8b6f7e7b9b1fd2791f6037a479797`
- Imported cgroup guard: `be84ff3d90d369248673f27b99875137a93c32cb6d541abb52a22e9cbd3bd52f`
- Selected checker script: `c8278b989f72e845460fdf9a025f501b5ccd520f12f5270652a714b050940ed8`
- Checker stage helper: `4d6356fa4f34f7d13e590315bcb14123aa2cdf4ff43defa034bd8d8672e697a7`

## Tests and lifecycle evidence

Ran the six `test_runner.py` unit tests under single-thread environment settings; all passed. They cover legacy identity evidence rejection, exact result binding, inherited-process-group cleanup on deadline, injection environment stripping, and disposable success/failure runner paths. The descendant cleanup case keeps the child in the supervisor's original process group; there is no successful-exit test for a separate process group.

The checked-in scope lifecycle receipt's source hash matches the fixture source. The fixture code requests a disposable user service with `MemoryMax=64M`, `MemorySwapMax=0`, `KillMode=control-group`, and verifies those effective properties and common cgroup membership before killing the service main process. It creates a SIGTERM-resistant child in a distinct process group, then checks both original PIDs (including start-time identity) are no longer live. This supports the narrow claim that that service configuration cleaned that fixture's escaped process-group child. It does not establish future run configuration or external supervision.

I also ran the actual `runner.run` success path against a temporary fake checker. The checker spawned a SIGTERM-resistant child with `start_new_session=True`, wrote a valid result, and exited successfully. The runner returned successfully and wrote `receipt.json` while that separate-process-group descendant was still alive. I then killed the disposable child. Thus, yes: a successful checker can leave an escaped process-group descendant alive through receipt creation if an external cgroup service manager is absent or fails.

Mechanism: the shared supervisor checks that the direct child's original process group no longer exists after successful exit. `killpg(child.pid, 0)` does not find a descendant that called `setsid`. Runner cgroup membership checking occurs only while the direct checker is still running; it does not enumerate/await every process in its cgroup after direct-child success. The runner's source and result checks do not close that process-lifecycle gap.

## Binding, guards, and limits

The runner pins the clean legacy site HEAD and a listed set of checker files, hashes Node and the configured Python executable before and after execution, strips common Node/Python/dynamic-loader injection variables, and revalidates metadata and source pins before receipt. This is a useful narrow source/runtime fingerprint, not a reproducible-environment or hostile-filesystem snapshot: system libraries and all ambient runtime components are not bound, as README acknowledges.

`scope_check` verifies unified non-root cgroup membership and a finite positive `memory.max` no greater than the caller's cap, with `memory.swap.max=0`. During the run, the runner compares the current scope snapshot and the direct child's cgroup. It does not create the cgroup, prove the callback holds a lock, reserve RAM, measure resource peaks, guarantee sufficient headroom, or independently kill escaped descendants. The separate headroom tool is not connected. A hard cap can still terminate work; partial private output remains for review. The README states these boundaries and requires external continuously supervised cgroup cleanup, fresh coordination, and held resource lock.

## Assessment

The narrow tests and fixture receipts support the claims they make. The successful-exit escaped-process probe exposes a real lifecycle condition that must be contained by the separately arranged `KillMode=control-group` service manager; it is not prevented by `runner.py` itself. That limitation is explicitly disclosed in the packet, so this is not a contradiction in its stated contract, but it is a material precondition for any future run. No run authorization or forecast qualification follows from this review.
