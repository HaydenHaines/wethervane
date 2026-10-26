# Independent review: compressed M01 actual-equivalence probe

Review target: frozen `m01-compressed-actual-equivalence-probe-sol-v1` (evidence SHA-256 `4613346d3166a43835bc3499410811af37cc6dd064ea2a10b1ab49e5a5e95d4a`). No author files were edited and no actual model graph was imported, built, compiled, or evaluated.

## Decision

**Block actual execution pending a child-entry/resource-enforcement fix.** The prepared runner applies the fresh review/resource gate and the RLIMITs/watchdogs in `launch.py`, but the actual worker `child.py` can be directly invoked with a hand-created `authorization.json`. Its “controller binding” checks only contract digest, the two arm names, output-directory name, and `os.getppid() == authorization['controller_pid']`. A wrapper can create that file and spawn `child.py`, satisfying the parent-PID check without executing the reviewed controller. The worker then imports the scientific stack and builds/compiles the complete actual graph without `launch.py`’s CPU, address-space, RSS, wall-time, per-file/arm disk controls or caller review/resource receipt. This contradicts the documented “controller-only invocation” and “no launch without externally pinned gates” claims, and is a material resource-control bypass. I did not run this path.

Before any actual run, make child entry fail closed unless it is started by the approved controller, and enforce critical resource limits inside the worker before importing/building the graph as defense in depth. A direct-entry guard alone is not a same-user security boundary; document the operational trust model and ensure the worker cannot accidentally perform an unbounded graph build. Re-freeze and review the successor separately.

## Supported claims and limits

Static source review supports a well-scoped computational design: the selected arm compares the original and compressed providers on the same frozen actual Problem, exact source-bound six-poll block, same first parent, and same two 662-dimensional points. Parent selection validation retains all eight `.125` masses; the calculation itself evaluates only the first parent and does not insert the parent mass in logp. The comparison requires matching complete finite 662-vectors, factor values, poll logp, rates, exact state/layout, unchanged nonmeasurement ownership and distinct scientific/composition IDs. It does not claim all-parent normalization, model aliasing, prediction skill, or a forecast. Finite differences check only six selected full-joint coordinates, not every gradient coordinate.

The source-pinned compressed provider and native wrapper are explicitly loaded from paths whose bytes are checked through the 89-entry contract and predecessor selection pins. Library version/path checks are also present. This supports source identity under the packet’s stated local non-hostile-filesystem trust model; local SHA checks/locks do not protect against hostile concurrent coherent rewrites.

The two-run design uses separate controller invocations; compressed requires the caller-passed digest of the original controller-success receipt and fresh resources. Source inspection shows success stdout is returned only after arm completion (and compressed comparison). However, the authored tiny suite does not exercise the CLI gate/receipt sequence end to end; it tests the admission helper and worker cleanup fixtures. That coverage gap is secondary to the direct child bypass.

The controller uses one CPU affinity, `nice`, one-thread environment, `RLIMIT_CPU`, `RLIMIT_AS`, `RLIMIT_FSIZE`, sampled RSS/thread/disk watchdogs, and process-group TERM/KILL/reap. The design correctly notes sampled limits can overshoot and cannot guarantee cleanup after SIGKILL/power loss. None of these protections apply to a direct worker invocation because they are installed by the controller’s `Popen` setup and monitoring loop.

## Independent bounded reproduction

Copied the six frozen source/contract files into `/tmp/luna-compressed-probe.VJKkFS` and ran only `test_tiny.py` with the packet’s calibration Python, `nice -n 15`, a 30-second wall timeout, and BLAS/OMP/Numba threads set to one. All 18 invented-object/source checks passed. The two real sleeping subprocesses were reaped after injected launch-receipt write failure and controller `KeyboardInterrupt`; terminal records captured PID, `-15` exit, controller exception and `no_retry=true`. The tests explicitly report `actual_graph_imported=false` and `actual_graph_executed=false`.

This reproduction supports the tiny helper behavior only. It cannot establish actual graph compile success, actual `662`-gradient parity, full factor correctness, runtime resource sufficiency, or the advertised controller gate against direct invocation.
