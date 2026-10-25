# Single-attempt execution and recovery

`execution.py` supplies durable execution support for **future, explicitly selected**
owner commands. It launches a real subprocess and preserves its lifecycle across
caller loss. It does not schedule scientific comparisons, adopt existing jobs,
retry automatically, interpret numerical gates, advance journal phases, qualify
forecasts, activate models, or publish anything. Existing House/Senate jobs and
completed CPI/gasoline/momentum experiments are untouched.

## Contract

Use one persistent local store for cooperating execution clients. `attempt_id`
is globally unique within that store. Job locks and retry histories are scoped
by the exact `(cycle_id, job_key)` pair; another cycle may reuse a job label.
An output-directory claim is global within the store, preventing cross-cycle
reuse of an owner output path. Owner runners still create their own fresh output
directory and enforce their scientific contracts.

`prepare(root, specification, expected_spec_sha256)` validates and durably records
an immutable specification. The digest is SHA256 of UTF-8 `ledger.canonical(spec)`.
Repeating the same preparation returns its original receipt; a changed spec under
the same attempt ID is rejected. The required fields are:

| Field | Meaning |
| --- | --- |
| `schema` | `LocalExecutionSpecV1` |
| `cycle_id`, `job_key`, `attempt_id` | Explicit research cycle, job and attempt identities |
| `argv` | Exact argument array; absolute invoked executable path, no shell |
| `executable` | `executable_pin(argv[0])`: invoked path plus authenticated resolved binary |
| `cwd` | Canonical absolute working directory |
| `environment` | Complete, deliberately minimal environment mapping; inherited variables are excluded |
| `input_pins` | Nonempty explicit source/plan/input pins made with `pin(path)` |
| `output_directory` | Fresh canonical absolute native output directory |
| `native_receipts` | Nonempty list of `{path: relative_json_path, required_fields: {...}}` |
| `timeout_seconds`, `grace_seconds` | Positive finite execution/termination bounds |
| `previous_attempt` | `null` initially; otherwise `{attempt_id, receipt}` binding the prior terminal/recovery pin |
| `retry_reason` | Empty initially; explicit nonempty reason for a successor attempt |

Ordinary input pins contain absolute canonical `path`, `sha256` and `size`.
Input verification streams file bytes, including large posterior files. Compact
JSON documents are authenticated against the exact bytes returned to their parser.
Executable symlink invocation is retained: invoking `.venv/bin/python` still
selects that venv. Retargeting its symlink or changing the resolved binary is rejected.
Preparation also pins the manager, its local `ledger.py`/`run_receipts.py` helpers,
and its own supervisor/helper interpreter. Supply source snapshots and existing
owner manifests/certificates as inputs; pins detect drift and do not create an
immutable filesystem or copy large scientific banks.

Native receipt checks authenticate JSON and exact declared top-level metadata
(for example schema and plan SHA). They do not interpret a scientific `status`,
posterior diagnostics, score or promotion decision. All execution observations
and terminals retain `qualification: "unassessed"`.

## Launch and observe

`launch(root, attempt_id)` starts a detached supervisor and returns an observation.
Repeated calls reconnect to that attempt's records; they never launch it again
once a durable launch intent exists. The per-job lock passes through the supervisor,
startup helper and native `execve`. A startup pipe releases the native command
only after the helper's Linux boot ID, PID and process start ticks are durably
recorded. Pipe EOF before release exits the helper without executing the command.

`inspect(root, attempt_id)` verifies retained receipt evidence and reports these
separate facts:

- `process_observation`: birth-identity observations, held cooperating job lock,
  and surviving members of the isolated process session.
- `attempt_terminal`: recorded `return_code` or explicit `exit_unknown`; native
  output metadata verification, missing/invalid files and cleanup observation.
- `qualification: "unassessed"`: unchanged regardless of process return/output.

Return code zero can coexist with missing or malformed native receipts. A native
failure receipt can be valid execution evidence. A verified native receipt after
supervisor loss can coexist with unknown process exit. None establishes a useful
forecast or a qualified comparison.

The command's stdout/stderr are retained in `command.log`; lifecycle JSON records
are immutable publications, flushed with their directory entries. Terminal records
pin the specification, lifecycle/stop observations, log and declared native files.
Unexpected output drift is rejected. There is no overwrite or deletion operation.

CLI equivalents, from the repository root:

```sh
python3 -B research/improvement-loop-20261024/execution.py --root STORE prepare --specification SPEC.json --sha256 CANONICAL_SHA
python3 -B research/improvement-loop-20261024/execution.py --root STORE launch ATTEMPT
python3 -B research/improvement-loop-20261024/execution.py --root STORE inspect ATTEMPT
```

Future commands wrap existing owner CLIs verbatim. For example, momentum's native
runner takes `--receipt PATH --receipt-sha256 SHA`; E01/G01 takes `--plan PATH
--plan-sha256 SHA --run FRESH_PATH --runtime-certificate PATH
--runtime-certificate-sha256 SHA`. Their native guards continue to own source/runtime
verification, label access, matched retry rules and scientific diagnostics.
Include only the necessary explicit environment, such as their declared BLAS
thread limits; never copy the caller's environment or credentials into a spec.
These examples are interfaces for new authorized experiments, not instructions
to rerun the preserved experiments.

## Recovery and limits

Caller termination leaves the detached supervisor running. While it survives,
the declared timeout sends TERM, then KILL after the grace period, to its owned
unreaped process group; the leader's PID cannot be reused before `wait()`. Remaining
live session members prevent a terminal being sealed. A supervisor exception with
surviving descendants preserves `supervisor-error.json` and their birth identities.

**Killing the supervisor removes its timeout enforcement.** The child may continue
to run and hold the lock. Inspection/recovery does not fabricate a failure or start
a replacement. To explicitly stop only this manager's own orphaned attempt:

```sh
python3 -B research/improvement-loop-20261024/execution.py --root STORE stop ATTEMPT
python3 -B research/improvement-loop-20261024/execution.py --root STORE inspect ATTEMPT
python3 -B research/improvement-loop-20261024/execution.py --root STORE recover ATTEMPT
```

`stop` requires supervisor loss and a verified live ownership anchor: the native
leader, a surviving descendant recorded by the supervisor, or a survivor holding
the exact inherited job-lock file description. Preparation pins that lock's
device/inode. The last route checks same boot/session, birth identity, the matching
descriptor's inode and its `FLOCK ADVISORY WRITE` entry before and after PIDFD
acquisition. Opening the same inode separately does not supply that lock evidence.
Linux documents these per-descriptor locks in
[proc fdinfo](https://www.kernel.org/doc/html/latest/filesystems/proc.html#proc-pid-fdinfo-fd-information-about-opened-file).
It uses
PID-safe Linux descriptors, confirms stopped states, and rescans the frozen session
before killing individually verified members. Each signal delivery/absence and
stabilization result is preserved. It never signals a numeric recovered PID/group.
If no birth-verified or inherited-lock ownership anchor survives, it refuses to
signal an unanchored session; explicit operator investigation is required.
Inaccessible procfs evidence also fails closed. Daemonized descendants that
escape the isolated session or deliberately close the cooperating lock are outside
this contract. This is process supervision for cooperating local commands, not a
container/cgroup boundary.

`recover` acquires the free job lock and verifies the recorded processes and
session are absent before sealing an immutable recovery receipt.
It captures the native files currently present without claiming to know the lost
wait status. A supervisor's preserved wait code/timeout remains recorded; otherwise
the result is explicitly `exit_unknown`. Stop delivery alone never supplies an
orphan's exit status. An active
lock, surviving process or uncertain ownership blocks replacement. A subsequent
attempt needs its own fresh output path, the exact previous terminal/recovery
pin and a retry reason; owner-specific retry authorization remains independent.

A hard crash during preparation may leave retained unregistered files. Such an
attempt cannot launch. A launch intent with no surviving process is never rerun
in place: reconcile it and create an explicitly bound successor. Copy/backup the
whole store together with external owner artifacts; local fsync is not a backup
service. Multiple stores do not coordinate locks, and distributed/network
filesystems are unsupported.

`attach(ledger, command_id, root, attempt_id)` verifies recorded execution evidence
and copies compact specification/prepared/terminal records with `Ledger.put_artifact`.
Large native artifacts remain in owner storage under their verified pins. Attachment
does not append or advance a cycle checkpoint. Scientific result adapters and
independent review remain separate.

## Verification

```sh
python3 -B -m unittest discover -s research/improvement-loop-20261024 -p test_execution.py -v
```

The suite runs tiny invented owner-shaped commands. It covers concurrent launches,
caller loss, startup/gate crashes, supervisor loss, saved descendant recovery,
PID reuse/rechecked PIDFD acquisition, timeout tree cleanup, venv symlink semantics,
cross-cycle job labels, immutable retry binding, metadata/output drift, streamed
input verification and artifact-only journal attachment. No scientific fit runs.
