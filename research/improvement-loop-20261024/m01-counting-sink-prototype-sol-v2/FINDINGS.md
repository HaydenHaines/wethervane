# V2 fixes controller teardown and CPU labeling

V1 remains unchanged as issued. V2 is a separate invented-object prototype with two corrections: SIGXCPU is labeled `cpu_soft_limit_sigxcpu`, and every post-spawn exit path enters owned-child teardown and terminal publication before returning or propagating a controller exception.

The controller obtains a pidfd for its own unreaped direct child. Catchable INT/TERM signals are deferred across Popen/handle assignment; the child restores its old signal mask before exec. On a controller exception, handled interruption or watchdog limit, the controller sends TERM through its child handle, waits up to one second, escalates to KILL if needed, and reaps the child. A target that races to exit is still waited/reaped. Repeated INT/TERM are temporarily ignored during bounded cleanup, then original handlers are restored. Completion is forbidden unless cleanup is verified.

The terminal records child PID, return code, cleanup signals/reap outcome, interruption signal or controller exception, and resource observations. A terminal-write failure triggers a best-effort JSON terminal on inherited stderr and prevents completion. That fallback is not a guarantee when storage and stderr are both unavailable.

# Executed failure tests

Eleven invented-object tests passed in 3.905 seconds. All fixture directories, including deliberate failures and metadata/version corruptions, remain preserved. Earlier passing test runs are also retained. No TX graph, scientific compile, sampler, prior incomplete pickle or owner job was accessed.

Two new real-process tests address the review findings:

- An injected launcher `launch.json` write exception occurs **after actual Popen**. The owned child (PID 1955455 in the final run) is terminated by TERM and reaped (return −15). `terminal.json` records `controller_exception`, the original OSError, and verified cleanup. The exception propagates; no completion marker exists. The early failure preceded monitoring, so its zero peak counters mean unobserved, not zero resource use.
- A separate real temporary controller receives SIGTERM after its actual worker has published startup progress. Its handler enters teardown, terminates/reaps worker PID 1955460 (return −15), and writes `controller_interruption`, signal 15 and verified cleanup. The test holds a pidfd opened while that worker is live and observes its exit independently. The temporary controller then exits 84 through the test wrapper; no completion marker exists.

The CPU-busy fixture receives SIGXCPU (return −24) at the **2-CPU-second soft limit**, with a 3-second hard ceiling. The terminal now uses the precise soft-limit label. Its wall time was 3.130 seconds under scheduling. This is a kernel-enforced CPU limit, not a claim that the hard CPU ceiling fired.

Other tests retain V1's exact tiny pickle-byte count/hash oracle, pre-write budget rejection, post-write failure prefix, worker exit after candidate metadata, source/request SHA refusal, duplicate output refusal and post-completion metadata-tamper rejection. V2 completion/candidate/terminal schemas are versioned; a V1 completion schema is rejected even when its caller-supplied hash is recomputed. The TERM→KILL escalation branch and interruption during the masked spawn window are not independently fault-injected here; the successful new signal test is after startup.

# Boundaries remain narrow

The worker accepts only a fixed deterministic invented fixture description, at most 64 KiB payload and a 1 MiB logical count budget. It never spawns descendants, so verified cleanup covers its dedicated direct child. This is not a general subprocess-tree controller. SIGKILL of the controller, power loss and uncatchable termination cannot execute Python teardown or guarantee a terminal receipt; those cases remain unsupported. A kernel/OS cleanup error produces an unresolved cleanup receipt and forbids completion rather than asserting success.

Each child runs nice15 with five one-thread settings, CPU limits 2/3 seconds, 256 MiB address-space hard limit, 1 MiB per-file hard limit and no core dump. Watchdogs enforce 10 seconds wall, 128 MiB observed RSS and 4 MiB observed output, at 0.05-second intervals. Polling can miss short peaks. File/address-space exhaustion is not tested; effective resource settings and CPU termination are tested.

Caller-selected source/request SHAs, source snapshots, successful child termination, and V2 completion file pins provide trusted-local byte identity. They are not signatures, malicious-storage protection, scientific authorization or permission to deserialize. There is **no serialized payload**, no loader, no chunk reconstruction test and no actual graph size/feasibility result. Candidate records remain distinct from authenticated complete metadata. Deliberately tampered/version-corrupted test outputs remain invalid under their saved completion contracts.

Progress still fsyncs on every Pickler write: temporary JSON write/fsync, rename and directory fsync, or **two fsyncs per callback**. Tiny successful fixtures use one write; actual graph write counts/runtime overhead are unknown. No throttling was implemented. Any later throttling needs explicit lower-bound-prefix crash semantics, maximum checkpoint lag and final exact-counter publication; a prefix hash is never a resumable count or deserialization authorization.

`evidence.json` freezes source, final and earlier test logs/results, all retained fixture artifacts, and checks that V1's original packet files remain unchanged. Reproduce only the invented tests with `nice -n 15 python3 -B .../test_counting.py`; each run creates a fresh UUID directory and never adopts old attempts.
