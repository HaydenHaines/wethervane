"""Durable single-attempt Linux command execution; scientific qualification is unassessed.

Only newly prepared explicit commands are launched. There is no scheduler, adoption,
automatic retry, scientific receipt interpreter, checkpoint transition, or publication.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import ledger as ownership_ledger
import run_receipts as ownership_receipts
from ledger import canonical
from run_receipts import decode

_SUPERVISORS = []

class ExecutionError(ValueError):
    pass


def require(value, message):
    if not value:
        raise ExecutionError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def pin(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "regular non-symlink file required: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path.absolute()), sha256=digest.hexdigest(), size=path.stat().st_size)


def checked(item):
    """Read a compact document and authenticate exactly the returned bytes."""
    path = Path(item["path"])
    require(path.is_file() and not path.is_symlink(), "regular document required")
    data = path.read_bytes()
    require(len(data) == item["size"] and hashlib.sha256(data).hexdigest() == item["sha256"], "document fingerprint changed: " + str(path))
    return data


def verify_pin(item):
    require(pin(item["path"]) == item, "artifact fingerprint changed: " + item["path"])


def executable_pin(path):
    return dict(path=str(Path(path).absolute()), target=pin(Path(path).resolve()))


def verify_executable(item):
    require(Path(item["path"]).is_absolute() and str(Path(item["path"]).resolve()) == item["target"]["path"], "executable invocation target changed")
    verify_pin(item["target"])


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def ensure_directory(path):
    path = Path(path)
    if path.is_dir():
        return
    ensure_directory(path.parent)
    try:
        path.mkdir()
    except FileExistsError:
        require(path.is_dir(), "directory path occupied")
    sync_directory(path.parent)


def publish(path, value):
    """Publish one complete immutable record, including its directory entry."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = stream.name
            stream.write(canonical(value).encode())
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        sync_directory(path.parent)
    finally:
        if temporary:
            os.unlink(temporary)


def identifier(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value), "invalid attempt ID")
    return value


def attempt(root, attempt_id):
    return Path(root).resolve() / "attempts" / identifier(attempt_id)


def absolute(value):
    require(isinstance(value, str) and Path(value).is_absolute(), "absolute path required")
    require(str(Path(value).resolve()) == value, "canonical path required")
    return Path(value)


class LiveInputDrift(ExecutionError):pass
class LockBusy(ExecutionError):pass
LOCK_WAIT_SECONDS=.5

def lexical_absolute(value):
    require(isinstance(value,str) and Path(value).is_absolute() and os.path.normpath(value)==value,'canonical lexical path required')
    return Path(value)

def validate(spec):
    _validate(spec,False)
    try:return _validate(spec,True)
    except (OSError,ValueError) as error:raise LiveInputDrift(str(error)) from error

def bounded_flock(fd,seconds=LOCK_WAIT_SECONDS):
    deadline=time.monotonic()+seconds
    while True:
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);return
        except BlockingIOError:
            if time.monotonic()>=deadline:raise LockBusy('bounded ownership/dispatcher lock wait expired') from None
            time.sleep(.005)

def _validate(spec, live):
    require(set(spec) == {
        "schema", "cycle_id", "job_key", "attempt_id", "argv", "executable", "cwd", "environment", "input_pins",
        "output_directory", "native_receipts", "timeout_seconds", "grace_seconds", "previous_attempt", "retry_reason",
    }, "unknown or missing execution specification fields")
    require(spec["schema"] == "LocalExecutionSpecV1", "unsupported execution specification")
    identifier(spec["attempt_id"])
    require(all(isinstance(spec[k], str) and spec[k].strip() for k in ("cycle_id", "job_key")), "cycle/job identity required")
    argv = spec["argv"]
    require(isinstance(argv, list) and argv and all(isinstance(x, str) and "\0" not in x for x in argv), "explicit argv required")
    require(Path(argv[0]).is_absolute() and os.path.normpath(argv[0]) == argv[0], "absolute executable invocation required")
    require(spec["executable"]["path"] == argv[0], "executable invocation pin mismatch")
    if live:verify_executable(spec["executable"])
    require(absolute(spec["cwd"]).is_dir(), "working directory missing") if live else lexical_absolute(spec["cwd"])
    output = absolute(spec["output_directory"]) if live else lexical_absolute(spec["output_directory"])
    env = spec["environment"]
    require(isinstance(env, dict) and all(
        isinstance(k, str) and k and "=" not in k and "\0" not in k and isinstance(v, str) and "\0" not in v
        for k, v in env.items()
    ), "explicit environment mapping required")
    require(isinstance(spec["input_pins"], list) and spec["input_pins"], "input/source/plan pins required")
    for item in spec["input_pins"]:
        absolute(item["path"]) if live else lexical_absolute(item["path"])
        if live:verify_pin(item)
    receipts = spec["native_receipts"]
    require(isinstance(receipts, list) and receipts, "declared native receipts required")
    require(all(isinstance(r, dict) and set(r) == {"path", "required_fields"} and isinstance(r["required_fields"], dict)
                and r["required_fields"] for r in receipts), "explicit native JSON metadata bindings required")
    names = [r["path"] for r in receipts]
    require(len(set(names)) == len(names), "duplicate native receipt")
    for name in names:
        require(isinstance(name, str) and name and not Path(name).is_absolute() and ".." not in Path(name).parts,
                "native receipt must stay inside output directory")
        require((output / name).resolve().is_relative_to(output), "native receipt escapes output") if live else None
    for key in ("timeout_seconds", "grace_seconds"):
        require(type(spec[key]) in (int, float) and math.isfinite(spec[key]) and 0 < spec[key] <= 604800, "invalid execution bound")
    require(isinstance(spec["retry_reason"], str), "retry reason must be a string")
    previous = spec["previous_attempt"]
    require(previous is None or isinstance(previous, dict) and set(previous) == {"attempt_id", "receipt"}, "invalid previous attempt")
    require(bool(previous) == bool(spec["retry_reason"].strip()), "new attempts require an explicit retry reason")
    return spec


def load(root, attempt_id, verify_inputs=False):
    ownership_binding(root)
    directory = attempt(root, attempt_id)
    prepared = decode((directory / "prepared.json").read_bytes())
    require(prepared["schema"]=="LocalExecutionPreparedOwnedIntegratedResearchV1","unsupported preparation schema")
    require(prepared["manager_source"]==pin(Path(__file__).resolve()),"executing manager source differs")
    spec = decode(checked(prepared["specification"]))
    require(spec["attempt_id"] == attempt_id and prepared["attempt_id"] == attempt_id, "attempt identity changed")
    intent_pin=prepared['ownership_intent']
    require(intent_pin['path']==str(Path(root).resolve()/'ownership-intents'/(attempt_id+'.json')),'ownership intent path differs')
    intent=decode(checked(intent_pin))
    require(intent['schema']=='LocalExecutionOwnershipIntentIntegratedResearchV1' and intent['specification']==spec
            and intent['store']==pin(Path(root)/'ownership-store.json') and intent['job_lock']==prepared['job_lock'],'loaded ownership intent differs')
    if verify_inputs:
        validate(spec)
        verify_pin(prepared["manager_source"])
        verify_executable(prepared["manager_executable"])
        for item in prepared["manager_dependencies"]:
            verify_pin(item)
    return directory, spec, prepared


def registered(root, directory, spec, require_leaf=False):
    ownership_binding(root)
    prepared=decode((directory/'prepared.json').read_bytes())
    intent=decode(checked(prepared['ownership_intent']))
    require(prepared['ownership_intent']['path']==str(Path(root).resolve()/'ownership-intents'/(spec['attempt_id']+'.json'))
            and intent['specification']==spec and intent['job_lock']==prepared['job_lock']
            and intent['store']==pin(Path(root)/'ownership-store.json'),'preparation ownership intent differs')
    job = job_directory(root, spec["cycle_id"], spec["job_key"])
    claim = decode((job / ("claim-" + spec["attempt_id"] + ".json")).read_bytes())
    require(claim["prepared"] == pin(directory / "prepared.json"), "attempt preparation registration changed")
    prepared = decode((directory / "prepared.json").read_bytes())
    require(lock_identity(job / "lock") == prepared["job_lock"], "job lock identity changed")
    output = Path(root).resolve() / "outputs" / hashlib.sha256(spec["output_directory"].encode()).hexdigest()
    require(decode(output.read_bytes()) == dict(attempt_id=spec["attempt_id"], output_directory=spec["output_directory"]),
            "output claim registration changed")
    if require_leaf:
        claims = [decode(path.read_bytes()) for path in job.glob("claim-*.json")]
        parents = {c["previous_attempt"] for c in claims if c["previous_attempt"]}
        require({c["attempt_id"] for c in claims} - parents == {spec["attempt_id"]}, "attempt is no longer the job claim leaf")


def job_directory(root, cycle_id, job_key):
    return Path(root).resolve() / "jobs" / hashlib.sha256(canonical([cycle_id, job_key]).encode()).hexdigest()


def lock_identity(path):
    info = Path(path).stat()
    return dict(device=info.st_dev, inode=info.st_ino)


def holds_lock(identity, expected):
    """Require the inherited FLOCK open-file description, not merely its inode."""
    try:
        for path in Path(f"/proc/{identity['pid']}/fdinfo").iterdir():
            target = Path(f"/proc/{identity['pid']}/fd/{path.name}")
            try:
                if lock_identity(target) != expected:
                    continue
                for line in path.read_text().splitlines():
                    match = re.fullmatch(r"lock:\s+\d+:\s+FLOCK\s+ADVISORY\s+WRITE\s+-?\d+\s+([0-9a-fA-F]+):([0-9a-fA-F]+):(\d+)\s+0\s+EOF", line)
                    if match and (os.makedev(int(match[1], 16), int(match[2], 16)), int(match[3])) == (expected["device"], expected["inode"]):
                        return observe(identity)["state"] == "alive"
            except (FileNotFoundError, ProcessLookupError):
                continue
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False
    return False


def lock(root, cycle_id, job_key):
    directory = job_directory(root, cycle_id, job_key)
    ensure_directory(directory)
    fd = os.open(directory / "lock", os.O_CREAT | os.O_RDWR, 0o600)
    sync_directory(directory)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise ExecutionError("job lock held by a cooperating process") from None
    return fd, directory


def process_identity(pid):
    """Linux birth identity; command text is observational, never the birth key."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return dict(pid=pid, boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                    start_ticks=int(fields[19]), session=int(fields[3]), process_group=int(fields[2]), state=fields[0])
    except (FileNotFoundError, ProcessLookupError):
        return None


def observe(identity):
    current = process_identity(identity["pid"])
    if current is None:
        state = "absent"
    elif any(current[k] != identity[k] for k in ("boot_id", "pid", "start_ticks")):
        state = "identity_changed"
    else:
        state = "zombie" if current["state"] == "Z" else "alive"
    return dict(state=state, expected=identity, observed=current)


def session_members(identity):
    if identity["boot_id"] != Path("/proc/sys/kernel/random/boot_id").read_text().strip():
        return []
    members = []
    for path in Path("/proc").iterdir():
        if path.name.isdecimal():
            process = process_identity(int(path.name))
            if process and process["session"] == identity["session"] and process["state"] != "Z":
                members.append(process)
    return members


def inspect(root, attempt_id):
    _SUPERVISORS[:] = [process for process in _SUPERVISORS if process.poll() is None]
    directory, spec, prepared = load(root, attempt_id)
    require(lock_identity(job_directory(root, spec["cycle_id"], spec["job_key"]) / "lock") == prepared["job_lock"], "job lock identity changed")
    observations = {}
    for name in ("supervisor", "child"):
        path = directory / (name + ".json")
        if path.exists():
            observations[name] = observe(decode(path.read_bytes())["identity"])
    try:
        descriptor, _ = lock(root, spec["cycle_id"], spec["job_key"])
    except ExecutionError:
        held = True
    else:
        os.close(descriptor)
        held = False
    members = session_members(observations["child"]["expected"]) if "child" in observations else []
    terminal = None
    terminal_path = next((directory / name for name in ("terminal.json", "recovery.json") if (directory / name).exists()), None)
    if terminal_path:
        terminal = decode(terminal_path.read_bytes())
        require(terminal["specification"] == prepared["specification"], "terminal/specification mismatch")
        validate_terminal(terminal, spec)
        for item in terminal["evidence"]:
            verify_pin(item)
    live = held or members or any(o["state"] == "alive" for o in observations.values())
    state = "terminal" if terminal else "running_or_starting" if live else "orphaned_incomplete" if (directory / "launch-intent.json").exists() else "prepared"
    return dict(schema="LocalExecutionObservationV1", attempt_id=attempt_id, state=state,
                process_observation=observations, job_lock_held=held, session_members=members,
                attempt_terminal=terminal, terminal_receipt=pin(terminal_path) if terminal_path else None,
                qualification="unassessed", recorded_at=now())


def ownership_sources():
    directory=Path(__file__).resolve().parent
    require(Path(ownership_ledger.__file__).resolve()==directory/'ledger.py' and
            Path(ownership_receipts.__file__).resolve()==directory/'run_receipts.py','mixed ownership source imports')
    return [pin(directory/name) for name in ('execution.py','ledger.py','run_receipts.py')]


def ownership_binding(root,create=False):
    root=Path(root).resolve();path=root/'ownership-store.json'
    expected=dict(schema='LocalExecutionOwnershipIntegratedResearchV1',root=str(root),sources=ownership_sources(),domain=domain_pin(root))
    if path.exists():require(decode(path.read_bytes())==expected,'ownership store source/root drift')
    else:
        require(create and {p.name for p in root.iterdir()} <= {'ownership.lock'},'old/missing ownership store; no adoption')
        publish(path,expected)
    return expected


def ownership_crash(point):
    if os.environ.get('OWNERSHIP_CRASH')==point:os._exit(93)


def overlap(left,right):
    a,b=Path(left),Path(right)
    return a.is_relative_to(b) or b.is_relative_to(a)


def ownership_intents(root):
    directory=Path(root)/'ownership-intents';values=[]
    if directory.exists():
        for path in sorted(directory.iterdir()):
            require(path.is_file() and not path.is_symlink() and path.suffix=='.json','unknown ownership intent entry')
            value=decode(path.read_bytes());spec=value['specification']
            require(value['schema']=='LocalExecutionOwnershipIntentIntegratedResearchV1' and value['store']==pin(Path(root)/'ownership-store.json')
                    and path.name==identifier(spec['attempt_id'])+'.json','ownership intent binding differs')
            _validate(spec,False)
            values.append((path,value))
    for i,(a,x) in enumerate(values):
        for b,y in values[i+1:]:
            require(not overlap(x['specification']['output_directory'],y['specification']['output_directory']),'stored overlapping ownership intents')
    known={v['specification']['attempt_id']:v for p,v in values}
    attempts=Path(root)/'attempts'
    if attempts.exists():
        require(all(p.is_dir() and not p.is_symlink() and p.name in known for p in attempts.iterdir()),'attempt without durable ownership intent')
    outputs=Path(root)/'outputs'
    by_output={hashlib.sha256(v['specification']['output_directory'].encode()).hexdigest():v['specification']for p,v in values}
    if outputs.exists():
        for path in outputs.iterdir():
            require(path.is_file() and not path.is_symlink() and path.name in by_output,'output claim without durable ownership intent')
            spec=by_output[path.name]
            require(decode(path.read_bytes())==dict(attempt_id=spec['attempt_id'],output_directory=spec['output_directory']),'output claim drift')
    jobs=Path(root)/'jobs'
    if jobs.exists():
        for path in jobs.glob('*/claim-*.json'):
            value=decode(path.read_bytes());require(value['attempt_id']in known,'job claim without durable ownership intent')
            spec=known[value['attempt_id']]['specification']
            require(path.parent==job_directory(root,spec['cycle_id'],spec['job_key']) and value['prepared']==pin(attempt(root,spec['attempt_id'])/'prepared.json'),'job ownership claim drift')
    return values


def publish_exact(path,value):
    if Path(path).exists():
        require(decode(Path(path).read_bytes())==value,'partial ownership scaffold drift')
        with Path(path).open('rb')as stream:os.fsync(stream.fileno())
        sync_directory(Path(path).parent)
    else:publish(path,value)


def prepare(root, spec, expected_spec_sha256):
    assert_manager(root,spec)
    ownership_sources()
    validate(spec)
    require(hashlib.sha256(canonical(spec).encode()).hexdigest()==expected_spec_sha256,'specification digest mismatch')
    root=Path(root).resolve()
    if root.exists():
        if (root/'ownership-store.json').exists():ownership_binding(root)
        else:require({p.name for p in root.iterdir()} <= {'ownership.lock'},'old/missing ownership store; no adoption')
    ensure_directory(root)
    global_fd=os.open(root/'ownership.lock',os.O_CREAT|os.O_RDWR,0o600)
    try:
        bounded_flock(global_fd)
        ownership_binding(root,True)
        intents=ownership_intents(root)
        existing=next(((p,v)for p,v in intents if v['specification']['attempt_id']==spec['attempt_id']),None)
        if existing:require(existing[1]['specification']==spec,'attempt ID reused with changed ownership specification')
        else:
            require(all(not overlap(spec['output_directory'],v['specification']['output_directory'])for p,v in intents),'owner output containment already claimed')
            for path,value in intents:
                other=value['specification']
                if (other['cycle_id'],other['job_key'])==(spec['cycle_id'],spec['job_key']):
                    require((job_directory(root,other['cycle_id'],other['job_key'])/('claim-'+other['attempt_id']+'.json')).is_file(),'pending intent owns job; exact repair required')
        ensure_directory(root/'attempts');directory=attempt(root,spec['attempt_id'])
        # A finished exact replay is safe while the job's inherited lock is live.
        if existing and (directory/'prepared.json').exists() and (job_directory(root,spec['cycle_id'],spec['job_key'])/('claim-'+spec['attempt_id']+'.json')).exists():
            _,old,_=load(root,spec['attempt_id']);require(old==spec,'attempt specification differs');registered(root,directory,spec);return pin(directory/'prepared.json')
        descriptor,job=lock(root,spec['cycle_id'],spec['job_key'])
        try:
            if existing:
                intent_path,value=existing
                require(value['job_lock']==lock_identity(job/'lock'),'interrupted ownership lock identity changed')
            else:
                claims = [decode(path.read_bytes()) for path in job.glob("claim-*.json")]
                previous = spec["previous_attempt"]
                if claims:
                    parents = {c["previous_attempt"] for c in claims if c["previous_attempt"]}
                    leaves = {c["attempt_id"] for c in claims} - parents
                    require(previous is not None and leaves == {previous["attempt_id"]}, "retry must bind the last retained job attempt")
                    _, prior_spec, _ = load(root, previous["attempt_id"])
                    require((prior_spec["cycle_id"], prior_spec["job_key"]) == (spec["cycle_id"], spec["job_key"]), "previous attempt scope mismatch")
                    status = inspect(root, previous["attempt_id"])
                    # Our own job lock is held here; inspect reports that lock, so use the
                    # independently recorded process/session observations for this check.
                    require(not status["session_members"] and all(o["state"] != "alive" for o in status["process_observation"].values()), "previous process remains alive")
                    require(status["terminal_receipt"] == previous["receipt"], "previous terminal/recovery receipt required")
                    require(status["attempt_terminal"] is not None, "previous attempt outcome is unresolved")
                else:
                    require(previous is None, "previous attempt belongs to a different job")
                output=Path(spec['output_directory'])
                if output.exists():raise LiveInputDrift('owner output directory must be fresh')
                require(not directory.exists(),'unclaimed attempt directory; no adoption')
                ensure_directory(root/'ownership-intents');intent_path=root/'ownership-intents'/(spec['attempt_id']+'.json')
                value=dict(schema='LocalExecutionOwnershipIntentIntegratedResearchV1',store=pin(root/'ownership-store.json'),
                           specification=spec,job_lock=lock_identity(job/'lock'),recorded_at=now())
                ownership_crash('before_intent');publish(intent_path,value);ownership_crash('after_intent')
            # No launch can start until the final job claim is published.
            require(not (directory/'launch-intent.json').exists(),'partial repair cannot adopt launched execution')
            require(not Path(spec['output_directory']).exists(),'partial repair output must remain fresh')
            ensure_directory(directory);ownership_crash('after_directory')
            publish_exact(directory/'specification.json',spec);ownership_crash('after_specification')
            prepared=dict(schema='LocalExecutionPreparedOwnedIntegratedResearchV1',attempt_id=spec['attempt_id'],
                          specification=pin(directory/'specification.json'),manager_source=pin(Path(__file__).resolve()),
                          manager_executable=executable_pin(sys.executable),job_lock=value['job_lock'],
                          manager_dependencies=[pin(Path(__file__).with_name(name))for name in ('ledger.py','run_receipts.py')],
                          ownership_intent=pin(intent_path),recorded_at=value['recorded_at'])
            publish_exact(directory/'prepared.json',prepared);ownership_crash('after_prepared')
            ensure_directory(root/'outputs');claim=root/'outputs'/hashlib.sha256(spec['output_directory'].encode()).hexdigest()
            publish_exact(claim,dict(attempt_id=spec['attempt_id'],output_directory=spec['output_directory']));ownership_crash('after_output_claim')
            previous=spec['previous_attempt']
            publish_exact(job/('claim-'+spec['attempt_id']+'.json'),dict(attempt_id=spec['attempt_id'],previous_attempt=previous['attempt_id']if previous else None,prepared=pin(directory/'prepared.json')))
            ownership_crash('after_job_claim');registered(root,directory,spec)
            return pin(directory/'prepared.json')
        finally:os.close(descriptor)
    finally:os.close(global_fd)


def launch(root, attempt_id):
    directory, spec, prepared = load(root, attempt_id, verify_inputs=True)
    assert_manager(root,spec)
    registered(root, directory, spec)
    if (directory / "launch-intent.json").exists():
        return inspect(root, attempt_id)
    descriptor, _ = lock(root, spec["cycle_id"], spec["job_key"])
    try:
        if (directory / "launch-intent.json").exists():
            return inspect(root, attempt_id)
        registered(root, directory, spec, require_leaf=True)
        require(not Path(spec["output_directory"]).exists(), "owner output directory must be fresh")
        publish(directory / "launch-intent.json", dict(schema="LocalExecutionLaunchIntentV1", specification=prepared["specification"], recorded_at=now()))
        with (directory / "supervisor.log").open("xb") as log:
            process = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "_supervise", str(Path(root).resolve()), attempt_id, str(descriptor)],
                             cwd=spec["cwd"], env={}, stdout=log, stderr=subprocess.STDOUT,
                             pass_fds=(descriptor,), start_new_session=True)
            _SUPERVISORS.append(process)
    finally:
        # Closing, rather than LOCK_UN, leaves the inherited flock with the worker.
        os.close(descriptor)
    return inspect(root, attempt_id)


def collect(directory, spec, prepared, return_code, timed_out, cleanup, error=None):
    outputs, missing, invalid = [], [], []
    for declaration in spec["native_receipts"]:
        name = declaration["path"]
        path = Path(spec["output_directory"]) / name
        if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(Path(spec["output_directory"])):
            item = pin(path)
            outputs.append(item)
            try:
                document = decode(checked(item))
                require(isinstance(document, dict) and all(k in document and canonical(document[k]) == canonical(v)
                        for k, v in declaration["required_fields"].items()), "native metadata binding mismatch")
            except (ValueError, TypeError) as exc:
                invalid.append(dict(path=name, error=str(exc)))
        else:
            missing.append(name)
    names = ("prepared.json", "launch-intent.json", "supervisor.json", "child.json", "release.json", "exec-intent.json", "command.log", "supervisor-error.json")
    lifecycle = [pin(directory / name) for name in names if (directory / name).is_file()]
    lifecycle.extend(pin(path) for path in sorted(directory.glob("stop-*.json")))
    return dict(schema="LocalExecutionTerminalV1", attempt_id=spec["attempt_id"], specification=prepared["specification"],
                process_outcome="exit_unknown" if return_code is None else "returned", return_code=return_code,
                timed_out=timed_out, native_outputs="invalid" if invalid else "incomplete" if missing else "metadata_verified_science_unassessed",
                native_receipts=outputs, missing_native_receipts=missing, invalid_native_receipts=invalid, cleanup=cleanup,
                evidence=[prepared["specification"], *lifecycle, *outputs], error=error,
                qualification="unassessed", recorded_at=now())


def validate_terminal(value, spec):
    require(set(value) == {"schema", "attempt_id", "specification", "process_outcome", "return_code", "timed_out",
            "native_outputs", "native_receipts", "missing_native_receipts", "invalid_native_receipts", "cleanup",
            "evidence", "error", "qualification", "recorded_at"}, "malformed execution terminal fields")
    require(value["schema"] == "LocalExecutionTerminalV1" and value["attempt_id"] == spec["attempt_id"]
            and value["qualification"] == "unassessed", "malformed execution terminal identity/qualification")
    code = value["return_code"]
    require(code is None or type(code) is int, "invalid return code")
    require(value["process_outcome"] == ("exit_unknown" if code is None else "returned") and type(value["timed_out"]) is bool,
            "invalid execution terminal outcome")
    require(value["native_outputs"] in ("invalid", "incomplete", "metadata_verified_science_unassessed"), "invalid native output status")
    require(isinstance(value["cleanup"], dict) and type(value["cleanup"].get("verified")) is bool, "invalid cleanup receipt")
    require(value["cleanup"]["verified"] == (value["cleanup"].get("remaining") == []), "contradictory cleanup receipt")
    require(isinstance(value["evidence"], list) and value["specification"] in value["evidence"], "missing terminal evidence")
    require(all(item in value["evidence"] for item in value["native_receipts"]), "native evidence missing")
    declared = {r["path"] for r in spec["native_receipts"]}
    output = Path(spec["output_directory"])
    present = []
    for item in value["native_receipts"]:
        require(Path(item["path"]).is_relative_to(output), "undeclared native receipt path")
        present.append(Path(item["path"]).relative_to(output).as_posix())
    missing = value["missing_native_receipts"]
    invalid = [item["path"] for item in value["invalid_native_receipts"]]
    require(len(set(present)) == len(present) and len(set(missing)) == len(missing)
            and len(set(invalid)) == len(invalid) and not set(present) & set(missing)
            and set(present) | set(missing) == declared and set(invalid) <= set(present), "contradictory native receipt inventory")
    require(value["native_outputs"] == ("invalid" if invalid else "incomplete" if missing else "metadata_verified_science_unassessed"),
            "contradictory native output status")


def supervise(root, attempt_id, descriptor):
    directory, spec, prepared = load(root, attempt_id, verify_inputs=True)
    publish(directory / "supervisor.json", dict(identity=process_identity(os.getpid()), recorded_at=now()))
    code, timed_out, error, child = None, False, None, None
    read_gate, write_gate = os.pipe()
    try:
        require(not Path(spec["output_directory"]).exists(), "owner output directory must be fresh")
        with (directory / "command.log").open("xb") as log:
            child = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "_child", str(root), attempt_id, str(descriptor), str(read_gate)],
                                     cwd=spec["cwd"], env={}, stdout=log, stderr=subprocess.STDOUT,
                                     pass_fds=(descriptor, read_gate), start_new_session=True)
            os.close(read_gate)
            read_gate = None
            deadline = time.monotonic() + 10
            while not (directory / "child.json").exists():
                require(time.monotonic() < deadline, "child identity publication timed out")
                time.sleep(0.01)
            identity = decode((directory / "child.json").read_bytes())["identity"]
            require(identity["pid"] == child.pid and observe(identity)["state"] == "alive", "child identity mismatch")
            publish(directory / "release.json", dict(child=pin(directory / "child.json"), specification=prepared["specification"], recorded_at=now()))
            os.write(write_gate, b"G")
            os.close(write_gate)
            write_gate = None
            try:
                code = child.wait(timeout=spec["timeout_seconds"])
            except subprocess.TimeoutExpired:
                timed_out = True
                # This is our unreaped session leader. Its numeric PID cannot be
                # reused until wait(), including if it becomes a zombie here.
                for signum, delay in ((signal.SIGTERM, spec["grace_seconds"]), (signal.SIGKILL, 0)):
                    try:
                        os.killpg(child.pid, signum)
                    except ProcessLookupError:
                        pass
                    if delay:
                        time.sleep(delay)
                code = child.wait(timeout=10)
            log.flush()
            os.fsync(log.fileno())
    except BaseException as exc:
        error = type(exc).__name__ + ": " + str(exc)
        # Before release, closing the pipe makes the helper exit without exec.
        if write_gate is not None:
            os.close(write_gate)
            write_gate = None
        if child is not None and code is None:
            try:
                code = child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                error += "; command outcome unknown; no recovery signal sent"
    finally:
        for fd in (read_gate, write_gate):
            if fd is not None:
                os.close(fd)
    child_path = directory / "child.json"
    remaining = session_members(decode(child_path.read_bytes())["identity"]) if child_path.exists() else []
    cleanup = dict(scope="owned_process_session_without_daemonization", verified=not remaining, remaining=remaining)
    if remaining:
        publish(directory / "supervisor-error.json", dict(error=error, return_code=code, timed_out=timed_out, cleanup=cleanup,
                specification=prepared["specification"], child=pin(child_path), recorded_at=now()))
    else:
        publish(directory / "terminal.json", collect(directory, spec, prepared, code, timed_out, cleanup, error))
    os.close(descriptor)


def child_main(root, attempt_id, descriptor, read_gate):
    directory, spec, prepared = load(root, attempt_id, verify_inputs=True)
    publish(directory / "child.json", dict(identity=process_identity(os.getpid()), specification=prepared["specification"], recorded_at=now()))
    go = os.read(read_gate, 1)
    os.close(read_gate)
    if go != b"G":
        return
    publish(directory / "exec-intent.json", dict(specification=prepared["specification"], recorded_at=now()))
    os.set_inheritable(descriptor, True)
    os.execve(spec["argv"][0], spec["argv"], spec["environment"])


def recover(root, attempt_id):
    """Seal an explicitly unknown exit after proving cooperating processes absent."""
    directory, spec, prepared = load(root, attempt_id)
    descriptor, _ = lock(root, spec["cycle_id"], spec["job_key"])
    try:
        state = inspect(root, attempt_id)
        if state["attempt_terminal"]:
            return state["terminal_receipt"]
        require((directory / "launch-intent.json").exists(), "unlaunched attempt requires no recovery")
        require(not state["session_members"] and all(o["state"] != "alive" for o in state["process_observation"].values()), "owned process remains alive")
        code, timed_out = None, False
        error_path = directory / "supervisor-error.json"
        if error_path.is_file():
            observed = decode(error_path.read_bytes())
            require(observed["specification"] == prepared["specification"] and observed["child"] == pin(directory / "child.json"), "supervisor wait observation identity mismatch")
            code, timed_out = observed["return_code"], observed["timed_out"]
        publish(directory / "recovery.json", collect(directory, spec, prepared, code, timed_out,
                dict(scope="observed_owned_session_and_cooperating_lock", verified=True, remaining=[]),
                "Preserved supervisor wait observation" if code is not None else "No wait status survived; absence does not establish process success or failure"))
        return pin(directory / "recovery.json")
    finally:
        os.close(descriptor)


def open_pidfd(identity):
    require(observe(identity)["state"] == "alive", "original process identity is not alive")
    descriptor = os.pidfd_open(identity["pid"])
    current = observe(identity)
    if current["state"] != "alive" or current["observed"]["session"] != identity["session"]:
        os.close(descriptor)
        raise ExecutionError("process identity changed during pidfd acquisition")
    return descriptor


def wait_stopped(identity, deadline):
    while time.monotonic() < deadline:
        current = observe(identity)
        if current["state"] != "alive" or current["observed"]["state"] in ("T", "t"):
            return
        time.sleep(0.005)
    raise ExecutionError("owned process did not confirm stopped state")


def stop(root, attempt_id):
    directory = attempt(root, attempt_id)
    descriptor = os.open(directory / "stop.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ExecutionError("another stop request owns this attempt") from None
        return stop_owned(root, attempt_id)
    finally:
        os.close(descriptor)


def stop_owned(root, attempt_id):
    """Explicit SIGKILL of this manager's observed session through verified pidfds.

    Used only after supervisor loss. Detached/daemonized descendants are unsupported.
    No process group is signaled after recovery; PID reuse cannot redirect pidfds.
    """
    directory, _, prepared = load(root, attempt_id)
    state = inspect(root, attempt_id)
    require(state["process_observation"].get("supervisor", {}).get("state") != "alive", "supervisor still owns execution")
    require("child" in state["process_observation"], "no recorded owned child identity")
    identity = state["process_observation"]["child"]["expected"]
    anchor_kind = "original_child_birth"
    leader = None
    if observe(identity)["state"] != "alive":
        error_path = directory / "supervisor-error.json"
        candidates = []
        if error_path.is_file():
            error = decode(error_path.read_bytes())
            require(error["specification"] == prepared["specification"], "descendant observation specification mismatch")
            require(error["child"] == pin(directory / "child.json"), "descendant observation child mismatch")
            candidates = [member for member in error["cleanup"]["remaining"] if member["session"] == identity["session"]
                          and member["boot_id"] == identity["boot_id"] and observe(member)["state"] == "alive"]
        if candidates:
            identity, anchor_kind = candidates[0], "saved_descendant_birth"
        else:
            for member in session_members(identity):
                if not holds_lock(member, prepared["job_lock"]):
                    continue
                try:
                    descriptor = open_pidfd(member)
                except (OSError, ExecutionError):
                    continue
                if holds_lock(member, prepared["job_lock"]):
                    identity, leader, anchor_kind = member, descriptor, "inherited_flock_verified"
                    break
                os.close(descriptor)
            require(leader is not None, "no recorded birth or inherited-lock anchor; ownership unresolved")
    if leader is None:
        leader = open_pidfd(identity)
    targets = [(leader, identity)]
    actions = {leader: dict(identity=identity, ownership=anchor_kind, sigstop="not_sent", sigkill="not_sent")}
    stable, error = False, None
    try:
        signal.pidfd_send_signal(leader, signal.SIGSTOP)
        actions[leader]["sigstop"] = "delivered"
        deadline = time.monotonic() + 3
        wait_stopped(identity, deadline)
        # Freeze each observed descendant before killing. Only pidfd handles are
        # signaled; a later reused PID cannot redirect a signal to another process.
        for _ in range(16):
            known = {(p["pid"], p["start_ticks"]) for _, p in targets}
            added = False
            for member in session_members(identity):
                if (member["pid"], member["start_ticks"]) in known:
                    continue
                try:
                    descriptor = open_pidfd(member)
                except (ProcessLookupError, ExecutionError):
                    continue
                if observe(identity)["state"] != "alive":
                    os.close(descriptor)
                    raise ExecutionError("original session leader disappeared during stop")
                targets.append((descriptor, member))
                actions[descriptor] = dict(identity=member, ownership="session_while_verified_anchor_frozen", sigstop="not_sent", sigkill="not_sent")
                try:
                    signal.pidfd_send_signal(descriptor, signal.SIGSTOP)
                    actions[descriptor]["sigstop"] = "delivered"
                except ProcessLookupError:
                    actions[descriptor]["sigstop"] = "already_gone"
                wait_stopped(member, deadline)
                added = True
            if not added:
                stable = True
                break
        require(stable, "owned session did not stabilize while stopped")
    except (OSError, ExecutionError) as exc:
        error = str(exc)
    finally:
        for descriptor, _ in reversed(targets):
            actions[descriptor]["before_kill"] = observe(actions[descriptor]["identity"])
            try:
                signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                actions[descriptor]["sigkill"] = "delivered"
            except ProcessLookupError:
                actions[descriptor]["sigkill"] = "already_gone"
        try:
            publish(directory / ("stop-" + str(time.time_ns()) + ".json"), dict(schema="LocalExecutionStopObservationV1",
                    targets=list(actions.values()), stopped_session_stable=stable,
                    error=error, recorded_at=now(), qualification="unassessed"))
        finally:
            for descriptor, _ in targets:
                os.close(descriptor)
    return inspect(root, attempt_id)


def attach(ledger, command_id, root, attempt_id):
    """Copy compact immutable execution evidence only; never append a checkpoint."""
    state = inspect(root, attempt_id)
    require(state["terminal_receipt"] is not None, "terminal execution evidence required")
    directory, _, prepared = load(root, attempt_id)
    # Native files stay in owner storage. The terminal pins their verified bytes.
    records = [prepared["specification"], pin(directory / "prepared.json"), state["terminal_receipt"]]
    with tempfile.TemporaryDirectory() as temporary:
        staged = []
        for index, item in enumerate(records):
            path = Path(temporary) / str(index)
            path.write_bytes(checked(item))
            staged.append((path, item["sha256"]))
        result = None
        for path, digest in staged:
            result = ledger.put_artifact(command_id + ":" + digest, path)
        return result


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("_supervise", "_child"):
        operation, root, attempt_id, descriptor, *rest = sys.argv[1:]
        if operation == "_supervise":
            supervise(root, attempt_id, int(descriptor))
        else:
            child_main(root, attempt_id, int(descriptor), int(rest[0]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    sub = parser.add_subparsers(dest="operation", required=True)
    preparation = sub.add_parser("prepare")
    preparation.add_argument("--specification", required=True, type=Path)
    preparation.add_argument("--sha256", required=True)
    for name in ("launch", "inspect", "recover", "stop"):
        sub.add_parser(name).add_argument("attempt_id")
    args = parser.parse_args()
    if args.operation == "prepare":
        value = prepare(args.root, decode(args.specification.read_bytes()), args.sha256)
    else:
        value = globals()[args.operation](args.root, args.attempt_id)
    print(canonical(value))





def domain_pin(root):
    raw=Path(os.path.abspath(root));raw_store=raw.parent/'store.json'
    if raw.name=='executions' and raw_store.is_file() and decode(raw_store.read_bytes()).get('schema')=='LocalExperimentQueueStoreFiniteOwnedResearchV1':
        require(raw.resolve()==raw,'managed execution root cannot be a symlink or namespace alias')
    root=raw.resolve();path=root.parent/'execution-domain.json'
    queue_store=root.parent/'store.json'
    recognized=(root.name=='executions' and queue_store.is_file() and decode(queue_store.read_bytes()).get('schema')=='LocalExperimentQueueStoreFiniteOwnedResearchV1')
    if recognized:require(path.is_file(),'managed domain marker missing')
    if not path.exists():return None
    value=decode(path.read_bytes())
    if value['execution_root']!=str(root):
        require(not recognized,'managed domain execution-root mismatch')
        return None
    require(value['schema']=='QueueOwnedExecutionDomainResearchV1' and value['queue_root']==str(root.parent),'invalid managed domain')
    verify_pin(value['queue_store']);require(value['queue_store']['path']==str(root.parent/'store.json'),'domain queue-store path mismatch')
    require(value['dispatcher']['path']==str(root.parent/'dispatcher.lock') and lock_identity(value['dispatcher']['path'])==value['dispatcher']['identity'],'domain dispatcher identity changed')
    return pin(path)

def assert_manager(root,spec=None):
    item=domain_pin(root)
    if item:
        domain=decode(checked(item));require(holds_lock(process_identity(os.getpid()),domain['dispatcher']['identity']),'managed execution requires held dispatcher flock')
        if spec is not None:
            admitted=managed_admissions(domain)
            matched=False
            for reservation in Path(domain['queue_root']).glob('queues/*/jobs/*/reservation.json'):
                if(reservation.parent/'release.json').exists():continue
                value=decode(reservation.read_bytes());binding=decode(checked(value['binding']))
                require(value['binding']['path']==str(reservation.parent/'binding.json'),'managed binding path differs')
                if binding['execution_spec']==spec:
                    plan=decode(checked(binding['plan']))
                    require(plan['queue_id']in admitted and plan==admitted[plan['queue_id']],'managed admission is not accepted')
                    require(binding['plan']['path']==str(reservation.parents[2]/'plan.json') and binding['job_id']==reservation.parent.name
                            and any(j['id']==binding['job_id']and j['execution_spec']['attempt_id']==spec['attempt_id']for j in plan['jobs'])
                            and binding['specification_sha256']==hashlib.sha256(canonical(spec).encode()).hexdigest(),'managed effective spec reservation differs')
                    matched=True
            require(matched,'managed execution requires matching unreleased queue reservation')

def managed_admissions(domain):
    """Bounded read-only lifecycle check, before O; requires caller's D."""
    root=Path(domain['queue_root']);store=domain['queue_store']
    require(decode((root/'intake-created.json').read_bytes())==dict(schema='FiniteIntakeCreatedOwnedV1',store=store),'managed lifecycle marker differs')
    db=sqlite3.connect((root/'intake.sqlite').as_uri()+'?mode=ro',uri=True,timeout=.5,isolation_level=None)
    try:
        db.execute('BEGIN')
        binding=db.execute('SELECT value FROM binding').fetchall()
        require(len(binding)==1 and decode(binding[0][0].encode())==dict(schema='FiniteQueueIntakeOwnedV1',root=str(root),store=store),'managed lifecycle store binding differs')
        run=db.execute('SELECT state,seal FROM run').fetchall();require(len(run)==1 and run[0][0]in ('OPEN','SEALED'),'managed lifecycle is not runnable')
        rows=db.execute('SELECT id,plan,state,receipt FROM intents ORDER BY id').fetchall();plans={};sealed=[]
        require(all(p.is_dir()and p.name in {r[0]for r in rows}for p in (root/'queues').iterdir()),'managed unknown admission directory')
        for key,body,state,receipt in rows:
            require(state=='ACCEPTED','managed pending admission blocks execution')
            plan=decode(body.encode());directory=root/'queues'/key
            require(plan['queue_id']==key and decode((directory/'plan.json').read_bytes())==plan,'managed accepted plan differs')
            item=decode(receipt.encode());require(item['path']==str(directory/'receipt.json'),'managed receipt path differs');value=decode(checked(item))
            require(value['schema']=='LocalExperimentQueueAdmissionV1' and value['qualification']=='unassessed' and value['plan']==pin(directory/'plan.json')and value['store']==store,'managed accepted receipt differs')
            require((directory/'jobs').is_dir()and {p.name for p in (directory/'jobs').iterdir()}=={j['id']for j in plan['jobs']},'managed accepted scaffold differs')
            plans[key]=plan;sealed.append(dict(queue_id=key,plan_sha256=hashlib.sha256(canonical(plan).encode()).hexdigest()))
        if run[0][0]=='SEALED':require(run[0][1]==hashlib.sha256(canonical(sealed).encode()).hexdigest(),'managed sealed set changed')
        return plans
    finally:db.close()

def initialize_domain(queue_root):
    root=Path(queue_root).resolve();lockpath=root/'dispatcher.lock'
    require(holds_lock(process_identity(os.getpid()),lock_identity(lockpath)),'initialize domain requires dispatcher flock')
    publish_exact(root/'execution-domain.json',dict(schema='QueueOwnedExecutionDomainResearchV1',queue_root=str(root),execution_root=str(root/'executions'),queue_store=pin(root/'store.json'),dispatcher=dict(path=str(lockpath),identity=lock_identity(lockpath))))

def ownership_audit(root, expected):
    """Read-only structural audit. expected maps id to effective spec + required flag."""
    root=Path(root).resolve();assert_manager(root);domain=domain_pin(root)
    evidence=[domain]if domain else[]
    if not root.exists():
        require(not any(v['required']for v in expected.values()),'previously acknowledged execution store vanished')
        return dict(evidence=evidence,historical_input_drift={})
    ownership_binding(root)
    fd=os.open(root/'ownership.lock',os.O_RDWR)
    try:
        bounded_flock(fd);values=ownership_intents(root);actual={v['specification']['attempt_id']:(p,v)for p,v in values}
        require(set(actual)<=set(expected),'unknown external execution intent in queue domain')
        require(all(not v['required']or k in actual for k,v in expected.items()),'acknowledged execution intent missing')
        evidence.append(pin(root/'ownership-store.json'));drift={}
        for name,(path,value)in actual.items():
            spec=value['specification'];require(spec==expected[name]['spec'],'ownership intent differs from effective queue binding')
            evidence.append(pin(path));directory=attempt(root,name)
            for filename in ('specification.json','prepared.json'):
                if(directory/filename).exists():evidence.append(pin(directory/filename))
            jobclaim=job_directory(root,spec['cycle_id'],spec['job_key'])/('claim-'+name+'.json')
            outputclaim=root/'outputs'/hashlib.sha256(spec['output_directory'].encode()).hexdigest()
            if jobclaim.exists():
                registered(root,directory,spec);evidence.extend([pin(jobclaim),pin(outputclaim)])
            else:require(not expected[name]['required'],'acknowledged preparation has incomplete claims')
            try:validate(spec)
            except LiveInputDrift as error:drift[name]=str(error)
        return dict(evidence=evidence,historical_input_drift=drift)
    finally:os.close(fd)

if __name__ == "__main__":
    main()
