"""Portable, read-only evidence archive for completed local execution attempts.

This copies authenticated bytes. It never restores an execution store, adopts a
process, runs archived code, or changes scientific qualification.
"""

import argparse
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path

import execution
from ledger import canonical
from run_receipts import decode


class ArchiveError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ArchiveError(message)


def _sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _pin_shape(value):
    require(type(value) is dict and set(value) == {"path", "sha256", "size"}, "invalid pin fields")
    require(type(value["path"]) is str and value["path"].startswith("/")
            and os.path.normpath(value["path"]) == value["path"], "invalid original absolute path")
    require(type(value["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]), "invalid pin digest")
    require(type(value["size"]) is int and value["size"] >= 0, "invalid pin size")
    return value


def _add_pin(pins, item):
    item = _pin_shape(item)
    prior = pins.get(item["path"])
    require(prior is None or prior == item, "conflicting pin for original path: " + item["path"])
    pins[item["path"]] = item


def _check_executable(value, pins):
    require(type(value) is dict and set(value) == {"path", "target"}, "invalid executable pin")
    require(type(value["path"]) is str and value["path"].startswith("/") and
            os.path.normpath(value["path"]) == value["path"], "invalid executable invocation path")
    _add_pin(pins, value["target"])


def _read_document(item, objects, limit=16 * 1024 * 1024):
    _pin_shape(item)
    require(item["size"] <= limit, "compact JSON document exceeds archive limit")
    data = objects(item)
    require(len(data) == item["size"] and _sha_bytes(data) == item["sha256"], "JSON object pin mismatch")
    return decode(data)


def _validate_spec(spec, pins):
    fields = {"schema", "cycle_id", "job_key", "attempt_id", "argv", "executable", "cwd", "environment",
              "input_pins", "output_directory", "native_receipts", "timeout_seconds", "grace_seconds",
              "previous_attempt", "retry_reason"}
    require(type(spec) is dict and set(spec) == fields and spec["schema"] == "LocalExecutionSpecV1", "invalid specification")
    require(type(spec["attempt_id"]) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", spec["attempt_id"]),
            "invalid attempt ID")
    require(all(type(spec[key]) is str and spec[key].strip() for key in ("cycle_id", "job_key")), "invalid job scope")
    require(type(spec["argv"]) is list and spec["argv"] and all(type(x) is str and "\0" not in x for x in spec["argv"]),
            "invalid argv")
    require(spec["argv"][0].startswith("/") and os.path.normpath(spec["argv"][0]) == spec["argv"][0],
            "invalid invoked executable path")
    _check_executable(spec["executable"], pins)
    require(spec["executable"]["path"] == spec["argv"][0], "executable invocation mismatch")
    for key in ("cwd", "output_directory"):
        require(type(spec[key]) is str and spec[key].startswith("/") and os.path.normpath(spec[key]) == spec[key],
                "invalid original " + key)
    require(type(spec["environment"]) is dict and all(type(k) is str and k and "=" not in k and "\0" not in k
            and type(v) is str and "\0" not in v for k, v in spec["environment"].items()), "invalid environment")
    require(type(spec["input_pins"]) is list and spec["input_pins"], "missing input pins")
    for item in spec["input_pins"]:
        _add_pin(pins, item)
    receipts = spec["native_receipts"]
    require(type(receipts) is list and receipts, "missing native receipt declarations")
    names = []
    for item in receipts:
        require(type(item) is dict and set(item) == {"path", "required_fields"} and
                type(item["required_fields"]) is dict and item["required_fields"], "invalid native declaration")
        name = item["path"]
        require(type(name) is str and name and not name.startswith("/") and
                ".." not in Path(name).parts and os.path.normpath(name) == name,
                "invalid native receipt relative path")
        names.append(name)
    require(len(set(names)) == len(names), "duplicate native receipt declaration")
    for key in ("timeout_seconds", "grace_seconds"):
        value = spec[key]
        require(type(value) in (int, float) and math.isfinite(value) and 0 < value <= 604800, "invalid execution bound")
    previous = spec["previous_attempt"]
    require(previous is None or type(previous) is dict and set(previous) == {"attempt_id", "receipt"}, "invalid predecessor")
    require(type(spec["retry_reason"]) is str and bool(previous) == bool(spec["retry_reason"].strip()), "invalid retry reason")
    if previous:
        require(type(previous["attempt_id"]) is str and
                re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", previous["attempt_id"]),
                "invalid predecessor ID")
        _add_pin(pins, previous["receipt"])


def _validate_prepared(prepared, record, pins):
    require(type(prepared) is dict and set(prepared) == {"schema", "attempt_id", "specification", "manager_source",
            "manager_executable", "job_lock", "manager_dependencies", "recorded_at"}, "invalid prepared record")
    require(prepared["schema"] == "LocalExecutionPreparedV1" and prepared["attempt_id"] == record["attempt_id"]
            and prepared["specification"] == record["specification"], "prepared binding mismatch")
    lock = prepared["job_lock"]
    require(type(lock) is dict and set(lock) == {"device", "inode"} and
            all(type(lock[key]) is int and lock[key] >= 0 for key in ("device", "inode")), "invalid original lock metadata")
    _add_pin(pins, prepared["manager_source"])
    _check_executable(prepared["manager_executable"], pins)
    require(type(prepared["manager_dependencies"]) is list and prepared["manager_dependencies"], "missing manager dependencies")
    for item in prepared["manager_dependencies"]:
        _add_pin(pins, item)


def _validate_terminal(terminal, spec, record, pins):
    require(terminal["specification"] == record["specification"], "terminal specification binding mismatch")
    execution.validate_terminal(terminal, spec)  # Pure structural check; no source path or procfs lookup.
    require(terminal["cleanup"]["verified"] and terminal["cleanup"]["remaining"] == [],
            "completed archive requires verified empty session")
    attempt_dir = Path(record["prepared"]["path"]).parent
    output_dir = Path(spec["output_directory"])
    native = {item["path"] for item in terminal["native_receipts"]}
    allowed = {"prepared.json", "launch-intent.json", "supervisor.json", "child.json", "release.json",
               "exec-intent.json", "command.log", "supervisor-error.json", "specification.json"}
    for item in terminal["evidence"]:
        _add_pin(pins, item)
        path = Path(item["path"])
        require(item["path"] in native or path == Path(record["specification"]["path"]) or
                path.parent == attempt_dir and (path.name in allowed or re.fullmatch(r"stop-\d+\.json", path.name)),
                "terminal evidence outside declared execution scope")
    for item in terminal["native_receipts"]:
        _add_pin(pins, item)
        require(Path(item["path"]).is_relative_to(output_dir), "native receipt escaped output directory")


def _metadata_status(spec, terminal, objects):
    declarations = {item["path"]: item["required_fields"] for item in spec["native_receipts"]}
    actual_invalid = set()
    output = Path(spec["output_directory"])
    for item in terminal["native_receipts"]:
        name = Path(item["path"]).relative_to(output).as_posix()
        try:
            document = _read_document(item, objects)
            require(type(document) is dict and all(key in document and canonical(document[key]) == canonical(value)
                    for key, value in declarations[name].items()), "native metadata binding mismatch")
        except (ArchiveError, ValueError, TypeError, KeyError):
            actual_invalid.add(name)
    declared_invalid = {item["path"] for item in terminal["invalid_native_receipts"]}
    require(actual_invalid == declared_invalid, "native declared metadata status disagrees with archived bytes")


def _validate_graph(manifest, objects):
    require(type(manifest) is dict and set(manifest) == {"schema", "root_attempt_id", "source_store_root", "attempts",
            "objects", "qualification"} and manifest["schema"] == "LocalExecutionArchiveManifestV1" and
            manifest["qualification"] == "unassessed", "invalid archive manifest")
    require(type(manifest["source_store_root"]) is str and manifest["source_store_root"].startswith("/") and
            os.path.normpath(manifest["source_store_root"]) == manifest["source_store_root"],
            "invalid original store root")
    records = manifest["attempts"]
    require(type(records) is list and records, "archive has no attempts")
    ids = [record.get("attempt_id") for record in records if type(record) is dict]
    require(len(ids) == len(records) and len(set(ids)) == len(ids) and ids[0] == manifest["root_attempt_id"],
            "invalid predecessor chain")
    pins = {}
    for index, record in enumerate(records):
        require(set(record) == {"attempt_id", "specification", "prepared", "terminal", "job_claim", "output_claim"},
                "invalid attempt archive record")
        for key in ("specification", "prepared", "terminal", "job_claim", "output_claim"):
            _add_pin(pins, record[key])
        spec = _read_document(record["specification"], objects)
        prepared = _read_document(record["prepared"], objects)
        terminal = _read_document(record["terminal"], objects)
        claim = _read_document(record["job_claim"], objects)
        output_claim = _read_document(record["output_claim"], objects)
        _validate_spec(spec, pins)
        require(spec["attempt_id"] == record["attempt_id"], "attempt/specification mismatch")
        attempt_dir = Path(manifest["source_store_root"]) / "attempts" / record["attempt_id"]
        require(record["specification"]["path"] == str(attempt_dir / "specification.json") and
                record["prepared"]["path"] == str(attempt_dir / "prepared.json") and
                record["terminal"]["path"] in (str(attempt_dir / "terminal.json"),
                                                str(attempt_dir / "recovery.json")),
                "executor record path identity mismatch")
        job_digest = hashlib.sha256(canonical([spec["cycle_id"], spec["job_key"]]).encode()).hexdigest()
        output_digest = hashlib.sha256(spec["output_directory"].encode()).hexdigest()
        require(record["job_claim"]["path"] == str(Path(manifest["source_store_root"]) / "jobs" /
                job_digest / ("claim-" + record["attempt_id"] + ".json")) and
                record["output_claim"]["path"] == str(Path(manifest["source_store_root"]) / "outputs" / output_digest),
                "executor claim path identity mismatch")
        _validate_prepared(prepared, record, pins)
        _validate_terminal(terminal, spec, record, pins)
        _metadata_status(spec, terminal, objects)
        prior = spec["previous_attempt"]
        if index + 1 < len(records):
            next_record = records[index + 1]
            require(prior is not None and prior == {"attempt_id": next_record["attempt_id"],
                    "receipt": next_record["terminal"]}, "predecessor receipt mismatch")
            prior_spec = _read_document(next_record["specification"], objects)
            require((spec["cycle_id"], spec["job_key"]) == (prior_spec["cycle_id"], prior_spec["job_key"]),
                    "predecessor job scope mismatch")
        else:
            require(prior is None, "predecessor chain truncated")
        require(claim == {"attempt_id": record["attempt_id"], "prepared": record["prepared"],
                "previous_attempt": prior["attempt_id"] if prior else None}, "job claim mismatch")
        require(output_claim == {"attempt_id": record["attempt_id"],
                "output_directory": spec["output_directory"]}, "output claim mismatch")
    entries = manifest["objects"]
    require(type(entries) is list and entries == sorted(entries, key=lambda item: item["path"]),
            "archive object map must be sorted")
    mapped = {}
    for item in entries:
        _add_pin(mapped, item)
    require(len(mapped) == len(entries) and mapped == pins, "archive object map does not match exact reference graph")
    return records, pins


def _source_records(root, attempt_id):
    root = Path(root).resolve()
    records = []
    seen = set()
    while attempt_id is not None:
        require(attempt_id not in seen, "cyclic previous-attempt chain")
        seen.add(attempt_id)
        observation = execution.inspect(root, attempt_id)
        require(observation["attempt_terminal"] is not None and observation["state"] == "terminal" and
                not observation["job_lock_held"] and not observation["session_members"] and
                all(value["state"] != "alive" for value in observation["process_observation"].values()),
                "source attempt is not completed and quiescent")
        directory, spec, prepared = execution.load(root, attempt_id, verify_inputs=True)
        execution.registered(root, directory, spec)
        claim = execution.job_directory(root, spec["cycle_id"], spec["job_key"]) / ("claim-" + attempt_id + ".json")
        output_claim = root / "outputs" / hashlib.sha256(spec["output_directory"].encode()).hexdigest()
        record = {"attempt_id": attempt_id, "specification": prepared["specification"],
                  "prepared": execution.pin(directory / "prepared.json"),
                  "terminal": observation["terminal_receipt"], "job_claim": execution.pin(claim),
                  "output_claim": execution.pin(output_claim)}
        records.append(record)
        attempt_id = spec["previous_attempt"]["attempt_id"] if spec["previous_attempt"] else None
    return records


def _copy_pinned(item, destination):
    _pin_shape(item)
    source_fd = os.open(item["path"], os.O_RDONLY | os.O_NOFOLLOW)
    try:
        source_stat = os.fstat(source_fd)
        require(stat.S_ISREG(source_stat.st_mode), "pinned source is not regular")
        target = destination / "objects" / "sha256" / item["sha256"]
        first = not target.exists()
        if first:
            target_fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        else:
            target_fd = None
        sha = hashlib.sha256()
        size = 0
        try:
            while True:
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    break
                sha.update(chunk)
                size += len(chunk)
                if target_fd is not None:
                    remaining = memoryview(chunk)
                    while remaining:
                        remaining = remaining[os.write(target_fd, remaining):]
            require(size == item["size"] and sha.hexdigest() == item["sha256"],
                    "source artifact fingerprint changed: " + item["path"])
            after = os.fstat(source_fd)
            require((source_stat.st_dev, source_stat.st_ino, source_stat.st_size, source_stat.st_mtime_ns) ==
                    (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "source changed during copy")
            if target_fd is not None:
                os.fsync(target_fd)
        finally:
            if target_fd is not None:
                os.close(target_fd)
    finally:
        os.close(source_fd)


def _archive_object_reader(directory):
    root = Path(directory)
    def read(item):
        target = root / "objects" / "sha256" / item["sha256"]
        require(not target.is_symlink() and target.is_file(), "archive object missing or symlinked")
        require(item["size"] <= 16 * 1024 * 1024, "compact JSON document exceeds archive limit")
        return target.read_bytes()
    return read


def export_completed_attempt(root, attempt_id, destination):
    """Return an external manifest pin; destination must not exist."""
    root = Path(root).resolve()
    requested = Path(destination).absolute()
    require(requested.name not in ("", ".", "..") and requested.parent.is_dir(),
            "fresh archive destination required")
    destination = requested.parent.resolve() / requested.name
    require(not destination.exists() and not destination.is_symlink(), "fresh archive destination required")
    require(not destination.is_relative_to(root), "archive destination must stay outside source store")
    records = _source_records(root, attempt_id)
    for record in records:
        source_spec = decode(execution.checked(record["specification"]))
        require(not destination.is_relative_to(Path(source_spec["output_directory"])),
                "archive destination must stay outside native output")
    destination.mkdir()
    (destination / "objects").mkdir()
    (destination / "objects" / "sha256").mkdir()
    try:
        # Read source JSON through their immutable pins to derive the exact graph.
        source_reader = lambda item: execution.checked(item)
        seed = {"schema": "LocalExecutionArchiveManifestV1", "root_attempt_id": attempt_id,
                "source_store_root": str(root), "attempts": records, "objects": [], "qualification": "unassessed"}
        # First discover referenced pins. The graph validator deliberately checks
        # the final map, so use a local pin recorder for source traversal.
        entries = _discover_source_pins(seed, source_reader)
        for item in entries:
            _copy_pinned(item, destination)
        manifest = dict(seed, objects=entries)
        _verify_copied_objects(destination, entries)
        _validate_graph(manifest, _archive_object_reader(destination))
        # The source job must still be idle before a complete manifest appears.
        _source_records(root, attempt_id)
        # Make all object bytes and directory entries durable before publishing
        # the single complete manifest link.
        execution.sync_directory(destination / "objects" / "sha256")
        execution.sync_directory(destination / "objects")
        execution.sync_directory(destination)
        execution.sync_directory(destination.parent)
        data = canonical(manifest).encode()
        manifest_pin = {"sha256": _sha_bytes(data), "size": len(data)}
        execution.publish(destination / "manifest.json", manifest)
        return {"schema": "LocalExecutionArchiveExportV1", "archive": str(destination),
                "manifest": manifest_pin, "attempt_count": len(records),
                "object_count": len({item["sha256"] for item in entries}), "qualification": "unassessed"}
    except BaseException:
        # A partial destination remains plainly incomplete (manifest is last).
        raise


def _discover_source_pins(manifest, objects):
    pins = {}
    records = manifest["attempts"]
    for index, record in enumerate(records):
        for key in ("specification", "prepared", "terminal", "job_claim", "output_claim"):
            _add_pin(pins, record[key])
        spec = _read_document(record["specification"], objects)
        prepared = _read_document(record["prepared"], objects)
        terminal = _read_document(record["terminal"], objects)
        _validate_spec(spec, pins)
        _validate_prepared(prepared, record, pins)
        _validate_terminal(terminal, spec, record, pins)
        if index + 1 < len(records):
            require(spec["previous_attempt"] == {"attempt_id": records[index + 1]["attempt_id"],
                    "receipt": records[index + 1]["terminal"]}, "source predecessor mismatch")
        else:
            require(spec["previous_attempt"] is None, "source predecessor chain truncated")
    return sorted(pins.values(), key=lambda item: item["path"])


def inspect_archive(directory, manifest_sha256, manifest_size=None):
    """Inspect copied bytes only. Never consult original paths or process state."""
    directory = Path(directory).absolute()
    require(directory.is_dir() and not directory.is_symlink(), "archive directory missing or symlinked")
    require(type(manifest_sha256) is str and re.fullmatch(r"[0-9a-f]{64}", manifest_sha256),
            "external manifest SHA256 required")
    manifest_path = directory / "manifest.json"
    require(manifest_path.is_file() and not manifest_path.is_symlink(), "archive manifest missing or symlinked")
    data = manifest_path.read_bytes()
    require(_sha_bytes(data) == manifest_sha256 and
            (manifest_size is None or type(manifest_size) is int and len(data) == manifest_size),
            "external manifest pin mismatch")
    manifest = decode(data)
    entries = manifest.get("objects") if type(manifest) is dict else None
    require(type(entries) is list, "archive object map missing")
    digest_sizes = _verify_copied_objects(directory, entries)
    records, _ = _validate_graph(manifest, _archive_object_reader(directory))
    return {"schema": "LocalExecutionArchiveInspectionV1", "root_attempt_id": manifest["root_attempt_id"],
            "attempt_count": len(records), "object_count": len(digest_sizes),
            "manifest": {"sha256": manifest_sha256, "size": len(data)},
            "qualification": "unassessed", "portable_evidence_verified": True}


def _verify_copied_objects(directory, entries):
    object_dir = Path(directory) / "objects" / "sha256"
    require(object_dir.is_dir() and not object_dir.is_symlink() and not object_dir.parent.is_symlink(),
            "archive object directory missing or symlinked")
    digest_sizes = {}
    for item in entries:
        _pin_shape(item)
        size = digest_sizes.get(item["sha256"])
        require(size is None or size == item["size"], "conflicting size for content digest")
        digest_sizes[item["sha256"]] = item["size"]
    require({path.name for path in object_dir.iterdir()} == set(digest_sizes), "archive object inventory mismatch")
    for digest, size in digest_sizes.items():
        path = object_dir / digest
        require(path.is_file() and not path.is_symlink(), "archive object missing or symlinked")
        sha = hashlib.sha256()
        seen = 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(chunk)
                seen += len(chunk)
        require(seen == size and sha.hexdigest() == digest, "archive object fingerprint mismatch: " + digest)
    return digest_sizes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--root", required=True)
    export.add_argument("--attempt", required=True)
    export.add_argument("--destination", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--archive", required=True)
    inspect.add_argument("--manifest-sha256", required=True)
    inspect.add_argument("--manifest-size", type=int)
    args = parser.parse_args()
    if args.command == "export":
        result = export_completed_attempt(args.root, args.attempt, args.destination)
    else:
        result = inspect_archive(args.archive, args.manifest_sha256, args.manifest_size)
    print(canonical(result))


if __name__ == "__main__":
    main()
