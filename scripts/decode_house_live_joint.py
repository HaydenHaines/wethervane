#!/usr/bin/env python3
"""Offline decoder for forecast.py House winner archives; stdout is raw rows.

Accepts one JSON request on stdin. Each successful row emits exactly 435 bytes,
one D=0/R=1/O=2 code per ordered race. This authenticates and validates the
archive format; it does not qualify a forecast or a publication decision.
For combined archives, an authenticated local plan supplies source-row layout.
V1 original source receipts/plans remain references. V2 additionally requires an
explicit offline storage root and authenticates its manifest, original receipts,
plans and child bytes; no research root is inferred from a receipt.

NPY specification: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
ZIP reader: https://docs.python.org/3/library/zipfile.html
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import stat
import struct
import sys
import zipfile
from contextlib import ExitStack
from pathlib import Path


RACES = 435
MAX_REQUEST_BYTES = 128_000
MAX_HEADER_BYTES = 16_384
MAX_PLAN_BYTES = 1_000_000
# Bound the stream while allowing five-source precision extensions without row loss.
MAX_ROWS = 5_000_000
BASE_MEMBERS = {
    "winner_codes.npy": ("|i1", lambda count: (count, RACES), 1),
    "seat_counts.npy": ("<i2", lambda count: (count, 3), 2),
    "race_ids.npy": ("<U24", lambda count: (RACES,), 96),
    "categorical_bank_index.npy": ("<i8", lambda count: (count,), 8),
    "core_posterior_index.npy": ("<i8", lambda count: (count,), 8),
    "national_z.npy": ("<f8", lambda count: (count,), 8),
    "joint_draw_id.npy": ("<i8", lambda count: (count,), 8),
    "predictive_chain.npy": ("<i8", lambda count: (count,), 8),
    "predictive_draw.npy": ("<i8", lambda count: (count,), 8),
}
SOURCE_MEMBERS = {
    "source_run_index.npy": ("<i2", lambda count: (count,), 2),
    "source_joint_draw_id.npy": ("<i8", lambda count: (count,), 8),
}
MEMBERS = {**BASE_MEMBERS, **SOURCE_MEMBERS}
PIN_PATH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
RACE_ID = re.compile(r"^2026:[A-Z]{2}-\d{2}:house:regular$")
SHA = re.compile(r"^[0-9a-f]{64}$")


class DecodeError(Exception):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise DecodeError(message)


def valid_pin(pin, suffix: str | None = None) -> bool:
    if not isinstance(pin, dict) or set(pin) != {"path", "sha256"}:
        return False
    path, sha = pin["path"], pin["sha256"]
    return (isinstance(path, str) and PIN_PATH.fullmatch(path) is not None and
            (suffix is None or path.endswith(suffix)) and
            all(part not in ("", ".", "..") for part in path.split("/")) and
            isinstance(sha, str) and SHA.fullmatch(sha) is not None)


def valid_configuration(value) -> bool:
    if not isinstance(value, dict) or set(value) != {"id", "base", "polling", "uncertainty"}:
        return False
    base, polling, uncertainty = value["base"], value["polling"], value["uncertainty"]
    bases = ("fundamentals", "recent_calendar_history", "house_history_refreshed_v1",
             "house_presidential_slope_v1", "house_relative_lean_v1", "house_relative_context_v1")
    return (isinstance(value["id"], str) and 0 < len(value["id"].strip()) <= 200 and
            base in bases and polling in ("none", "standard") and
            uncertainty in ("standard", "cycle-low", "cycle-high", "error-low", "error-high") and
            (uncertainty == "standard" or (base == "fundamentals" and polling == "standard")))


def read_request() -> dict:
    data = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    need(len(data) <= MAX_REQUEST_BYTES, "Request is too large.")
    try:
        request = json.loads(data, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (json.JSONDecodeError, ValueError) as error:
        raise DecodeError("Request is not valid JSON.") from error
    base_keys = {"schema", "directory", "artifact", "count", "chainCount", "orderedRaceIds"}
    combined_keys = base_keys | {"combinedPlan", "configuration"}
    need(isinstance(request, dict) and set(request) in (base_keys, combined_keys, combined_keys | {"conditionalStorage"}),
         "Request fields do not match HouseLiveJointDecodeRequestV1.")
    need(request["schema"] == "HouseLiveJointDecodeRequestV1", "Unknown joint decode request schema.")
    directory = request["directory"]
    need(isinstance(directory, str) and os.path.isabs(directory) and "\x00" not in directory,
         "A safe absolute publication directory is required.")
    need(valid_pin(request["artifact"], ".npz"), "Artifact pin has an unsafe path or digest.")
    if "combinedPlan" in request:
        need(valid_pin(request["combinedPlan"], ".json"), "Combined plan pin has an unsafe path or digest.")
        need(valid_configuration(request["configuration"]), "Expected combined configuration is invalid.")
    if "conditionalStorage" in request:
        storage = request["conditionalStorage"]
        need(isinstance(storage, dict) and set(storage) == {"directory", "manifest", "plan", "receipt", "orderedPairRaceIds"} and
             isinstance(storage["directory"], str) and os.path.isabs(storage["directory"]) and
             all(valid_pin(storage[key], ".json") for key in ("manifest", "plan", "receipt")) and
             storage["plan"]["sha256"] == request["combinedPlan"]["sha256"] and
             isinstance(storage["orderedPairRaceIds"], list) and storage["orderedPairRaceIds"] and
             len(set(storage["orderedPairRaceIds"])) == len(storage["orderedPairRaceIds"]) and
             all(isinstance(race, str) and RACE_ID.fullmatch(race) for race in storage["orderedPairRaceIds"]),
             "V2 conditional storage requires an explicit root and exact source/ordered-pair pins.")
    count, chains = request["count"], request["chainCount"]
    need(type(count) is int and 0 < count <= MAX_ROWS and type(chains) is int and
         0 < chains <= count and count % chains == 0,
         "Draw count or chain count is invalid.")
    race_ids = request["orderedRaceIds"]
    need(isinstance(race_ids, list) and len(race_ids) == RACES and
         all(isinstance(race, str) and RACE_ID.fullmatch(race) for race in race_ids) and
         len(set(race_ids)) == RACES, "Exactly 435 unique ordered current House race IDs are required.")
    return request


def file_identity(info: os.stat_result) -> tuple[int, ...]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def opened_artifact(directory: str, pin: dict, label: str = "Joint archive"):
    root = Path(directory).resolve(strict=True)
    need(root.is_dir(), "Publication root must be a directory.")
    artifact = (root / pin["path"]).resolve(strict=True)
    need(artifact != root and os.path.commonpath((root, artifact)) == str(root),
         f"{label} must stay inside the publication directory.")
    descriptor = os.open(artifact, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    opened = os.fdopen(descriptor, "rb")
    initial = os.fstat(opened.fileno())
    need(stat.S_ISREG(initial.st_mode), f"{label} must be a regular file.")
    need(file_identity(initial) == file_identity(os.stat(artifact)), f"{label} changed while opening.")
    return opened, artifact, initial


def verify_sha(opened, expected: str, initial: os.stat_result, artifact: Path,
               label: str = "Joint archive") -> None:
    need(file_identity(os.fstat(opened.fileno())) == file_identity(initial) and
         file_identity(os.stat(artifact)) == file_identity(initial), f"{label} changed during decoding.")
    opened.seek(0)
    digest = hashlib.sha256()
    while chunk := opened.read(1024 * 1024):
        digest.update(chunk)
    need(digest.hexdigest() == expected, f"{label} fingerprint mismatch.")
    need(file_identity(os.fstat(opened.fileno())) == file_identity(initial) and
         file_identity(os.stat(artifact)) == file_identity(initial), f"{label} changed during decoding.")


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        need(key not in value, f"Duplicate JSON field in combined plan: {key}.")
        value[key] = item
    return value


def integer(value, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def read_combined_plan(opened, request: dict) -> dict:
    opened.seek(0)
    raw = opened.read(MAX_PLAN_BYTES + 1)
    need(0 < len(raw) <= MAX_PLAN_BYTES, "Combined plan exceeds the size bound.")
    try:
        plan = json.loads(raw, object_pairs_hook=unique_object,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (UnicodeError, ValueError) as error:
        raise DecodeError("Combined plan is not valid JSON.") from error
    expected_keys = {"schema", "issued_at", "configuration", "pilot", "settings", "source_runs",
                     "inputs", "source", "polling_error_scales", "exact_electoral_rule_simulation",
                     "source_2026_outcomes_loaded", "combination"}
    segmented = isinstance(plan, dict) and plan.get("schema") == "HouseLiveCombinedPredictionPlanV2"
    if segmented:
        expected_keys |= {"conditional_storage", "group_completion"}
        need("conditionalStorage" in request, "V2 plan requires an explicit offline conditional storage root.")
        need(plan.get("conditional_storage") == {"schema": "ConditionalSegmentsManifestV1", "manifest": request["conditionalStorage"]["manifest"],
             "weights": "equal_weight_per_original_joint_row"}, "V2 conditional manifest or weights differ.")
    else:
        need("conditionalStorage" not in request, "V1 archive must not be reinterpreted as V2 storage.")
    need(isinstance(plan, dict) and set(plan) == expected_keys and
         plan["schema"] in {"HouseLiveCombinedPredictionPlanV1", "HouseLiveCombinedPredictionPlanV2"} and plan["pilot"] is False and
         plan["source_2026_outcomes_loaded"] is False and
         plan["exact_electoral_rule_simulation"] is False and
         isinstance(plan["issued_at"], str) and bool(plan["issued_at"]) and
         isinstance(plan["combination"], str) and bool(plan["combination"]),
         "Combined plan schema or safety fields differ.")
    need(valid_configuration(plan["configuration"]) and
         plan["configuration"] == request["configuration"],
         "Combined plan configuration differs from requested result.")
    settings = plan["settings"]
    need(isinstance(settings, dict) and set(settings) == {"chains", "draws", "warmup_per_source"} and
         integer(settings["chains"], 1) and integer(settings["draws"], 1) and
         integer(settings["warmup_per_source"]) and
         settings["chains"] == request["chainCount"] and
         settings["chains"] * settings["draws"] == request["count"],
         "Combined plan total draw or chain layout differs.")
    scales = plan["polling_error_scales"]
    need(isinstance(scales, dict) and set(scales) == {"independent", "cycle", "race", "house", "movement"} and
         all(type(value) in (float, int) and math.isfinite(value) and value > 0 for value in scales.values()),
         "Combined plan polling error scales are invalid.")
    inputs = plan["inputs"]
    need(isinstance(inputs, list) and all(valid_pin(item) for item in inputs) and
         len({item["path"] for item in inputs}) == len(inputs) and valid_pin(plan["source"], ".py"),
         "Combined plan evidence pins are invalid.")
    records = plan["source_runs"]
    need(isinstance(records, list) and 2 <= len(records) <= 32_768,
         "Combined plan must name at least two source runs.")
    offset = chain_offset = 0
    seeds = set()
    directories = set()
    evidence_paths = set()
    input_pins = {(item["path"], item["sha256"]) for item in inputs}
    for index, record in enumerate(records):
        need(isinstance(record, dict) and set(record) == {"directory", "receipt", "plan", "settings",
                                                         "joint_draw_offset", "predictive_chain_offset"},
             f"Combined source record {index} has invalid fields.")
        directory = record["directory"]
        source_settings = record["settings"]
        need(isinstance(directory, str) and PIN_PATH.fullmatch(directory) is not None and
             all(part not in ("", ".", "..") for part in directory.split("/")) and
             directory not in directories and
             valid_pin(record["receipt"], "/receipt.json") and
             valid_pin(record["plan"], "/plan.json") and
             record["receipt"]["path"] not in evidence_paths and
             record["plan"]["path"] not in evidence_paths and
             (record["receipt"]["path"], record["receipt"]["sha256"]) in input_pins,
             f"Combined source record {index} evidence pins differ.")
        need(isinstance(source_settings, dict) and
             set(source_settings) == {"chains", "draws", "warmup", "seed"} and
             integer(source_settings["chains"], 1) and integer(source_settings["draws"], 1) and
             integer(source_settings["warmup"]) and integer(source_settings["seed"]) and
             source_settings["draws"] == settings["draws"] and
             source_settings["warmup"] == settings["warmup_per_source"] and
             source_settings["seed"] not in seeds and
             integer(record["joint_draw_offset"]) and integer(record["predictive_chain_offset"]) and
             record["joint_draw_offset"] == offset and record["predictive_chain_offset"] == chain_offset,
             f"Combined source record {index} layout or seed differs.")
        directories.add(directory)
        evidence_paths.update((record["receipt"]["path"], record["plan"]["path"]))
        seeds.add(source_settings["seed"])
        offset += source_settings["chains"] * source_settings["draws"]
        chain_offset += source_settings["chains"]
    need(offset == request["count"] and chain_offset == request["chainCount"],
         "Combined source segments do not cover every row and chain.")
    return plan


def exact(stream, count: int, label: str) -> bytes:
    data = stream.read(count)
    need(len(data) == count, f"Truncated {label} member.")
    return data


def header(stream, info: zipfile.ZipInfo, name: str, count: int) -> None:
    need(exact(stream, 6, name) == b"\x93NUMPY", f"Invalid NPY magic in {name}.")
    major, minor = exact(stream, 2, name)
    need(major in (1, 2, 3) and minor == 0, f"Unsupported NPY version in {name}.")
    length_bytes = 2 if major == 1 else 4
    length = struct.unpack("<H" if major == 1 else "<I", exact(stream, length_bytes, name))[0]
    need(0 < length <= MAX_HEADER_BYTES, f"NPY header in {name} exceeds the bound.")
    encoding = "utf-8" if major == 3 else "latin1"
    try:
        value = ast.literal_eval(exact(stream, length, name).decode(encoding).strip())
    except (SyntaxError, ValueError, UnicodeError) as error:
        raise DecodeError(f"Invalid NPY header in {name}.") from error
    need(isinstance(value, dict) and set(value) == {"descr", "fortran_order", "shape"},
         f"Unrecognized NPY header fields in {name}.")
    dtype, expected_shape, item_bytes = MEMBERS[name]
    expected = expected_shape(count)
    shape = value["shape"]
    need(value["descr"] == dtype and value["fortran_order"] is False and
         isinstance(shape, tuple) and len(shape) == len(expected) and
         all(type(dimension) is int and dimension == required for dimension, required in zip(shape, expected)),
         f"NPY dtype, shape, or order differs in {name}.")
    payload = item_bytes * math.prod(expected)
    need(info.file_size == 6 + 2 + length_bytes + length + payload,
         f"NPY member length differs in {name}.")


def validate_archive(opened, request: dict, combined_plan: dict | None) -> None:
    count, chains = request["count"], request["chainCount"]
    per_chain = count // chains
    expected_members = {**BASE_MEMBERS, **SOURCE_MEMBERS} if combined_plan else BASE_MEMBERS
    source_records = combined_plan["source_runs"] if combined_plan else None
    source_index = 0
    with zipfile.ZipFile(opened) as archive:
        infos = archive.infolist()
        need(len(infos) == len(expected_members) and
             {info.filename for info in infos} == set(expected_members),
             "Joint archive needs the exact nine original or eleven combined members.")
        for info in infos:
            need(not info.is_dir() and info.compress_type == zipfile.ZIP_DEFLATED and
                 not (info.flag_bits & 1), "Joint archive has an encrypted or unsupported member.")
        with ExitStack() as stack:
            streams = {name: stack.enter_context(archive.open(name)) for name in expected_members}
            for name, stream in streams.items():
                header(stream, archive.getinfo(name), name, count)
            race_stream = streams["race_ids.npy"]
            for index, expected_id in enumerate(request["orderedRaceIds"]):
                raw = exact(race_stream, 96, "race_ids")
                try:
                    actual = raw.decode("utf-32-le").rstrip("\x00")
                except UnicodeError as error:
                    raise DecodeError("Invalid Unicode race ID in joint archive.") from error
                need(actual == expected_id, f"Joint race order differs at position {index}.")
            need(race_stream.read(1) == b"", "Race ID member has trailing bytes.")
            output = bytearray()
            for index in range(count):
                row = exact(streams["winner_codes.npy"], RACES, "winner_codes")
                counts = (row.count(0), row.count(1), row.count(2))
                need(sum(counts) == RACES, f"Invalid winner code in row {index}.")
                seats = struct.unpack("<hhh", exact(streams["seat_counts.npy"], 6, "seat_counts"))
                need(seats == counts,
                     f"Seat sidecar differs from winner row {index}.")
                for name in ("categorical_bank_index.npy", "core_posterior_index.npy"):
                    value = struct.unpack("<q", exact(streams[name], 8, name))[0]
                    need(value >= 0, f"Negative index in {name}, row {index}.")
                z = struct.unpack("<d", exact(streams["national_z.npy"], 8, "national_z"))[0]
                need(math.isfinite(z), f"Nonfinite national shock in row {index}.")
                draw_id = struct.unpack("<q", exact(streams["joint_draw_id.npy"], 8, "joint_draw_id"))[0]
                chain = struct.unpack("<q", exact(streams["predictive_chain.npy"], 8, "predictive_chain"))[0]
                draw = struct.unpack("<q", exact(streams["predictive_draw.npy"], 8, "predictive_draw"))[0]
                need(draw_id == index and chain == index // per_chain and draw == index % per_chain,
                     f"Joint draw or chain identity differs at row {index}.")
                if source_records is not None:
                    if index >= (source_records[source_index]["joint_draw_offset"] +
                                 source_records[source_index]["settings"]["chains"] * per_chain):
                        source_index += 1
                    source = source_records[source_index]
                    run_index = struct.unpack("<h", exact(streams["source_run_index.npy"], 2,
                                                          "source_run_index"))[0]
                    local_id = struct.unpack("<q", exact(streams["source_joint_draw_id.npy"], 8,
                                                         "source_joint_draw_id"))[0]
                    need(run_index == source_index and local_id == index - source["joint_draw_offset"],
                         f"Combined source-run or local-draw identity differs at row {index}.")
                output.extend(row)
                if len(output) >= 64 * RACES:
                    sys.stdout.buffer.write(output)
                    output.clear()
            for name, stream in streams.items():
                if name != "race_ids.npy":
                    need(stream.read(1) == b"", f"Unexpected trailing bytes in {name}.")
            if output:
                sys.stdout.buffer.write(output)


def main() -> int:
    try:
        request = read_request()
        with ExitStack() as stack:
            opened, artifact, initial = opened_artifact(request["directory"], request["artifact"])
            stack.enter_context(opened)
            combined_plan = None
            if "combinedPlan" in request:
                plan_opened, plan_path, plan_initial = opened_artifact(
                    request["directory"], request["combinedPlan"], "Combined plan")
                stack.enter_context(plan_opened)
                verify_sha(plan_opened, request["combinedPlan"]["sha256"], plan_initial,
                           plan_path, "Combined plan")
                combined_plan = read_combined_plan(plan_opened, request)
                if combined_plan["schema"] == "HouseLiveCombinedPredictionPlanV2":
                    from house_live_conditional_storage import verify_prediction_storage
                    storage = request["conditionalStorage"]
                    verified = verify_prediction_storage(storage["directory"], storage["manifest"], storage["plan"], storage["receipt"], storage["orderedPairRaceIds"])
                    need(verified["rows"] == request["count"] and verified["chains"] == request["chainCount"], "V2 conditional and joint row/chain counts differ.")
            verify_sha(opened, request["artifact"]["sha256"], initial, artifact)
            validate_archive(opened, request, combined_plan)
            verify_sha(opened, request["artifact"]["sha256"], initial, artifact)
            if combined_plan is not None:
                verify_sha(plan_opened, request["combinedPlan"]["sha256"], plan_initial,
                           plan_path, "Combined plan")
        return 0
    except Exception as error:
        # The caller must discard any streamed bytes unless the exit code is zero.
        message = (str(error).replace("\n", " ") or type(error).__name__).encode(
            "ascii", "backslashreplace"
        )[:240].decode("ascii", "ignore")
        print(f"House joint decode error: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
