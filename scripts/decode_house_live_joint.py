#!/usr/bin/env python3
"""Offline decoder for forecast.py House winner archives; stdout is raw rows.

Accepts one JSON request on stdin. Each successful row emits exactly 435 bytes,
one D=0/R=1/O=2 code per ordered race. This authenticates and validates the
archive format; it does not qualify a forecast or a publication decision.

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
MAX_ROWS = 2_000_000
MEMBERS = {
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
PIN_PATH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
RACE_ID = re.compile(r"^2026:[A-Z]{2}-\d{2}:house:regular$")
SHA = re.compile(r"^[0-9a-f]{64}$")


class DecodeError(Exception):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise DecodeError(message)


def read_request() -> dict:
    data = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    need(len(data) <= MAX_REQUEST_BYTES, "Request is too large.")
    try:
        request = json.loads(data, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (json.JSONDecodeError, ValueError) as error:
        raise DecodeError("Request is not valid JSON.") from error
    need(isinstance(request, dict) and set(request) == {"schema", "directory", "artifact", "count", "chainCount", "orderedRaceIds"},
         "Request fields do not match HouseLiveJointDecodeRequestV1.")
    need(request["schema"] == "HouseLiveJointDecodeRequestV1", "Unknown joint decode request schema.")
    directory = request["directory"]
    need(isinstance(directory, str) and os.path.isabs(directory) and "\x00" not in directory,
         "A safe absolute publication directory is required.")
    pin = request["artifact"]
    need(isinstance(pin, dict) and set(pin) == {"path", "sha256"}, "Artifact pin is invalid.")
    path, sha = pin["path"], pin["sha256"]
    need(isinstance(path, str) and PIN_PATH.fullmatch(path) is not None and path.endswith(".npz") and
         all(part not in ("", ".", "..") for part in path.split("/")) and
         isinstance(sha, str) and SHA.fullmatch(sha) is not None,
         "Artifact pin has an unsafe path or digest.")
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


def opened_artifact(directory: str, pin: dict):
    root = Path(directory).resolve(strict=True)
    need(root.is_dir(), "Publication root must be a directory.")
    artifact = (root / pin["path"]).resolve(strict=True)
    need(artifact != root and os.path.commonpath((root, artifact)) == str(root),
         "Joint archive must stay inside the publication directory.")
    descriptor = os.open(artifact, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    opened = os.fdopen(descriptor, "rb")
    initial = os.fstat(opened.fileno())
    need(stat.S_ISREG(initial.st_mode), "Joint archive must be a regular file.")
    need(file_identity(initial) == file_identity(os.stat(artifact)), "Joint archive changed while opening.")
    return opened, artifact, initial


def verify_sha(opened, expected: str, initial: os.stat_result, artifact: Path) -> None:
    need(file_identity(os.fstat(opened.fileno())) == file_identity(initial) and
         file_identity(os.stat(artifact)) == file_identity(initial), "Joint archive changed during decoding.")
    opened.seek(0)
    digest = hashlib.sha256()
    while chunk := opened.read(1024 * 1024):
        digest.update(chunk)
    need(digest.hexdigest() == expected, "Joint archive fingerprint mismatch.")
    need(file_identity(os.fstat(opened.fileno())) == file_identity(initial) and
         file_identity(os.stat(artifact)) == file_identity(initial), "Joint archive changed during decoding.")


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


def validate_archive(opened, request: dict) -> None:
    count, chains = request["count"], request["chainCount"]
    per_chain = count // chains
    with zipfile.ZipFile(opened) as archive:
        infos = archive.infolist()
        need(len(infos) == len(MEMBERS) and {info.filename for info in infos} == set(MEMBERS),
             "Joint archive needs the exact nine unique forecast.py members.")
        for info in infos:
            need(not info.is_dir() and info.compress_type == zipfile.ZIP_DEFLATED and
                 not (info.flag_bits & 1), "Joint archive has an encrypted or unsupported member.")
        with ExitStack() as stack:
            streams = {name: stack.enter_context(archive.open(name)) for name in MEMBERS}
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
            verify_sha(opened, request["artifact"]["sha256"], initial, artifact)
            validate_archive(opened, request)
            verify_sha(opened, request["artifact"]["sha256"], initial, artifact)
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
