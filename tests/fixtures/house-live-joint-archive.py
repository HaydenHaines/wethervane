"""Independent tiny NPY/NPZ writer for invented Node integration fixtures only."""
import json
import struct
import sys
import zipfile


def npy(descr, shape, payload):
    header = repr({"descr": descr, "fortran_order": False, "shape": tuple(shape)}).encode("latin1")
    header += b" " * ((64 - (10 + len(header) + 1) % 64) % 64) + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + payload


request = json.load(sys.stdin)
rows, ids, chains = request["rows"], request["raceIds"], request["chains"]
count, width = len(rows), max(map(len, ids))
seats = [row.count(party) for row in rows for party in range(3)]
integer = lambda values: struct.pack("<" + "q" * len(values), *values)
members = {
    "winner_codes": npy("|i1", (count, 435), bytes(value for row in rows for value in row)),
    "seat_counts": npy("<i2", (count, 3), struct.pack("<" + "h" * len(seats), *seats)),
    "race_ids": npy(f"<U{width}", (435,), b"".join(value.ljust(width, "\0").encode("utf-32-le") for value in ids)),
    "joint_draw_id": npy("<i8", (count,), integer(list(range(count)))),
    "predictive_chain": npy("<i8", (count,), integer([i // (count // chains) for i in range(count)])),
    "predictive_draw": npy("<i8", (count,), integer([i % (count // chains) for i in range(count)])),
    "categorical_bank_index": npy("<i8", (count,), integer([0] * count)),
    "core_posterior_index": npy("<i8", (count,), integer([0] * count)),
    "national_z": npy("<f8", (count,), struct.pack("<" + "d" * count, *([0.0] * count))),
}
if "sourceRunLengths" in request:
    lengths = request["sourceRunLengths"]
    assert len(lengths) >= 2 and sum(lengths) == count
    source_indices = [i for i, length in enumerate(lengths) for _ in range(length)]
    local_ids = [i for length in lengths for i in range(length)]
    members["source_run_index"] = npy("<i2", (count,), struct.pack("<" + "h" * count, *source_indices))
    members["source_joint_draw_id"] = npy("<i8", (count,), integer(local_ids))
with zipfile.ZipFile(request["destination"], "x", compression=zipfile.ZIP_DEFLATED) as archive:
    for name, data in members.items():
        archive.writestr(name + ".npy", data)
