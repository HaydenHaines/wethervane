"""Invented NPZ bytes for the narrow offline House winner decoder."""
from __future__ import annotations

import hashlib
import json
import math
import struct
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DECODER = ROOT / "scripts/decode_house_live_joint.py"
RACE_IDS = [f"2026:{'ABCDE'[index // 100]}A-{index % 100:02d}:house:regular"
            for index in range(435)]
ROWS = [bytes([0] * 435), bytes([1] * 434 + [2]),
        bytes([2] + [0] * 434), bytes([1] * 435)]


def npy(dtype: str, shape: tuple[int, ...], payload: bytes, version: int = 1) -> bytes:
    encoded = repr({"descr": dtype, "fortran_order": False, "shape": shape}).encode("utf-8" if version == 3 else "latin1")
    length_bytes = 2 if version == 1 else 4
    padding = (-(6 + 2 + length_bytes + len(encoded) + 1)) % 16
    header = encoded + b" " * padding + b"\n"
    return (b"\x93NUMPY" + bytes((version, 0)) +
            struct.pack("<H" if version == 1 else "<I", len(header)) + header + payload)


def members(version: int = 1, rows: list[bytes] = ROWS) -> dict[str, bytes]:
    count = len(rows)
    codes = b"".join(rows)
    seats = b"".join(struct.pack("<hhh", row.count(0), row.count(1), row.count(2)) for row in rows)
    race_ids = b"".join(race.encode("utf-32-le").ljust(24 * 4, b"\0") for race in RACE_IDS)
    i64 = lambda values: b"".join(struct.pack("<q", value) for value in values)
    return {
        "winner_codes.npy": npy("|i1", (count, 435), codes, version),
        "seat_counts.npy": npy("<i2", (count, 3), seats, version),
        "race_ids.npy": npy("<U24", (435,), race_ids, version),
        "categorical_bank_index.npy": npy("<i8", (count,), i64([0, 1, 1, 2][:count]), version),
        "core_posterior_index.npy": npy("<i8", (count,), i64([10, 11, 12, 13][:count]), version),
        "national_z.npy": npy("<f8", (count,), b"".join(struct.pack("<d", value) for value in [0., .5, -.5, 1.][:count]), version),
        "joint_draw_id.npy": npy("<i8", (count,), i64(range(count)), version),
        "predictive_chain.npy": npy("<i8", (count,), i64([0, 0, 1, 1][:count]), version),
        "predictive_draw.npy": npy("<i8", (count,), i64([0, 1, 0, 1][:count]), version),
    }


class JointDecoderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="house-joint-decoder-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def archive(self, parts: dict[str, bytes] | None = None) -> dict:
        path = self.directory / "joint.npz"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, value in (parts or members()).items():
                    archive.writestr(name, value)
        return {"path": "joint.npz", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def request(self, artifact: dict, **changes) -> dict:
        return {"schema": "HouseLiveJointDecodeRequestV1", "directory": str(self.directory),
                "artifact": artifact, "count": len(ROWS), "chainCount": 2,
                "orderedRaceIds": RACE_IDS, **changes}

    def run_decoder(self, request: dict) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(DECODER)], input=json.dumps(request).encode(),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=15)

    def rejected(self, request: dict, phrase: str) -> None:
        result = self.run_decoder(request)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(phrase, result.stderr.decode())
        self.assertLessEqual(len(result.stderr), 280)

    def test_valid_versions_stream_exact_original_rows(self) -> None:
        for version in (1, 2, 3):
            with self.subTest(version=version):
                artifact = self.archive(members(version))
                result = self.run_decoder(self.request(artifact))
                self.assertEqual(result.returncode, 0, result.stderr.decode())
                self.assertEqual(result.stdout, b"".join(ROWS))
                self.assertEqual(result.stderr, b"")

    def test_hash_and_path_binding(self) -> None:
        artifact = self.archive()
        self.rejected(self.request({**artifact, "sha256": "0" * 64}), "fingerprint")
        self.rejected(self.request({**artifact, "path": "../joint.npz"}), "unsafe path")
        outside = self.directory.parent / (self.directory.name + "-outside.npz")
        outside.write_bytes((self.directory / "joint.npz").read_bytes())
        self.addCleanup(outside.unlink)
        (self.directory / "escape.npz").symlink_to(outside)
        self.rejected(self.request({"path": "escape.npz", "sha256": artifact["sha256"]}), "inside")
        (self.directory / "joint.npz").write_bytes(b"changed")
        self.rejected(self.request(artifact), "fingerprint")

    def test_race_identity_and_npy_shape(self) -> None:
        artifact = self.archive()
        wrong = list(RACE_IDS)
        wrong[0], wrong[1] = wrong[1], wrong[0]
        self.rejected(self.request(artifact, orderedRaceIds=wrong), "race order")
        altered = members()
        altered["winner_codes.npy"] = npy("|i1", (4, 434), b"".join(ROWS))
        self.rejected(self.request(self.archive(altered)), "dtype, shape, or order")
        altered = members()
        altered["winner_codes.npy"] = npy("|O8", (4, 435), b"".join(ROWS))
        self.rejected(self.request(self.archive(altered)), "dtype, shape, or order")
        altered = members()
        altered["winner_codes.npy"] = npy("|i1", (4, 435.0), b"".join(ROWS))
        self.rejected(self.request(self.archive(altered)), "dtype, shape, or order")
        altered = members(rows=ROWS[:1])
        altered["joint_draw_id.npy"] = npy("<i8", (True,), struct.pack("<q", 0))
        self.rejected(self.request(self.archive(altered), count=1, chainCount=1), "dtype, shape, or order")

    def test_winner_seats_and_chain_sidecars_must_agree(self) -> None:
        altered = members()
        wrong_rows = [bytes([3]) + ROWS[0][1:], *ROWS[1:]]
        altered["winner_codes.npy"] = npy("|i1", (4, 435), b"".join(wrong_rows))
        self.rejected(self.request(self.archive(altered)), "winner code")
        altered = members()
        altered["seat_counts.npy"] = npy("<i2", (4, 3), struct.pack("<hhh", 434, 1, 0) +
                                           b"".join(struct.pack("<hhh", row.count(0), row.count(1), row.count(2)) for row in ROWS[1:]))
        self.rejected(self.request(self.archive(altered)), "Seat sidecar")
        altered = members()
        altered["predictive_chain.npy"] = npy("<i8", (4,), b"".join(struct.pack("<q", x) for x in [0, 1, 1, 1]))
        self.rejected(self.request(self.archive(altered)), "chain identity")
        altered = members()
        altered["national_z.npy"] = npy("<f8", (4,), b"".join(struct.pack("<d", x) for x in [0., math.nan, 0., 1.]))
        self.rejected(self.request(self.archive(altered)), "Nonfinite")

    def test_exact_members_and_bounded_header(self) -> None:
        altered = members()
        altered["unexpected.npy"] = npy("|i1", (1,), b"\0")
        self.rejected(self.request(self.archive(altered)), "exact nine")
        altered = members()
        altered["race_ids.npy"] = b"\x93NUMPY\x04\x00" + b"\0" * 64
        self.rejected(self.request(self.archive(altered)), "Unsupported NPY version")
        altered = members()
        oversized = bytearray(altered["winner_codes.npy"])
        oversized[8:10] = struct.pack("<H", 16_385)
        altered["winner_codes.npy"] = bytes(oversized)
        self.rejected(self.request(self.archive(altered)), "header in winner_codes.npy exceeds the bound")
        altered = members()
        altered["joint_draw_id.npy"] = npy("<i8", (4,), b"".join(struct.pack("<q", x) for x in [0, 1, 3, 3]))
        self.rejected(self.request(self.archive(altered)), "draw or chain identity")

    def test_member_length_and_crc_binding(self) -> None:
        altered = members()
        altered["winner_codes.npy"] = altered["winner_codes.npy"][:-1]
        self.rejected(self.request(self.archive(altered)), "member length differs")
        altered = members()
        altered["winner_codes.npy"] += b"\0"
        self.rejected(self.request(self.archive(altered)), "member length differs")

        artifact = self.archive()
        path = self.directory / artifact["path"]
        raw = bytearray(path.read_bytes())
        central = raw.find(b"PK\x01\x02")
        self.assertGreaterEqual(central, 0)
        # Change the central-directory CRC while keeping compressed bytes intact.
        raw[central + 16] ^= 1
        path.write_bytes(raw)
        artifact["sha256"] = hashlib.sha256(raw).hexdigest()
        self.rejected(self.request(artifact), "CRC")

    def test_duplicate_and_encrypted_members(self) -> None:
        artifact = self.archive()
        path = self.directory / artifact["path"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("winner_codes.npy", members()["winner_codes.npy"])
        artifact["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.rejected(self.request(artifact), "exact nine")

        artifact = self.archive()
        raw = bytearray(path.read_bytes())
        central = raw.find(b"PK\x01\x02")
        self.assertGreaterEqual(central, 0)
        # ZIP general-purpose flag bit 0 marks encryption in both headers.
        raw[6] |= 1
        raw[central + 8] |= 1
        path.write_bytes(raw)
        artifact["sha256"] = hashlib.sha256(raw).hexdigest()
        self.rejected(self.request(artifact), "encrypted or unsupported")


if __name__ == "__main__":
    unittest.main()
