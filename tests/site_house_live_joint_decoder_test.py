"""Invented NPZ bytes for the narrow offline House winner decoder."""
from __future__ import annotations

import hashlib
import json
import math
import copy
import struct
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile
from unittest.mock import patch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DECODER = ROOT / "scripts/decode_house_live_joint.py"
RACE_IDS = [f"2026:{'ABCDE'[index // 100]}A-{index % 100:02d}:house:regular"
            for index in range(435)]
ROWS = [bytes([0] * 435), bytes([1] * 434 + [2]),
        bytes([2] + [0] * 434), bytes([1] * 435)]
CONFIGURATION = {"id": "invented-current-fundamentals", "base": "fundamentals",
                 "polling": "none", "uncertainty": "standard"}


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


def combined_members() -> dict[str, bytes]:
    parts = members()
    parts["source_run_index.npy"] = npy("<i2", (4,), struct.pack("<hhhh", 0, 0, 1, 1))
    parts["source_joint_draw_id.npy"] = npy("<i8", (4,), struct.pack("<qqqq", 0, 1, 0, 1))
    return parts


def combined_plan() -> dict:
    sources = []
    for index in range(2):
        directory = f"predictions-v{index + 1}/invented"
        prefix = f"research/house-live-20261025/{directory}"
        sources.append({"directory": directory,
                        "receipt": {"path": f"{prefix}/receipt.json", "sha256": f"{index + 1:064x}"},
                        "plan": {"path": f"{prefix}/plan.json", "sha256": f"{index + 3:064x}"},
                        "settings": {"chains": 1, "draws": 2, "warmup": 0, "seed": 100 + index},
                        "joint_draw_offset": 2 * index, "predictive_chain_offset": index})
    return {"schema": "HouseLiveCombinedPredictionPlanV1", "issued_at": "2026-10-25T16:00:00Z",
            "configuration": dict(CONFIGURATION), "pilot": False,
            "settings": {"chains": 2, "draws": 2, "warmup_per_source": 0},
            "source_runs": sources, "inputs": [row["receipt"] for row in sources],
            "source": {"path": "research/house-live-20261025/combine_predictions.py", "sha256": "a" * 64},
            "polling_error_scales": {name: .1 for name in ("independent", "cycle", "race", "house", "movement")},
            "exact_electoral_rule_simulation": False, "source_2026_outcomes_loaded": False,
            "combination": "All source rows retained in source order."}


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

    def plan_pin(self, plan: dict) -> dict:
        path = self.directory / "combined-plan.json"
        path.write_text(json.dumps(plan))
        return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def combined_request(self, artifact: dict, plan: dict | None = None) -> dict:
        return self.request(artifact, combinedPlan=self.plan_pin(plan or combined_plan()),
                            configuration=dict(CONFIGURATION))

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

    def test_seven_million_draw_plan_is_admitted_by_request_bound_but_archive_rows_are_still_proven(self) -> None:
        plan = combined_plan()
        plan["settings"] = {"chains": 20, "draws": 350_000, "warmup_per_source": 5000}
        template = plan["source_runs"][0]
        plan["source_runs"] = []
        for index in range(5):
            source = copy.deepcopy(template)
            source["directory"] = f"predictions-v{index + 1}/invented"
            prefix = f"research/house-live-20261025/{source['directory']}"
            source["receipt"]["path"] = f"{prefix}/receipt.json"
            source["plan"]["path"] = f"{prefix}/plan.json"
            source["settings"] = {"chains": 4, "draws": 350_000, "warmup": 5000, "seed": 100 + index}
            source["joint_draw_offset"] = index * 1_400_000
            source["predictive_chain_offset"] = index * 4
            plan["source_runs"].append(source)
        plan["inputs"] = [source["receipt"] for source in plan["source_runs"]]
        artifact = self.archive(combined_members())
        request = self.combined_request(artifact, plan)
        request.update(count=7_000_000, chainCount=20)
        # The declared size is accepted, but four actual rows cannot prove 7M.
        self.rejected(request, "dtype, shape, or order")
        plan["source_runs"][4]["settings"]["seed"] = 100
        request["combinedPlan"] = self.plan_pin(plan)
        self.rejected(request, "layout or seed")
        self.rejected({**request, "count": 7_000_001, "chainCount": 1}, "Draw count or chain count")
        self.rejected({**request, "chainCount": 21}, "Draw count or chain count")

    def test_zip64_local_headers_and_zip64_end_directory_are_supported(self) -> None:
        path = self.directory / "joint-zip64.npz"
        with patch.object(zipfile, "ZIP64_LIMIT", 1024):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    for name, value in members().items():
                        with archive.open(name, "w", force_zip64=True) as stream:
                            stream.write(value)
        archive_bytes = path.read_bytes()
        self.assertIn(b"PK\x06\x06", archive_bytes, "fixture must contain the ZIP64 end-of-central-directory record")
        self.assertEqual(archive_bytes[:4], b"PK\x03\x04")
        self.assertEqual(struct.unpack_from("<II", archive_bytes, 18), (0xFFFFFFFF, 0xFFFFFFFF))
        name_length, extra_length = struct.unpack_from("<HH", archive_bytes, 26)
        local_extra = archive_bytes[30 + name_length:30 + name_length + extra_length]
        self.assertIn(b"\x01\x00", local_extra, "fixture must contain a local ZIP64 extra field")
        artifact = {"path": path.name, "sha256": hashlib.sha256(archive_bytes).hexdigest()}
        result = self.run_decoder(self.request(artifact))
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertEqual(result.stdout, b"".join(ROWS))

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

    def test_combined_archive_preserves_source_identity(self) -> None:
        artifact = self.archive(combined_members())
        result = self.run_decoder(self.combined_request(artifact))
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertEqual(result.stdout, b"".join(ROWS))
        self.assertEqual(result.stderr, b"")
        self.rejected(self.request(artifact), "exact nine original or eleven combined")
        original = self.archive(members())
        self.rejected(self.combined_request(original), "exact nine original or eleven combined")

    def test_combined_provenance_arrays_and_plan_pin(self) -> None:
        altered = combined_members()
        altered["source_run_index.npy"] = npy("<i2", (4,), struct.pack("<hhhh", 0, 0, 0, 1))
        self.rejected(self.combined_request(self.archive(altered)), "source-run or local-draw")
        altered = combined_members()
        altered["source_joint_draw_id.npy"] = npy("<i8", (4,), struct.pack("<qqqq", 0, 1, 2, 1))
        self.rejected(self.combined_request(self.archive(altered)), "source-run or local-draw")
        altered = combined_members()
        altered["source_joint_draw_id.npy"] = npy("<i4", (4,), struct.pack("<iiii", 0, 1, 0, 1))
        self.rejected(self.combined_request(self.archive(altered)), "dtype, shape, or order")

        artifact = self.archive(combined_members())
        request = self.combined_request(artifact)
        request["combinedPlan"]["sha256"] = "0" * 64
        self.rejected(request, "Combined plan fingerprint mismatch")
        request = self.combined_request(artifact)
        request["combinedPlan"]["path"] = "../combined-plan.json"
        self.rejected(request, "Combined plan pin has an unsafe path")
        request = self.combined_request(artifact)
        (self.directory / "combined-plan.json").write_text("{}")
        self.rejected(request, "Combined plan fingerprint mismatch")
        request = self.combined_request(artifact)
        request["configuration"]["id"] = "different-valid-id"
        self.rejected(request, "configuration differs")

    def test_combined_plan_requires_complete_independent_layout(self) -> None:
        artifact = self.archive(combined_members())
        bad = copy.deepcopy(combined_plan())
        bad["source_runs"][1]["joint_draw_offset"] = 1
        self.rejected(self.combined_request(artifact, bad), "layout or seed")
        bad = copy.deepcopy(combined_plan())
        bad["source_runs"][1]["predictive_chain_offset"] = 0
        self.rejected(self.combined_request(artifact, bad), "layout or seed")
        bad = copy.deepcopy(combined_plan())
        bad["source_runs"][1]["settings"]["seed"] = 100
        self.rejected(self.combined_request(artifact, bad), "layout or seed")
        bad = copy.deepcopy(combined_plan())
        bad["source_runs"][1]["settings"]["draws"] = 1
        self.rejected(self.combined_request(artifact, bad), "layout or seed")
        bad = copy.deepcopy(combined_plan())
        bad["inputs"].pop()
        self.rejected(self.combined_request(artifact, bad), "evidence pins differ")
        bad = copy.deepcopy(combined_plan())
        bad["settings"]["chains"] = 4
        self.rejected(self.combined_request(artifact, bad), "total draw or chain layout")


if __name__ == "__main__":
    unittest.main()
