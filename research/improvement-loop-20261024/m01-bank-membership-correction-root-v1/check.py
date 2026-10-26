#!/usr/bin/env python3
"""Independently compare the pinned compatibility artifact's H0/F0 key sets."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "momentum-preflight-bank-compatibility-sol-v1/compatibility.json"
EXPECTED = "53337fcbd640c1ece9c31f4c2c44cf107bbdcd60bc5bee912716109a3806ba05"


def main():
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED:
        raise ValueError("compatibility artifact changed")
    banks = json.loads(raw)["banks"]
    rows = []
    for year in (2018, 2020, 2022, 2024):
        matches = [b for b in banks if b["year"] == year and b["arm"] in {"H0", "F0"}]
        if len(matches) != 2 or {b["arm"] for b in matches} != {"H0", "F0"}:
            raise ValueError("expected one bank per arm")
        groups = {b["arm"]: set(b["training_race_ids"]) for b in matches}
        if any(len(b["training_race_ids"]) != len(groups[b["arm"]]) for b in matches):
            raise ValueError("duplicate event key")
        rows.append({"target_cycle": year, "H0_count": len(groups["H0"]), "F0_count": len(groups["F0"]),
                     "H0_only": sorted(groups["H0"] - groups["F0"]), "F0_only": sorted(groups["F0"] - groups["H0"])})
    print(json.dumps({"schema": "M01BankMembershipCorrectionV1", "compatibility_sha256": EXPECTED,
                      "results": rows,
                      "scope": "Equality in the pinned compatibility artifact only; raw-bank equality and actual density/factor ownership require their own evidence."}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
