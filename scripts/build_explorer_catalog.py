#!/usr/bin/env python3
"""Join exported run indexes after verifying every public bundle's bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "public" / "data" / "explorer"
DEFAULTS = {
    "2026": "current-polling",
    "2022": "history-2022-FP-1",
    "2018": "history-2018-FP-1",
}


def build_catalog(data_dir: Path) -> dict:
    entries: list[dict] = []
    for index_name in ("current-index.json", "history-index.json"):
        rows = json.loads((data_dir / index_name).read_text())
        if not isinstance(rows, list):
            raise ValueError(f"{index_name} must contain a run-entry array")
        entries.extend(rows)
    ids: set[str] = set()
    configurations: set[tuple] = set()
    for entry in entries:
        run_id = entry["id"]
        if run_id in ids:
            raise ValueError(f"Duplicate run ID: {run_id}")
        ids.add(run_id)
        relative = entry["href"].removeprefix("/data/explorer/")
        if (not entry["href"].startswith("/data/explorer/")
                or relative != Path(relative).name or not relative.endswith(".json")):
            raise ValueError(f"Unsafe public bundle reference: {entry['href']}")
        raw = (data_dir / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"Bundle hash mismatch: {run_id}")
        bundle = json.loads(raw)
        if bundle["schemaVersion"] != 1:
            raise ValueError(f"Unsupported bundle schema: {run_id}")
        metadata = {key: value for key, value in entry.items() if key not in {"href", "sha256"}}
        if bundle["run"] != metadata:
            raise ValueError(f"Catalog metadata differs from bundle: {run_id}")
        components = entry["components"]
        configuration = (
            entry["year"], entry["dateId"],
            components["base"], components["polling"], components["uncertainty"],
        )
        if configuration in configurations:
            raise ValueError(f"Ambiguous selectable configuration: {configuration}")
        configurations.add(configuration)
        if entry["historical"] != (entry["year"] < 2026):
            raise ValueError(f"Incorrect historical status: {run_id}")
        if entry["historical"] and (entry["strictAsOf"] or bundle["tipping"] is not None):
            raise ValueError(f"Historical archive claims unsupported timing or margins: {run_id}")
    for year, default_id in DEFAULTS.items():
        if not any(entry["id"] == default_id and str(entry["year"]) == year for entry in entries):
            raise ValueError(f"Missing default for {year}: {default_id}")
    return {
        "schemaVersion": 1,
        "defaults": DEFAULTS,
        "runs": sorted(entries, key=lambda entry: (-entry["year"], entry["id"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    catalog = build_catalog(args.data_dir)
    output = args.data_dir / "registry.json"
    output.write_text(json.dumps(catalog, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {len(catalog['runs'])} validated configurations to {output}")


if __name__ == "__main__":
    main()
