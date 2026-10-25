"""Exact H14-midnight CPI selection using the reviewed v1 parser/inventory.

This version adds only the missing earlier August 2004 release, then applies
the count graph's UTC midnight cutoff. It never edits the v1 packet.
"""

import hashlib
import importlib.util
import json
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
V1 = HERE.parent / "economic-vintages-v1"
PARSER_PATH = V1 / "cpi_vintages.py"
SERIES = {"headline": "CUUR0000SA0", "core": "CUUR0000SA0L1E"}
NEW_SOURCE_URL = "https://fraser.stlouisfed.org/files/docs/publications/bls/newsreleases/cpi/cpi_20040916.pdf"
EXPECTED_V1 = {
    "cpi_vintages.py": "42473ae777f1e17c1704b93894d9cb37bf92ea456a0a6245d7958da29198dafc",
    "rows.json": "02d54e39994fe41500c33c881b0fc70389deea41f66073d35ebb62e3e89099c4",
    "selections.json": "18cb3bd821ac036658423e6d1495d2295fd5d9430b65c58bbc7b200804b918cf",
    "receipt.json": "7635903576d1154634fe96b9fcab9fcadc261f01827af92eb3ddd75810e22f6f",
}
NEW_RELEASE = {
    "reference_month": "2004-08",
    "release_date": "2004-09-16",
    "original_url": "https://www.bls.gov/news.release/history/cpi_091604.txt",
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_v1_parser():
    validate_v1_inputs()
    spec = importlib.util.spec_from_file_location("economic_vintages_v1_parser", PARSER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load pinned v1 parser")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_v1_inputs(v1_dir=None):
    """Fail closed against reviewed v1 identities before parsing or writes."""
    root = Path(v1_dir) if v1_dir is not None else V1
    for name, expected_hash in EXPECTED_V1.items():
        path = root / name
        if not path.is_file() or _sha(path) != expected_hash:
            raise ValueError(f"pinned v1 input changed or missing: {name}")
    receipt = json.loads((root / "receipt.json").read_text())
    if not isinstance(receipt.get("source_files"), list) or len(receipt["source_files"]) != 20:
        raise ValueError("pinned v1 receipt source inventory is malformed")
    for entry in receipt["source_files"]:
        rel = Path(entry.get("file", ""))
        if rel.is_absolute() or not rel.parts or any(part in ("", ".", "..") for part in rel.parts):
            raise ValueError("unsafe path in pinned v1 source inventory")
        path = (root / rel).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"pinned v1 source missing or outside bundle: {entry.get('file')}")
        if path.stat().st_size != entry.get("bytes") or _sha(path) != entry.get("sha256"):
            raise ValueError(f"pinned v1 source changed: {entry.get('file')}")


v1 = _load_v1_parser()


def cutoff_for(year, horizon_days=14):
    if type(horizon_days) is not int or horizon_days < 0:
        raise ValueError("horizon must be nonnegative integer calendar days")
    day = v1.election_day(year) - timedelta(days=horizon_days)
    return datetime.combine(day, time(0, 0), timezone.utc).isoformat()


def input_manifest():
    return {
        "schema": "economic_cutoff_adapter_v2_inputs_v1",
        "cutoff_convention": "inclusive_H14_calendar_day_start_00_00_UTC",
        "v1_directory": "../economic-vintages-v1",
        "v1_parser": {"file": "../economic-vintages-v1/cpi_vintages.py", "sha256": EXPECTED_V1["cpi_vintages.py"]},
        "v1_rows": {"file": "../economic-vintages-v1/rows.json", "sha256": EXPECTED_V1["rows.json"]},
        "v1_selections": {"file": "../economic-vintages-v1/selections.json", "sha256": EXPECTED_V1["selections.json"]},
        "v1_receipt": {"file": "../economic-vintages-v1/receipt.json", "sha256": EXPECTED_V1["receipt.json"]},
    }


def build():
    validate_v1_inputs()
    pinned_manifest = input_manifest()
    existing_manifest = json.loads((HERE / "inputs.json").read_text())
    if existing_manifest != pinned_manifest:
        raise ValueError("v2 pinned inputs manifest changed; refusing to re-pin")
    acquisition = json.loads((HERE / "acquisition.json").read_text())
    source = acquisition["source"]
    if source["url"] != NEW_SOURCE_URL or source["status"] != "downloaded_and_verified":
        raise ValueError("new 2004 source acquisition receipt mismatch")
    pdf = HERE / source["file"]
    if _sha(pdf) != source["sha256"] or pdf.stat().st_size != source["bytes"]:
        raise ValueError("new source bytes differ from acquisition receipt")
    base = json.loads((V1 / "rows.json").read_text())["rows"]
    added = v1.parse_release(pdf, NEW_RELEASE, source)
    if [(r["kind"], r["value"], r["release_utc"]) for r in added] != [
        ("headline", 2.7, "2004-09-16T12:30:00+00:00"),
        ("core", 1.7, "2004-09-16T12:30:00+00:00"),
    ]:
        raise ValueError("independently reviewed August 2004 CPI cells changed")
    all_rows = [*base, *added]
    selections = []
    for year in range(1998, 2025, 2):
        for kind, series_id in SERIES.items():
            cutoff = cutoff_for(year)
            selection = v1.select(
                all_rows,
                cutoff_utc=cutoff,
                election_date=v1.election_day(year).isoformat(),
                horizon_days=14,
                series_id=series_id,
            )
            selections.append({
                "cycle": year,
                "kind": kind,
                "cutoff_utc": cutoff,
                "election_date": v1.election_day(year).isoformat(),
                "horizon_days": 14,
                **selection,
            })

    baseline = json.loads((V1 / "selections.json").read_text())["selections"]
    before = {(s["cycle"], s["kind"]): s for s in baseline}
    deltas = []
    for selected in selections:
        old = before[(selected["cycle"], selected["kind"])]
        oldrow, newrow = old.get("row"), selected.get("row")
        oldsig = None if oldrow is None else (oldrow["reference_month"], oldrow["value"], oldrow["release_utc"])
        newsig = None if newrow is None else (newrow["reference_month"], newrow["value"], newrow["release_utc"])
        if oldsig != newsig:
            deltas.append({
                "cycle": selected["cycle"],
                "kind": selected["kind"],
                "v1": None if oldrow is None else {"reference_month": oldrow["reference_month"], "value": oldrow["value"], "release_utc": oldrow["release_utc"]},
                "v2": None if newrow is None else {"reference_month": newrow["reference_month"], "value": newrow["value"], "release_utc": newrow["release_utc"]},
            })
    if [(d["cycle"], d["kind"]) for d in deltas] != [(2004, "headline"), (2004, "core")]:
        raise ValueError("expected only the two 2004 CPI selections to change")
    if len(all_rows) != 32 or len(selections) != 28 or any(s["status"] != "available" for s in selections):
        raise ValueError("unexpected v2 source or selection coverage")

    (HERE / "rows.json").write_text(json.dumps({
        "schema": "bls_cpi_release_rows_v2_from_v1_plus_2004_august_v1",
        "v1_base_rows": 30,
        "v2_added_rows": 2,
        "rows": all_rows,
    }, indent=2) + "\n")
    (HERE / "selections.json").write_text(json.dumps({
        "schema": "bls_cpi_h14_midnight_selection_v2",
        "cutoff_convention": "inclusive_H14_calendar_day_start_00_00_UTC",
        "selections": selections,
    }, indent=2) + "\n")
    (HERE / "deltas.json").write_text(json.dumps({
        "schema": "bls_cpi_h14_midnight_v1_comparison_v1",
        "changed_selections": deltas,
    }, indent=2) + "\n")
    summary = {
        "schema": "economic_cutoff_adapter_v2_build_summary_v1",
        "rows": len(all_rows),
        "selections": len(selections),
        "available": sum(s["status"] == "available" for s in selections),
        "changed_from_v1": [(d["cycle"], d["kind"]) for d in deltas],
        "2004_values": {s["kind"]: s["row"]["value"] for s in selections if s["cycle"] == 2004},
        "fit": False,
    }
    (HERE / "build-output.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_receipt()
    print(json.dumps(summary, indent=2))


def _write_receipt():
    paths = [
        "README.md", "acquisition.json", "adapter.py", "inputs.json", "rows.json",
        "selections.json", "deltas.json", "build-output.json", "test_adapter.py",
        "raw/cpi_20040916.pdf",
    ]
    receipt = {
        "schema": "economic_cutoff_adapter_v2_receipt_v1",
        "fit": False,
        "exact_cutoff": "inclusive 00:00 UTC at election day minus fourteen calendar days",
        "input_manifest": json.loads((HERE / "inputs.json").read_text()),
        "files": [{"file": name, "sha256": _sha(HERE / name), "bytes": (HERE / name).stat().st_size} for name in paths if (HERE / name).exists()],
    }
    (HERE / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    build()
