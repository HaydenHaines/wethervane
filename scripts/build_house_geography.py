#!/usr/bin/env python3
"""Build pinned Census display geometry without changing any forecast artifacts.

Run with .venv/bin/python; existing geopandas/pyogrio/shapely suffice. Source
archives stay in the requested cache (default /tmp), never in the repository.
Census supplies the generalized 1:20m boundaries; no local simplification occurs.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
FIPS = dict(zip("01 02 04 05 06 08 09 10 12 13 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 44 45 46 47 48 49 50 51 53 54 55 56".split(), "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split()))

def digest(data):
    return hashlib.sha256(data).hexdigest()

def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()

def build(source, cache, output):
    import geopandas as gpd
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / source.get("cacheFile", source["url"].rsplit("/", 1)[1])
    if not archive.exists():
        with urllib.request.urlopen(urllib.request.Request(source["url"], headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as response:
            archive.write_bytes(response.read())
    raw = archive.read_bytes()
    if digest(raw) != source["sha256"]:
        raise ValueError(f"Source fingerprint changed: {source['url']}")
    frame = gpd.read_file(archive)
    original_count = len(frame)
    excluded = sorted(frame.loc[~frame.STATEFP.isin(FIPS), "STATEFP"].tolist())
    frame = frame[frame.STATEFP.isin(FIPS)].copy().to_crs("EPSG:4326")
    replacement_receipt = None
    if source.get("replacement"):
        import shapely
        replacement = source["replacement"]
        replacement_archive = cache / replacement["cacheFile"]
        if not replacement_archive.exists():
            request = urllib.request.Request(replacement["url"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                replacement_archive.write_bytes(response.read())
        if digest(replacement_archive.read_bytes()) != replacement["sha256"]:
            raise ValueError("Replacement source fingerprint changed")
        state = gpd.read_file(replacement_archive)
        if len(state) != 13 or state.crs.to_epsg() != 32119 or not state.geometry.is_valid.all() or not shapely.coverage_is_valid(state.geometry.array):
            raise ValueError("Expected valid 13-district North Carolina 2019 coverage in EPSG:32119")
        # Coverage simplification preserves shared NC district edges, unlike per-ring thinning.
        simplified = shapely.coverage_simplify(state.geometry.array, replacement["simplificationToleranceMeters"], simplify_boundary=True)
        if not all(shapely.is_valid(simplified)) or not shapely.coverage_is_valid(simplified):
            raise ValueError("NC topology-aware simplification failed")
        state.geometry = simplified
        state = state.to_crs("EPSG:4326")
        state["STATEFP"] = "37"
        state[source["districtField"]] = state["DISTRICT"].astype(str).str.zfill(2)
        if set(state[source["districtField"]]) != {str(i).zfill(2) for i in range(1, 14)}:
            raise ValueError("NC district support differs from 01 through 13")
        frame = frame[frame.STATEFP != "37"]
        import pandas as pd
        frame = gpd.GeoDataFrame(pd.concat([frame, state], ignore_index=True), crs="EPSG:4326")
        replacement_receipt = {"source": replacement, "featureCount": 13, "coverageValidBefore": True, "coverageValidAfter": True, "method": "shapely.coverage_simplify, simplify_boundary=True, EPSG32119 meters"}
    field = source["districtField"]
    frame["districtKey"] = [FIPS[state] + "-" + str(district).zfill(2) for state, district in zip(frame.STATEFP, frame[field])]
    if len(frame) != 435 or frame.districtKey.duplicated().any() or frame.STATEFP.nunique() != 50:
        raise ValueError("Exactly 435 distinct districts across 50 states are required")
    for state_code in FIPS:
        keys = set(frame.loc[frame.STATEFP == state_code, "districtKey"])
        expected = {FIPS[state_code] + "-00"} if len(keys) == 1 else {FIPS[state_code] + "-" + str(i).zfill(2) for i in range(1, len(keys) + 1)}
        if keys != expected:
            raise ValueError(f"Noncanonical district numbering in {FIPS[state_code]}")
    if not frame.geometry.is_valid.all() or frame.geometry.is_empty.any() or not frame.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError("Source polygons are invalid or empty")
    # At-large IDs are supplied as 00 by Census, never inferred from missing district IDs.
    if any(key.endswith("-98") or key.endswith("-ZZ") for key in frame.districtKey):
        raise ValueError("Nonvoting or unassigned district survived filtering")
    features = [{"type": "Feature", "properties": {"districtKey": row.districtKey}, "geometry": row.geometry.__geo_interface__} for row in frame.sort_values("districtKey").itertuples()]
    data = encoded({"type": "FeatureCollection", "features": features})
    file = f"house-{source['cycle']}.geojson"
    output.mkdir(parents=True, exist_ok=True)
    (output / file).write_bytes(data)
    receipt = {"schemaVersion": 1, "cycle": source["cycle"], "mapVintage": f"house_election_map_{source['cycle']}", "admission": source["admission"], "source": source, "sourceFeatureCount": original_count, "excludedNonStateFips": excluded, "featureCount": len(features), "stateCount": 50, "atLargeDistricts": sorted(key for key in frame.districtKey if key.endswith("-00")), "file": file, "sha256": digest(data), "bytes": len(data), "crs": "EPSG:4326", "processing": "Filter explicitly to 50 state FIPS; preserve Census district 00; transform NAD83 to WGS84; sort canonical keys; retain Census 1:20m generalized topology without further simplification.", "limitations": ["Generalized boundaries for map display; consult official state plans for precise legal boundaries.", "Geography admission does not establish forecast support, scores, chamber control or tipping."]}
    if replacement_receipt:
        receipt["replacement"] = replacement_receipt
        receipt["processing"] += " For the 2020 election only, replace North Carolina’s 13 earlier districts with its official plan enacted in 2019. Simplify the 13 districts together as one valid coverage with a 500 meter tolerance before transforming to WGS84."
        receipt["limitations"].append("The 2020 map combines Census Bureau boundaries for 49 states with separately generalized North Carolina boundaries; shoreline and state-border detail may differ where the sources meet.")
    (output / f"house-{source['cycle']}.receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return {"cycle": source["cycle"], "file": file, "sha256": digest(data), "bytes": len(data), "admission": source["admission"]}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=[2024])
    parser.add_argument("--cache", type=Path, default=Path("/tmp"))
    parser.add_argument("--output", type=Path, default=ROOT / "data/house-geography")
    args = parser.parse_args()
    if not all(importlib.util.find_spec(module) for module in ["geopandas", "pyogrio", "shapely"]):
        parser.error("Use the existing .venv/bin/python with geopandas, pyogrio and shapely; no app dependency install is required.")
    manifest = json.loads((ROOT / "data/house-geography/sources.json").read_text())
    sources = {source["cycle"]: source for source in manifest["sources"]}
    for year in args.years:
        if year not in sources:
            parser.error(f"No independently resolved source for {year}; do not reuse another election's boundary edition.")
        print(json.dumps(build(sources[year], args.cache, args.output)))
    editions, pending = [], list(manifest.get("unresolved", []))
    for source in manifest["sources"]:
        receipt_path = args.output / f"house-{source['cycle']}.receipt.json"
        if source["admission"] != "verified":
            pending.append({"cycle": source["cycle"], "reason": source["electionEvidence"], "preparedFile": f"house-{source['cycle']}.geojson"})
            continue
        if not receipt_path.exists():
            continue
        receipt = json.loads(receipt_path.read_text())
        editions.append({"cycle": source["cycle"], "mapVintage": receipt["mapVintage"], "file": receipt["file"], "sha256": receipt["sha256"], "sourceName": ("U.S. Census Bureau and North Carolina General Assembly, 2020 election districts" if source.get("replacement") else f"U.S. Census Bureau, {source['cycle']} election districts"), "sourceUrl": source["url"], "sourceNote": source["electionEvidence"] + " Generalized for map display. Includes the 435 voting districts in the 50 states; excludes Washington, D.C. and the territories.", "electionEvidenceUrl": source["electionEvidenceUrl"], "sourceSha256": source["sha256"], "receipt": receipt_path.name, "featureCount": 435})
        if source.get("replacement"):
            editions[-1]["replacement"] = source["replacement"]
    (args.output / "manifest.json").write_text(json.dumps({"schemaVersion": 1, "editions": editions, "pending": pending}, indent=2) + "\n")

if __name__ == "__main__":
    main()
