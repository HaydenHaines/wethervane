"""Independent arithmetic check of the pinned Ohio descriptive comparison."""
import hashlib
import json
from io import BytesIO
from pathlib import Path

import geopandas as gpd

ROOT = Path("/home/hayden/projects/ballotmancy-house-benchmark")
RESULT = ROOT / "research/house-oh-geometry-comparison-20261027-v1/comparison-v2.json"
RESULT_SHA = "2d74c8846435f1bbc5d6a43214945f7d05bb5ee4f3240de5b8433f16d6b202fd"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    raw = RESULT.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == RESULT_SHA, "comparison result differs")
    saved = json.loads(raw)
    sources = saved["sources"]
    need(len(sources) == 9 and len({item["path"] for item in sources}) == 9, "source roster differs")
    for item in sources:
        path = Path(item["path"])
        need(not path.is_absolute() and ".." not in path.parts and sha(ROOT / path) == item["sha256"], "source pin differs")
    records = {Path(item["path"]).name: item["path"] for item in sources}
    state_path = ROOT / records["districts.geojson"]
    census_path = ROOT / records["body"]
    display_path = ROOT / records["house-2026.geojson"]
    state = gpd.read_file(state_path).to_crs(5070).set_index("district")
    census = gpd.read_file(BytesIO(census_path.read_bytes())).to_crs(5070).set_index("CD120FP")
    frame = gpd.read_file(display_path)
    display = frame[frame.districtKey.str.startswith("OH-")].to_crs(5070).set_index("districtKey")
    expected = {f"OH-{i:02}" for i in range(1, 16)}
    rows = saved["comparisons"]
    need(len(rows) == 15 and {row["district"] for row in rows} == expected, "saved district coverage differs")
    need(set(state.index) == {str(i) for i in range(1, 16)} and len(state) == 15 and
         set(census.index) == {f"{i:02}" for i in range(1, 16)} and len(census) == 15 and
         set(display.index) == expected and len(display) == 15, "geometry district coverage differs")
    absolute_errors, fraction_errors = [], []
    for row in rows:
        district = int(row["district"][-2:])
        a = state.loc[str(district)].geometry
        for name, b in (("state_vs_census", census.loc[f"{district:02}"].geometry),
                        ("state_vs_simplified_display", display.loc[f"OH-{district:02}"].geometry)):
            # Independent decomposition of symmetric difference and union areas.
            diff = a.difference(b).area + b.difference(a).area
            union = a.area + b.area - a.intersection(b).area
            need(union > 0, "empty polygon union")
            absolute_errors.append(abs(diff / 1e6 - row[name]["symmetric_difference_km2"]))
            fraction_errors.append(abs(diff / union - row[name]["fraction_of_union"]))
    need(max(absolute_errors) < 1e-8 and max(fraction_errors) < 1e-10, "saved arithmetic differs")
    need(sha(RESULT) == RESULT_SHA and all(sha(ROOT / i["path"]) == i["sha256"] for i in sources),
         "sources changed during review")
    print(json.dumps({"schema": "HouseOhioGeometryIndependentArithmeticV1", "result_sha256": RESULT_SHA,
        "source_pins_checked": len(sources), "districts": len(rows), "pairs": len(absolute_errors),
        "max_absolute_km2_difference": max(absolute_errors), "max_fraction_difference": max(fraction_errors),
        "legal_map_verified": False, "affected_voters_verified": False,
        "historical_capture_vintage_verified": False, "publication_qualified": False}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
