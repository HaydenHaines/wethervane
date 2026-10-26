#!/usr/bin/env python3
"""Extract the five Natural Earth Great Lakes into the pinned display mask.

Uses the repository's existing geopandas/Shapely environment. The downloaded
archive is cached outside the repository and must match the Natural Earth 5.0.0
SHA256 recorded below before any geometry is emitted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_lakes.zip"
ARCHIVE_SHA256 = "0803a06f9c3cb4671d89b68c48b142aad9366ba40f665245e12a913fbc61722a"
NAMES = {"Lake Erie", "Lake Huron", "Lake Michigan", "Lake Ontario", "Lake Superior"}


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build(archive: Path, output: Path) -> dict[str, str | int]:
    import geopandas as gpd

    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            archive.write_bytes(response.read())
    raw = archive.read_bytes()
    if digest(raw) != ARCHIVE_SHA256:
        raise ValueError(f"Natural Earth source archive fingerprint changed: {URL}")
    frame = gpd.read_file(archive)
    selected = frame[frame["name"].isin(NAMES)].sort_values("name")
    if set(selected["name"]) != NAMES or len(selected) != 5 or selected.crs.to_epsg() != 4326:
        raise ValueError("Expected the five named Great Lakes in WGS84 from Natural Earth 5.0.0")
    features = [{"type": "Feature", "properties": {"name": row["name"]},
                 "geometry": row.geometry.__geo_interface__} for _, row in selected.iterrows()]
    content = (json.dumps({"type": "FeatureCollection", "features": features},
                          separators=(",", ":"), allow_nan=False) + "\n").encode()
    output.mkdir(parents=True, exist_ok=True)
    (output / "great-lakes-display.json").write_bytes(content)
    receipt = {
        "schema": "HouseGreatLakesDisplayGeometryV1",
        "purpose": "Display-only water mask drawn above forecast district paths; not a district assignment or election boundary.",
        "sourceName": "Natural Earth 1:10m Lakes + Reservoirs",
        "sourceVersion": "5.0.0",
        "sourceUrl": "https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-lakes/",
        "downloadUrl": URL,
        "sourceArchiveSha256": ARCHIVE_SHA256,
        "sourceNote": "Selected the five named Great Lakes from the Natural Earth 5.0.0 1:10m natural/artificial lake dataset. Geometry is generalized for small-scale cartographic display. Natural Earth describes the basic lakes dataset as suitable to approximately 1:30,000,000; verify shoreline precision against a more detailed official source for legal or analytical uses.",
        "license": "Public domain; Natural Earth data is public domain.",
        "features": sorted(NAMES),
        "file": "great-lakes-display.json",
        "sha256": digest(content),
        "featureCount": len(features),
        "builder": "scripts/build_great_lakes_display.py",
        "rendering": "Each lake feature is passed through lib/district-projection.ts for D3 ring winding normalization on a display copy; source geometry bytes and district identity are unchanged.",
    }
    (output / "great-lakes-display.receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return {"file": receipt["file"], "sha256": receipt["sha256"], "featureCount": receipt["featureCount"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("/tmp/ne_10m_lakes.zip"))
    parser.add_argument("--output", type=Path, default=ROOT / "data/house-live/geography")
    args = parser.parse_args()
    print(json.dumps(build(args.archive, args.output)))
