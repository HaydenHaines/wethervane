"""Seal the final packet plus explicit correction layers; refuses overwrite."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
DEST = OUT / "artifact-correction-manifest.json"


def digest(p):
    b = p.read_bytes()
    return {"path": p.name, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}


def main():
    if DEST.exists():
        raise RuntimeError("refusing to replace correction manifest")
    first = OUT / "artifact-manifest.json"
    index = OUT / "authoritative-index-v1.json"
    if not first.exists() or not index.exists():
        raise RuntimeError("initial packet and authoritative index are required")
    files = sorted((p for p in OUT.iterdir() if p.is_file() and p != DEST), key=lambda p: p.name)
    doc = {
        "schema": "MomentumPreflightCoverageFinalCorrectionManifestV1",
        "first_pass_manifest_sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
        "authoritative_index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        "files": [digest(p) for p in files],
        "scope": "No model fitting, optimization, scoring, or new poll acquisition.",
    }
    with DEST.open("x") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
