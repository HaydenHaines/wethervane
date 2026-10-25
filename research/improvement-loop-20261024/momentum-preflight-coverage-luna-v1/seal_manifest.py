"""Create the final immutable file inventory once; refuses replacement."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
DEST = OUT / "artifact-manifest.json"


def digest(path):
    raw = path.read_bytes()
    return {"path": path.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def main():
    if DEST.exists():
        raise RuntimeError("refusing to replace the final artifact manifest")
    files = sorted((p for p in OUT.iterdir() if p.is_file() and p != DEST), key=lambda p: p.name)
    manifest = {
        "schema": "MomentumPreflightCoverageLunaArtifactManifestV1",
        "root_selection_sha256": "5de976e5facbc84aa9db48e64fcc43864d1b8741fc5d1ddc134cc2e0fa05e5e9",
        "files": [digest(p) for p in files],
        "scope": "No fitting, optimization, scoring, new poll acquisition, or source mutation.",
    }
    with DEST.open("x") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
