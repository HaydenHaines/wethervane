#!/usr/bin/env python3
"""Compare two 2019 author files without creating or admitting a training set."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
FILES = {
    "combined": (HERE / "raw/CNNData1992to2018-20191120.csv", "f761a376033cb9b5b9e67e53e07bfd260c9865519d0bf45e30755b6b473dc6ca"),
    "2018": (BASE / "m01-chen-archive-root-v1/raw/CNNData2018-20191120.csv", "23e22ac1a64962d21d2f31fdfb8ca903dd3ecd9b7c9514661c726be610000c6c"),
}


def read(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"source changed: {path}")
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def main():
    columns, combined = read(*FILES["combined"])
    standalone_columns, standalone = read(*FILES["2018"])
    if not all(c in standalone_columns for c in columns):
        raise ValueError("standalone columns do not cover combined columns")
    source_rows = [r for r in combined if r["cycle"] == "2018"]
    left = [tuple(row[c] for c in columns) for row in source_rows]
    right = [tuple(row[c] for c in columns) for row in standalone]
    result = {
        "schema": "Chen2019Combined2018ProjectionV1",
        "source_sha256": {k: v[1] for k, v in FILES.items()},
        "combined_rows": len(combined), "combined_2018_rows": len(source_rows),
        "standalone_rows": len(standalone), "compared_columns": columns,
        "standalone_extra_columns": [c for c in standalone_columns if c not in columns],
        "ordered_2018_projection_equals_standalone": left == right,
        "scope": "Exact CSV field equality for the compared projection, not raw byte equality, source publication date, poll identity or historical forecast admission. Both files include previously observed 2018 outcome fields and are fully parsed.",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
