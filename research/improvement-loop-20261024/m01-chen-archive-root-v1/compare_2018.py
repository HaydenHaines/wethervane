#!/usr/bin/env python3
"""Screen candidate row pairs in two dated Chen snapshots; do not admit polls."""
import collections
import csv
import hashlib
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAINING = HERE.parent / "momentum-calibration-preparation-sol-v1/prepared-v1/2020/selected-training-polls.csv"
SOURCES = {
    "2019": (HERE / "raw/CNNData2018-20191120.csv", "23e22ac1a64962d21d2f31fdfb8ca903dd3ecd9b7c9514661c726be610000c6c"),
    "2021": (HERE / "raw/CNNdata2018-20210223.csv", "b6fb022254df4a3fe0624b050e04c6e7911c2e3ba593bf3ff9333f798c5a1ea3"),
}
TRAINING_SHA = "22d1e4f6b325fe28a53b97a0d4a4bcd9102ff48e869d4667dbe75f14fb454a56"


def read_pinned(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"input changed: {path}")
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def whole(value):
    try:
        number = float(value)
        return int(number) if math.isfinite(number) and number.is_integer() else None
    except (TypeError, ValueError):
        return None


def name(value):
    return re.sub(r"[^a-z0-9]", "", value.lower())


def screen(source, prepared):
    index = collections.defaultdict(list)
    skipped = collections.Counter()
    for line, row in enumerate(source, 2):
        ident = row["Candidateidentifier"]
        days = whole(row["daysLeft"])
        if len(ident) < 6 or ident[:4] != "2018" or row["cycle"] != "2018":
            skipped["non_2018_or_bad_candidate_prefix"] += 1
        elif days is None:
            skipped["unparseable_or_fractional_days"] += 1
        elif days > 0:
            # Some could concern runoffs or conventions after November 6;
            # do not reinterpret them as pre-election days by abs().
            skipped["positive_days_excluded"] += 1
        else:
            index[("2018", ident[4:6], str(-days), name(row["pollster"]))].append((line, row))
    counts = collections.Counter()
    links = []
    for row in prepared:
        key = (row["cycle"], row["state"], row["days_to_election"], name(row["pollster"]))
        group = index[key]
        if not group:
            continue
        counts["matching_coarse_key"] += 1
        d = [(line, r) for line, r in group if whole(r["Democrat"]) == 1 and whole(r["numberSupport"]) == whole(row["source_democratic_value"]) and whole(r["numberSupport"]) is not None]
        r = [(line, v) for line, v in group if whole(v["Republican"]) == 1 and whole(v["numberSupport"]) == whole(row["source_republican_value"]) and whole(v["numberSupport"]) is not None]
        pairs = [(a, b) for a in d for b in r if whole(a[1]["samplesize"]) is not None and whole(a[1]["samplesize"]) == whole(b[1]["samplesize"])]
        if pairs:
            counts["at_least_one_candidate_pair"] += 1
        if len(pairs) == 1:
            counts["one_candidate_pair_on_screen"] += 1
            (dline, dr), (rline, rr) = pairs[0]
            links.append({"tierney_observation_id": row["observation_id"], "source_dem_line": dline, "source_rep_line": rline, "source_sample_size": whole(dr["samplesize"]), "tierney_reported_n": whole(row["reported_n"])})
    uses = collections.Counter((v["source_dem_line"], v["source_rep_line"]) for v in links)
    return {"source_candidate_rows": len(source), "screen_exclusions": dict(skipped),
            "counts": dict(counts), "distinct_source_line_pairs": len(uses),
            "source_line_pairs_used_multiple_times": sum(v > 1 for v in uses.values()), "candidate_links": links}


def main():
    prepared = [r for r in read_pinned(TRAINING, TRAINING_SHA) if r["cycle"] == "2018"]
    result = {"schema": "M01Chen2018SnapshotCandidateScreenV1", "prepared_sha256": TRAINING_SHA,
              "prepared_2018_rows": len(prepared), "sources": {}}
    for key, (path, expected) in SOURCES.items():
        result["sources"][key] = {"sha256": expected, **screen(read_pinned(path, expected), prepared)}
    result["scope"] = ("Deterministic heuristic candidates, not verified poll/sample identity, original release, Tierney lineage, "
                       "or strict-as-of source admission. Positive source days are conservatively excluded, not classified as erroneous. "
                       "The later source commit postdates the 2020 target. Raw source records contain earlier outcomes but no target predictions are fitted or scored.")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
