#!/usr/bin/env python3
"""Independent, label-free G01 prior geometry recomputation; standard library only."""
import csv, hashlib, json, math, statistics
from pathlib import Path

LAB = Path("/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024")
OWNER_PANEL = Path("/home/hayden/projects/wethervane-redesign-20261006/integration/research/fundamentals-iteration-20261020/panel-sol/features/all-cycles.csv")
TWIP = LAB / "gasoline-twip-cutoff-v1/panel.json"
PINNED = {
    OWNER_PANEL: "0a2a1b9c15a83f33406828e7b7e2c1b24230570ed0639c995c410b2bbe9ec786",
    TWIP: "9f4627170f846bfa6768e2bda43659a6218c4a577a2daf92989b14bd78ed3902",
}
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
for path, expected in PINNED.items():
    assert digest(path) == expected, f"source hash mismatch: {path}"

panel = json.loads(TWIP.read_text())
raw = {row["cycle"]: row["log_change_13_weeks"] for row in panel["panel"]}
assert len(raw) == 12 and set(raw) == set(range(2002, 2026, 2))
by_cycle = {}
with OWNER_PANEL.open(newline="") as f:
    for row in csv.DictReader(f):
        c = int(row["cycle"])
        if c not in raw:
            continue
        by_cycle.setdefault(c, []).append(row)
assert set(by_cycle) == set(raw)
cycles = {}
for c, rows in by_cycle.items():
    assert len({r["president_party_asof"] for r in rows}) == 1
    assert len({r["midterm"] for r in rows}) == 1
    assert len({r["prior_national_presidential_dem_share"] for r in rows}) == 1
    p, m = int(rows[0]["president_party_asof"]), int(rows[0]["midterm"])
    share = float(rows[0]["prior_national_presidential_dem_share"])
    assert p in (-1, 1) and m in (0, 1) and 0 < share < 1
    assert not {"actual_dem_share", "dem_votes", "rep_votes", "winner"} & set(rows[0])
    cycles[c] = {
        "raw": float(raw[c]), "p": p, "m": m,
        "national": math.log(share / (1-share)), "race_rows": len(rows),
    }

def corr(a, b):
    am, bm = statistics.mean(a), statistics.mean(b)
    da, db = [x-am for x in a], [x-bm for x in b]
    return sum(x*y for x,y in zip(da,db)) / math.sqrt(sum(x*x for x in da)*sum(y*y for y in db))

results = []
for target in (2018, 2020, 2022, 2024):
    train = [cycles[c] for c in sorted(cycles) if c < target]
    assert len(train) == {2018:8, 2020:9, 2022:10, 2024:11}[target]
    u = [r["p"] * r["raw"] for r in train]
    mean, sd = statistics.mean(u), statistics.pstdev(u)
    tr = cycles[target]
    ut = tr["p"] * tr["raw"]
    z = (ut-mean)/sd
    logit_bound = 1.96 * 0.05 * abs(z)
    share_bound = 1/(1+math.exp(-logit_bound)) - 0.5
    x = [(v-mean)/sd for v in u]
    results.append({
        "target": target,
        "training_cycles": sorted(c for c in cycles if c < target),
        "training_cycle_count": len(train),
        "training_race_rows": sum(r["race_rows"] for c,r in cycles.items() if c < target),
        "oriented_train_mean": mean,
        "oriented_train_population_sd": sd,
        "training_raw_change_minmax": [min(r["raw"] for r in train), max(r["raw"] for r in train)],
        "target_raw_change": tr["raw"],
        "target_raw_change_outside_training_range": not min(r["raw"] for r in train) <= tr["raw"] <= max(r["raw"] for r in train),
        "training_oriented_change_minmax": [min(u), max(u)],
        "target_oriented_change": ut,
        "target_standardized_z": z,
        "target_oriented_value_outside_training_range": not min(u) <= ut <= max(u),
        "target_95pct_incremental_logit_shift": [-logit_bound, logit_bound],
        "target_95pct_share_shift_at_fixed_baseline_0_5": [-share_bound, share_bound],
        "training_unique_cycle_correlations": {
            "gas_z_vs_party": corr(x, [r["p"] for r in train]),
            "gas_z_vs_party_x_midterm": corr(x, [r["p"]*r["m"] for r in train]),
            "gas_z_vs_national_prior_presidential_logit": corr(x, [r["national"] for r in train]),
        },
    })
print(json.dumps({"feature": "z(pres_party * log_gas_change_13wk)", "results": results}, indent=2, sort_keys=True))
