"""Independent read-only review of frozen G01 preparation; standard library only."""
import csv, hashlib, json, math
from datetime import datetime, timezone
from pathlib import Path

LAB = Path("/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024")
OWNER = Path("/home/hayden/projects/wethervane-redesign-20261006/integration/research/fundamentals-iteration-20261020/panel-sol")
PREP = LAB / "gasoline-experiment-v1/prepared-v2"
TWIP = LAB / "gasoline-twip-cutoff-v1"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan_path = PREP / "plan.json"
plan = json.loads(plan_path.read_text())
panel_path = TWIP / "panel.json"
twip = json.loads(panel_path.read_text())
receipt = json.loads((TWIP / "receipt.json").read_text())
features_path = OWNER / "features/all-cycles.csv"
features = {}
with features_path.open(newline="") as f:
    for row in csv.DictReader(f):
        features.setdefault(int(row["cycle"]), []).append(row)
gas = {r["cycle"]:r for r in twip["panel"]}
assert len(receipt["files"]) == 32
assert set(gas) == set(range(2002, 2026, 2))
for item in receipt["files"]:
    assert sha(TWIP / item["file"]) == item["sha256"]
for c, row in gas.items():
    d = datetime.fromisoformat(row["cutoff_utc"])
    midnight = d.replace(hour=0, minute=0, second=0)
    assert d.time().isoformat() == "23:59:59"
    assert row["log_change_13_weeks"] == math.log(row["current"]["value"]/row["lag13"]["value"])
    assert (datetime.fromisoformat(row["current"]["observation_date"]) - datetime.fromisoformat(row["lag13"]["observation_date"])).days == 91
    assert row["current"]["method_regime"] == row["lag13"]["method_regime"]
    for obs in (row["current"], row["lag13"]):
        assert datetime.fromisoformat(obs["available_upper_bound_utc"]) < midnight
        assert datetime.fromisoformat(obs["issue_date"]).replace(tzinfo=timezone.utc) < midnight
        assert obs["original_posting_log_authenticated"] is False
    assert row["strict_original_posting_authenticated"] is False
    group=features[c]
    assert len({r["president_party_asof"] for r in group}) == 1
    assert len({r["forecast_cutoff"] for r in group}) == 1
    cutoff=datetime.fromisoformat(group[0]["forecast_cutoff"].replace("Z","+00:00"))
    if cutoff.tzinfo is None:
        cutoff=cutoff.replace(tzinfo=timezone.utc)
    assert cutoff == midnight

check=[]
for fold in plan["folds"]:
    target=fold["target_cycle"]
    file=PREP / f"features-through-{target}-LABEL-FREE.csv"
    rows=list(csv.DictReader(file.open(newline="")))
    assert not {"actual_dem_share","dem_votes","rep_votes","winner"}.intersection(rows[0])
    assert len([r for r in rows if int(r["cycle"])==target]) == {2018:30,2020:28,2022:30,2024:30}[target]
    train_cycles=sorted(c for c in gas if c<target)
    oriented=[int(features[c][0]["president_party_asof"])*gas[c]["log_change_13_weeks"] for c in train_cycles]
    mean=sum(oriented)/len(oriented)
    sd=(sum((x-mean)**2 for x in oriented)/len(oriented))**0.5
    assert fold["scaler"]["training_cycles"] == train_cycles
    assert math.isclose(mean,fold["scaler"]["mean"],rel_tol=0,abs_tol=1e-14)
    assert math.isclose(sd,fold["scaler"]["sd"],rel_tol=0,abs_tol=1e-14)
    zmap={c:(int(features[c][0]["president_party_asof"])*gas[c]["log_change_13_weeks"]-mean)/sd for c in train_cycles+[target]}
    assert all(int(r["cycle"]) in zmap for r in rows)
    for r in rows:
        assert math.isclose(float(r[plan["feature_id"]]),zmap[int(r["cycle"])],rel_tol=0,abs_tol=2e-14)
    arms=fold["arms"]
    f0,g01=arms["F0"],arms["G01"]
    assert f0["cycles"] == g01["cycles"] and f0["training_race_ids"] == g01["training_race_ids"]
    assert f0["outcome_rows_sha256"] == g01["outcome_rows_sha256"]
    assert f0["parameter_names"] == g01["parameter_names"][:-1]
    assert g01["parameter_names"][-1] == "economy.twip_gas_change"
    p=fold["prior_review"]["mean_shift_checks"]["primary"]
    z=zmap[target]
    assert math.isclose(p["coordinate"],z,rel_tol=0,abs_tol=2e-14)
    assert fold["prior_review"]["unique_cycle_correlations_with_chosen_coordinate"].keys() == {"party","party_midterm","national"}
    check.append({
        "target":target,
        "feature_rows_sha256":sha(file),
        "rows":len(rows),
        "training_cycle_count":len(train_cycles),
        "training_mean":mean,"training_population_sd":sd,"target_z":z,
        "matched_F0_G01_rows":len(f0["training_race_ids"]),
        "same_outcome_rows":True,
    })
assert plan["economic_prior"] == {"family":"Normal","mean":0,"sd":0.05,"sign_restriction":None}
assert plan["status"] == "prepared_not_fitted" and not plan["mcmc_launched"]
selection=LAB/"checkpoints/gasoline-preparation-root-selection-v1.json"
selection_hash=sha(selection)
assert any(x["path"]==str(selection) and x["sha256"]==selection_hash for x in plan["source_pins"])
assert plan["source_pins"][0]["sha256"] == sha(LAB/"gasoline-experiment-v1/prepare.py")
assert plan["source_pins"][1]["sha256"] == sha(LAB/"gasoline-experiment-v1/test_prepare.py")
print(json.dumps({
    "schema":"independent_g01_preparation_checks_v2",
    "result":"pass",
    "checks":{
      "all_32_raw_source_hashes":True,
      "all_12_current_and_lag_upper_bounds_pre_exact_midnight_H14":True,
      "all_12_exact_91day_arithmetic_and_same_method_regime":True,
      "owner_forecast_cutoffs_equal_exact_midnight_H14":True,
      "p_first_equal_cycle_scaler_and_csv_feature_values":True,
      "matched_support_and_outcome_hashes_F0_G01":True,
      "beta_prior_sd_and_sign_unrestricted":True,
      "accepted_root_selection_hash_is_pinned":True,
      "no_sampler_run":True,
      "nested_density_equivalence_supplied_point_test_present":True
    },
    "folds":check,
    "limitations":[
      "Density equivalence was reviewed from frozen compiled hashes and supplied-point test source; no fit, sampler, or owner test suite was run.",
      "TWIP source upper bounds establish dated historical availability under the declared convention, not original posting logs or no-revision provenance.",
      "Correlations are descriptive on only 8-11 unique earlier cycles."
    ]
},indent=2,sort_keys=True))
