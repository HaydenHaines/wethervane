"""Compare forecast predictions with and without xt_* crosstab demographics.

Loads the full prediction pipeline, runs it twice:
  1. With xt_ data (current production — Tier 1 W vectors for Emerson polls)
  2. Without xt_ data (strip xt_ keys — all polls fall back to Tier 3)

Reports per-race theta_national and state_pred differences.
"""
from __future__ import annotations

import copy
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.prediction.county_priors import load_county_priors_with_ridge
from src.prediction.forecast_engine import run_forecast
from src.prediction.generic_ballot import compute_gb_shift

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_data():
    """Load all inputs needed for run_forecast (mirrors predict_2026_types.run)."""
    from src.core import config as _cfg
    from src.assembly.define_races import load_races

    # Type data
    ta_df = pd.read_parquet(PROJECT_ROOT / "data" / "communities" / "type_assignments.parquet")
    county_fips = ta_df["county_fips"].astype(str).str.zfill(5).tolist()
    score_cols = sorted([c for c in ta_df.columns if c.endswith("_score")])
    type_scores = ta_df[score_cols].values

    # County metadata
    states = [_cfg.STATE_ABBR.get(f[:2], "??") for f in county_fips]

    # County votes
    county_votes = np.ones(len(county_fips))
    votes_path = PROJECT_ROOT / "data" / "raw" / "medsl_county_presidential_2024.parquet"
    if votes_path.exists():
        vdf = pd.read_parquet(votes_path)
        vmap = dict(zip(
            vdf["county_fips"].astype(str).str.zfill(5), vdf["totalvotes"],
        ))
        county_votes = np.array([vmap.get(f, 1.0) for f in county_fips])

    # County priors
    county_priors = load_county_priors_with_ridge(county_fips)

    # Type profiles
    tp_path = PROJECT_ROOT / "data" / "communities" / "type_profiles.parquet"
    type_profiles = pd.read_parquet(tp_path) if tp_path.exists() else None

    # Polls
    polls_path = PROJECT_ROOT / "data" / "polls" / "polls_2026.csv"
    polls_df = pd.read_csv(polls_path)
    if "geography" in polls_df.columns and "state" not in polls_df.columns:
        polls_df = polls_df.rename(columns={"geography": "state"})
    if "geo_level" in polls_df.columns:
        polls_df = polls_df[polls_df["geo_level"] == "state"].copy()

    xt_cols = [c for c in polls_df.columns if c.startswith("xt_")]

    polls_by_race: dict[str, list[dict]] = {}
    for race_id, grp in polls_df.groupby("race"):
        if str(race_id).startswith("2026 Generic Ballot"):
            continue
        race_dicts = []
        for _, row in grp.iterrows():
            d = {
                "dem_share": float(row["dem_share"]),
                "n_sample": int(row["n_sample"]) if pd.notna(row["n_sample"]) else 600,
                "state": str(row["state"]),
            }
            for col in xt_cols:
                val = row.get(col)
                if val is not None and pd.notna(val):
                    d[col] = float(val)
            race_dicts.append(d)
        if race_dicts:
            polls_by_race[str(race_id)] = race_dicts

    # Prediction params
    params_path = PROJECT_ROOT / "data" / "config" / "prediction_params.json"
    import json
    params = json.loads(params_path.read_text()) if params_path.exists() else {}

    # Races
    registry = load_races(2026)
    all_race_ids = [r.race_id for r in registry]

    # Generic ballot
    gb_shift = compute_gb_shift().shift

    return {
        "type_scores": type_scores,
        "county_priors": county_priors,
        "states": states,
        "county_fips": county_fips,
        "county_votes": county_votes,
        "type_profiles": type_profiles,
        "polls_by_race": polls_by_race,
        "all_race_ids": all_race_ids,
        "gb_shift": gb_shift,
        "params": params,
        "xt_cols": xt_cols,
    }


def _strip_xt(polls_by_race: dict) -> dict:
    """Return a deep copy with all xt_ keys removed from poll dicts."""
    stripped = {}
    for race_id, polls in polls_by_race.items():
        stripped[race_id] = []
        for p in polls:
            new_p = {k: v for k, v in p.items() if not k.startswith("xt_")}
            stripped[race_id].append(new_p)
    return stripped


def _run_comparison(data: dict) -> None:
    params = data["params"]
    accuracy_path = PROJECT_ROOT / "data" / "config" / "pollster_accuracy.json"

    common_kwargs = dict(
        type_scores=data["type_scores"],
        county_priors=data["county_priors"],
        states=data["states"],
        county_votes=data["county_votes"],
        races=data["all_race_ids"],
        lam=params.get("lam", 1.0),
        mu=params.get("mu", 1.0),
        generic_ballot_shift=data["gb_shift"],
        w_vector_mode=params.get("w_vector_mode", "core"),
        reference_date=str(date.today()),
        type_profiles=data["type_profiles"],
        half_life_days=params.get("half_life_days", 30.0),
        pre_primary_discount=params.get("pre_primary_discount", 0.5),
        accuracy_path=accuracy_path if accuracy_path.exists() else None,
        methodology_weights=params.get("methodology_weights"),
    )

    # Run WITH xt_ data (current production)
    results_with = run_forecast(polls_by_race=data["polls_by_race"], **common_kwargs)

    # Run WITHOUT xt_ data (Tier 3 fallback for all polls)
    polls_no_xt = _strip_xt(data["polls_by_race"])
    results_without = run_forecast(polls_by_race=polls_no_xt, **common_kwargs)

    # Count polls with xt_ data per race
    xt_counts = {}
    for race_id, polls in data["polls_by_race"].items():
        xt_counts[race_id] = sum(
            1 for p in polls if any(k.startswith("xt_") for k in p)
        )

    # Compare
    print("\n" + "=" * 80)
    print("XT_ CROSSTAB IMPACT ANALYSIS")
    print("=" * 80)
    print(f"\nTotal xt_ columns: {len(data['xt_cols'])}")
    print(f"Races with xt_ polls: {sum(1 for v in xt_counts.values() if v > 0)}/{len(xt_counts)}")

    states_arr = np.array(data["states"])
    county_votes = data["county_votes"]
    county_fips = data["county_fips"]

    rows = []
    for race_id in sorted(results_with.keys()):
        if race_id not in results_without:
            continue
        fw = results_with[race_id]
        fwo = results_without[race_id]

        # theta_national difference
        theta_diff = fw.theta_national - fwo.theta_national
        theta_diff_mean = np.mean(np.abs(theta_diff))

        # State prediction (vote-weighted) for the race's state
        # Extract state from race name
        parts = race_id.split()
        state_abbr = parts[1] if len(parts) > 1 else None
        state_mask = states_arr == state_abbr

        if state_mask.any():
            with_local = fw.county_preds_local[state_mask]
            without_local = fwo.county_preds_local[state_mask]
            votes = county_votes[state_mask]

            # Vote-weighted state pred
            sp_with = np.average(with_local, weights=votes)
            sp_without = np.average(without_local, weights=votes)
            sp_diff = (sp_with - sp_without) * 100  # pp
        else:
            sp_with = sp_without = sp_diff = 0.0

        n_xt = xt_counts.get(race_id, 0)
        n_total = len(data["polls_by_race"].get(race_id, []))

        rows.append({
            "race": race_id,
            "xt_polls": n_xt,
            "total_polls": n_total,
            "theta_mean_abs_diff": theta_diff_mean,
            "state_pred_with": sp_with,
            "state_pred_without": sp_without,
            "state_pred_diff_pp": sp_diff,
        })

    # Print results
    print(f"\n{'Race':<35} {'xt/tot':>8} {'θ mean Δ':>10} {'pred_w':>8} {'pred_wo':>8} {'Δ (pp)':>8}")
    print("-" * 80)

    total_abs_diff = 0
    n_affected = 0
    for r in rows:
        marker = " *" if r["xt_polls"] > 0 else ""
        print(
            f"{r['race']:<35} "
            f"{r['xt_polls']:>3}/{r['total_polls']:<4} "
            f"{r['theta_mean_abs_diff']:>10.6f} "
            f"{r['state_pred_with']:>8.4f} "
            f"{r['state_pred_without']:>8.4f} "
            f"{r['state_pred_diff_pp']:>+8.3f}{marker}"
        )
        if abs(r["state_pred_diff_pp"]) > 0.001:
            total_abs_diff += abs(r["state_pred_diff_pp"])
            n_affected += 1

    print("-" * 80)
    print(f"* = race has polls with xt_ data")
    print(f"\nSummary:")
    print(f"  Races affected: {n_affected}")
    print(f"  Mean |state_pred Δ|: {total_abs_diff / max(n_affected, 1):.3f} pp")
    print(f"  Max |state_pred Δ|:  {max(abs(r['state_pred_diff_pp']) for r in rows):.3f} pp")

    # Breakdown by whether the race itself has xt_ polls
    xt_races = [r for r in rows if r["xt_polls"] > 0]
    non_xt_races = [r for r in rows if r["xt_polls"] == 0]
    if xt_races:
        print(f"\n  Races WITH xt_ polls ({len(xt_races)}):")
        print(f"    Mean |Δ|: {np.mean([abs(r['state_pred_diff_pp']) for r in xt_races]):.3f} pp")
    if non_xt_races:
        diffs = [abs(r["state_pred_diff_pp"]) for r in non_xt_races]
        if any(d > 0.001 for d in diffs):
            print(f"\n  Races WITHOUT xt_ polls but still affected ({sum(1 for d in diffs if d > 0.001)}):")
            print(f"    Mean |Δ|: {np.mean([d for d in diffs if d > 0.001]):.3f} pp")
            print(f"    (Spillover via shared θ_national)")


if __name__ == "__main__":
    data = _load_data()
    _run_comparison(data)
