"""Spot check: county-prior vs type-mean-prior prediction comparison.

Computes 2024 presidential prediction error for:
  1. county_prior approach: each county's own historical baseline (2024 actual as prior)
  2. type_mean_prior approach: score-weighted average of type means (the old behavior)

Since there are no 2026 polls to propagate, we use the prior-only prediction (no
Bayesian update). This isolates the effect of using county baselines vs type means.

The "county_prior" approach in production uses 2024 actual as the county prior when
predicting 2026 -- so for a 2024 in-sample check, we use 2020 actual as the county
prior (the most recent available at prediction time).

For the type_mean approach, we compute score-weighted type means derived from
2024 actual results (population-weighted per type) -- this replicates what the
old type-mean logic produced.

Usage:
    uv run python scripts/spot_check_county_priors.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Black Belt FIPS
# ---------------------------------------------------------------------------
BLACK_BELT_FIPS = {
    "01005",  # Barbour, AL
    "01063",  # Greene, AL
    "01067",  # Hale, AL
    "01069",  # Henry, AL
    "01085",  # Lowndes, AL
    "01087",  # Macon, AL
    "01091",  # Marengo, AL
    "13001",  # Appling, GA
    "13037",  # Calhoun, GA
    "13061",  # Stewart, GA
}

SPOTLIGHT_FIPS = {
    "13089": "DeKalb, GA",
    "13067": "Cobb, GA",
    "12086": "Miami-Dade, FL",
    "12103": "Pinellas, FL",
}


def load_data() -> dict:
    """Load all required data files."""
    assembled = PROJECT_ROOT / "data" / "assembled"
    communities = PROJECT_ROOT / "data" / "communities"

    # Type assignments (293 counties x 43 type scores)
    ta = pd.read_parquet(communities / "type_assignments.parquet")
    ta["county_fips"] = ta["county_fips"].astype(str).str.zfill(5)

    score_cols = sorted([c for c in ta.columns if c.endswith("_score")])
    J = len(score_cols)
    type_scores = ta[score_cols].values  # (N, J)
    county_fips = ta["county_fips"].tolist()
    dominant_type = ta["dominant_type"].values
    super_type = ta["super_type"].values

    # 2024 actual presidential results (ground truth)
    pres2024 = pd.read_parquet(assembled / "medsl_county_presidential_2024.parquet")
    pres2024["county_fips"] = pres2024["county_fips"].astype(str).str.zfill(5)
    actual_2024 = dict(zip(pres2024["county_fips"], pres2024["pres_dem_share_2024"]))
    votes_2024 = dict(zip(pres2024["county_fips"], pres2024["pres_total_2024"]))

    # 2020 actual presidential results (county_prior for 2024 prediction)
    pres2020 = pd.read_parquet(assembled / "medsl_county_presidential_2020.parquet")
    pres2020["county_fips"] = pres2020["county_fips"].astype(str).str.zfill(5)
    share_col_2020 = "pres_dem_share_2020"
    actual_2020 = dict(zip(pres2020["county_fips"], pres2020[share_col_2020]))

    # State abbreviations from 2024 data
    state_map = dict(zip(pres2024["county_fips"], pres2024["state_abbr"]))

    # County names from crosswalk if available
    crosswalk_path = PROJECT_ROOT / "data" / "raw" / "fips_county_crosswalk.csv"
    name_map: dict[str, str] = {}
    if crosswalk_path.exists():
        xwalk = pd.read_csv(crosswalk_path, dtype=str)
        xwalk["county_fips"] = xwalk["county_fips"].str.zfill(5)
        name_map = dict(zip(xwalk["county_fips"], xwalk["county_name"]))

    return {
        "county_fips": county_fips,
        "type_scores": type_scores,
        "J": J,
        "dominant_type": dominant_type,
        "super_type": super_type,
        "actual_2024": actual_2024,
        "actual_2020": actual_2020,
        "votes_2024": votes_2024,
        "state_map": state_map,
        "name_map": name_map,
    }


def compute_type_means(
    county_fips: list[str],
    type_scores: np.ndarray,
    actual_2024: dict[str, float],
    votes_2024: dict[str, float],
) -> np.ndarray:
    """Compute population-weighted type mean dem share for each of J types.

    This replicates the 'old' type-mean prior: for each type, the weighted
    average 2024 dem share of counties, weighted by type score * vote count.
    """
    J = type_scores.shape[1]
    type_means = np.full(J, 0.45)

    actuals = np.array([actual_2024.get(f, np.nan) for f in county_fips])
    votes = np.array([votes_2024.get(f, 1.0) for f in county_fips])
    valid = ~np.isnan(actuals)

    for j in range(J):
        scores_j = type_scores[:, j]
        # Population-weighted: score * vote count
        w = scores_j * votes
        w_valid = w[valid]
        a_valid = actuals[valid]
        wsum = w_valid.sum()
        if wsum > 0:
            type_means[j] = (w_valid * a_valid).sum() / wsum

    return type_means


def predict_county_prior(
    county_fips: list[str],
    type_scores: np.ndarray,
    actual_2020: dict[str, float],
    fallback: float = 0.45,
) -> np.ndarray:
    """County-prior approach: each county's 2020 actual as the prior.

    No poll update — prior-only prediction.
    In the production system, this uses each county's own historical baseline.
    For 2024 prediction we use 2020 actual (the most recent available before 2024).
    """
    priors = np.array([actual_2020.get(f, fallback) for f in county_fips])
    return priors  # No adjustment without polls


def predict_type_mean(
    county_fips: list[str],
    type_scores: np.ndarray,
    type_means: np.ndarray,
) -> np.ndarray:
    """Type-mean approach: score-weighted average of type means.

    This is the 'old' behavior before county-level priors were introduced.
    No poll update — prior-only prediction.
    """
    abs_scores = np.abs(type_scores)
    weight_sums = abs_scores.sum(axis=1)
    weight_sums = np.where(weight_sums == 0, 1.0, weight_sums)
    pred = (abs_scores * type_means[None, :]).sum(axis=1) / weight_sums
    return pred


def rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    valid = ~np.isnan(actual)
    return float(np.sqrt(np.mean((pred[valid] - actual[valid]) ** 2)))


def build_results_df(
    county_fips: list[str],
    pred_county: np.ndarray,
    pred_type: np.ndarray,
    actual_2024: dict[str, float],
    actual_2020: dict[str, float],
    dominant_type: np.ndarray,
    super_type: np.ndarray,
    state_map: dict[str, str],
    name_map: dict[str, str],
) -> pd.DataFrame:
    actuals = np.array([actual_2024.get(f, np.nan) for f in county_fips])
    prior_2020 = np.array([actual_2020.get(f, np.nan) for f in county_fips])

    df = pd.DataFrame({
        "county_fips": county_fips,
        "state": [state_map.get(f, "??") for f in county_fips],
        "county_name": [name_map.get(f, f) for f in county_fips],
        "dominant_type": dominant_type,
        "super_type": super_type,
        "actual_2024": actuals,
        "prior_2020": prior_2020,
        "pred_county_prior": pred_county,
        "pred_type_mean": pred_type,
        "err_county_prior": pred_county - actuals,
        "err_type_mean": pred_type - actuals,
        "abs_err_county": np.abs(pred_county - actuals),
        "abs_err_type": np.abs(pred_type - actuals),
    })
    return df


def main() -> None:
    print("Loading data...")
    data = load_data()

    county_fips = data["county_fips"]
    type_scores = data["type_scores"]
    actual_2024 = data["actual_2024"]
    actual_2020 = data["actual_2020"]
    votes_2024 = data["votes_2024"]
    dominant_type = data["dominant_type"]
    super_type = data["super_type"]
    state_map = data["state_map"]
    name_map = data["name_map"]
    J = data["J"]

    print(f"Counties: {len(county_fips)}, Types: {J}")
    print(f"Counties with 2020 data: {sum(f in actual_2020 for f in county_fips)}")
    print(f"Counties with 2024 data: {sum(f in actual_2024 for f in county_fips)}")

    # Compute type means from 2024 actuals (for type-mean approach)
    type_means = compute_type_means(county_fips, type_scores, actual_2024, votes_2024)
    print(f"\nType means range: [{type_means.min():.3f}, {type_means.max():.3f}]")

    # Predictions
    pred_county = predict_county_prior(county_fips, type_scores, actual_2020)
    pred_type = predict_type_mean(county_fips, type_scores, type_means)

    actuals = np.array([actual_2024.get(f, np.nan) for f in county_fips])

    # Overall RMSE
    rmse_county = rmse(pred_county, actuals)
    rmse_type = rmse(pred_type, actuals)

    print(f"\n{'='*60}")
    print("OVERALL RMSE COMPARISON (2024 presidential, no poll update)")
    print(f"{'='*60}")
    print(f"County-prior approach (2020 actual as prior): {rmse_county*100:.2f} pp")
    print(f"Type-mean approach (score-weighted type means): {rmse_type*100:.2f} pp")
    print(f"Improvement: {(rmse_type - rmse_county)*100:.2f} pp")

    # Build full results dataframe
    df = build_results_df(
        county_fips, pred_county, pred_type, actual_2024, actual_2020,
        dominant_type, super_type, state_map, name_map,
    )

    valid_mask = df["actual_2024"].notna()
    df_valid = df[valid_mask].copy()

    print(f"\nValid counties for comparison: {len(df_valid)}")

    # Top 10 worst errors - county prior approach
    print(f"\n{'='*60}")
    print("TOP 10 WORST ERRORS — County-Prior Approach")
    print(f"{'='*60}")
    worst_county = df_valid.nlargest(10, "abs_err_county")[
        ["county_name", "state", "county_fips", "dominant_type",
         "actual_2024", "prior_2020", "pred_county_prior", "err_county_prior"]
    ]
    worst_county = worst_county.copy()
    worst_county["actual_2024_pct"] = (worst_county["actual_2024"] * 100).round(1)
    worst_county["prior_2020_pct"] = (worst_county["prior_2020"] * 100).round(1)
    worst_county["pred_county_pct"] = (worst_county["pred_county_prior"] * 100).round(1)
    worst_county["err_pp"] = (worst_county["err_county_prior"] * 100).round(1)
    print(worst_county[["county_name", "state", "county_fips", "dominant_type",
                         "actual_2024_pct", "prior_2020_pct", "pred_county_pct", "err_pp"]].to_string(index=False))

    # Top 10 worst errors - type mean approach
    print(f"\n{'='*60}")
    print("TOP 10 WORST ERRORS — Type-Mean Approach")
    print(f"{'='*60}")
    worst_type = df_valid.nlargest(10, "abs_err_type")[
        ["county_name", "state", "county_fips", "dominant_type",
         "actual_2024", "pred_type_mean", "err_type_mean"]
    ]
    worst_type = worst_type.copy()
    worst_type["actual_2024_pct"] = (worst_type["actual_2024"] * 100).round(1)
    worst_type["pred_type_pct"] = (worst_type["pred_type_mean"] * 100).round(1)
    worst_type["err_pp"] = (worst_type["err_type_mean"] * 100).round(1)
    print(worst_type[["county_name", "state", "county_fips", "dominant_type",
                       "actual_2024_pct", "pred_type_pct", "err_pp"]].to_string(index=False))

    # Black Belt counties
    print(f"\n{'='*60}")
    print("BLACK BELT COUNTY ERRORS")
    print(f"{'='*60}")
    bb_df = df_valid[df_valid["county_fips"].isin(BLACK_BELT_FIPS)].copy()
    bb_df["actual_pct"] = (bb_df["actual_2024"] * 100).round(1)
    bb_df["prior_2020_pct"] = (bb_df["prior_2020"] * 100).round(1)
    bb_df["pred_county_pct"] = (bb_df["pred_county_prior"] * 100).round(1)
    bb_df["pred_type_pct"] = (bb_df["pred_type_mean"] * 100).round(1)
    bb_df["err_county_pp"] = (bb_df["err_county_prior"] * 100).round(1)
    bb_df["err_type_pp"] = (bb_df["err_type_mean"] * 100).round(1)
    bb_rmse_county = rmse(bb_df["pred_county_prior"].values, bb_df["actual_2024"].values)
    bb_rmse_type = rmse(bb_df["pred_type_mean"].values, bb_df["actual_2024"].values)
    print(bb_df[["county_name", "state", "county_fips", "dominant_type",
                 "actual_pct", "prior_2020_pct", "pred_county_pct", "pred_type_pct",
                 "err_county_pp", "err_type_pp"]].to_string(index=False))
    print(f"\nBlack Belt RMSE — county_prior: {bb_rmse_county*100:.2f} pp, type_mean: {bb_rmse_type*100:.2f} pp")

    # Spotlight counties: DeKalb, Cobb, Miami-Dade, Pinellas
    print(f"\n{'='*60}")
    print("SPOTLIGHT COUNTIES: DeKalb, Cobb, Miami-Dade, Pinellas")
    print(f"{'='*60}")
    spot_df = df_valid[df_valid["county_fips"].isin(SPOTLIGHT_FIPS)].copy()
    spot_df["actual_pct"] = (spot_df["actual_2024"] * 100).round(1)
    spot_df["prior_2020_pct"] = (spot_df["prior_2020"] * 100).round(1)
    spot_df["pred_county_pct"] = (spot_df["pred_county_prior"] * 100).round(1)
    spot_df["pred_type_pct"] = (spot_df["pred_type_mean"] * 100).round(1)
    spot_df["err_county_pp"] = (spot_df["err_county_prior"] * 100).round(1)
    spot_df["err_type_pp"] = (spot_df["err_type_mean"] * 100).round(1)
    print(spot_df[["county_name", "state", "county_fips", "dominant_type",
                   "actual_pct", "prior_2020_pct", "pred_county_pct", "pred_type_pct",
                   "err_county_pp", "err_type_pp"]].to_string(index=False))

    # Comparison: how many counties improved?
    improved = (df_valid["abs_err_county"] < df_valid["abs_err_type"]).sum()
    worsened = (df_valid["abs_err_county"] > df_valid["abs_err_type"]).sum()
    tied = (df_valid["abs_err_county"] == df_valid["abs_err_type"]).sum()
    print(f"\n{'='*60}")
    print("COUNTY-BY-COUNTY COMPARISON")
    print(f"{'='*60}")
    print(f"County-prior better: {improved}/{len(df_valid)} ({improved/len(df_valid)*100:.0f}%)")
    print(f"Type-mean better:    {worsened}/{len(df_valid)} ({worsened/len(df_valid)*100:.0f}%)")
    print(f"Tied:                {tied}")

    # Error distribution summary
    print(f"\n{'='*60}")
    print("ERROR DISTRIBUTION SUMMARY (absolute error in pp)")
    print(f"{'='*60}")
    for label, col in [("County-prior", "abs_err_county"), ("Type-mean", "abs_err_type")]:
        errs = df_valid[col] * 100
        print(f"{label:14s}: median={errs.median():.2f}pp  p75={errs.quantile(0.75):.2f}pp  p90={errs.quantile(0.90):.2f}pp  max={errs.max():.2f}pp")

    # --- Output markdown report ---
    report_lines = build_markdown_report(
        df_valid=df_valid,
        rmse_county=rmse_county,
        rmse_type=rmse_type,
        worst_county=worst_county,
        worst_type=worst_type,
        bb_df=bb_df,
        bb_rmse_county=bb_rmse_county,
        bb_rmse_type=bb_rmse_type,
        spot_df=spot_df,
        improved=improved,
        worsened=worsened,
        tied=tied,
        total=len(df_valid),
        J=J,
    )
    out_path = PROJECT_ROOT / "docs" / "spot-check-county-priors-S164.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(report_lines))
    print(f"\nReport saved to: {out_path}")


def build_markdown_report(
    df_valid: pd.DataFrame,
    rmse_county: float,
    rmse_type: float,
    worst_county: pd.DataFrame,
    worst_type: pd.DataFrame,
    bb_df: pd.DataFrame,
    bb_rmse_county: float,
    bb_rmse_type: float,
    spot_df: pd.DataFrame,
    improved: int,
    worsened: int,
    tied: int,
    total: int,
    J: int,
) -> list[str]:
    lines = []
    lines.append("# County-Prior vs Type-Mean Prior Spot Check — S164 (2026-03-22)")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"**N counties:** {total}  ")
    lines.append(f"**J types:** {J}  ")
    lines.append("**Prediction mode:** Prior-only (no poll update — isolates baseline quality)  ")
    lines.append("**County-prior:** each county's 2020 actual Dem share (most recent available before 2024)  ")
    lines.append("**Type-mean prior:** score-weighted average of population-weighted type means (computed from 2024 actuals)  ")
    lines.append("**Ground truth:** 2024 presidential Dem share")
    lines.append("")
    lines.append("## Overall RMSE")
    lines.append("")
    lines.append("| Approach | RMSE (pp) |")
    lines.append("|----------|-----------|")
    lines.append(f"| County-prior (2020 actual) | {rmse_county*100:.2f} |")
    lines.append(f"| Type-mean prior | {rmse_type*100:.2f} |")
    lines.append(f"| Improvement (type_mean - county_prior) | {(rmse_type - rmse_county)*100:.2f} |")
    lines.append("")
    lines.append(f"County-prior better in **{improved}/{total} counties ({improved/total*100:.0f}%)**.  ")
    lines.append(f"Type-mean better in {worsened}/{total} counties.")

    def _err_dist_row(label: str, col: str) -> str:
        errs = df_valid[col] * 100
        return (f"| {label} | {errs.median():.2f} | {errs.quantile(0.75):.2f} | "
                f"{errs.quantile(0.90):.2f} | {errs.max():.2f} |")

    lines.append("")
    lines.append("### Error Distribution (absolute error, pp)")
    lines.append("")
    lines.append("| Approach | Median | p75 | p90 | Max |")
    lines.append("|----------|--------|-----|-----|-----|")
    lines.append(_err_dist_row("County-prior", "abs_err_county"))
    lines.append(_err_dist_row("Type-mean", "abs_err_type"))

    lines.append("")
    lines.append("## Top 10 Worst Errors — County-Prior Approach")
    lines.append("")
    lines.append("| County | State | FIPS | Dom.Type | Actual 2024 | Prior 2020 | Pred | Error |")
    lines.append("|--------|-------|------|----------|-------------|------------|------|-------|")
    for _, r in worst_county.iterrows():
        lines.append(f"| {r['county_name']} | {r['state']} | {r['county_fips']} | {r['dominant_type']} "
                     f"| {r['actual_2024_pct']}% | {r['prior_2020_pct']}% | {r['pred_county_pct']}% | {r['err_pp']:+.1f}pp |")

    lines.append("")
    lines.append("## Top 10 Worst Errors — Type-Mean Approach")
    lines.append("")
    lines.append("| County | State | FIPS | Dom.Type | Actual 2024 | Pred | Error |")
    lines.append("|--------|-------|------|----------|-------------|------|-------|")
    for _, r in worst_type.iterrows():
        lines.append(f"| {r['county_name']} | {r['state']} | {r['county_fips']} | {r['dominant_type']} "
                     f"| {r['actual_2024_pct']}% | {r['pred_type_pct']}% | {r['err_pp']:+.1f}pp |")

    lines.append("")
    lines.append("## Black Belt County Errors")
    lines.append("")
    lines.append(f"Black Belt RMSE — county_prior: **{bb_rmse_county*100:.2f} pp**, type_mean: **{bb_rmse_type*100:.2f} pp**")
    lines.append("")
    lines.append("| County | State | FIPS | Dom.Type | Actual | Prior 2020 | County-Prior Pred | Type-Mean Pred | Err(County) | Err(Type) |")
    lines.append("|--------|-------|------|----------|--------|------------|-------------------|----------------|-------------|-----------|")
    for _, r in bb_df.sort_values("err_county_pp").iterrows():
        lines.append(f"| {r['county_name']} | {r['state']} | {r['county_fips']} | {r['dominant_type']} "
                     f"| {r['actual_pct']}% | {r['prior_2020_pct']}% | {r['pred_county_pct']}% "
                     f"| {r['pred_type_pct']}% | {r['err_county_pp']:+.1f}pp | {r['err_type_pp']:+.1f}pp |")

    lines.append("")
    lines.append("## Spotlight Counties: DeKalb, Cobb, Miami-Dade, Pinellas")
    lines.append("")
    lines.append("| County | State | FIPS | Dom.Type | Actual | Prior 2020 | County-Prior Pred | Type-Mean Pred | Err(County) | Err(Type) |")
    lines.append("|--------|-------|------|----------|--------|------------|-------------------|----------------|-------------|-----------|")
    for _, r in spot_df.sort_values("county_fips").iterrows():
        lines.append(f"| {r['county_name']} | {r['state']} | {r['county_fips']} | {r['dominant_type']} "
                     f"| {r['actual_pct']}% | {r['prior_2020_pct']}% | {r['pred_county_pct']}% "
                     f"| {r['pred_type_pct']}% | {r['err_county_pp']:+.1f}pp | {r['err_type_pp']:+.1f}pp |")

    lines.append("")
    lines.append("## Key Findings")
    lines.append("")
    lines.append("_(Auto-generated section — see printed output for the full interpretation)_")
    lines.append("")

    improvement_pp = (rmse_type - rmse_county) * 100
    if improvement_pp > 0:
        lines.append(f"1. **County-prior reduces RMSE by {improvement_pp:.2f}pp** ({rmse_type*100:.2f}pp → {rmse_county*100:.2f}pp).")
    else:
        lines.append(f"1. **Type-mean slightly better** by {-improvement_pp:.2f}pp ({rmse_county*100:.2f}pp vs {rmse_type*100:.2f}pp) — county prior did not improve overall RMSE.")

    bb_improvement = (bb_rmse_type - bb_rmse_county) * 100
    if bb_improvement > 0:
        lines.append(f"2. **Black Belt improvement: {bb_improvement:.2f}pp** ({bb_rmse_type*100:.2f}pp → {bb_rmse_county*100:.2f}pp). County priors anchor Black Belt counties at their actual baseline, not the type mean.")
    else:
        lines.append(f"2. **Black Belt not improved** ({bb_rmse_county*100:.2f}pp county vs {bb_rmse_type*100:.2f}pp type-mean). Both approaches struggle with Black Belt counties.")

    lines.append(f"3. **{improved}/{total} counties ({improved/total*100:.0f}%) improved** with county-prior approach.")
    lines.append("")
    lines.append("---")
    lines.append("_Generated by scripts/spot_check_county_priors.py_")

    return lines


if __name__ == "__main__":
    main()
