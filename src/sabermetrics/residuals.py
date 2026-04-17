"""Candidate residual computation.

The residual = actual - expected is the raw material for all
electoral sabermetric stats. When the community-covariance model
is available, residuals are decomposed by community type to produce
the CTOV (Community-Type Overperformance Vector).

Phase 2 implementation: compute_mvd, compute_ctov, compute_cec using
the backtest harness as the source of structural predictions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Ridge regularization for CTOV decomposition — same shrinkage as estimate_delta_race.
# Higher mu = more shrinkage toward zero (flatter CTOV vectors).
# The backtest residuals are at county level but we only have O(30) "observations"
# (one poll per state) so aggressive regularization prevents overfitting.
_CTOV_RIDGE_MU = 1.0

# State FIPS prefix → abbreviation (used to infer state from county FIPS).
# Loaded lazily from core config on first call.
_STATE_ABBR_CACHE: dict[str, str] | None = None


def _get_state_abbr_map() -> dict[str, str]:
    """Load state FIPS → abbreviation from core config (lazy singleton)."""
    global _STATE_ABBR_CACHE
    if _STATE_ABBR_CACHE is None:
        from src.core import config as cfg

        _STATE_ABBR_CACHE = cfg.STATE_ABBR
    return _STATE_ABBR_CACHE


# ---------------------------------------------------------------------------
# County-level prediction loader
# ---------------------------------------------------------------------------


def _run_county_level_backtest(year: int, race_type: str) -> dict:
    """Run the structural model for one year+race_type and return county-level predictions.

    This is a richer version of backtest_harness.run_backtest() that exposes
    the county-level (fips → pred_dem_share) mapping for each state, which is
    what we need for CTOV decomposition.

    Returns a dict mapping state_abbr → pd.DataFrame with columns:
        county_fips, actual_dem_share, pred_dem_share, residual

    States with missing actuals or no counties are omitted.
    """
    from src.prediction.county_priors import (
        load_county_priors_with_ridge,
        load_county_priors_with_ridge_governor,
    )
    from src.prediction.forecast_engine import run_forecast
    from src.validation.backtest_harness import (
        _county_metadata,
        _load_type_data_for_backtest,
        load_historic_actuals,
        load_historic_polls,
    )

    race_type = race_type.lower()
    log.info("Running county-level backtest: %s %d", race_type, year)

    polls_by_race = load_historic_polls(year, race_type)
    if not polls_by_race:
        log.warning("%s %d: no polls, skipping", race_type, year)
        return {}

    county_fips, type_scores, _ = _load_type_data_for_backtest()
    states = _county_metadata(county_fips)

    # Use 2024 presidential vote counts for population weighting — this is frozen
    # model infrastructure; the counts reflect population size, not outcome.
    county_votes_arr = np.ones(len(county_fips))
    votes_path = PROJECT_ROOT / "data" / "assembled" / "medsl_county_presidential_2024.parquet"
    if votes_path.exists():
        vdf = pd.read_parquet(votes_path)
        if "county_fips" in vdf.columns and "pres_total_2024" in vdf.columns:
            vmap = dict(
                zip(
                    vdf["county_fips"].astype(str).str.zfill(5),
                    vdf["pres_total_2024"],
                )
            )
            county_votes_arr = np.array([float(vmap.get(f, 1.0)) for f in county_fips])

    if race_type == "governor":
        county_priors = load_county_priors_with_ridge_governor(county_fips)
    else:
        county_priors = load_county_priors_with_ridge(county_fips)

    forecast_results = run_forecast(
        type_scores=type_scores,
        county_priors=county_priors,
        states=states,
        county_votes=county_votes_arr,
        polls_by_race=polls_by_race,
        races=list(polls_by_race.keys()),
        lam=1.0,
        mu=1.0,
        generic_ballot_shift=0.0,
        w_vector_mode="core",
        reference_date=f"{year}-11-01",
    )

    actuals_df = load_historic_actuals(year, race_type)

    county_preds_by_race: dict[str, np.ndarray] = {
        race_id: fr.county_preds_national for race_id, fr in forecast_results.items()
    }

    result: dict[str, pd.DataFrame] = {}
    for race_id, preds_arr in county_preds_by_race.items():
        parts = race_id.split(" ")
        if len(parts) < 3:
            continue
        state_abbr = parts[1]

        state_actuals = actuals_df[actuals_df["state_abbr"] == state_abbr].copy()
        if len(state_actuals) == 0:
            continue

        fips_to_pred_dem = dict(zip(county_fips, preds_arr))
        state_actuals["pred_dem_share"] = state_actuals["county_fips"].map(fips_to_pred_dem)
        state_actuals = state_actuals.dropna(subset=["pred_dem_share", "actual_dem_share"])
        if len(state_actuals) == 0:
            continue

        state_actuals["residual"] = state_actuals["actual_dem_share"] - state_actuals["pred_dem_share"]
        result[state_abbr] = state_actuals.reset_index(drop=True)

    log.info(
        "%s %d: county-level predictions ready for %d states",
        race_type,
        year,
        len(result),
    )
    return result


# ---------------------------------------------------------------------------
# Type weight matrix loader (for CTOV ridge decomposition)
# ---------------------------------------------------------------------------


def _load_type_weight_matrix() -> tuple[list[str], np.ndarray]:
    """Load the county × type soft-membership matrix (W matrix).

    Returns
    -------
    county_fips : list[str]
        Zero-padded 5-digit FIPS codes, one per row of W.
    W : ndarray of shape (N_counties, J)
        Soft type membership weights (each row sums to 1).
    """
    ta_path = PROJECT_ROOT / "data" / "communities" / "type_assignments.parquet"
    ta_df = pd.read_parquet(ta_path)
    county_fips = ta_df["county_fips"].astype(str).str.zfill(5).tolist()
    score_cols = sorted([c for c in ta_df.columns if c.endswith("_score")])
    W = ta_df[score_cols].values
    return county_fips, W


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _build_backtest_cache(
    registry: dict,
) -> dict[tuple[int, str], dict[str, pd.DataFrame]]:
    """Build and populate the backtest cache for all (year, office) pairs in the registry.

    This is the expensive step (~0.8 sec per year×office combination).
    We run it once and share the results between compute_mvd and compute_ctov
    to avoid redundant backtest runs.

    Returns
    -------
    dict mapping (year, office_lowercase) → dict[state_abbr, county_DataFrame]
    """
    needed: set[tuple[int, str]] = set()
    for person in registry["persons"].values():
        for race in person["races"]:
            if race["actual_dem_share_2party"] is None:
                continue
            year = race["year"]
            if year < 2014:
                # 2008-2012 538 data lacks candidate names — skip gracefully
                continue
            needed.add((year, race["office"].lower()))

    cache: dict[tuple[int, str], dict[str, pd.DataFrame]] = {}
    for year, office in sorted(needed):
        log.info("Caching backtest: %s %d", office, year)
        cache[(year, office)] = _run_county_level_backtest(year, office)
    return cache


def compute_mvd(
    registry: dict,
    backtest_cache: dict[tuple[int, str], dict[str, pd.DataFrame]] | None = None,
) -> pd.DataFrame:
    """Compute Marginal Vote Delivery (MVD) for all candidates in the registry.

    MVD = actual_dem_share - model_predicted_dem_share at the state level.

    Positive MVD means the Democrat outperformed the structural model
    (candidate overperformance, net of environment, net of district lean).
    For Republican candidates the sign is: positive MVD = Dems outperformed
    the model, which is R underperformance.

    The "model prediction" comes from running the backtest harness with the
    frozen structural model — the same polls-plus final forecast as the
    model uses, but without any candidate effect signal. This is exactly
    what a generic D or R would be expected to get given the structural
    landscape and that year's environment.

    Parameters
    ----------
    registry : dict
        Loaded candidate_registry.json dict (keys: "persons", "_meta").
    backtest_cache : dict | None
        Pre-built backtest cache from _build_backtest_cache(). If None,
        the cache is built internally. Pass an existing cache to avoid
        redundant backtest runs when also calling compute_ctov().

    Returns
    -------
    pd.DataFrame
        Columns: person_id, name, party, year, state, office,
                 actual_dem_share, pred_dem_share, mvd
        One row per candidate-race with a known actual_dem_share.
    """
    if backtest_cache is None:
        backtest_cache = _build_backtest_cache(registry)

    persons = registry["persons"]
    rows: list[dict] = []
    for person_id, person in persons.items():
        for race in person["races"]:
            actual = race["actual_dem_share_2party"]
            if actual is None:
                continue
            year = race["year"]
            office = race["office"].lower()
            state = race["state"]
            if year < 2014:
                continue

            county_df = backtest_cache.get((year, office), {}).get(state)
            if county_df is None or len(county_df) == 0:
                log.debug("No county data for %s %s %d %s", person["name"], state, year, office)
                continue

            # State-level prediction = unweighted county mean (same approximation
            # as backtest_harness.run_backtest uses for state-level comparisons).
            pred_state_dem = float(county_df["pred_dem_share"].mean())
            mvd = actual - pred_state_dem

            rows.append(
                {
                    "person_id": person_id,
                    "name": person["name"],
                    "party": person["party"],
                    "year": year,
                    "state": state,
                    "office": race["office"],
                    "actual_dem_share": actual,
                    "pred_dem_share": pred_state_dem,
                    "mvd": mvd,
                }
            )

    df = pd.DataFrame(rows)
    log.info("compute_mvd: %d candidate-races with MVD", len(df))
    return df


def compute_ctov(
    mvd_df: pd.DataFrame,
    registry: dict,
    backtest_cache: dict[tuple[int, str], dict[str, pd.DataFrame]] | None = None,
) -> pd.DataFrame:
    """Compute Community-Type Overperformance Vectors (CTOV) for all candidates.

    For each candidate-race, decomposes the state-level residual into 100
    type-level components using Ridge regression on county-level residuals.

    The math is identical to estimate_delta_race() in candidate_effects.py:
        minimize ||W_state · ctov - residuals_state||² + μ·||ctov||²
    where W_state is the county × type weight matrix (N_counties_in_state × J)
    and residuals_state is the per-county (actual - predicted) vector.

    A positive CTOV component for type k means: the Democrat overperformed
    the structural model in counties dominated by type k. For Republican
    candidates, negate the CTOV to get their overperformance direction.

    Parameters
    ----------
    mvd_df : pd.DataFrame
        Output of compute_mvd() — must contain person_id, year, state, office.
    registry : dict
        Loaded candidate_registry.json dict.
    backtest_cache : dict | None
        Pre-built backtest cache from _build_backtest_cache(). If None,
        the cache is built internally.

    Returns
    -------
    pd.DataFrame
        Columns: person_id, name, party, year, state, office, mvd,
                 ctov_type_0, ..., ctov_type_99
        One row per candidate-race (same rows as mvd_df, with CTOV appended).
    """
    from src.prediction.candidate_effects import estimate_delta_race

    county_fips_all, W_all = _load_type_weight_matrix()
    J = W_all.shape[1]

    # Build a fast lookup: county_fips → row index in W_all
    fips_to_row: dict[str, int] = {f: i for i, f in enumerate(county_fips_all)}

    # Use pre-built cache if provided; otherwise build lazily per (year, office).
    # The lazy path avoids double-loading when called standalone, but the pipeline
    # always passes a pre-built cache to avoid duplicate backtest runs.
    county_data_cache: dict[tuple[int, str], dict[str, pd.DataFrame]] = (
        dict(backtest_cache) if backtest_cache is not None else {}
    )

    rows: list[dict] = []

    for _, mvd_row in mvd_df.iterrows():
        year = int(mvd_row["year"])
        office = str(mvd_row["office"]).lower()
        state = str(mvd_row["state"])

        cache_key = (year, office)
        if cache_key not in county_data_cache:
            county_data_cache[cache_key] = _run_county_level_backtest(year, office)

        county_df = county_data_cache[cache_key].get(state)
        if county_df is None or len(county_df) == 0:
            log.debug("CTOV: no county data for %s %s %d", state, office, year)
            ctov = np.zeros(J)
        else:
            # Build W_state: rows of W corresponding to this state's counties
            row_indices = [
                fips_to_row[f]
                for f in county_df["county_fips"]
                if f in fips_to_row
            ]

            if len(row_indices) == 0:
                ctov = np.zeros(J)
            else:
                W_state = W_all[row_indices, :]  # (N_counties_in_state, J)
                residuals = county_df["residual"].values[: len(row_indices)]

                # Uniform county weights: each county is equally trusted
                # (no poll-noise estimate available at county level, so σ=1 everywhere).
                sigma_counties = np.ones(len(row_indices))

                ctov = estimate_delta_race(
                    W_polls=W_state,
                    residuals=residuals,
                    sigma_polls=sigma_counties,
                    J=J,
                    mu=_CTOV_RIDGE_MU,
                )

        row = dict(mvd_row)
        for j, val in enumerate(ctov):
            row[f"ctov_type_{j}"] = float(val)
        rows.append(row)

    df = pd.DataFrame(rows)
    log.info("compute_ctov: %d candidate-races with CTOV vectors", len(df))
    return df


def compute_cec(ctov_history: list[np.ndarray]) -> float:
    """Compute Cross-Election Consistency (CEC) from a list of CTOV vectors.

    Mean pairwise Pearson correlation of CTOV vectors across elections for
    the same candidate. High CEC = the candidate consistently over/underperforms
    the same community types across elections — a genuine skill signal.
    Low CEC = the overperformance pattern is election-specific noise.

    Edge cases:
    - Single election: CEC = 1.0 by convention (can't measure consistency
      with no pairs, but a candidate who ran once is trivially "consistent").
    - Zero-variance CTOV (all zeros after Ridge shrinkage): correlation
      is undefined; treat as 0.0 (no signal, no consistency).

    Parameters
    ----------
    ctov_history : list[np.ndarray]
        List of CTOV vectors from successive elections (each J-length).

    Returns
    -------
    float
        CEC score in [-1, 1]. 1.0 for single-election candidates.
    """
    if len(ctov_history) == 0:
        return float("nan")
    if len(ctov_history) == 1:
        # Degenerate: no pair to correlate. Convention: 1.0 (trivially consistent).
        return 1.0

    pairwise_r: list[float] = []
    for i in range(len(ctov_history)):
        for j in range(i + 1, len(ctov_history)):
            v1 = ctov_history[i]
            v2 = ctov_history[j]
            std1 = float(np.std(v1))
            std2 = float(np.std(v2))
            if std1 < 1e-9 or std2 < 1e-9:
                # Zero-variance vector = Ridge shrunk everything to ~0.
                # Treat as 0 correlation: no pattern to be consistent about.
                pairwise_r.append(0.0)
            else:
                r = float(np.corrcoef(v1, v2)[0, 1])
                pairwise_r.append(r)

    return float(np.mean(pairwise_r))


def compute_cec_for_all_candidates(ctov_df: pd.DataFrame) -> pd.DataFrame:
    """Compute CEC for every candidate, returning a per-candidate DataFrame.

    For single-race candidates, CEC = 1.0 (convention).
    For multi-race candidates, CEC = mean pairwise Pearson correlation of CTOVs.

    Parameters
    ----------
    ctov_df : pd.DataFrame
        Output of compute_ctov().

    Returns
    -------
    pd.DataFrame
        Columns: person_id, name, party, n_races, cec
    """
    ctov_cols = [c for c in ctov_df.columns if c.startswith("ctov_type_")]
    rows: list[dict] = []

    for person_id, group in ctov_df.groupby("person_id"):
        ctov_vectors = [row[ctov_cols].values.astype(float) for _, row in group.iterrows()]
        cec = compute_cec(ctov_vectors)
        rows.append(
            {
                "person_id": person_id,
                "name": group["name"].iloc[0],
                "party": group["party"].iloc[0],
                "n_races": len(group),
                "cec": cec,
            }
        )

    df = pd.DataFrame(rows).sort_values("n_races", ascending=False).reset_index(drop=True)
    log.info(
        "compute_cec_for_all_candidates: %d candidates, %d multi-race",
        len(df),
        (df["n_races"] > 1).sum(),
    )
    return df


def compute_polling_gap(
    actual_results: "pd.DataFrame",
    final_polling_averages: "pd.DataFrame",
    cycle_systematic_error: float | None = None,
) -> "pd.DataFrame":
    """Compute cycle-adjusted polling gap.

    Raw gap: actual - final_polling_average
    Adjusted: raw_gap - cycle_median_error

    The adjustment removes the within-cycle correlated polling
    error (Gelman et al.), isolating the candidate-specific signal.

    Parameters
    ----------
    actual_results : pd.DataFrame
        Columns: candidate_id, district_id, vote_share.
    final_polling_averages : pd.DataFrame
        Columns: district_id, polling_avg, n_polls.
    cycle_systematic_error : float | None
        Median (actual - polling_avg) across all races in the
        cycle. If None, computed from the data.

    Returns
    -------
    pd.DataFrame
        Columns: candidate_id, district_id, raw_gap, adjusted_gap,
        n_polls_in_district.
    """
    raise NotImplementedError
