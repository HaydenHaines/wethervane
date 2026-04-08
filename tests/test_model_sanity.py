"""Model sanity checks — regression tests for prediction quality.

These tests verify that predictions are directionally correct and catch
systematic bugs like the type-compression issue (#139) where unpolled
races collapsed to type means regardless of actual partisan lean.

The tests load the generated predictions parquet and the DuckDB database
to compute vote-weighted state predictions, then assert that known
partisan leanings are reflected in the model output.

Run after: predict_2026_types.py + build_database.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "predictions" / "county_predictions_2026_types.parquet"
DB_PATH = PROJECT_ROOT / "data" / "wethervane.duckdb"


def _skip_if_no_predictions():
    if not PREDICTIONS_PATH.exists():
        pytest.skip("No predictions parquet found — run predict_2026_types.py first")


def _load_state_predictions() -> pd.DataFrame:
    """Compute vote-weighted state-level predictions for all Senate races.

    Returns a DataFrame with columns: race, state, pred_dem_share, n_counties.
    """
    _skip_if_no_predictions()

    import duckdb

    if not DB_PATH.exists():
        pytest.skip("No DuckDB found — run build_database.py first")

    df = pd.read_parquet(PREDICTIONS_PATH)
    con = duckdb.connect(str(DB_PATH), read_only=True)
    votes = con.execute("SELECT county_fips, total_votes_2024, state_abbr FROM counties").fetchdf()
    votes["county_fips"] = votes["county_fips"].astype(str).str.zfill(5)
    con.close()

    # Filter to local mode Senate races, in-state counties only
    senate = df[
        (df["race"].str.contains("Senate"))
        & (df["forecast_mode"] == "local")
    ].copy()
    senate = senate.merge(votes, on="county_fips", how="left")

    # Keep only in-state counties (race state matches county state)
    senate["race_state"] = senate["race"].str.split().str[1]
    senate = senate[senate["state"] == senate["race_state"]]

    senate["total_votes_2024"] = senate["total_votes_2024"].fillna(1)

    results = []
    for race, grp in senate.groupby("race"):
        wt = np.average(grp["pred_dem_share"], weights=grp["total_votes_2024"])
        results.append({
            "race": race,
            "state": race.split()[1],
            "pred_dem_share": wt,
            "n_counties": len(grp),
        })

    return pd.DataFrame(results)


@pytest.fixture(scope="module")
def state_preds() -> pd.DataFrame:
    return _load_state_predictions()


def _get_pred(state_preds: pd.DataFrame, state: str) -> float:
    row = state_preds[state_preds["state"] == state]
    if row.empty:
        pytest.skip(f"No prediction found for {state}")
    return float(row.iloc[0]["pred_dem_share"])


# ──────────────────────────────────────────────────────────────────────
# Tier 1: Hard constraints — obvious reality checks.
# If any of these fail, the model has a serious bug.
# ──────────────────────────────────────────────────────────────────────


class TestSafeSeats:
    """States that are so partisan they should never flip in any model."""

    @pytest.mark.parametrize("state,min_dem", [
        ("MA", 0.55),  # MA hasn't gone R in a Senate race since 2010 (Brown special)
        ("DE", 0.53),  # Biden's home state, D+19 in 2020
        ("RI", 0.53),  # Deep blue, D+20+ at presidential level
        ("IL", 0.53),  # D+17 at presidential level
    ])
    def test_safe_dem_states_above_threshold(self, state_preds, state, min_dem):
        """Safe D states must predict Dem share above threshold."""
        pred = _get_pred(state_preds, state)
        assert pred > min_dem, (
            f"{state}: predicted {pred:.3f} (D+{(pred-0.5)*200:.1f}pp), "
            f"expected > {min_dem:.2f}. Model may have type-compression bug."
        )

    @pytest.mark.parametrize("state,max_dem", [
        ("WY", 0.35),  # Deepest red state, R+40+ at presidential level
        ("WV", 0.40),  # Was D, now deep R
        ("OK", 0.40),  # Consistently R+20+
        ("ID", 0.40),  # R+30+ at presidential level
    ])
    def test_safe_r_states_below_threshold(self, state_preds, state, max_dem):
        """Safe R states must predict Dem share below threshold."""
        pred = _get_pred(state_preds, state)
        assert pred < max_dem, (
            f"{state}: predicted {pred:.3f} (R+{(0.5-pred)*200:.1f}pp), "
            f"expected < {max_dem:.2f}. Model may be over-shifting D."
        )


class TestPredictionSpread:
    """Predictions should have meaningful variation across states."""

    def test_prediction_std_above_floor(self, state_preds):
        """Predictions should not collapse to a narrow range (type-compression)."""
        std = state_preds["pred_dem_share"].std()
        assert std > 0.05, (
            f"Prediction std={std:.4f} — too compressed. "
            f"Range: {state_preds['pred_dem_share'].min():.3f} to "
            f"{state_preds['pred_dem_share'].max():.3f}. "
            f"Possible type-compression bug."
        )

    def test_dem_range_spans_50(self, state_preds):
        """Should have both D-leaning and R-leaning states."""
        has_d = (state_preds["pred_dem_share"] > 0.55).any()
        has_r = (state_preds["pred_dem_share"] < 0.45).any()
        assert has_d and has_r, (
            "Model should predict both D-leaning (>0.55) and R-leaning (<0.45) "
            f"states. D-leaning: {has_d}, R-leaning: {has_r}."
        )


# ──────────────────────────────────────────────────────────────────────
# Tier 2: NJ regression test — the canary in the coal mine (#140).
# NJ is a consistently D state. Any model predicting NJ R-leaning
# for a Senate race has a structural bug.
# ──────────────────────────────────────────────────────────────────────


class TestNJRegression:
    """NJ-specific checks — the state that exposed the type-compression bug."""

    def test_nj_predicts_dem(self, state_preds):
        """NJ Senate should predict D-leaning (>0.50 Dem share)."""
        pred = _get_pred(state_preds, "NJ")
        assert pred > 0.50, (
            f"NJ predicted {pred:.3f} (R+{(0.5-pred)*200:.1f}pp). "
            f"NJ is a D+16 state — this indicates a model bug."
        )

    def test_nj_not_tossup(self, state_preds):
        """NJ should not be rated as a tossup (>0.52 Dem share)."""
        pred = _get_pred(state_preds, "NJ")
        margin = abs(pred - 0.5)
        assert margin > 0.02, (
            f"NJ margin is only {margin*200:.1f}pp — effectively a tossup. "
            f"NJ has been D+10-20 for decades."
        )


# ──────────────────────────────────────────────────────────────────────
# Tier 3: Cross-state correlation with partisan lean.
# The model should correlate with PVI/historical results.
# ──────────────────────────────────────────────────────────────────────


class TestCrossStateCorrelation:
    """Model predictions should correlate with known state partisan leans."""

    # Approximate 2024 presidential Dem share by state (simplified).
    # These are rough enough to test correlation, not exact margins.
    _PRESIDENTIAL_2024_APPROX = {
        "MA": 0.63, "RI": 0.60, "DE": 0.58, "IL": 0.57, "OR": 0.56,
        "NJ": 0.57, "CO": 0.56, "VA": 0.54, "NM": 0.54, "MN": 0.53,
        "NH": 0.52, "ME": 0.53, "MI": 0.49, "GA": 0.49, "NC": 0.48,
        "TX": 0.46, "IA": 0.44, "SC": 0.44, "AK": 0.43, "KS": 0.42,
        "MT": 0.40, "MS": 0.40, "LA": 0.40, "KY": 0.37, "AL": 0.37,
        "AR": 0.36, "TN": 0.36, "SD": 0.36, "NE": 0.40, "OK": 0.34,
        "ID": 0.33, "WV": 0.30, "WY": 0.27,
    }

    def test_correlation_with_presidential_lean(self, state_preds):
        """Senate predictions should correlate r > 0.80 with 2024 pres results."""
        pres_vals = []
        pred_vals = []
        for _, row in state_preds.iterrows():
            st = row["state"]
            if st in self._PRESIDENTIAL_2024_APPROX:
                pres_vals.append(self._PRESIDENTIAL_2024_APPROX[st])
                pred_vals.append(row["pred_dem_share"])

        if len(pres_vals) < 10:
            pytest.skip("Insufficient state overlap for correlation test")

        from scipy import stats
        r, _ = stats.pearsonr(pres_vals, pred_vals)
        assert r > 0.80, (
            f"Senate predictions correlate r={r:.3f} with 2024 presidential. "
            f"Expected r > 0.80. Model may have systematic issues."
        )
