"""Tests for county_priors.py weighted multi-election priors."""
import numpy as np
import pandas as pd

from src.prediction.county_priors import (
    _FALLBACK_DEM_SHARE,
    WEIGHTED_PRIOR_YEARS,
    compute_weighted_priors_from_data,
)


def test_single_year_equals_that_year():
    """With only one year of data, result equals that year's share."""
    shares = {"01001": {2024: 0.35}}
    result = compute_weighted_priors_from_data(["01001"], shares)
    np.testing.assert_allclose(result, [0.35])


def test_equal_shares_across_years():
    """With identical shares every year, decay doesn't matter."""
    shares = {"01001": {y: 0.50 for y in WEIGHTED_PRIOR_YEARS}}
    result = compute_weighted_priors_from_data(["01001"], shares)
    np.testing.assert_allclose(result, [0.50], atol=1e-10)


def test_recency_weighting():
    """2024 should dominate over older elections."""
    shares = {"01001": {2024: 0.60, 2020: 0.40, 2016: 0.40, 2012: 0.40, 2008: 0.40}}
    result = compute_weighted_priors_from_data(["01001"], shares, decay=0.7)
    # w = decay^((2024 - year) / 4): 1.0, 0.7, 0.49, 0.343, 0.2401
    expected = (0.6 + 0.4 * (0.7 + 0.49 + 0.343 + 0.2401)) / (1.0 + 0.7 + 0.49 + 0.343 + 0.2401)
    np.testing.assert_allclose(result, [expected], atol=1e-6)


def test_missing_years_graceful():
    """Counties missing some years use available data only."""
    shares = {"01001": {2024: 0.55, 2016: 0.45}}
    result = compute_weighted_priors_from_data(["01001"], shares, decay=0.7)
    # 2024: w=1.0, 2016: w=0.7^2=0.49
    expected = (0.55 + 0.45 * 0.49) / (1.0 + 0.49)
    np.testing.assert_allclose(result, [expected], atol=1e-6)


def test_no_data_returns_fallback():
    """Counties with zero data get _FALLBACK_DEM_SHARE."""
    result = compute_weighted_priors_from_data(["99999"], {})
    np.testing.assert_allclose(result, [_FALLBACK_DEM_SHARE])


def test_output_shape():
    """Output shape matches input county_fips length."""
    shares = {"A": {2024: 0.5}, "B": {2024: 0.6}, "C": {2024: 0.4}}
    result = compute_weighted_priors_from_data(["A", "B", "C"], shares)
    assert result.shape == (3,)


def test_compute_weighted_county_priors(tmp_path):
    """File-based loader produces weighted mean from parquet files."""
    fips = ["01001", "01002"]
    for year in [2024, 2020]:
        df = pd.DataFrame({
            "county_fips": fips,
            f"pres_dem_share_{year}": [0.60 if year == 2024 else 0.40, 0.50],
        })
        df.to_parquet(tmp_path / f"medsl_county_presidential_{year}.parquet")

    from src.prediction.county_priors import compute_weighted_county_priors

    result = compute_weighted_county_priors(fips, assembled_dir=tmp_path)
    assert result.shape == (2,)
    # County 01001: 2024=0.60, 2020=0.40 — 2024 dominates, so result > 0.50
    assert result[0] > 0.50
