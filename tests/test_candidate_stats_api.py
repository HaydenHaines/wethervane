"""Tests for Phase 5 Step 4: campaign_stats and legislative_stats in GET /candidates/{id}.

Verifies:
  - Response shape includes both new optional sections
  - Candidates with data return plausible numeric values
  - Candidates without data return has_fec_record=False / has_legislative_record=False
  - No 500 errors on missing data (always returns 200 when candidate exists)
  - Stat value ranges are internally consistent (e.g. burn_rate near 1.0 for well-run campaigns)
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from api.routers.candidates import (
    _CAMPAIGN_STATS,
    _LEGISLATIVE_STATS,
    _build_campaign_stats,
    _build_legislative_stats,
)
from api.models import CampaignStats, LegislativeStats


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def known_senator_id() -> str:
    """Return a bioguide ID known to exist in both parquets (Ron Johnson, R-WI)."""
    return "J000293"


@pytest.fixture()
def unknown_id() -> str:
    """Return a bioguide ID that is not in any parquet."""
    return "XXXXXX_NOT_REAL"


# ── CampaignStats builder tests ──────────────────────────────────────────────


def test_campaign_stats_data_loaded():
    """Campaign stats parquet must have been loaded at import time."""
    assert not _CAMPAIGN_STATS.empty, (
        "campaign_stats.parquet must be present in data/sabermetrics/ — "
        "run the Phase 5 pipeline to generate it"
    )


def test_campaign_stats_returns_none_for_unknown(unknown_id):
    """Unknown bioguide IDs produce has_fec_record=False, not a 500."""
    stats = _build_campaign_stats(unknown_id)
    # Not None — we still return a model with has_fec_record=False
    assert stats is not None
    assert isinstance(stats, CampaignStats)
    assert stats.has_fec_record is False
    assert stats.sdr is None
    assert stats.fer is None
    assert stats.burn_rate is None
    assert stats.cycle is None


def test_campaign_stats_returns_valid_data_for_known(known_senator_id):
    """Known senator returns non-null SDR/FER/burn_rate and cycle."""
    stats = _build_campaign_stats(known_senator_id)
    assert stats is not None
    assert stats.has_fec_record is True
    assert stats.cycle in (2022, 2024), f"unexpected cycle: {stats.cycle}"
    # SDR: fraction 0–1
    assert stats.sdr is not None
    assert 0.0 <= stats.sdr <= 1.0, f"SDR out of range: {stats.sdr}"
    # FER: should be positive
    assert stats.fer is not None
    assert stats.fer > 0.0, f"FER should be positive, got {stats.fer}"
    # burn_rate: should be positive
    assert stats.burn_rate is not None
    assert stats.burn_rate > 0.0, f"burn_rate should be positive, got {stats.burn_rate}"


def test_campaign_stats_burn_rate_fer_inverse(known_senator_id):
    """burn_rate and FER should be rough inverses (within floating-point rounding).

    FER = receipts / disbursements; burn_rate = disbursements / receipts.
    Their product should equal 1.0 (up to rounding from separate parquet columns).
    """
    stats = _build_campaign_stats(known_senator_id)
    assert stats is not None and stats.fer and stats.burn_rate
    product = stats.fer * stats.burn_rate
    assert abs(product - 1.0) < 0.01, (
        f"FER × burn_rate should ≈ 1.0, got {product} "
        f"(fer={stats.fer}, burn_rate={stats.burn_rate})"
    )


def test_campaign_stats_sdr_is_fraction_not_percent():
    """SDR must be stored as a 0–1 fraction, not a 0–100 percentage."""
    for bioguide_id in _CAMPAIGN_STATS.index[:20]:
        stats = _build_campaign_stats(str(bioguide_id))
        if stats and stats.has_fec_record and stats.sdr is not None:
            assert stats.sdr <= 1.0, (
                f"SDR for {bioguide_id} = {stats.sdr} looks like a percentage, not a fraction"
            )


# ── LegislativeStats builder tests ──────────────────────────────────────────


def test_legislative_stats_data_loaded():
    """Legislative stats parquet must have been loaded at import time."""
    assert not _LEGISLATIVE_STATS.empty, (
        "legislative_stats.parquet must be present in data/sabermetrics/ — "
        "run the Phase 5 pipeline to generate it"
    )


def test_legislative_stats_returns_no_record_for_unknown(unknown_id):
    """Unknown bioguide IDs produce has_legislative_record=False, not a 500."""
    stats = _build_legislative_stats(unknown_id)
    assert stats is not None
    assert isinstance(stats, LegislativeStats)
    assert stats.has_legislative_record is False
    assert stats.nominate_dim1 is None
    assert stats.les_score is None
    assert stats.congresses_served is None


def test_legislative_stats_returns_valid_data_for_known(known_senator_id):
    """Known senator returns NOMINATE score in [-1, +1] range."""
    stats = _build_legislative_stats(known_senator_id)
    assert stats is not None
    assert stats.has_legislative_record is True
    assert stats.nominate_dim1 is not None
    assert -1.0 <= stats.nominate_dim1 <= 1.0, (
        f"NOMINATE dim1 = {stats.nominate_dim1} out of [-1, +1] range"
    )
    assert stats.congresses_served is not None
    assert stats.congresses_served > 0


def test_legislative_stats_les_score_non_negative(known_senator_id):
    """LES is a count-based score and should be ≥ 0 for serving members."""
    stats = _build_legislative_stats(known_senator_id)
    assert stats is not None and stats.les_score is not None
    assert stats.les_score >= 0.0, f"LES score {stats.les_score} should be non-negative"


def test_legislative_stats_nominate_range_across_all():
    """All NOMINATE scores in the parquet should be within the valid [-1, +1] range."""
    if _LEGISLATIVE_STATS.empty:
        pytest.skip("Legislative stats not loaded")
    col = "career_nominate_dim1"
    if col not in _LEGISLATIVE_STATS.columns:
        pytest.skip(f"Column {col} missing from legislative stats parquet")
    valid = _LEGISLATIVE_STATS[col].dropna()
    out_of_range = valid[(valid < -1.0) | (valid > 1.0)]
    assert len(out_of_range) == 0, (
        f"{len(out_of_range)} NOMINATE scores are outside [-1, +1]: "
        f"{out_of_range.head().tolist()}"
    )


def test_nan_values_become_none():
    """NaN values in the parquet should be converted to None, not propagated as NaN."""
    # Find a row that may have NaN for LES (e.g., senators don't get LES)
    if _LEGISLATIVE_STATS.empty:
        pytest.skip("Legislative stats not loaded")
    for bid in _LEGISLATIVE_STATS.index[:50]:
        stats = _build_legislative_stats(str(bid))
        if stats:
            # If a value is None, it should be truly None — not NaN
            for attr in ("nominate_dim1", "les_score", "les2_score"):
                val = getattr(stats, attr)
                if val is not None:
                    assert not (isinstance(val, float) and math.isnan(val)), (
                        f"{attr} for {bid} is NaN instead of None"
                    )
