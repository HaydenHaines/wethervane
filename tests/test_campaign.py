"""Tests for campaign finance stats in src/sabermetrics/campaign.py.

Tests use synthetic DataFrames and mock file I/O so no network access or
actual FEC downloads are required. Coverage:

1. compute_sdr() — small-dollar ratio math and edge cases
2. compute_fer() — fundraising efficiency and edge cases
3. compute_burn_rate() — burn rate math and edge cases
4. build_fec_id_crosswalk() — bioguide → FEC ID linkage
5. build_campaign_stats() — full pipeline integration with mocked FEC data
6. Graceful handling of state-only / no-FEC candidates
7. load_weball_file() — column parsing and dollar coercion
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.sabermetrics.campaign import (
    _safe_float,
    build_campaign_stats,
    build_fec_id_crosswalk,
    compute_burn_rate,
    compute_fer,
    compute_sdr,
    load_weball_file,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic FEC summary rows
# ---------------------------------------------------------------------------


@pytest.fixture
def ossoff_summary() -> pd.Series:
    """Synthetic FEC summary row resembling a competitive Senate D candidate."""
    return pd.Series(
        {
            "fec_candidate_id": "S0GA00161",
            "candidate_name": "OSSOFF, JON",
            "state": "GA",
            "total_receipts": 106_000_000.0,
            "total_disbursements": 95_000_000.0,
            "total_individual_contributions": 88_000_000.0,
            "candidate_contributions": 0.0,
            "candidate_loans": 0.0,
            "other_loans": 0.0,
            "individual_refunds": 500_000.0,
            "committee_refunds": 100_000.0,
            "other_committee_contributions": 2_000_000.0,
            "party_committee_contributions": 500_000.0,
            "cash_on_hand_bop": 5_000_000.0,
            "cash_on_hand_cop": 11_000_000.0,
        }
    )


@pytest.fixture
def cornyn_summary() -> pd.Series:
    """Synthetic FEC summary row resembling a well-funded Republican incumbent."""
    return pd.Series(
        {
            "fec_candidate_id": "S2TX00418",
            "candidate_name": "CORNYN, JOHN",
            "state": "TX",
            "total_receipts": 32_000_000.0,
            "total_disbursements": 28_000_000.0,
            "total_individual_contributions": 20_000_000.0,
            "candidate_contributions": 0.0,
            "candidate_loans": 0.0,
            "other_loans": 0.0,
            "individual_refunds": 100_000.0,
            "committee_refunds": 50_000.0,
            "other_committee_contributions": 4_000_000.0,
            "party_committee_contributions": 2_000_000.0,
            "cash_on_hand_bop": 4_000_000.0,
            "cash_on_hand_cop": 8_000_000.0,
        }
    )


@pytest.fixture
def empty_summary() -> pd.Series:
    """Candidate with no financial activity — all zeros."""
    return pd.Series(
        {
            "fec_candidate_id": "S0XX00000",
            "candidate_name": "EMPTY, CAND",
            "state": "XX",
            "total_receipts": 0.0,
            "total_disbursements": 0.0,
            "total_individual_contributions": 0.0,
            "candidate_contributions": 0.0,
            "candidate_loans": 0.0,
            "other_loans": 0.0,
            "individual_refunds": 0.0,
            "committee_refunds": 0.0,
            "other_committee_contributions": 0.0,
            "party_committee_contributions": 0.0,
        }
    )


@pytest.fixture
def self_funded_summary() -> pd.Series:
    """Self-funded candidate: all receipts come from candidate loans."""
    return pd.Series(
        {
            "fec_candidate_id": "H0XX00001",
            "candidate_name": "SELFMADE, RICH",
            "state": "TX",
            "total_receipts": 5_000_000.0,
            "total_disbursements": 4_800_000.0,
            "total_individual_contributions": 0.0,
            "candidate_contributions": 500_000.0,
            "candidate_loans": 4_500_000.0,
            "other_loans": 0.0,
            "individual_refunds": 0.0,
            "committee_refunds": 0.0,
            "other_committee_contributions": 0.0,
            "party_committee_contributions": 0.0,
        }
    )


# ---------------------------------------------------------------------------
# 1. compute_sdr() tests
# ---------------------------------------------------------------------------


def test_sdr_returns_float_between_0_and_1(ossoff_summary):
    """SDR should be a float in [0, 1] for a normal fundraising profile."""
    sdr = compute_sdr(ossoff_summary)
    assert sdr is not None
    assert 0.0 <= sdr <= 1.0


def test_sdr_higher_for_grassroots_candidate(ossoff_summary, cornyn_summary):
    """Ossoff-style fundraising (higher % small donors) should yield higher SDR
    than a PAC-heavy Republican incumbent with more party committee contributions."""
    sdr_ossoff = compute_sdr(ossoff_summary)
    sdr_cornyn = compute_sdr(cornyn_summary)
    assert sdr_ossoff is not None
    assert sdr_cornyn is not None
    # Ossoff has lower party/committee contributions relative to total indiv,
    # so SDR proxy should be higher.
    assert sdr_ossoff > sdr_cornyn, (
        f"Ossoff SDR ({sdr_ossoff:.3f}) should exceed Cornyn SDR ({sdr_cornyn:.3f})"
    )


def test_sdr_returns_none_when_no_individual_contributions(empty_summary):
    """SDR must return None when denominator is zero (no individual contributions)."""
    sdr = compute_sdr(empty_summary)
    assert sdr is None


def test_sdr_returns_none_for_missing_field():
    """SDR must return None when total_individual_contributions is missing."""
    row = pd.Series({"fec_candidate_id": "X0XX00000"})  # no financial fields
    assert compute_sdr(row) is None


def test_sdr_formula_exact():
    """SDR mathematical definition: (total_indiv - other_cmte - party_cmte) / total_indiv."""
    row = pd.Series(
        {
            "total_individual_contributions": 1_000_000.0,
            "other_committee_contributions": 100_000.0,
            "party_committee_contributions": 50_000.0,
        }
    )
    expected = (1_000_000 - 100_000 - 50_000) / 1_000_000
    result = compute_sdr(row)
    assert result is not None
    assert abs(result - expected) < 1e-10


# ---------------------------------------------------------------------------
# 2. compute_fer() tests
# ---------------------------------------------------------------------------


def test_fer_returns_float_near_1_for_efficient_campaign(ossoff_summary):
    """A campaign with minimal refunds should have FER close to 1."""
    fer = compute_fer(ossoff_summary)
    assert fer is not None
    assert 0.5 <= fer <= 1.0


def test_fer_returns_none_when_zero_receipts(empty_summary):
    """FER must return None when total_receipts is zero."""
    assert compute_fer(empty_summary) is None


def test_fer_returns_none_when_self_funded(self_funded_summary):
    """FER must return None when all receipts come from candidate loans
    (net_raised after subtracting self-funding = 0)."""
    fer = compute_fer(self_funded_summary)
    assert fer is None


def test_fer_formula_exact():
    """FER mathematical definition check with known inputs."""
    row = pd.Series(
        {
            "total_receipts": 1_000_000.0,
            "candidate_contributions": 0.0,
            "candidate_loans": 0.0,
            "other_loans": 0.0,
            "individual_refunds": 20_000.0,
            "committee_refunds": 5_000.0,
        }
    )
    # net_raised = 1_000_000, fundraising_costs = 25_000
    expected = (1_000_000 - 25_000) / 1_000_000
    result = compute_fer(row)
    assert result is not None
    assert abs(result - expected) < 1e-10


# ---------------------------------------------------------------------------
# 3. compute_burn_rate() tests
# ---------------------------------------------------------------------------


def test_burn_rate_below_1_for_solvent_campaign(ossoff_summary):
    """A campaign spending less than it raises has burn_rate < 1."""
    br = compute_burn_rate(ossoff_summary)
    assert br is not None
    assert br < 1.0


def test_burn_rate_above_1_when_spending_exceeds_receipts():
    """Burn rate > 1 is possible (drawing down reserves / taking on debt)."""
    row = pd.Series(
        {
            "total_receipts": 1_000_000.0,
            "total_disbursements": 1_200_000.0,
        }
    )
    br = compute_burn_rate(row)
    assert br is not None
    assert br > 1.0
    assert abs(br - 1.2) < 1e-10


def test_burn_rate_exact_formula(ossoff_summary):
    """Burn rate = total_disbursements / total_receipts exactly."""
    br = compute_burn_rate(ossoff_summary)
    expected = ossoff_summary["total_disbursements"] / ossoff_summary["total_receipts"]
    assert br is not None
    assert abs(br - expected) < 1e-10


def test_burn_rate_returns_none_for_zero_receipts(empty_summary):
    """Burn rate must return None when total_receipts is zero."""
    assert compute_burn_rate(empty_summary) is None


# ---------------------------------------------------------------------------
# 4. _safe_float helper
# ---------------------------------------------------------------------------


def test_safe_float_handles_none():
    assert _safe_float(None) is None


def test_safe_float_handles_nan_string():
    assert _safe_float("nan") is None


def test_safe_float_handles_empty_string():
    assert _safe_float("") is None


def test_safe_float_converts_valid_number():
    assert _safe_float("1234.56") == pytest.approx(1234.56)


def test_safe_float_with_default():
    assert _safe_float(None, default=0.0) == 0.0


# ---------------------------------------------------------------------------
# 5. load_weball_file() — zip parsing
# ---------------------------------------------------------------------------


def _make_weball_zip(rows: list[str]) -> bytes:
    """Build a synthetic weball zip in memory for testing."""
    content = "\n".join(rows).encode("latin-1")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("weball22.txt", content)
    return buf.getvalue()


def test_load_weball_parses_columns(tmp_path):
    """load_weball_file should parse pipe-delimited rows and return correct columns."""
    # Build a minimal weball row (30 fields, pipe-separated)
    # Field positions: 0=CAND_ID, 5=TTL_RECEIPTS, 7=TTL_DISB, 17=TTL_INDIV_CONTRIB, 18=STATE
    fields = [""] * 30
    fields[0] = "S0GA00161"
    fields[1] = "OSSOFF, JON"
    fields[2] = "C"
    fields[4] = "DEM"
    fields[5] = "106000000"
    fields[7] = "95000000"
    fields[17] = "88000000"
    fields[18] = "GA"
    row = "|".join(fields)

    zip_bytes = _make_weball_zip([row])
    zip_path = tmp_path / "weball22.zip"
    zip_path.write_bytes(zip_bytes)

    df = load_weball_file(zip_path)

    assert len(df) == 1
    assert df["fec_candidate_id"].iloc[0] == "S0GA00161"
    assert df["state"].iloc[0] == "GA"
    assert df["total_receipts"].iloc[0] == pytest.approx(106_000_000.0)
    assert df["total_disbursements"].iloc[0] == pytest.approx(95_000_000.0)
    assert df["total_individual_contributions"].iloc[0] == pytest.approx(88_000_000.0)


def test_load_weball_drops_rows_with_no_candidate_id(tmp_path):
    """Rows with empty CAND_ID (e.g. summary lines) should be dropped."""
    fields_valid = [""] * 30
    fields_valid[0] = "S0GA00161"
    fields_valid[5] = "1000000"

    fields_empty = [""] * 30
    # Leave fields_empty[0] empty

    zip_bytes = _make_weball_zip(["|".join(fields_valid), "|".join(fields_empty)])
    zip_path = tmp_path / "weball22.zip"
    zip_path.write_bytes(zip_bytes)

    df = load_weball_file(zip_path)
    assert len(df) == 1
    assert df["fec_candidate_id"].iloc[0] == "S0GA00161"


# ---------------------------------------------------------------------------
# 6. build_fec_id_crosswalk() — YAML-based linkage
# ---------------------------------------------------------------------------


def test_build_fec_id_crosswalk_maps_bioguide_to_fec(tmp_path):
    """Crosswalk should map bioguide IDs → FEC candidate IDs from YAML."""
    # Write a minimal legislators YAML with known FEC IDs
    yaml_content = """
- id:
    bioguide: O000174
    fec: ["S0GA00161"]
  name:
    official_full: Jon Ossoff
    last: Ossoff
    first: Jon
  terms:
    - type: sen
      state: GA
      party: Democrat
      start: "2021-01-20"
      end: "2027-01-03"
"""
    yaml_path = tmp_path / "legislators-current.yaml"
    yaml_path.write_text(yaml_content)
    (tmp_path / "legislators-historical.yaml").write_text("[]")

    registry = {
        "persons": {
            "O000174": {
                "name": "Jon Ossoff",
                "party": "D",
                "bioguide_id": "O000174",
                "races": [{"year": 2020, "state": "GA", "office": "Senate"}],
            },
            "gen_abc123": {
                "name": "Some Governor",
                "party": "D",
                "bioguide_id": None,
                "races": [{"year": 2022, "state": "GA", "office": "Governor"}],
            },
        }
    }

    crosswalk = build_fec_id_crosswalk(registry, legislators_dir=tmp_path)

    assert crosswalk["O000174"] == ["S0GA00161"]
    assert crosswalk["gen_abc123"] == []


def test_build_fec_id_crosswalk_empty_registry(tmp_path):
    """Crosswalk with empty registry should return empty dict."""
    # Use a real tmp_path with empty YAML files to simulate missing legislators
    (tmp_path / "legislators-current.yaml").write_text("[]")
    (tmp_path / "legislators-historical.yaml").write_text("[]")
    crosswalk = build_fec_id_crosswalk({"persons": {}}, legislators_dir=tmp_path)
    assert crosswalk == {}


# ---------------------------------------------------------------------------
# 7. build_campaign_stats() — integration test with mocked FEC data
# ---------------------------------------------------------------------------


def test_build_campaign_stats_returns_dataframe_with_expected_columns(tmp_path):
    """build_campaign_stats should return a DataFrame with all required columns."""
    # Build minimal registry
    registry = {
        "persons": {
            "O000174": {
                "name": "Jon Ossoff",
                "party": "D",
                "bioguide_id": "O000174",
                "races": [{"year": 2020, "state": "GA", "office": "Senate"}],
            }
        }
    }

    # Build a synthetic weball zip for cycle 2022
    fields = [""] * 30
    fields[0] = "S0GA00161"
    fields[5] = "106000000"
    fields[7] = "95000000"
    fields[17] = "88000000"
    fields[18] = "GA"
    fields[25] = "2000000"
    fields[26] = "500000"
    zip_bytes = _make_weball_zip(["|".join(fields)])
    zip_path = tmp_path / "weball22.zip"
    zip_path.write_bytes(zip_bytes)

    # Minimal YAML crosswalk
    yaml_content = """
- id:
    bioguide: O000174
    fec: ["S0GA00161"]
  name:
    official_full: Jon Ossoff
    last: Ossoff
    first: Jon
  terms:
    - type: sen
      state: GA
      party: Democrat
      start: "2021-01-20"
"""
    (tmp_path / "legislators-current.yaml").write_text(yaml_content)
    (tmp_path / "legislators-historical.yaml").write_text("[]")

    output_path = tmp_path / "campaign_stats.parquet"

    df = build_campaign_stats(
        candidate_registry=registry,
        cycles=["2022"],
        fec_data_dir=str(tmp_path),
        legislators_dir=str(tmp_path),
        output_path=str(output_path),
        download_if_missing=False,
    )

    required_columns = {
        "person_id",
        "name",
        "party",
        "cycle",
        "fec_candidate_id",
        "total_receipts",
        "total_disbursements",
        "total_individual_contributions",
        "sdr",
        "fer",
        "burn_rate",
        "has_fec_record",
    }
    assert required_columns.issubset(set(df.columns))


def test_build_campaign_stats_matches_fec_record(tmp_path):
    """Candidates with FEC IDs should be matched and have non-null stats."""
    registry = {
        "persons": {
            "O000174": {
                "name": "Jon Ossoff",
                "party": "D",
                "bioguide_id": "O000174",
                "races": [{"year": 2022, "state": "GA", "office": "Senate"}],
            }
        }
    }

    fields = [""] * 30
    fields[0] = "S0GA00161"
    fields[5] = "106000000"
    fields[7] = "95000000"
    fields[17] = "88000000"
    fields[18] = "GA"
    fields[25] = "2000000"
    fields[26] = "500000"
    fields[28] = "500000"  # individual_refunds
    fields[29] = "100000"  # committee_refunds
    zip_bytes = _make_weball_zip(["|".join(fields)])
    zip_path = tmp_path / "weball22.zip"
    zip_path.write_bytes(zip_bytes)

    yaml_content = """
- id:
    bioguide: O000174
    fec: ["S0GA00161"]
  name:
    official_full: Jon Ossoff
    last: Ossoff
    first: Jon
  terms:
    - type: sen
      state: GA
      party: Democrat
      start: "2021-01-20"
"""
    (tmp_path / "legislators-current.yaml").write_text(yaml_content)
    (tmp_path / "legislators-historical.yaml").write_text("[]")

    df = build_campaign_stats(
        candidate_registry=registry,
        cycles=["2022"],
        fec_data_dir=str(tmp_path),
        legislators_dir=str(tmp_path),
        output_path=str(tmp_path / "out.parquet"),
        download_if_missing=False,
    )

    ossoff_row = df[(df["person_id"] == "O000174") & (df["cycle"] == 2022)]
    assert len(ossoff_row) == 1
    row = ossoff_row.iloc[0]

    assert row["has_fec_record"] is True or row["has_fec_record"] == True
    assert row["sdr"] is not None and not math.isnan(float(row["sdr"]))
    assert row["fer"] is not None and not math.isnan(float(row["fer"]))
    assert row["burn_rate"] is not None and not math.isnan(float(row["burn_rate"]))

    # Spot-check burn_rate = 95M / 106M ≈ 0.896
    assert abs(float(row["burn_rate"]) - 95_000_000 / 106_000_000) < 0.01


def test_build_campaign_stats_graceful_no_fec_for_governor(tmp_path):
    """State-only candidates (governors) with no FEC ID should appear with NaN stats."""
    registry = {
        "persons": {
            "gen_abc123": {
                "name": "Brian Kemp",
                "party": "R",
                "bioguide_id": None,
                "races": [{"year": 2022, "state": "GA", "office": "Governor"}],
            }
        }
    }

    # Empty weball file (no records matching this governor)
    zip_bytes = _make_weball_zip([])
    zip_path = tmp_path / "weball22.zip"
    zip_path.write_bytes(zip_bytes)

    (tmp_path / "legislators-current.yaml").write_text("[]")
    (tmp_path / "legislators-historical.yaml").write_text("[]")

    df = build_campaign_stats(
        candidate_registry=registry,
        cycles=["2022"],
        fec_data_dir=str(tmp_path),
        legislators_dir=str(tmp_path),
        output_path=str(tmp_path / "out.parquet"),
        download_if_missing=False,
    )

    assert len(df) == 1
    row = df.iloc[0]
    # Governor has no FEC record — stats should be None/NaN
    assert row["has_fec_record"] is False or row["has_fec_record"] == False
    assert row["fec_candidate_id"] is None or (
        isinstance(row["fec_candidate_id"], float) and math.isnan(row["fec_candidate_id"])
    )


def test_build_campaign_stats_output_written_to_parquet(tmp_path):
    """build_campaign_stats should write a parquet file at the specified path."""
    registry = {"persons": {}}

    df = build_campaign_stats(
        candidate_registry=registry,
        cycles=["2022"],
        fec_data_dir=str(tmp_path),
        legislators_dir=str(tmp_path),
        output_path=str(tmp_path / "campaign_stats.parquet"),
        download_if_missing=False,
    )

    assert (tmp_path / "campaign_stats.parquet").exists()
    reloaded = pd.read_parquet(tmp_path / "campaign_stats.parquet")
    assert len(reloaded) == len(df)


def test_build_campaign_stats_multiple_cycles(tmp_path):
    """Processing two cycles should produce 2 rows per candidate."""
    registry = {
        "persons": {
            "W000790": {
                "name": "Raphael Warnock",
                "party": "D",
                "bioguide_id": "W000790",
                "races": [
                    {"year": 2020, "state": "GA", "office": "Senate"},
                    {"year": 2022, "state": "GA", "office": "Senate"},
                ],
            }
        }
    }

    yaml_content = """
- id:
    bioguide: W000790
    fec: ["S0GA00350"]
  name:
    official_full: Raphael G. Warnock
    last: Warnock
    first: Raphael
  terms:
    - type: sen
      state: GA
      party: Democrat
      start: "2021-01-20"
"""
    (tmp_path / "legislators-current.yaml").write_text(yaml_content)
    (tmp_path / "legislators-historical.yaml").write_text("[]")

    # Create weball files for both cycles
    for suffix, fec_id in [("20", "S0GA00350"), ("22", "S0GA00350")]:
        fields = [""] * 30
        fields[0] = fec_id
        fields[5] = "50000000"
        fields[7] = "45000000"
        fields[17] = "40000000"
        zip_bytes = _make_weball_zip(["|".join(fields)])
        (tmp_path / f"weball{suffix}.zip").write_bytes(zip_bytes)

    df = build_campaign_stats(
        candidate_registry=registry,
        cycles=["2020", "2022"],
        fec_data_dir=str(tmp_path),
        legislators_dir=str(tmp_path),
        output_path=str(tmp_path / "out.parquet"),
        download_if_missing=False,
    )

    warnock_rows = df[df["person_id"] == "W000790"]
    assert len(warnock_rows) == 2
    assert set(warnock_rows["cycle"].tolist()) == {2020, 2022}
