"""Tests for the canonical Cygnal report parser entrypoint."""

from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import parse_cygnal_crosstab
from parse_cygnal_report import (
    parse_cygnal_report,
    parse_cygnal_text,
    parse_demographic_vote_shares,
    parse_header,
    parse_sample_composition,
    two_party_dem_share,
)

FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "cygnal_generic_ballot_extract.txt"
)


class TestTwoPartyConversion:
    def test_basic_conversion(self):
        assert two_party_dem_share(43, 51) == pytest.approx(43 / 94)

    def test_both_zero_returns_none(self):
        assert two_party_dem_share(0, 0) is None


class TestHeaderParsing:
    def test_extracts_sample_and_dates(self):
        result = parse_header(FIXTURE.read_text())
        assert result["n_sample"] == 1500
        assert result["date_start"] == "2026-08-06"
        assert result["date_end"] == "2026-08-07"


class TestSampleComposition:
    def test_race_composition(self):
        result = parse_sample_composition(FIXTURE.read_text())
        assert result["xt_race_white"] == pytest.approx(0.709)
        assert result["xt_race_black"] == pytest.approx(0.116)
        assert result["xt_race_hispanic"] == pytest.approx(0.109)
        assert result["xt_race_asian"] == pytest.approx(0.020)

    def test_education_age_and_urbanicity(self):
        result = parse_sample_composition(FIXTURE.read_text())
        assert result["xt_education_college"] == pytest.approx(0.473)
        assert result["xt_education_noncollege"] == pytest.approx(0.520)
        assert result["xt_age_senior"] == pytest.approx(0.296)
        assert result["xt_urbanicity_urban"] == pytest.approx(0.211)
        assert result["xt_urbanicity_rural"] == pytest.approx(0.260)

    def test_empty_text_returns_empty(self):
        assert parse_sample_composition("") == {}


class TestDemographicVoteShares:
    def test_vote_shares(self):
        result = parse_demographic_vote_shares(FIXTURE.read_text())
        assert result["xt_vote_race_white"] == pytest.approx(43 / (43 + 51))
        assert result["xt_vote_race_black"] == pytest.approx(78 / (78 + 13))
        assert result["xt_vote_race_hispanic"] == pytest.approx(52 / (52 + 39))
        assert result["xt_vote_race_asian"] == pytest.approx(58 / (58 + 32))
        assert result["xt_vote_age_senior"] == pytest.approx(45 / (45 + 47))
        assert result["xt_vote_education_college"] == pytest.approx(50 / (50 + 41))
        assert result["xt_vote_education_noncollege"] == pytest.approx(43 / (43 + 47))
        assert result["xt_vote_urbanicity_urban"] == pytest.approx(55 / (55 + 36))
        assert result["xt_vote_urbanicity_rural"] == pytest.approx(33 / (33 + 58))

    def test_inline_rows_work_without_response_labels(self):
        text = """\
        White or Caucasian 51% 43% 6%
        Black or African American 13% 78% 9%
        65+ 47% 45% 8%
        """
        result = parse_demographic_vote_shares(text)
        assert result["xt_vote_race_white"] == pytest.approx(43 / (43 + 51))
        assert result["xt_vote_race_black"] == pytest.approx(78 / (78 + 13))
        assert result["xt_vote_age_senior"] == pytest.approx(45 / (45 + 47))


class TestFullParsing:
    def test_canonical_text_parser_returns_poll_compatible_fields(self):
        result = parse_cygnal_text(FIXTURE.read_text())
        assert result["pollster"] == "Cygnal"
        assert result["n_sample"] == 1500
        assert result["xt_race_white"] == pytest.approx(0.709)
        assert result["xt_education_college"] == pytest.approx(0.473)
        assert result["xt_age_senior"] == pytest.approx(0.296)
        assert result["xt_vote_race_white"] == pytest.approx(43 / (43 + 51))
        assert result["xt_vote_education_noncollege"] == pytest.approx(43 / (43 + 47))

    def test_canonical_file_parser(self):
        result = parse_cygnal_report(FIXTURE)
        assert result["xt_vote_race_black"] == pytest.approx(78 / (78 + 13))

    def test_legacy_crosstab_module_redirects_to_canonical_parser(self):
        assert parse_cygnal_crosstab.parse_cygnal_report(FIXTURE) == parse_cygnal_report(
            FIXTURE
        )
