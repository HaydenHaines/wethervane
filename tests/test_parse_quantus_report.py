"""Tests for the Quantus Insights report crosstab parser."""

from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from parse_quantus_report import (
    parse_demographic_vote_shares,
    parse_header,
    parse_quantus_report,
    parse_quantus_text,
    parse_sample_composition,
    two_party_dem_share,
)

FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "quantus_generic_ballot_extract.txt"
)


class TestTwoPartyConversion:
    def test_basic_conversion(self):
        assert two_party_dem_share(34, 55) == pytest.approx(0.3820, abs=0.001)

    def test_both_zero_returns_none(self):
        assert two_party_dem_share(0, 0) is None


class TestHeaderParsing:
    def test_extracts_sample_and_dates(self):
        text = FIXTURE.read_text()
        result = parse_header(text)
        assert result["n_sample"] == 1000
        assert result["date_start"] == "2025-06-23"
        assert result["date_end"] == "2025-06-25"
        assert result["notes"] == "registered voters"


class TestSampleComposition:
    def test_race_composition(self):
        result = parse_sample_composition(FIXTURE.read_text())
        assert result["xt_race_white"] == pytest.approx(0.72)
        assert result["xt_race_black"] == pytest.approx(0.11)
        assert result["xt_race_hispanic"] == pytest.approx(0.11)

    def test_education_composition(self):
        result = parse_sample_composition(FIXTURE.read_text())
        assert result["xt_education_college"] == pytest.approx(0.39)
        assert result["xt_education_noncollege"] == pytest.approx(0.62)

    def test_senior_age_composition(self):
        result = parse_sample_composition(FIXTURE.read_text())
        assert result["xt_age_senior"] == pytest.approx(0.28)

    def test_urbanicity_when_present(self):
        result = parse_sample_composition(FIXTURE.read_text())
        assert result["xt_urbanicity_urban"] == pytest.approx(0.28)
        assert result["xt_urbanicity_rural"] == pytest.approx(0.20)

    def test_empty_text_returns_empty(self):
        assert parse_sample_composition("") == {}


class TestDemographicVoteShares:
    def test_race_vote_shares(self):
        result = parse_demographic_vote_shares(FIXTURE.read_text())
        assert result["xt_vote_race_white"] == pytest.approx(34 / (34 + 55))
        assert result["xt_vote_race_black"] == pytest.approx(76 / (76 + 12))
        assert result["xt_vote_race_hispanic"] == pytest.approx(49 / (49 + 34))

    def test_education_vote_shares(self):
        result = parse_demographic_vote_shares(FIXTURE.read_text())
        assert result["xt_vote_education_college"] == pytest.approx(51 / (51 + 38))
        assert result["xt_vote_education_noncollege"] == pytest.approx(39 / (39 + 50))

    def test_senior_age_vote_share(self):
        result = parse_demographic_vote_shares(FIXTURE.read_text())
        assert result["xt_vote_age_senior"] == pytest.approx(40 / (40 + 50))

    def test_compact_percent_rows(self):
        text = """\
        Generic Congressional Ballot by Demographic
        White Black Hispanic 65+ College Non-college
        34% 76% 49% 40% 51% 39%
        55% 12% 34% 50% 38% 50%
        4% 3% 5% 3% 3% 4%
        """
        result = parse_demographic_vote_shares(text)
        assert result["xt_vote_race_white"] == pytest.approx(34 / (34 + 55))
        assert result["xt_vote_age_senior"] == pytest.approx(40 / (40 + 50))
        assert result["xt_vote_education_noncollege"] == pytest.approx(39 / (39 + 50))


class TestFullParsing:
    def test_parse_text_returns_poll_compatible_fields(self):
        result = parse_quantus_text(FIXTURE.read_text())
        assert result["pollster"] == "Quantus Insights"
        assert result["n_sample"] == 1000
        assert result["xt_race_white"] == pytest.approx(0.72)
        assert result["xt_education_college"] == pytest.approx(0.39)
        assert result["xt_age_senior"] == pytest.approx(0.28)
        assert result["xt_vote_race_white"] == pytest.approx(34 / (34 + 55))
        assert result["xt_vote_education_noncollege"] == pytest.approx(39 / (39 + 50))
        assert result["xt_vote_age_senior"] == pytest.approx(40 / (40 + 50))

    def test_parse_text_file(self):
        result = parse_quantus_report(FIXTURE)
        assert result["xt_vote_race_black"] == pytest.approx(76 / (76 + 12))
