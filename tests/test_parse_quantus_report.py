"""Tests for the Quantus Insights crosstab parser."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from parse_quantus_report import (  # noqa: E402
    _normalize_label,
    parse_quantus_composition_text,
    parse_quantus_crosstab_text,
    parse_quantus_report,
    parse_quantus_report_text,
    two_party_dem_share,
)

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "quantus_ga_senate_2025_09.txt"


# ---------------------------------------------------------------------------
# Label normalization
# ---------------------------------------------------------------------------


class TestLabelNormalization:
    def test_lowercase(self):
        assert _normalize_label("COLLEGE") == "college"

    def test_strips_whitespace(self):
        assert _normalize_label("  Hispanic  ") == "hispanic"

    def test_collapses_internal_whitespace(self):
        assert _normalize_label("65   or   older") == "65 or older"

    def test_preserves_punctuation(self):
        # "65+" is a distinct canonical label key — we must not strip the '+'.
        assert _normalize_label("65+") == "65+"


# ---------------------------------------------------------------------------
# Two-party conversion
# ---------------------------------------------------------------------------


class TestTwoPartyConversion:
    def test_even_split(self):
        assert two_party_dem_share(50, 50) == pytest.approx(0.5)

    def test_all_dem(self):
        assert two_party_dem_share(100, 0) == pytest.approx(1.0)

    def test_all_rep(self):
        assert two_party_dem_share(0, 100) == pytest.approx(0.0)

    def test_both_zero_returns_none(self):
        assert two_party_dem_share(0, 0) is None

    def test_quantus_topline(self):
        # Ossoff 40% vs Collins 37% -> 40/77 = 0.5195
        assert two_party_dem_share(40, 37) == pytest.approx(0.5195, abs=0.001)


# ---------------------------------------------------------------------------
# Crosstab vote-share parsing
# ---------------------------------------------------------------------------


class TestCrosstabVoteShares:
    @pytest.fixture(scope="class")
    def text(self):
        return FIXTURE_PATH.read_text(encoding="utf-8")

    @pytest.fixture(scope="class")
    def extracted(self, text):
        return parse_quantus_crosstab_text(text)

    def test_topline_dem_share(self, extracted):
        # Overall: Ossoff 40% / Collins 37% -> 40/77 ≈ 0.5195
        assert extracted["dem_share_topline"] == pytest.approx(0.5195, abs=0.001)

    def test_white_share(self, extracted):
        # White: 24/78 ≈ 0.3077
        assert extracted["xt_vote_race_white"] == pytest.approx(0.3077, abs=0.001)

    def test_black_share(self, extracted):
        # Black: 86/92 ≈ 0.9348
        assert extracted["xt_vote_race_black"] == pytest.approx(0.9348, abs=0.001)

    def test_hispanic_share(self, extracted):
        # Hispanic: 55/85 ≈ 0.6471
        assert extracted["xt_vote_race_hispanic"] == pytest.approx(0.6471, abs=0.001)

    def test_college_share(self, extracted):
        # College graduate: 52/86 ≈ 0.6047
        assert extracted["xt_vote_education_college"] == pytest.approx(0.6047, abs=0.001)

    def test_noncollege_share(self, extracted):
        # No college: 33/73 ≈ 0.4521
        assert extracted["xt_vote_education_noncollege"] == pytest.approx(0.4521, abs=0.001)

    def test_urban_share(self, extracted):
        # Urban: 55/82 ≈ 0.6707
        assert extracted["xt_vote_urbanicity_urban"] == pytest.approx(0.6707, abs=0.001)

    def test_rural_share(self, extracted):
        # Rural: 23/78 ≈ 0.2949
        assert extracted["xt_vote_urbanicity_rural"] == pytest.approx(0.2949, abs=0.001)

    def test_at_least_five_xt_vote_values(self, extracted):
        xt_keys = [k for k in extracted if k.startswith("xt_vote_")]
        assert len(xt_keys) >= 5

    def test_no_asian_in_fixture(self, extracted):
        # The GA fixture does not break out Asian — so the column must be
        # absent, not present-with-None.
        assert "xt_vote_race_asian" not in extracted


class TestCrosstabEdgeCases:
    def test_empty_text_returns_empty(self):
        assert parse_quantus_crosstab_text("") == {}

    def test_text_without_percentages(self):
        assert parse_quantus_crosstab_text("Just some narrative prose.\n") == {}

    def test_partial_demographics(self):
        text = "Overall 48% 43% 9%\nWhite 32% 55% 13%\n"
        result = parse_quantus_crosstab_text(text)
        assert "dem_share_topline" in result
        assert "xt_vote_race_white" in result
        assert "xt_vote_race_black" not in result

    def test_alt_labels_latino_and_non_college(self):
        # Quantus sometimes uses "Latino" and "Non-college" — both must map
        # to the canonical columns.
        text = "Latino 60% 30%\nNon-college 35% 55%\n"
        result = parse_quantus_crosstab_text(text)
        assert "xt_vote_race_hispanic" in result
        assert "xt_vote_education_noncollege" in result

    def test_alt_age_label_65_or_older(self):
        text = "65 or older 40% 50%\n"
        result = parse_quantus_crosstab_text(text)
        assert result["xt_vote_age_senior"] == pytest.approx(40 / 90, abs=0.001)

    def test_two_column_rows_still_parse(self):
        # Quantus occasionally publishes D/R only with no undecided column.
        text = "White 30% 60%\n"
        result = parse_quantus_crosstab_text(text)
        assert result["xt_vote_race_white"] == pytest.approx(30 / 90, abs=0.001)

    def test_four_column_rows_still_parse(self):
        # Guard against reports that add an explicit "Other" column — we
        # should still take the first two percentages.
        text = "Black 85% 8% 4% 3%\n"
        result = parse_quantus_crosstab_text(text)
        assert result["xt_vote_race_black"] == pytest.approx(85 / 93, abs=0.001)

    def test_unknown_group_is_ignored(self):
        text = "LeftHanded Martians 99% 1%\n"
        assert parse_quantus_crosstab_text(text) == {}

    def test_zero_dem_zero_rep_returns_none(self):
        text = "White 0% 0%\n"
        result = parse_quantus_crosstab_text(text)
        assert result["xt_vote_race_white"] is None


# ---------------------------------------------------------------------------
# Sample-composition parsing
# ---------------------------------------------------------------------------


class TestSampleComposition:
    @pytest.fixture(scope="class")
    def text(self):
        return FIXTURE_PATH.read_text(encoding="utf-8")

    @pytest.fixture(scope="class")
    def composition(self, text):
        return parse_quantus_composition_text(text)

    def test_white(self, composition):
        assert composition["xt_race_white"] == pytest.approx(0.58)

    def test_black(self, composition):
        assert composition["xt_race_black"] == pytest.approx(0.31)

    def test_hispanic(self, composition):
        assert composition["xt_race_hispanic"] == pytest.approx(0.07)

    def test_college(self, composition):
        assert composition["xt_education_college"] == pytest.approx(0.34)

    def test_noncollege(self, composition):
        assert composition["xt_education_noncollege"] == pytest.approx(0.66)

    def test_senior(self, composition):
        # "65 or older" row -> xt_age_senior
        assert composition["xt_age_senior"] == pytest.approx(0.30)

    def test_urban(self, composition):
        assert composition["xt_urbanicity_urban"] == pytest.approx(0.30)

    def test_rural(self, composition):
        assert composition["xt_urbanicity_rural"] == pytest.approx(0.25)

    def test_all_values_are_fractions(self, composition):
        for value in composition.values():
            assert 0.0 <= value <= 1.0

    def test_returns_no_vote_columns(self, composition):
        # Composition keys must never include the ``_vote_`` infix.
        for key in composition:
            assert "_vote_" not in key


class TestCompositionEdgeCases:
    def test_empty_text_returns_empty(self):
        assert parse_quantus_composition_text("") == {}

    def test_text_without_header_returns_empty(self):
        # Even with recognizable "White 58%" lines, we should refuse to parse
        # them as composition unless preceded by a composition section header.
        text = "White 58%\nBlack 31%\n"
        assert parse_quantus_composition_text(text) == {}

    def test_header_variants(self):
        for header in (
            "SAMPLE COMPOSITION",
            "Sample composition (weighted)",
            "NATURE OF THE SAMPLE",
            "Weighted Sample",
        ):
            text = f"{header}\nWhite 58%\nBlack 31%\n"
            result = parse_quantus_composition_text(text)
            assert result.get("xt_race_white") == pytest.approx(0.58)
            assert result.get("xt_race_black") == pytest.approx(0.31)

    def test_unknown_group_is_ignored(self):
        text = "SAMPLE COMPOSITION\nMartians 4%\n"
        assert parse_quantus_composition_text(text) == {}


# ---------------------------------------------------------------------------
# Full-report parsing & file I/O
# ---------------------------------------------------------------------------


class TestFullReport:
    def test_merges_vote_and_composition_keys(self):
        text = FIXTURE_PATH.read_text(encoding="utf-8")
        merged = parse_quantus_report_text(text)
        # Both families of keys must appear in the merged output.
        assert any(k.startswith("xt_vote_") for k in merged)
        assert any(k.startswith("xt_") and not k.startswith("xt_vote_") for k in merged)
        # Topline too.
        assert "dem_share_topline" in merged

    def test_no_collisions_between_vote_and_composition(self):
        text = FIXTURE_PATH.read_text(encoding="utf-8")
        vote = parse_quantus_crosstab_text(text)
        comp = parse_quantus_composition_text(text)
        overlap = set(vote) & set(comp)
        assert overlap == set(), f"Unexpected overlap: {overlap}"

    def test_parse_report_from_txt_file(self):
        merged = parse_quantus_report(FIXTURE_PATH)
        assert merged["dem_share_topline"] == pytest.approx(0.5195, abs=0.001)
        assert merged["xt_race_white"] == pytest.approx(0.58)

    def test_parse_report_missing_file_raises(self, tmp_path):
        missing = tmp_path / "does_not_exist.txt"
        with pytest.raises(FileNotFoundError):
            parse_quantus_report(missing)
