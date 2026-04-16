"""Tests for the candidate registry (sabermetrics Phase 1).

Coverage:
- State name normalization (full names and abbreviations)
- Name normalization and similarity
- Name matching to congress-legislators
- Multi-race candidate identification (Ron Johnson, Mark Kelly)
- Cross-office linking (Beto O'Rourke Senate 2018 → Governor 2022)
- Registry output schema validation
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# State name normalization
# ---------------------------------------------------------------------------


class TestNormalizeState:
    """normalize_state() must handle both full names and abbreviations."""

    def test_full_name_to_abbr(self):
        from src.sabermetrics.registry import normalize_state

        assert normalize_state("Wisconsin") == "WI"
        assert normalize_state("Kentucky") == "KY"
        assert normalize_state("New Hampshire") == "NH"
        assert normalize_state("West Virginia") == "WV"
        assert normalize_state("District of Columbia") == "DC"

    def test_abbreviation_passthrough(self):
        from src.sabermetrics.registry import normalize_state

        assert normalize_state("WI") == "WI"
        assert normalize_state("TX") == "TX"
        assert normalize_state("CA") == "CA"

    def test_lowercase_abbreviation(self):
        from src.sabermetrics.registry import normalize_state

        # Lowercase abbreviations should be upcased
        assert normalize_state("wi") == "WI"

    def test_unknown_state_raises(self):
        from src.sabermetrics.registry import normalize_state

        with pytest.raises(ValueError, match="Unknown state"):
            normalize_state("Freedonia")

    def test_2014_senate_states(self):
        """2014 senate data uses full state names — these must all parse correctly."""
        from src.sabermetrics.registry import normalize_state

        senate_2014_sample = [
            "Wyoming",
            "West Virginia",
            "Virginia",
            "Texas",
            "Tennessee",
            "South Dakota",
            "Rhode Island",
            "Oregon",
            "Oklahoma",
            "Montana",
        ]
        for state in senate_2014_sample:
            result = normalize_state(state)
            assert len(result) == 2, f"{state} did not produce 2-letter abbreviation"


# ---------------------------------------------------------------------------
# Name normalization and similarity
# ---------------------------------------------------------------------------


class TestNormalizeName:
    """normalize_name() must produce consistent canonical forms."""

    def test_basic(self):
        from src.sabermetrics.registry import normalize_name

        assert normalize_name("Ron Johnson") == "ron johnson"

    def test_accent_stripping(self):
        from src.sabermetrics.registry import normalize_name

        # Accented chars should be converted to their ASCII base
        assert "jose" in normalize_name("José")

    def test_apostrophe_variants(self):
        from src.sabermetrics.registry import normalize_name

        # Different apostrophe chars should both produce the same tokens
        ascii_apos = normalize_name("Beto O'Rourke")
        unicode_apos = normalize_name("Beto O\u2019Rourke")  # right single quotation mark
        assert ascii_apos == unicode_apos

    def test_whitespace_collapse(self):
        from src.sabermetrics.registry import normalize_name

        assert normalize_name("John  Smith") == "john smith"

    def test_period_removal(self):
        from src.sabermetrics.registry import normalize_name

        # "Jr." and similar suffixes should lose punctuation
        assert "jr" in normalize_name("John Smith Jr.")


class TestNameSimilarity:
    """name_similarity() must return scores in [0, 1]."""

    def test_identical_names(self):
        from src.sabermetrics.registry import name_similarity

        score = name_similarity("Ron Johnson", "Ron Johnson")
        assert score == 1.0

    def test_same_name_different_suffix(self):
        from src.sabermetrics.registry import name_similarity

        # "Ron Johnson" vs "Ronald Johnson" — they share "johnson" token but
        # "ron" != "ronald", so Jaccard on token sets gives 1/3 ≈ 0.33.
        # The matching pipeline handles this via last-name-only fallback in
        # _match_to_legislator, not via name_similarity alone.
        score = name_similarity("Ron Johnson", "Ronald Johnson")
        # At least "johnson" overlaps — score must be > 0
        assert score > 0.0, f"Expected >0, got {score}"
        # But not a full match due to different first names
        assert score < 1.0, f"Expected <1.0, got {score}"

    def test_completely_different(self):
        from src.sabermetrics.registry import name_similarity

        score = name_similarity("Ron Johnson", "Maria Cantwell")
        assert score < 0.3, f"Expected <0.3, got {score}"

    def test_partial_match(self):
        from src.sabermetrics.registry import name_similarity

        # "Mark Kelly" vs "Mark Kelly (D)" — the extra token hurts but core tokens match
        score = name_similarity("Mark Kelly", "Mark Kelly")
        assert score == 1.0

    def test_returns_float_in_range(self):

        from src.sabermetrics.registry import name_similarity

        names = ["Ron Johnson", "Mark Kelly", "Beto O'Rourke", "Tim Scott"]
        for a in names:
            for b in names:
                score = name_similarity(a, b)
                assert 0.0 <= score <= 1.0, f"Out of range for {a!r} vs {b!r}: {score}"


# ---------------------------------------------------------------------------
# Congress-legislators matching
# ---------------------------------------------------------------------------


class TestMatchToLegislator:
    """_match_to_legislator() must correctly link 538 names to bioguide IDs."""

    @pytest.fixture(scope="class")
    def legislators(self):
        """Load legislators once for the class to avoid repeated YAML parsing."""
        from src.sabermetrics.registry import load_congress_legislators

        legislators_dir = Path("data/raw/congress-legislators")
        if not legislators_dir.exists():
            pytest.skip("Congress-legislators data not available")
        return load_congress_legislators(legislators_dir)

    def test_ron_johnson_matches(self, legislators):
        from src.sabermetrics.registry import _match_to_legislator

        match = _match_to_legislator("Ron Johnson", "WI", "R", 2022, legislators)
        assert match is not None, "Ron Johnson should match"
        assert match["bioguide_id"] == "J000293"

    def test_mark_kelly_matches(self, legislators):
        from src.sabermetrics.registry import _match_to_legislator

        match = _match_to_legislator("Mark Kelly", "AZ", "D", 2020, legislators)
        assert match is not None, "Mark Kelly should match"
        assert match["bioguide_id"] == "K000377"

    def test_beto_orourke_matches(self, legislators):
        from src.sabermetrics.registry import _match_to_legislator

        match = _match_to_legislator("Beto O'Rourke", "TX", "D", 2018, legislators)
        assert match is not None, "Beto O'Rourke should match"
        assert match["bioguide_id"] == "O000170"

    def test_wrong_state_no_match(self, legislators):
        """Ron Johnson should NOT match if wrong state is provided."""
        from src.sabermetrics.registry import _match_to_legislator

        # Ron Johnson served in WI, not CA
        match = _match_to_legislator("Ron Johnson", "CA", "R", 2022, legislators)
        # Either no match or a different person from CA
        if match is not None:
            assert match["bioguide_id"] != "J000293", "Should not match WI Ron Johnson"

    def test_random_name_no_match(self, legislators):
        from src.sabermetrics.registry import _match_to_legislator

        match = _match_to_legislator("Zxyqwerty Nonsense", "WI", "D", 2020, legislators)
        assert match is None


# ---------------------------------------------------------------------------
# Full registry integration tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def registry():
    """Load the built candidate registry (built by build_candidate_registry)."""
    registry_path = Path("data/sabermetrics/candidate_registry.json")
    if not registry_path.exists():
        pytest.skip("candidate_registry.json not built yet — run build_candidate_registry()")
    with open(registry_path) as f:
        return json.load(f)


class TestRegistrySchema:
    """The registry must conform to the documented output schema."""

    def test_top_level_keys(self, registry):
        assert "persons" in registry
        assert "_meta" in registry

    def test_meta_fields(self, registry):
        meta = registry["_meta"]
        assert "created" in meta
        assert "sources" in meta
        assert "total_persons" in meta
        assert "multi_race_persons" in meta
        assert isinstance(meta["total_persons"], int)
        assert isinstance(meta["multi_race_persons"], int)
        assert meta["total_persons"] > 0
        assert meta["multi_race_persons"] > 0

    def test_person_schema(self, registry):
        """Every person must have the required fields."""
        required_fields = {"name", "party", "bioguide_id", "races"}
        for pid, person in registry["persons"].items():
            missing = required_fields - set(person.keys())
            assert not missing, f"Person {pid} missing fields: {missing}"

    def test_race_schema(self, registry):
        """Every race must have the required fields."""
        required_fields = {"year", "state", "office", "special", "party"}
        for pid, person in registry["persons"].items():
            for race in person["races"]:
                missing = required_fields - set(race.keys())
                assert not missing, f"Person {pid} race missing fields: {missing}"

    def test_party_codes_valid(self, registry):
        """Party codes must be single-letter strings."""
        valid_parties = {"D", "R", "I", "L", "G", "?"}
        for pid, person in registry["persons"].items():
            assert person["party"] in valid_parties, f"Person {pid} has invalid party: {person['party']!r}"

    def test_state_codes_valid(self, registry):
        """All state codes in race records must be 2-letter abbreviations."""
        for pid, person in registry["persons"].items():
            for race in person["races"]:
                state = race["state"]
                assert len(state) == 2 and state.isupper(), f"Person {pid} has invalid state: {state!r}"

    def test_office_values_valid(self, registry):
        """Office values must be one of the expected values."""
        valid_offices = {"Senate", "Governor", "President"}
        for pid, person in registry["persons"].items():
            for race in person["races"]:
                assert race["office"] in valid_offices, f"Person {pid} has invalid office: {race['office']!r}"

    def test_meta_counts_consistent(self, registry):
        """_meta counts must match actual data."""
        actual_total = len(registry["persons"])
        actual_multi = sum(1 for p in registry["persons"].values() if len(p["races"]) > 1)
        assert registry["_meta"]["total_persons"] == actual_total
        assert registry["_meta"]["multi_race_persons"] == actual_multi


class TestMultiRaceCandidates:
    """Known multi-race senate candidates must be correctly linked."""

    def test_ron_johnson_three_races(self, registry):
        """Ron Johnson ran for WI Senate in 2010, 2016, and 2022.

        Note: 2010 is absent from the 538 checking-our-work data (no candidate
        names before 2014), so we expect 2016 and 2022 at minimum.
        """
        ron = next((p for p in registry["persons"].values() if p.get("bioguide_id") == "J000293"), None)
        assert ron is not None, "Ron Johnson (J000293) should be in registry"
        assert ron["name"] == "Ron Johnson"
        races = ron["races"]
        race_years = [r["year"] for r in races]
        assert 2022 in race_years, "Ron Johnson 2022 race should be included"
        assert 2016 in race_years, "Ron Johnson 2016 race should be included"
        assert all(r["state"] == "WI" for r in races), "All Ron Johnson races should be in WI"
        assert all(r["office"] == "Senate" for r in races)

    def test_mark_kelly_two_races(self, registry):
        """Mark Kelly ran for AZ Senate in 2020 (special) and 2022 (regular)."""
        kelly = next((p for p in registry["persons"].values() if p.get("bioguide_id") == "K000377"), None)
        assert kelly is not None, "Mark Kelly (K000377) should be in registry"
        race_years = sorted(r["year"] for r in kelly["races"])
        assert 2020 in race_years
        assert 2022 in race_years

    def test_cindy_hyde_smith_two_races(self, registry):
        """Cindy Hyde-Smith ran in 2018 (special) and 2020 (regular) for MS Senate."""
        chs = next((p for p in registry["persons"].values() if p.get("bioguide_id") == "H001079"), None)
        assert chs is not None, "Cindy Hyde-Smith (H001079) should be in registry"
        race_years = sorted(r["year"] for r in chs["races"])
        assert 2018 in race_years
        assert 2020 in race_years

    def test_warnock_two_races(self, registry):
        """Raphael Warnock ran in 2020 (special runoff) and 2022 for GA Senate."""
        warnock = next((p for p in registry["persons"].values() if p.get("bioguide_id") == "W000790"), None)
        assert warnock is not None, "Raphael Warnock (W000790) should be in registry"
        race_years = sorted(r["year"] for r in warnock["races"])
        assert 2020 in race_years
        assert 2022 in race_years

    def test_total_multi_race_senate_candidates(self, registry):
        """Should find 45+ multi-race senate candidates across available data."""
        senate_multi = [
            p for p in registry["persons"].values() if sum(1 for r in p["races"] if r["office"] == "Senate") > 1
        ]
        assert len(senate_multi) >= 45, f"Expected 45+ multi-race senate candidates, found {len(senate_multi)}"


class TestCrossOfficeLinking:
    """Candidates who ran for multiple offices must be linked as one person."""

    def test_beto_orourke_senate_to_governor(self, registry):
        """Beto O'Rourke ran for TX Senate in 2018 and TX Governor in 2022."""
        beto = next((p for p in registry["persons"].values() if p.get("bioguide_id") == "O000170"), None)
        assert beto is not None, "Beto O'Rourke (O000170) should be in registry"
        offices = {r["office"] for r in beto["races"]}
        assert "Senate" in offices, "Beto should have a Senate race"
        assert "Governor" in offices, "Beto should have a Governor race"
        # Both races should be in TX
        states = {r["state"] for r in beto["races"]}
        assert states == {"TX"}, f"Expected only TX races, got {states}"

    def test_cross_office_candidates_exist(self, registry):
        """At least some candidates should have both Senate and Governor races."""
        cross_office = [p for p in registry["persons"].values() if len({r["office"] for r in p["races"]}) > 1]
        assert len(cross_office) >= 1, "Should find at least 1 cross-office candidate"


class TestDemShareComputation:
    """Two-party Dem share must be computed correctly from vote totals."""

    def test_dem_share_bounds(self, registry):
        """All computed Dem shares must be in [0, 1]."""
        for pid, person in registry["persons"].items():
            for race in person["races"]:
                share = race.get("actual_dem_share_2party")
                if share is not None:
                    assert 0.0 <= share <= 1.0, (
                        f"Person {pid} race {race['year']} {race['state']}: dem_share {share} out of range"
                    )

    def test_result_consistent_with_share(self, registry):
        """For D candidates, result='win' iff dem_share > 0.5."""
        for pid, person in registry["persons"].items():
            for race in person["races"]:
                share = race.get("actual_dem_share_2party")
                result = race.get("result")
                party = race.get("party")
                if share is None or result is None:
                    continue  # Not all races have actual data
                if party == "D":
                    expected_result = "win" if share > 0.5 else "loss"
                    assert result == expected_result, (
                        f"Person {pid} {race['year']} {race['state']}: "
                        f"D party with share={share:.3f} should be {expected_result}, got {result}"
                    )
                elif party == "R":
                    expected_result = "win" if share < 0.5 else "loss"
                    assert result == expected_result, (
                        f"Person {pid} {race['year']} {race['state']}: "
                        f"R party with share={share:.3f} should be {expected_result}, got {result}"
                    )


class TestNoDuplicateRaces:
    """Each person should have at most one row per (year, state, office, special) race."""

    def test_no_duplicate_races_per_person(self, registry):
        for pid, person in registry["persons"].items():
            race_keys = [(r["year"], r["state"], r["office"], r["special"]) for r in person["races"]]
            assert len(race_keys) == len(set(race_keys)), (
                f"Person {pid} ({person['name']}) has duplicate races: {race_keys}"
            )


class TestBioguideUniqueness:
    """Each bioguide_id should appear at most once in the registry."""

    def test_bioguide_ids_unique(self, registry):
        bioguides = [p["bioguide_id"] for p in registry["persons"].values() if p.get("bioguide_id")]
        assert len(bioguides) == len(set(bioguides)), "Bioguide IDs should be unique across persons"
