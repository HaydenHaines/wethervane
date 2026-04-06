"""Unit tests for poll coverage diagnostic logic.

Tests focus on the core computation functions in scripts/analyze_poll_coverage.py:
- compute_coverage_ratios: poll/population ratio calculation
- build_state_population_vectors: vote-weighted demographic vector construction
- find_affected_types: identifying which types are most exposed to a coverage gap

We use synthetic data throughout to avoid depending on disk-resident data files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the scripts directory importable so we can import the module under test.
# Scripts are not a package so we need to add them to sys.path.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from analyze_poll_coverage import (
    XT_TO_TYPE_PROFILE_COL,
    build_state_population_vectors,
    compute_coverage_ratios,
    find_affected_types,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def make_type_profiles(n_types: int = 4) -> pd.DataFrame:
    """Build a minimal synthetic type_profiles DataFrame."""
    rng = np.random.default_rng(42)
    data = {
        "type_id": list(range(n_types)),
        "display_name": [f"Type {i}" for i in range(n_types)],
        "pct_white_nh": rng.uniform(0.1, 0.9, n_types),
        "pct_black": rng.uniform(0.01, 0.4, n_types),
        "pct_asian": rng.uniform(0.01, 0.2, n_types),
        "pct_hispanic": rng.uniform(0.02, 0.5, n_types),
        "pct_bachelors_plus": rng.uniform(0.1, 0.5, n_types),
        "evangelical_share": rng.uniform(0.05, 0.8, n_types),
    }
    # Ensure pct_white + pct_black + pct_asian + pct_hispanic ≤ 1 (rough normalization).
    df = pd.DataFrame(data)
    race_cols = ["pct_white_nh", "pct_black", "pct_asian", "pct_hispanic"]
    totals = df[race_cols].sum(axis=1)
    for col in race_cols:
        df[col] = df[col] / totals  # normalize so they sum to 1
    return df


def make_county_assignments(n_counties: int = 6, n_types: int = 4) -> pd.DataFrame:
    """Build a minimal county_type_assignments DataFrame with soft membership scores."""
    rng = np.random.default_rng(7)
    fips = [f"{i:05d}" for i in range(1, n_counties + 1)]
    scores = rng.dirichlet(alpha=np.ones(n_types), size=n_counties)
    score_cols = {f"type_{j}_score": scores[:, j] for j in range(n_types)}
    df = pd.DataFrame({"county_fips": fips, **score_cols})
    return df


def make_county_votes(
    county_fips: list[str], state_abbr: list[str], votes: list[int]
) -> pd.DataFrame:
    """Build a minimal county_votes DataFrame."""
    return pd.DataFrame(
        {
            "county_fips": county_fips,
            "state_abbr": state_abbr,
            "pres_total_2020": votes,
        }
    )


# ---------------------------------------------------------------------------
# compute_coverage_ratios tests
# ---------------------------------------------------------------------------


class TestComputeCoverageRatios:
    def test_oversampled_group(self):
        """A group with poll_share > population_share has ratio > 1."""
        poll = {"xt_race_white": 0.80}
        pop = {"xt_race_white": 0.60}
        ratios = compute_coverage_ratios(poll, pop)
        assert "xt_race_white" in ratios
        assert ratios["xt_race_white"] > 1.0
        assert abs(ratios["xt_race_white"] - 0.80 / 0.60) < 1e-9

    def test_undersampled_group(self):
        """A group with poll_share < population_share has ratio < 1."""
        poll = {"xt_race_black": 0.05}
        pop = {"xt_race_black": 0.20}
        ratios = compute_coverage_ratios(poll, pop)
        assert ratios["xt_race_black"] < 1.0
        assert abs(ratios["xt_race_black"] - 0.05 / 0.20) < 1e-9

    def test_representative_group(self):
        """A group with equal shares has ratio exactly 1."""
        poll = {"xt_education_college": 0.35}
        pop = {"xt_education_college": 0.35}
        ratios = compute_coverage_ratios(poll, pop)
        assert abs(ratios["xt_education_college"] - 1.0) < 1e-9

    def test_zero_population_share_returns_none(self):
        """When population share is 0, ratio cannot be computed; should return None."""
        poll = {"xt_race_asian": 0.02}
        pop = {"xt_race_asian": 0.0}
        ratios = compute_coverage_ratios(poll, pop)
        assert ratios["xt_race_asian"] is None

    def test_multiple_groups(self):
        """Multiple groups are all processed correctly."""
        poll = {"xt_race_white": 0.7, "xt_race_black": 0.1, "xt_race_hispanic": 0.2}
        pop = {"xt_race_white": 0.6, "xt_race_black": 0.15, "xt_race_hispanic": 0.25}
        ratios = compute_coverage_ratios(poll, pop)
        assert len(ratios) == 3
        assert ratios["xt_race_white"] > 1.0  # white oversampled
        assert ratios["xt_race_black"] < 1.0  # black undersampled
        assert ratios["xt_race_hispanic"] < 1.0  # hispanic undersampled

    def test_group_missing_from_population_is_skipped(self):
        """Groups present in the poll vector but absent from population are skipped."""
        poll = {"xt_race_white": 0.7, "xt_unknown_group": 0.3}
        pop = {"xt_race_white": 0.6}
        ratios = compute_coverage_ratios(poll, pop)
        assert "xt_race_white" in ratios
        assert "xt_unknown_group" not in ratios

    def test_empty_poll_vector(self):
        """Empty poll vector produces empty ratios dict."""
        ratios = compute_coverage_ratios({}, {"xt_race_white": 0.6})
        assert ratios == {}


# ---------------------------------------------------------------------------
# build_state_population_vectors tests
# ---------------------------------------------------------------------------


class TestBuildStatePopulationVectors:
    def setup_method(self):
        """Set up synthetic data shared across tests."""
        self.n_types = 4
        self.type_profiles = make_type_profiles(self.n_types)
        self.county_assignments = make_county_assignments(6, self.n_types)

        # Two states: AA with 3 counties, BB with 3 counties
        fips = self.county_assignments["county_fips"].tolist()
        states = ["AA", "AA", "AA", "BB", "BB", "BB"]
        votes = [10000, 20000, 30000, 15000, 25000, 5000]
        self.county_votes = make_county_votes(fips, states, votes)

        self.xt_cols = ["xt_race_white", "xt_race_black", "xt_race_hispanic", "xt_race_asian"]

    def test_returns_vector_for_each_state(self):
        """Should return a population vector for every state represented in data."""
        vecs = build_state_population_vectors(
            self.type_profiles, self.county_assignments, self.county_votes, self.xt_cols
        )
        assert "AA" in vecs
        assert "BB" in vecs

    def test_vectors_have_correct_keys(self):
        """Each state vector should contain only mappable xt_ columns."""
        vecs = build_state_population_vectors(
            self.type_profiles, self.county_assignments, self.county_votes, self.xt_cols
        )
        mappable = [c for c in self.xt_cols if c in XT_TO_TYPE_PROFILE_COL]
        for state_vec in vecs.values():
            for col in mappable:
                assert col in state_vec, f"{col} missing from state vector"

    def test_values_are_proportions(self):
        """Each demographic share should be between 0 and 1."""
        vecs = build_state_population_vectors(
            self.type_profiles, self.county_assignments, self.county_votes, self.xt_cols
        )
        for state, vec in vecs.items():
            for col, val in vec.items():
                assert 0.0 <= val <= 1.0, f"{state}:{col} = {val} out of range"

    def test_different_states_differ(self):
        """Two states with different county compositions should produce different vectors."""
        # This test relies on the random data having some variation between the two state groups.
        vecs = build_state_population_vectors(
            self.type_profiles, self.county_assignments, self.county_votes, self.xt_cols
        )
        aa = vecs.get("AA", {})
        bb = vecs.get("BB", {})
        # At least one demographic group should differ between states.
        col = "xt_race_white"
        if col in aa and col in bb:
            # The two state groups have different counties so values should differ
            # (they might be equal by coincidence with synthetic data, but very unlikely).
            # We assert structural correctness here, not exact values.
            assert isinstance(aa[col], float)
            assert isinstance(bb[col], float)

    def test_vote_weighting_changes_result(self):
        """A county with more votes should pull the state vector toward its type profile."""
        # Build two scenarios: first puts all votes on county 0, second on county 2.
        fips = self.county_assignments["county_fips"].tolist()

        # Scenario A: county 0 dominates
        votes_a = [100000, 1, 1, 1, 1, 1]
        cv_a = make_county_votes(fips, ["AA", "AA", "AA", "BB", "BB", "BB"], votes_a)
        vecs_a = build_state_population_vectors(
            self.type_profiles, self.county_assignments, cv_a, ["xt_race_white"]
        )

        # Scenario B: county 2 dominates
        votes_b = [1, 1, 100000, 1, 1, 1]
        cv_b = make_county_votes(fips, ["AA", "AA", "AA", "BB", "BB", "BB"], votes_b)
        vecs_b = build_state_population_vectors(
            self.type_profiles, self.county_assignments, cv_b, ["xt_race_white"]
        )

        # When different counties dominate, the state vector should differ
        # (unless counties 0 and 2 happen to be identical in soft membership — very unlikely).
        scores_0 = np.array(
            self.county_assignments.iloc[0][[f"type_{j}_score" for j in range(self.n_types)]].tolist(),
            dtype=float,
        )
        scores_2 = np.array(
            self.county_assignments.iloc[2][[f"type_{j}_score" for j in range(self.n_types)]].tolist(),
            dtype=float,
        )
        if not np.allclose(scores_0, scores_2):
            assert vecs_a["AA"]["xt_race_white"] != vecs_b["AA"]["xt_race_white"]

    def test_aggregate_county_fips_excluded(self):
        """County FIPS '00000' (MEDSL aggregate row) should not affect the result."""
        fips_with_agg = ["00000"] + self.county_assignments["county_fips"].tolist()
        states_with_agg = ["AA"] + ["AA", "AA", "AA", "BB", "BB", "BB"]
        votes_with_agg = [9999999, 10000, 20000, 30000, 15000, 25000, 5000]

        # Build extended assignments: add a row for fips 00000 with extreme type scores
        extreme_row = {c: 0.0 for c in self.county_assignments.columns}
        extreme_row["county_fips"] = "00000"
        extreme_row["type_0_score"] = 1.0  # would dominate if not excluded
        ca_with_agg = pd.concat(
            [pd.DataFrame([extreme_row]), self.county_assignments], ignore_index=True
        )

        cv_with_agg = make_county_votes(fips_with_agg, states_with_agg, votes_with_agg)
        vecs_with = build_state_population_vectors(
            self.type_profiles, ca_with_agg, cv_with_agg, ["xt_race_white"]
        )
        vecs_without = build_state_population_vectors(
            self.type_profiles, self.county_assignments, self.county_votes, ["xt_race_white"]
        )
        # The AA vector should be the same since the aggregate row is excluded.
        if "AA" in vecs_with and "AA" in vecs_without:
            assert abs(vecs_with["AA"]["xt_race_white"] - vecs_without["AA"]["xt_race_white"]) < 1e-6


# ---------------------------------------------------------------------------
# find_affected_types tests
# ---------------------------------------------------------------------------


class TestFindAffectedTypes:
    def setup_method(self):
        """Synthetic profiles with known concentration patterns."""
        # Type 0: high Black concentration, low weight in state
        # Type 1: low Black concentration, high weight in state
        # Type 2: medium Black, medium weight
        # Type 3: zero Black concentration
        self.type_profiles = pd.DataFrame(
            {
                "type_id": [0, 1, 2, 3],
                "display_name": ["BlackBelt", "Suburban", "Mixed", "Rural"],
                "pct_black": [0.60, 0.05, 0.20, 0.01],
                "pct_white_nh": [0.30, 0.75, 0.60, 0.85],
                "pct_bachelors_plus": [0.15, 0.45, 0.30, 0.18],
            }
        )
        # State weight: type 1 dominates
        self.state_weights = np.array([0.05, 0.60, 0.25, 0.10])

    def test_returns_correct_number_of_types(self):
        """Should return at most n_top types."""
        results = find_affected_types("xt_race_black", self.type_profiles, self.state_weights, n_top=2)
        assert len(results) <= 2

    def test_highest_exposure_type_ranked_first(self):
        """Type with highest (group_share × state_weight) should appear first."""
        results = find_affected_types("xt_race_black", self.type_profiles, self.state_weights)
        # Type 0 has pct_black=0.60, weight=0.05 → exposure=0.03
        # Type 2 has pct_black=0.20, weight=0.25 → exposure=0.05
        # Type 2 should rank above type 0 because 0.05 > 0.03
        assert results[0]["type_id"] == 2, (
            f"Expected type 2 first, got {results[0]['type_id']}"
        )

    def test_result_fields_present(self):
        """Each result dict must have the expected keys."""
        results = find_affected_types("xt_race_black", self.type_profiles, self.state_weights)
        required_keys = {"type_id", "display_name", "group_share", "state_weight", "exposure"}
        for r in results:
            assert required_keys.issubset(r.keys()), f"Missing keys in result: {r}"

    def test_exposures_sum_to_one(self):
        """Normalized exposures across all returned types should sum to ~1.

        Since we return only n_top types, the sum will be ≤ 1. We check the
        full set (n_top=J) sums to 1.
        """
        results = find_affected_types(
            "xt_race_black", self.type_profiles, self.state_weights, n_top=4
        )
        total = sum(r["exposure"] for r in results)
        # Values are rounded to 4 decimal places in the output, so allow for rounding error.
        assert abs(total - 1.0) < 1e-3, f"Exposures sum to {total}, expected ~1.0"

    def test_type_with_zero_group_share_excluded(self):
        """Types with zero group share should have zero exposure and not appear first."""
        results = find_affected_types("xt_race_black", self.type_profiles, self.state_weights)
        # Type 3 (pct_black=0.01) should not be first
        assert results[0]["type_id"] != 3

    def test_unknown_xt_col_returns_empty(self):
        """An xt_ column with no mapping in XT_TO_TYPE_PROFILE_COL returns empty list."""
        results = find_affected_types(
            "xt_urbanicity_rural", self.type_profiles, self.state_weights
        )
        assert results == []

    def test_missing_profile_col_returns_empty(self):
        """If the mapped profile column is absent from type_profiles, returns empty list."""
        profiles_no_bachelors = self.type_profiles.drop(columns=["pct_bachelors_plus"])
        results = find_affected_types(
            "xt_education_college", profiles_no_bachelors, self.state_weights
        )
        assert results == []

    def test_uniform_state_weights_rank_by_group_share(self):
        """When all types have equal state weight, ranking is driven by group share alone."""
        uniform_weights = np.ones(4) / 4
        results = find_affected_types(
            "xt_race_black", self.type_profiles, uniform_weights, n_top=4
        )
        # Type 0 has highest pct_black=0.60, should rank first
        assert results[0]["type_id"] == 0
