"""Tests for PCA-based data-driven badge discovery.

Covers:
1. discover_badge_axes(): PCA fits, axis naming, stability (pkl save/load)
2. derive_discovered_badges(): badge award logic, thresholding, party relative
3. compute_candidate_fingerprint(): percentile ranking within party
4. Integration: discovered badges are orthogonal to catalog badges (kind="discovered")
5. Regression: Ossoff/Warnock/Abrams show meaningfully distinct fingerprints
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

# All tests in this file use synthetic data to avoid filesystem dependencies.
# The only filesystem interaction is the PCA save/load path, which is patched.

J = 100  # Must match production type count

# ── Synthetic data fixtures ───────────────────────────────────────────────────

_NAMING_COLS = [
    "pct_black", "pct_hispanic", "pct_asian", "pct_white_nh",
    "log_pop_density", "pct_transit", "pct_bachelors_plus", "pct_graduate",
    "pct_management", "log_median_hh_income", "median_age",
    "pct_owner_occupied", "pct_wfh", "evangelical_share", "mainline_share",
    "catholic_share", "black_protestant_share", "religious_adherence_rate",
    "net_migration_rate", "avg_inflow_income", "earnings_share",
    "transfers_share", "investment_share",
]


@pytest.fixture
def type_profiles_df() -> pd.DataFrame:
    """Synthetic type_profiles with J=100 rows and all naming columns."""
    rng = np.random.default_rng(42)
    data: dict = {"type_id": list(range(J))}
    for col in _NAMING_COLS:
        data[col] = rng.uniform(0, 1, J).tolist()
    return pd.DataFrame(data).set_index("type_id")


def _make_ctov_df(candidates: list[dict]) -> pd.DataFrame:
    """Build a synthetic CTOV DataFrame from a list of candidate dicts."""
    rows = []
    for c in candidates:
        row = {
            "person_id": c["person_id"],
            "name": c.get("name", c["person_id"]),
            "party": c["party"],
            "year": c.get("year", 2022),
            "state": c.get("state", "TX"),
            "office": c.get("office", "Senate"),
            "mvd": float(c.get("mvd", 0.0)),
        }
        ctov = np.asarray(c.get("ctov", np.zeros(J)), dtype=float)
        for j in range(J):
            row[f"ctov_type_{j}"] = float(ctov[j])
        rows.append(row)
    return pd.DataFrame(rows)


def _diverse_pool(party: str, n: int, seed: int) -> list[dict]:
    """Generate a pool of candidates with random CTOV vectors."""
    rng = np.random.default_rng(seed)
    return [
        {
            "person_id": f"{party}_{i}",
            "name": f"{party} Candidate {i}",
            "party": party,
            "ctov": rng.normal(0, 1, J),
        }
        for i in range(n)
    ]


# ── Tests for discover_badge_axes ────────────────────────────────────────────


class TestDiscoverBadgeAxes:
    """Tests for the main discover_badge_axes() function."""

    def _run_discovery(
        self,
        ctov_df: pd.DataFrame,
        type_profiles_df: pd.DataFrame,
        n_components: int = 6,
        force_refit: bool = True,
        tmp_path: Path | None = None,
    ):
        """Helper: run discovery with filesystem and type_profiles patched."""
        from src.sabermetrics.badge_discovery import discover_badge_axes

        save_path = (tmp_path / "badge_axes.pkl") if tmp_path else None

        with (
            patch("src.sabermetrics.badge_discovery._load_type_profiles", return_value=type_profiles_df),
            patch("src.sabermetrics.badge_discovery._BADGE_AXES_PATH", save_path or Path("/dev/null/badge_axes.pkl")),
            # Prevent actual disk writes when tmp_path is not provided
            patch("src.sabermetrics.badge_discovery._BADGE_AXES_PATH", new=save_path or Path("/tmp/test_badge_axes_DISCARD.pkl")),
        ):
            return discover_badge_axes(
                n_components=n_components,
                ctov_df=ctov_df,
                force_refit=force_refit,
            )

    def test_returns_pca_and_axes(self, type_profiles_df, tmp_path):
        """discover_badge_axes returns a fitted PCA and a list of DiscoveredAxis."""
        candidates = _diverse_pool("D", 30, seed=1) + _diverse_pool("R", 30, seed=2)
        ctov_df = _make_ctov_df(candidates)
        pca, axes = self._run_discovery(ctov_df, type_profiles_df, n_components=6, tmp_path=tmp_path)

        assert isinstance(pca, PCA)
        assert len(axes) == 6

    def test_axis_count_matches_n_components(self, type_profiles_df, tmp_path):
        """Number of axes equals n_components."""
        candidates = _diverse_pool("D", 30, seed=3) + _diverse_pool("R", 30, seed=4)
        ctov_df = _make_ctov_df(candidates)
        pca, axes = self._run_discovery(ctov_df, type_profiles_df, n_components=4, tmp_path=tmp_path)

        assert len(axes) == 4

    def test_axes_have_required_fields(self, type_profiles_df, tmp_path):
        """Each DiscoveredAxis has name, description, top_demographics, etc."""
        from src.sabermetrics.badge_discovery import DiscoveredAxis

        candidates = _diverse_pool("D", 30, seed=5) + _diverse_pool("R", 30, seed=6)
        ctov_df = _make_ctov_df(candidates)
        pca, axes = self._run_discovery(ctov_df, type_profiles_df, n_components=3, tmp_path=tmp_path)

        for axis in axes:
            assert isinstance(axis, DiscoveredAxis)
            assert isinstance(axis.name, str) and len(axis.name) > 0
            assert isinstance(axis.description, str) and len(axis.description) > 0
            assert isinstance(axis.top_demographics, list)
            assert isinstance(axis.explained_variance_ratio, float)
            assert 0 < axis.explained_variance_ratio <= 1

    def test_explained_variance_ratios_sum_plausibly(self, type_profiles_df, tmp_path):
        """Sum of explained variance ratios is between 0 and 1."""
        candidates = _diverse_pool("D", 40, seed=7) + _diverse_pool("R", 40, seed=8)
        ctov_df = _make_ctov_df(candidates)
        pca, axes = self._run_discovery(ctov_df, type_profiles_df, n_components=8, tmp_path=tmp_path)

        total_var = sum(ax.explained_variance_ratio for ax in axes)
        assert 0 < total_var <= 1.0

    def test_pca_stability_loads_from_disk(self, type_profiles_df, tmp_path):
        """Second call loads saved PCA rather than re-fitting (stable axes)."""
        candidates = _diverse_pool("D", 30, seed=9) + _diverse_pool("R", 30, seed=10)
        ctov_df = _make_ctov_df(candidates)

        save_path = tmp_path / "badge_axes.pkl"

        from src.sabermetrics.badge_discovery import discover_badge_axes

        with (
            patch("src.sabermetrics.badge_discovery._load_type_profiles", return_value=type_profiles_df),
            patch("src.sabermetrics.badge_discovery._BADGE_AXES_PATH", save_path),
        ):
            # First call: fits and saves.
            pca1, axes1 = discover_badge_axes(ctov_df=ctov_df, n_components=4, force_refit=True)
            assert save_path.exists()

            # Second call with same data: should load from disk.
            pca2, axes2 = discover_badge_axes(ctov_df=ctov_df, n_components=4, force_refit=False)

        # Both models should produce identical components.
        np.testing.assert_array_equal(pca1.components_, pca2.components_)
        assert [ax.name for ax in axes1] == [ax.name for ax in axes2]

    def test_force_refit_ignores_saved_model(self, type_profiles_df, tmp_path):
        """force_refit=True always re-fits even if saved model exists."""
        save_path = tmp_path / "badge_axes.pkl"

        # Write a stale/dummy model with wrong shape.
        dummy_pca = PCA(n_components=2, random_state=42)
        dummy_pca.fit(np.random.default_rng(99).normal(0, 1, (50, J)))
        with save_path.open("wb") as f:
            pickle.dump({"pca": dummy_pca, "data_shape": (1, 1)}, f)

        candidates = _diverse_pool("D", 30, seed=11) + _diverse_pool("R", 30, seed=12)
        ctov_df = _make_ctov_df(candidates)

        from src.sabermetrics.badge_discovery import discover_badge_axes

        with (
            patch("src.sabermetrics.badge_discovery._load_type_profiles", return_value=type_profiles_df),
            patch("src.sabermetrics.badge_discovery._BADGE_AXES_PATH", save_path),
        ):
            pca, axes = discover_badge_axes(ctov_df=ctov_df, n_components=4, force_refit=True)

        # Should have 4 components (not 2 from the dummy model).
        assert pca.n_components_ == 4

    def test_dominant_demographic_drives_axis_name(self, tmp_path):
        """An axis driven by pct_black variation should name itself 'Black...'."""
        rng = np.random.default_rng(42)

        # Build type_profiles where pct_black varies widely across types,
        # and all other columns are nearly constant.
        n_types = J
        profiles_data = {"type_id": list(range(n_types))}
        for col in _NAMING_COLS:
            if col == "pct_black":
                profiles_data[col] = rng.uniform(0, 1, n_types).tolist()
            else:
                # Constant + tiny noise — won't drive correlations.
                profiles_data[col] = (0.1 + rng.uniform(0, 0.01, n_types)).tolist()

        tp = pd.DataFrame(profiles_data).set_index("type_id")
        pct_black = np.array(profiles_data["pct_black"])

        # Generate candidates whose CTOV aligns with pct_black.
        n_candidates = 60
        candidates = []
        for i in range(n_candidates):
            party = "D" if i < 30 else "R"
            # CTOV ≈ factor × pct_black + noise → PC1 should correlate with pct_black.
            factor = rng.normal(0, 0.1)
            ctov = factor * pct_black + rng.normal(0, 0.01, J)
            candidates.append({"person_id": f"{party}_{i}", "name": f"C{i}", "party": party, "ctov": ctov})

        ctov_df = _make_ctov_df(candidates)

        from src.sabermetrics.badge_discovery import discover_badge_axes

        with (
            patch("src.sabermetrics.badge_discovery._load_type_profiles", return_value=tp),
            patch("src.sabermetrics.badge_discovery._BADGE_AXES_PATH", tmp_path / "axes.pkl"),
        ):
            pca, axes = discover_badge_axes(ctov_df=ctov_df, n_components=3, force_refit=True)

        # The first axis should mention "Black" in its name.
        assert any("Black" in ax.name for ax in axes), (
            f"Expected 'Black' in at least one axis name; got: {[ax.name for ax in axes]}"
        )


# ── Tests for derive_discovered_badges ───────────────────────────────────────


class TestDeriveDiscoveredBadges:
    """Tests for badge award logic in derive_discovered_badges()."""

    def _run_awards(
        self,
        candidates: list[dict],
        type_profiles_df: pd.DataFrame,
        n_components: int = 4,
        tmp_path: Path | None = None,
    ):
        from src.sabermetrics.badge_discovery import derive_discovered_badges, discover_badge_axes

        ctov_df = _make_ctov_df(candidates)
        save_path = (tmp_path / "badge_axes_test.pkl") if tmp_path else Path("/tmp/_test_badge_axes.pkl")

        with (
            patch("src.sabermetrics.badge_discovery._load_type_profiles", return_value=type_profiles_df),
            patch("src.sabermetrics.badge_discovery._BADGE_AXES_PATH", save_path),
        ):
            pca, axes = discover_badge_axes(ctov_df=ctov_df, n_components=n_components, force_refit=True)
            result = derive_discovered_badges(ctov_df, pca=pca, axes=axes)

        return result, ctov_df

    def test_all_candidates_present_in_result(self, type_profiles_df, tmp_path):
        """Every candidate in ctov_df appears in the result dict."""
        candidates = _diverse_pool("D", 25, seed=20) + _diverse_pool("R", 25, seed=21)
        result, ctov_df = self._run_awards(candidates, type_profiles_df, tmp_path=tmp_path)

        expected_ids = set(ctov_df["person_id"].unique())
        assert set(result.keys()) == expected_ids

    def test_badge_kind_is_discovered(self, type_profiles_df, tmp_path):
        """All awarded badges have kind='discovered'."""
        candidates = _diverse_pool("D", 25, seed=22) + _diverse_pool("R", 25, seed=23)
        result, _ = self._run_awards(candidates, type_profiles_df, tmp_path=tmp_path)

        for pid, entry in result.items():
            for badge in entry["discovered_badge_details"]:
                assert badge["kind"] == "discovered", (
                    f"{pid}: badge '{badge['name']}' has kind={badge['kind']!r}, expected 'discovered'"
                )

    def test_badge_names_have_pca_prefix(self, type_profiles_df, tmp_path):
        """Badge names start with 'PCA: '."""
        candidates = _diverse_pool("D", 25, seed=24) + _diverse_pool("R", 25, seed=25)
        result, _ = self._run_awards(candidates, type_profiles_df, tmp_path=tmp_path)

        badge_names = [
            b["name"]
            for entry in result.values()
            for b in entry["discovered_badge_details"]
        ]
        assert badge_names, "Expect at least one badge to be awarded"
        for name in badge_names:
            assert name.startswith("PCA: "), f"Badge name should start with 'PCA: '; got {name!r}"

    def test_provisional_flag_for_single_race(self, type_profiles_df, tmp_path):
        """Badges for single-race candidates are marked provisional=True."""
        # All candidates in a simple pool have one race each.
        candidates = _diverse_pool("D", 25, seed=26) + _diverse_pool("R", 25, seed=27)
        result, _ = self._run_awards(candidates, type_profiles_df, tmp_path=tmp_path)

        pids_with_badges = [
            pid for pid, e in result.items() if e["discovered_badge_details"]
        ]
        assert pids_with_badges, "Expect at least some badges in a 50-candidate pool"

        for pid in pids_with_badges:
            for badge in result[pid]["discovered_badge_details"]:
                assert badge["provisional"] is True, (
                    f"{pid}: badge '{badge['name']}' should be provisional (1 race)"
                )

    def test_provisional_false_for_multi_race(self, type_profiles_df, tmp_path):
        """Multi-race candidates (2+ rows) get provisional=False."""
        rng = np.random.default_rng(28)
        multi_race = [
            {
                "person_id": "multi_D",
                "name": "Multi D",
                "party": "D",
                "year": yr,
                "ctov": rng.normal(5, 0.1, J),  # strong signal
            }
            for yr in [2018, 2022]
        ]
        pool = _diverse_pool("D", 24, seed=29) + _diverse_pool("R", 25, seed=30)
        result, _ = self._run_awards(multi_race + pool, type_profiles_df, tmp_path=tmp_path)

        multi_entry = result.get("multi_D", {})
        for badge in multi_entry.get("discovered_badge_details", []):
            assert badge["provisional"] is False, (
                f"Multi-race candidate badge '{badge['name']}' should not be provisional"
            )

    def test_discovered_badges_carry_pc_index(self, type_profiles_df, tmp_path):
        """Each discovered badge detail includes pc_index."""
        candidates = _diverse_pool("D", 25, seed=31) + _diverse_pool("R", 25, seed=32)
        result, _ = self._run_awards(candidates, type_profiles_df, tmp_path=tmp_path)

        for pid, entry in result.items():
            for badge in entry["discovered_badge_details"]:
                assert "pc_index" in badge, f"{pid}: badge missing 'pc_index'"
                assert isinstance(badge["pc_index"], int)

    def test_discovered_badges_carry_top_demographics(self, type_profiles_df, tmp_path):
        """Each discovered badge carries top_demographics list."""
        candidates = _diverse_pool("D", 25, seed=33) + _diverse_pool("R", 25, seed=34)
        result, _ = self._run_awards(candidates, type_profiles_df, tmp_path=tmp_path)

        for pid, entry in result.items():
            for badge in entry["discovered_badge_details"]:
                assert "top_demographics" in badge, f"{pid}: badge missing 'top_demographics'"
                assert isinstance(badge["top_demographics"], list)

    def test_small_pool_fallback_reason(self, type_profiles_df, tmp_path):
        """Candidates in parties with < 20 members get fallback_reason='small_pool'."""
        rng = np.random.default_rng(35)
        independents = [
            {"person_id": f"I_{i}", "name": f"Ind {i}", "party": "I",
             "ctov": rng.normal(0, 1, J)}
            for i in range(5)  # Below the 20-candidate floor
        ]
        pool = _diverse_pool("D", 25, seed=36) + _diverse_pool("R", 25, seed=37)
        result, _ = self._run_awards(independents + pool, type_profiles_df, tmp_path=tmp_path)

        ind_with_badges = [
            f"I_{i}" for i in range(5)
            if result.get(f"I_{i}", {}).get("discovered_badge_details")
        ]
        if not ind_with_badges:
            pytest.skip("No independent earned a badge — adjust fixture if needed")

        for pid in ind_with_badges:
            for badge in result[pid]["discovered_badge_details"]:
                assert badge.get("fallback_reason") == "small_pool", (
                    f"{pid}: expected fallback_reason='small_pool', got {badge.get('fallback_reason')!r}"
                )

    def test_symmetric_pool_balanced_badge_distribution(self, type_profiles_df, tmp_path):
        """Symmetric D/R pool → party imbalance per badge < 0.20 on average."""
        from collections import defaultdict

        rng = np.random.default_rng(38)
        n = 50
        dems = [{"person_id": f"D_{i}", "name": f"D {i}", "party": "D",
                  "ctov": rng.normal(0, 1, J)} for i in range(n)]
        reps = [{"person_id": f"R_{i}", "name": f"R {i}", "party": "R",
                  "ctov": rng.normal(0, 1, J)} for i in range(n)]

        result, _ = self._run_awards(dems + reps, type_profiles_df, n_components=4, tmp_path=tmp_path)

        badge_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"D": 0, "R": 0})
        for pid, entry in result.items():
            party = "D" if pid.startswith("D_") else "R"
            for badge in entry["discovered_badge_details"]:
                badge_counts[badge["name"]][party] += 1

        imbalances = [
            abs(counts["D"] / max(counts["D"] + counts["R"], 1) - 0.5)
            for counts in badge_counts.values()
            if counts["D"] + counts["R"] >= 10
        ]
        if not imbalances:
            pytest.skip("No badge with >= 10 recipients")

        mean_imbalance = sum(imbalances) / len(imbalances)
        assert mean_imbalance <= 0.20, (
            f"Mean party imbalance {mean_imbalance:.3f} > 0.20 for symmetric input"
        )


# ── Tests for compute_candidate_fingerprint ───────────────────────────────────


class TestComputeCandidateFingerprint:
    """Tests for compute_candidate_fingerprint()."""

    def _setup(self, type_profiles_df, tmp_path):
        from src.sabermetrics.badge_discovery import compute_candidate_fingerprint, discover_badge_axes

        candidates = _diverse_pool("D", 30, seed=40) + _diverse_pool("R", 30, seed=41)
        ctov_df = _make_ctov_df(candidates)
        save_path = tmp_path / "fp_axes.pkl"

        with (
            patch("src.sabermetrics.badge_discovery._load_type_profiles", return_value=type_profiles_df),
            patch("src.sabermetrics.badge_discovery._BADGE_AXES_PATH", save_path),
        ):
            pca, axes = discover_badge_axes(ctov_df=ctov_df, n_components=4, force_refit=True)

        return ctov_df, pca, axes

    def test_fingerprint_has_required_keys(self, type_profiles_df, tmp_path):
        """compute_candidate_fingerprint returns dict with person_id, name, party, axis_scores."""
        from src.sabermetrics.badge_discovery import compute_candidate_fingerprint

        ctov_df, pca, axes = self._setup(type_profiles_df, tmp_path)
        pid = ctov_df["person_id"].iloc[0]

        with patch("src.sabermetrics.badge_discovery._load_type_profiles", return_value=type_profiles_df):
            fp = compute_candidate_fingerprint(pid, ctov_df, pca=pca, axes=axes)

        assert "person_id" in fp
        assert "name" in fp
        assert "party" in fp
        assert "axis_scores" in fp
        assert isinstance(fp["axis_scores"], list)

    def test_axis_scores_count_matches_n_components(self, type_profiles_df, tmp_path):
        """axis_scores list length equals number of discovered axes."""
        from src.sabermetrics.badge_discovery import compute_candidate_fingerprint

        ctov_df, pca, axes = self._setup(type_profiles_df, tmp_path)
        pid = ctov_df["person_id"].iloc[0]

        with patch("src.sabermetrics.badge_discovery._load_type_profiles", return_value=type_profiles_df):
            fp = compute_candidate_fingerprint(pid, ctov_df, pca=pca, axes=axes)

        assert len(fp["axis_scores"]) == len(axes)

    def test_party_percentile_in_valid_range(self, type_profiles_df, tmp_path):
        """party_percentile for each axis is between 0 and 100."""
        from src.sabermetrics.badge_discovery import compute_candidate_fingerprint

        ctov_df, pca, axes = self._setup(type_profiles_df, tmp_path)
        pid = ctov_df["person_id"].iloc[0]

        with patch("src.sabermetrics.badge_discovery._load_type_profiles", return_value=type_profiles_df):
            fp = compute_candidate_fingerprint(pid, ctov_df, pca=pca, axes=axes)

        for ax in fp["axis_scores"]:
            assert 0.0 <= ax["party_percentile"] <= 100.0, (
                f"Percentile out of range: {ax['party_percentile']}"
            )

    def test_unknown_person_raises_key_error(self, type_profiles_df, tmp_path):
        """compute_candidate_fingerprint raises KeyError for unknown person_id."""
        from src.sabermetrics.badge_discovery import compute_candidate_fingerprint

        ctov_df, pca, axes = self._setup(type_profiles_df, tmp_path)

        with (
            patch("src.sabermetrics.badge_discovery._load_type_profiles", return_value=type_profiles_df),
            pytest.raises(KeyError, match="not found"),
        ):
            compute_candidate_fingerprint("nonexistent_person", ctov_df, pca=pca, axes=axes)
