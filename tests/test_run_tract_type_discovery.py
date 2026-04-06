"""Tests for tract-level type discovery pipeline.

Uses synthetic data exclusively — no real data files needed.
Real-data behavior is verified by running the main() entry point.

Test coverage:
- prepare_shift_matrix: NaN handling, holdout exclusion, all-NaN tract filtering
- scale_and_weight: presidential weighting correctness post-scaling
- evaluate_holdout: prediction correctness, edge cases
- save_tract_types: output structure, file creation
- End-to-end integration with small synthetic data
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.discovery.run_tract_type_discovery import (
    HOLDOUT_COLUMN,
    PRESIDENTIAL_PREFIX,
    evaluate_holdout,
    prepare_shift_matrix,
    run_discovery,
    save_tract_types,
    scale_and_weight,
)
from src.discovery.run_type_discovery import TypeDiscoveryResult


# ── Helpers ──────────────────────────────────────────────────────────────────


def make_synthetic_parquet(
    n_tracts: int = 50,
    n_pres_train: int = 3,
    n_offcycle: int = 5,
    nan_fraction: float = 0.3,
    all_nan_count: int = 3,
    tmp_dir: str | None = None,
    seed: int = 42,
) -> str:
    """Create a synthetic tract shift parquet with realistic column naming.

    Parameters
    ----------
    n_tracts : int
        Total number of tracts (rows).
    n_pres_train : int
        Number of presidential training shift columns.
    n_offcycle : int
        Number of centered off-cycle shift columns.
    nan_fraction : float
        Fraction of off-cycle values to set NaN.
    all_nan_count : int
        Number of tracts to make all-NaN across presidential training columns.
    tmp_dir : str, optional
        Directory for the temp file. If None, uses system temp.
    seed : int
        Random seed.

    Returns
    -------
    path : str
        Path to the written parquet file.
    """
    rng = np.random.default_rng(seed)

    geoids = [f"{str(i+1).zfill(11)}" for i in range(n_tracts)]

    # Build presidential training columns
    pres_train_cols = [f"pres_shift_{8+4*i:02d}_{8+4*(i+1):02d}" for i in range(n_pres_train)]
    # Build the holdout column
    holdout_col = HOLDOUT_COLUMN
    # Build centered off-cycle columns
    offcycle_cols = [f"gov_shift_{2014+i*2:d}_{2016+i*2:d}_centered" for i in range(n_offcycle)]

    all_cols = pres_train_cols + [holdout_col] + offcycle_cols

    data = {"tract_geoid": geoids}
    for col in all_cols:
        vals = rng.standard_normal(n_tracts)
        data[col] = vals

    df = pd.DataFrame(data)

    # Insert NaN in off-cycle columns
    for col in offcycle_cols:
        nan_idx = rng.choice(n_tracts, size=int(n_tracts * nan_fraction), replace=False)
        df.loc[nan_idx, col] = np.nan

    # Make all_nan_count tracts have all-NaN presidential training columns
    all_nan_idx = rng.choice(n_tracts, size=all_nan_count, replace=False)
    df.loc[all_nan_idx, pres_train_cols] = np.nan

    # Write to tempfile
    if tmp_dir:
        path = str(Path(tmp_dir) / "synthetic_tract_shifts.parquet")
    else:
        import tempfile as _tempfile
        f = _tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
        path = f.name
        f.close()

    df.to_parquet(path, index=False)
    return path


# ── Tests: prepare_shift_matrix ──────────────────────────────────────────────


class TestPrepareShiftMatrix:
    """Tests for the prepare_shift_matrix function."""

    def test_holdout_column_excluded_from_training(self, tmp_path):
        """pres_shift_20_24 (holdout) should not appear in returned column_names."""
        path = make_synthetic_parquet(tmp_dir=str(tmp_path))
        _, _, col_names = prepare_shift_matrix(path)
        assert HOLDOUT_COLUMN not in col_names, (
            f"Holdout column '{HOLDOUT_COLUMN}' should be excluded from training columns"
        )

    def test_presidential_columns_present_in_training(self, tmp_path):
        """All non-holdout presidential columns should be in column_names."""
        path = make_synthetic_parquet(n_pres_train=3, tmp_dir=str(tmp_path))
        _, _, col_names = prepare_shift_matrix(path)
        pres_cols = [c for c in col_names if c.startswith(PRESIDENTIAL_PREFIX)]
        # Should have 3 presidential training columns (holdout excluded)
        assert len(pres_cols) == 3

    def test_offcycle_columns_present_in_training(self, tmp_path):
        """Centered off-cycle columns should appear in column_names."""
        n_offcycle = 5
        path = make_synthetic_parquet(n_offcycle=n_offcycle, tmp_dir=str(tmp_path))
        _, _, col_names = prepare_shift_matrix(path)
        offcycle_cols = [c for c in col_names if c.endswith("_centered")]
        assert len(offcycle_cols) == n_offcycle

    def test_all_nan_tracts_dropped(self, tmp_path):
        """Tracts with all presidential training columns NaN should be dropped."""
        n_tracts = 50
        all_nan_count = 5
        path = make_synthetic_parquet(
            n_tracts=n_tracts, all_nan_count=all_nan_count, tmp_dir=str(tmp_path)
        )
        shift_matrix, tract_geoids, _ = prepare_shift_matrix(path)
        # Expect n_tracts - all_nan_count rows
        assert shift_matrix.shape[0] == n_tracts - all_nan_count
        assert len(tract_geoids) == n_tracts - all_nan_count

    def test_no_nan_in_output_matrix(self, tmp_path):
        """Output shift matrix should have zero NaN after filling."""
        path = make_synthetic_parquet(nan_fraction=0.5, tmp_dir=str(tmp_path))
        shift_matrix, _, _ = prepare_shift_matrix(path)
        assert not np.isnan(shift_matrix).any(), (
            "shift_matrix should have no NaN after 0-fill"
        )

    def test_geoids_aligned_with_matrix_rows(self, tmp_path):
        """tract_geoids length should match shift_matrix row count."""
        path = make_synthetic_parquet(n_tracts=40, all_nan_count=2, tmp_dir=str(tmp_path))
        shift_matrix, tract_geoids, _ = prepare_shift_matrix(path)
        assert shift_matrix.shape[0] == len(tract_geoids)

    def test_output_dtype_float64(self, tmp_path):
        """Output matrix should be float64."""
        path = make_synthetic_parquet(tmp_dir=str(tmp_path))
        shift_matrix, _, _ = prepare_shift_matrix(path)
        assert shift_matrix.dtype == np.float64

    def test_partial_pres_nan_not_dropped(self, tmp_path):
        """Tracts with SOME (but not all) presidential NaN should be kept."""
        n_tracts = 20
        rng = np.random.default_rng(0)
        geoids = [f"{i:011d}" for i in range(n_tracts)]
        pres_cols = ["pres_shift_08_12", "pres_shift_12_16", "pres_shift_16_20"]
        holdout_col = HOLDOUT_COLUMN
        offcycle_cols = ["gov_shift_2014_2016_centered"]

        data = {"tract_geoid": geoids}
        for c in pres_cols + [holdout_col] + offcycle_cols:
            data[c] = rng.standard_normal(n_tracts)

        df = pd.DataFrame(data)
        # Make first 5 tracts have NaN in only ONE presidential column (should be kept)
        df.loc[:4, "pres_shift_08_12"] = np.nan

        path = str(tmp_path / "partial_nan.parquet")
        df.to_parquet(path, index=False)

        shift_matrix, _, _ = prepare_shift_matrix(path)
        # All 20 tracts should be kept (only partial NaN, not all-NaN)
        assert shift_matrix.shape[0] == n_tracts

    def test_column_count_is_pres_plus_offcycle(self, tmp_path):
        """Column count = n_pres_train + n_offcycle (holdout excluded)."""
        n_pres_train, n_offcycle = 3, 7
        path = make_synthetic_parquet(
            n_pres_train=n_pres_train, n_offcycle=n_offcycle, tmp_dir=str(tmp_path)
        )
        shift_matrix, _, col_names = prepare_shift_matrix(path)
        expected_cols = n_pres_train + n_offcycle
        assert len(col_names) == expected_cols
        assert shift_matrix.shape[1] == expected_cols


# ── Tests: scale_and_weight ───────────────────────────────────────────────────


class TestScaleAndWeight:
    """Tests for the scale_and_weight function."""

    def _make_matrix_and_cols(self, n: int = 50, d_pres: int = 3, d_off: int = 5, seed: int = 0):
        rng = np.random.default_rng(seed)
        matrix = rng.standard_normal((n, d_pres + d_off))
        col_names = (
            [f"pres_shift_0{i}_0{i+1}" for i in range(d_pres)]
            + [f"gov_shift_col{i}_centered" for i in range(d_off)]
        )
        return matrix, col_names

    def test_presidential_cols_upweighted(self):
        """Presidential columns should have higher variance post-weighting."""
        weight = 8.0
        matrix, col_names = self._make_matrix_and_cols(d_pres=3, d_off=5)
        scaled = scale_and_weight(matrix, col_names, presidential_weight=weight)

        pres_indices = [i for i, c in enumerate(col_names) if PRESIDENTIAL_PREFIX in c]
        off_indices = [i for i, c in enumerate(col_names) if "_centered" in c]

        pres_std = np.std(scaled[:, pres_indices])
        off_std = np.std(scaled[:, off_indices])

        # Presidential columns should be significantly larger after weight=8
        assert pres_std > off_std * 2, (
            f"Presidential std ({pres_std:.2f}) should be >> off-cycle std ({off_std:.2f}) "
            f"with weight=8.0"
        )

    def test_weight_1_no_change_vs_scaler_only(self):
        """With presidential_weight=1.0, result equals StandardScaler output."""
        from sklearn.preprocessing import StandardScaler
        matrix, col_names = self._make_matrix_and_cols()
        scaled = scale_and_weight(matrix, col_names, presidential_weight=1.0)
        expected = StandardScaler().fit_transform(matrix)
        np.testing.assert_allclose(scaled, expected, atol=1e-10)

    def test_output_shape_preserved(self):
        """Output shape should equal input shape."""
        matrix, col_names = self._make_matrix_and_cols(n=30, d_pres=4, d_off=6)
        scaled = scale_and_weight(matrix, col_names, presidential_weight=8.0)
        assert scaled.shape == matrix.shape

    def test_offcycle_cols_unit_variance(self):
        """Off-cycle columns should have unit variance post-scaling (weight=1 for them)."""
        matrix, col_names = self._make_matrix_and_cols(n=200, d_pres=3, d_off=5)
        scaled = scale_and_weight(matrix, col_names, presidential_weight=8.0)

        off_indices = [i for i, c in enumerate(col_names) if "_centered" in c]
        for idx in off_indices:
            col_std = np.std(scaled[:, idx], ddof=0)
            # Off-cycle should be ~1.0 (StandardScaler output)
            assert abs(col_std - 1.0) < 0.1, (
                f"Off-cycle col {idx} std={col_std:.3f}, expected ~1.0"
            )

    def test_presidential_scaling_factor_correct(self):
        """Presidential columns should be exactly weight× the unit-scaled value."""
        weight = 4.0
        n = 200
        rng = np.random.default_rng(1)
        # One pres column, one off-cycle
        matrix = rng.standard_normal((n, 2))
        col_names = ["pres_shift_08_12", "gov_shift_2016_2018_centered"]

        from sklearn.preprocessing import StandardScaler
        unit_scaled = StandardScaler().fit_transform(matrix)
        weighted = scale_and_weight(matrix, col_names, presidential_weight=weight)

        np.testing.assert_allclose(weighted[:, 0], unit_scaled[:, 0] * weight, atol=1e-10)
        np.testing.assert_allclose(weighted[:, 1], unit_scaled[:, 1], atol=1e-10)


# ── Tests: evaluate_holdout ───────────────────────────────────────────────────


class TestEvaluateHoldout:
    """Tests for the evaluate_holdout function."""

    def _make_holdout_parquet(self, n_tracts: int, geoids: list, holdout_vals: np.ndarray, tmp_path: Path) -> str:
        """Create a minimal parquet with just tract_geoid and holdout column."""
        df = pd.DataFrame({
            "tract_geoid": geoids,
            "pres_shift_08_12": np.zeros(n_tracts),  # dummy pres col
            HOLDOUT_COLUMN: holdout_vals,
        })
        path = str(tmp_path / "holdout_test.parquet")
        df.to_parquet(path, index=False)
        return path

    def test_perfect_prediction_gives_r1(self, tmp_path):
        """If type means perfectly predict holdout, r should be 1.0."""
        n = 50
        j = 3
        rng = np.random.default_rng(7)
        geoids = [f"{i:011d}" for i in range(n)]

        # Hard assignment: each tract belongs to exactly one type
        labels = np.repeat([0, 1, 2], [20, 15, 15])
        scores = np.zeros((n, j))
        scores[np.arange(n), labels] = 1.0

        # Holdout = exact type mean (so prediction = actual)
        type_means = np.array([1.0, -1.0, 0.5])
        holdout_vals = type_means[labels]

        path = self._make_holdout_parquet(n, geoids, holdout_vals, tmp_path)
        r = evaluate_holdout(path, np.array(geoids), scores, j)
        assert abs(r - 1.0) < 1e-6, f"Perfect prediction should give r=1.0, got r={r:.6f}"

    def test_r_in_valid_range(self, tmp_path):
        """Holdout r should always be in [-1, 1]."""
        n = 60
        j = 5
        rng = np.random.default_rng(99)
        geoids = [f"{i:011d}" for i in range(n)]

        scores = rng.dirichlet(np.ones(j), size=n)
        holdout_vals = rng.standard_normal(n)

        path = self._make_holdout_parquet(n, geoids, holdout_vals, tmp_path)
        r = evaluate_holdout(path, np.array(geoids), scores, j)
        assert -1.0 <= r <= 1.0

    def test_returns_float(self, tmp_path):
        """evaluate_holdout should return a Python float."""
        n = 30
        j = 4
        rng = np.random.default_rng(5)
        geoids = [f"{i:011d}" for i in range(n)]

        scores = rng.dirichlet(np.ones(j), size=n)
        holdout_vals = rng.standard_normal(n)

        path = self._make_holdout_parquet(n, geoids, holdout_vals, tmp_path)
        r = evaluate_holdout(path, np.array(geoids), scores, j)
        assert isinstance(r, float)

    def test_missing_holdout_column_raises(self, tmp_path):
        """Should raise ValueError if holdout column is absent from parquet."""
        df = pd.DataFrame({
            "tract_geoid": ["01234567890"],
            "pres_shift_08_12": [0.1],
        })
        path = str(tmp_path / "no_holdout.parquet")
        df.to_parquet(path, index=False)

        scores = np.array([[0.5, 0.5]])
        with pytest.raises(ValueError, match=HOLDOUT_COLUMN):
            evaluate_holdout(path, np.array(["01234567890"]), scores, n_types=2)


# ── Tests: save_tract_types ───────────────────────────────────────────────────


class TestSaveTractTypes:
    """Tests for the save_tract_types function."""

    def _make_result(self, n: int, j: int) -> TypeDiscoveryResult:
        rng = np.random.default_rng(0)
        scores_raw = rng.random((n, j))
        scores = scores_raw / scores_raw.sum(axis=1, keepdims=True)
        dominant = np.argmax(scores, axis=1)
        return TypeDiscoveryResult(
            scores=scores,
            loadings=rng.standard_normal((j, 5)),
            dominant_types=dominant,
            explained_variance=np.ones(j) / j,
            rotation_matrix=np.eye(j),
        )

    def test_parquet_created(self, tmp_path):
        """tract_type_assignments.parquet should be created."""
        n, j = 20, 10
        rng = np.random.default_rng(0)
        geoids = np.array([f"{i:011d}" for i in range(n)])
        result = self._make_result(n, j)

        save_tract_types(geoids, result, j, str(tmp_path))

        assert (tmp_path / "tract_type_assignments.parquet").exists()

    def test_centroids_npy_created(self, tmp_path):
        """tract_type_centroids.npy should be created."""
        n, j = 20, 10
        geoids = np.array([f"{i:011d}" for i in range(n)])
        result = self._make_result(n, j)

        save_tract_types(geoids, result, j, str(tmp_path))

        assert (tmp_path / "tract_type_centroids.npy").exists()

    def test_parquet_has_correct_columns(self, tmp_path):
        """Parquet should have tract_geoid, type_N_score, and dominant_type columns."""
        n, j = 15, 5
        geoids = np.array([f"{i:011d}" for i in range(n)])
        result = self._make_result(n, j)

        save_tract_types(geoids, result, j, str(tmp_path))

        df = pd.read_parquet(tmp_path / "tract_type_assignments.parquet")
        assert "tract_geoid" in df.columns
        assert "dominant_type" in df.columns
        for i in range(j):
            assert f"type_{i}_score" in df.columns

    def test_parquet_row_count(self, tmp_path):
        """Parquet should have exactly N rows (one per tract)."""
        n, j = 25, 8
        geoids = np.array([f"{i:011d}" for i in range(n)])
        result = self._make_result(n, j)

        save_tract_types(geoids, result, j, str(tmp_path))

        df = pd.read_parquet(tmp_path / "tract_type_assignments.parquet")
        assert len(df) == n

    def test_centroids_shape(self, tmp_path):
        """Saved centroids should have shape (J, D_pca)."""
        n, j = 20, 7
        geoids = np.array([f"{i:011d}" for i in range(n)])
        result = self._make_result(n, j)

        save_tract_types(geoids, result, j, str(tmp_path))

        centroids = np.load(tmp_path / "tract_type_centroids.npy")
        assert centroids.shape == result.loadings.shape

    def test_wrong_n_types_raises(self, tmp_path):
        """Mismatched n_types should raise ValueError."""
        n, j = 20, 10
        geoids = np.array([f"{i:011d}" for i in range(n)])
        result = self._make_result(n, j)

        with pytest.raises(ValueError):
            save_tract_types(geoids, result, n_types=j + 1, output_dir=str(tmp_path))


# ── End-to-end integration test ───────────────────────────────────────────────


class TestEndToEnd:
    """End-to-end test with small synthetic data covering the full pipeline."""

    def test_full_pipeline_small_data(self, tmp_path):
        """Full pipeline: prepare → scale → discover → evaluate → save."""
        # Create synthetic data
        n_tracts = 80
        path = make_synthetic_parquet(
            n_tracts=n_tracts,
            n_pres_train=3,
            n_offcycle=5,
            nan_fraction=0.4,
            all_nan_count=3,
            tmp_dir=str(tmp_path),
        )

        # Step 1: prepare
        shift_matrix, tract_geoids, col_names = prepare_shift_matrix(path)
        expected_n = n_tracts - 3  # 3 all-NaN rows dropped
        assert shift_matrix.shape[0] == expected_n

        # Step 2: scale + weight
        weighted = scale_and_weight(shift_matrix, col_names, presidential_weight=8.0)
        assert weighted.shape == shift_matrix.shape

        # Step 3: discover (use small J for speed)
        j = 5
        result = run_discovery(
            weighted, j=j, temperature=10.0, pca_components=4, pca_whiten=True
        )
        assert result.scores.shape == (expected_n, j)
        assert result.dominant_types.shape == (expected_n,)
        assert np.allclose(result.scores.sum(axis=1), 1.0, atol=1e-6)

        # Step 4: evaluate holdout
        r = evaluate_holdout(path, tract_geoids, result.scores, j)
        assert isinstance(r, float)
        assert -1.0 <= r <= 1.0

        # Step 5: save
        save_tract_types(tract_geoids, result, j, str(tmp_path))
        assignments = pd.read_parquet(tmp_path / "tract_type_assignments.parquet")
        assert len(assignments) == expected_n
        assert f"type_{j-1}_score" in assignments.columns
        assert "dominant_type" in assignments.columns

    def test_scores_are_valid_probability_vectors(self, tmp_path):
        """Every tract's type scores must sum to 1 and be non-negative."""
        path = make_synthetic_parquet(n_tracts=60, tmp_dir=str(tmp_path))
        shift_matrix, _, col_names = prepare_shift_matrix(path)
        weighted = scale_and_weight(shift_matrix, col_names, presidential_weight=8.0)
        result = run_discovery(weighted, j=5, pca_components=3, pca_whiten=True)

        assert np.all(result.scores >= 0)
        assert np.all(result.scores <= 1 + 1e-6)
        row_sums = result.scores.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)
