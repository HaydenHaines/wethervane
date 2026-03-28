"""End-to-end test: tract type discovery → behavior layer → prediction."""
import numpy as np
import pytest
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_behavior_layer_files_exist():
    """Behavior layer artifacts must exist after training."""
    behavior_dir = DATA_DIR / "behavior"
    assert (behavior_dir / "tau.npy").exists(), "tau.npy not found"
    assert (behavior_dir / "delta.npy").exists(), "delta.npy not found"
    assert (behavior_dir / "summary.json").exists(), "summary.json not found"


def test_tau_reasonable_range():
    """τ values should be in reasonable range (0.1 - 1.5)."""
    tau = np.load(DATA_DIR / "behavior" / "tau.npy")
    assert tau.min() > 0.05, f"τ min too low: {tau.min()}"
    assert tau.max() < 1.6, f"τ max too high: {tau.max()}"
    assert 0.5 < tau.mean() < 1.1, f"τ mean out of range: {tau.mean()}"


def test_delta_reasonable_range():
    """δ values should be mostly small (95th percentile within ±0.10)."""
    delta = np.load(DATA_DIR / "behavior" / "delta.npy")
    p95 = np.percentile(np.abs(delta), 95)
    assert p95 < 0.10, f"95th percentile |δ| too large: {p95}"


def test_tract_assignments_exist():
    """Tract assignments must exist with expected shape."""
    import pandas as pd
    path = DATA_DIR / "tracts" / "national_tract_assignments.parquet"
    assert path.exists(), "national_tract_assignments.parquet not found"
    df = pd.read_parquet(path)
    assert len(df) > 50000, f"Too few tracts: {len(df)}"
    score_cols = [c for c in df.columns if c.startswith("type_") and c.endswith("_score")]
    assert len(score_cols) >= 100, f"Expected 100+ type scores, got {len(score_cols)}"


def test_tract_votes_cover_all_states():
    """Tract votes should cover all 50 states + DC."""
    import pandas as pd
    path = DATA_DIR / "tracts" / "tract_votes_dra.parquet"
    if not path.exists():
        pytest.skip("tract_votes_dra.parquet not found")
    df = pd.read_parquet(path, columns=["state"])
    states = df["state"].unique()
    assert len(states) >= 51, f"Expected 51 states, got {len(states)}: {sorted(states)}"


def test_behavior_adjustment_changes_offcycle():
    """apply_behavior_adjustment should modify priors for off-cycle races."""
    from src.behavior.voter_behavior import apply_behavior_adjustment

    tau = np.load(DATA_DIR / "behavior" / "tau.npy")
    delta = np.load(DATA_DIR / "behavior" / "delta.npy")

    # Synthetic priors and scores
    n_types = len(tau)
    priors = np.full(10, 0.45)
    scores = np.random.default_rng(42).dirichlet(np.ones(n_types), size=10)

    adjusted = apply_behavior_adjustment(priors, scores, tau, delta, is_offcycle=True)
    assert not np.allclose(adjusted, priors), "Behavior adjustment had no effect"
    assert (adjusted >= 0.0).all() and (adjusted <= 1.0).all()
