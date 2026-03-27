# National Environment Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hand-designed Bayesian poll update with a learned hierarchical decomposition: θ_national (citizen sentiment from all polls) + δ_race (candidate effects from race-specific polls), with a user-facing toggle on forecast pages.

**Architecture:** Two-stage regularized regression. Stage 1: estimate θ_national from all polls pooled across all races, regularized toward θ_prior. Stage 2: estimate per-race δ_race from residuals, regularized toward zero. λ and μ are hyperparameters (initially set to reasonable defaults, later learned from backtesting in Plan C).

**Tech Stack:** Python, numpy, scipy (Ridge regression), FastAPI, React/Next.js

**Spec:** `docs/superpowers/specs/2026-03-27-poll-calibration-national-forecast-design.md`

**Depends on:** Plan A (national expansion — race registry and all-race predictions must exist)

**Branch:** `feat/national-environment-model`

---

## File Structure

```
Create: src/prediction/national_environment.py     — θ_national estimation
Create: src/prediction/candidate_effects.py         — δ_race estimation
Create: src/prediction/forecast_engine.py           — Orchestrator: prior → θ_national → δ → predictions
Create: tests/prediction/test_national_environment.py
Create: tests/prediction/test_candidate_effects.py
Create: tests/prediction/test_forecast_engine.py
Modify: src/prediction/predict_2026_types.py        — Call forecast_engine instead of predict_race
Modify: api/routers/forecast.py                     — Add mode=national|local query param
Modify: web/app/forecast/[slug]/page.tsx            — Toggle UI component
```

---

### Task 1: θ_prior Computation

**Files:**
- Create: `src/prediction/forecast_engine.py`
- Create: `tests/prediction/test_forecast_engine.py`

Convert county-level priors to type-level priors. This is the "what the model thinks before polls" baseline.

- [ ] **Step 1: Write failing test**

```python
# tests/prediction/test_forecast_engine.py
import numpy as np
import pytest

from src.prediction.forecast_engine import compute_theta_prior


def test_theta_prior_shape():
    """θ_prior should have J elements."""
    J = 5
    n_counties = 10
    type_scores = np.random.rand(n_counties, J)
    type_scores = type_scores / type_scores.sum(axis=1, keepdims=True)
    county_priors = np.random.rand(n_counties) * 0.3 + 0.35  # ~[0.35, 0.65]
    theta = compute_theta_prior(type_scores, county_priors)
    assert theta.shape == (J,)


def test_theta_prior_weighted_average():
    """θ_prior[j] = weighted mean of county priors by type membership."""
    type_scores = np.array([
        [1.0, 0.0],  # county 0 is 100% type 0
        [0.0, 1.0],  # county 1 is 100% type 1
    ])
    county_priors = np.array([0.6, 0.4])
    theta = compute_theta_prior(type_scores, county_priors)
    np.testing.assert_allclose(theta, [0.6, 0.4])


def test_theta_prior_mixed_membership():
    """Mixed membership produces blended priors."""
    type_scores = np.array([
        [0.5, 0.5],
        [0.5, 0.5],
    ])
    county_priors = np.array([0.6, 0.4])
    theta = compute_theta_prior(type_scores, county_priors)
    np.testing.assert_allclose(theta, [0.5, 0.5])


def test_theta_prior_bounded():
    """θ_prior should be in [0, 1] (valid Dem share range)."""
    J = 20
    n_counties = 100
    type_scores = np.random.rand(n_counties, J)
    type_scores = type_scores / type_scores.sum(axis=1, keepdims=True)
    county_priors = np.random.rand(n_counties) * 0.6 + 0.2
    theta = compute_theta_prior(type_scores, county_priors)
    assert np.all(theta >= 0) and np.all(theta <= 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/prediction/test_forecast_engine.py::test_theta_prior_shape -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement compute_theta_prior**

```python
# src/prediction/forecast_engine.py
"""Forecast engine: θ_prior → θ_national → δ_race → county predictions.

This module orchestrates the hierarchical poll decomposition model.
Voters move slowly (θ_prior from decade of elections); polls move quickly
(θ_national captures current sentiment; δ_race captures candidate effects).
"""

import numpy as np


def compute_theta_prior(
    type_scores: np.ndarray,  # (n_counties, J) — soft membership
    county_priors: np.ndarray,  # (n_counties,) — baseline Dem share
) -> np.ndarray:
    """Convert county-level priors to type-level θ_prior.

    θ_prior[j] = Σ_c W[c,j] · prior[c] / Σ_c W[c,j]
    Weighted average of county priors by type membership.
    """
    # Avoid division by zero for types with no member counties
    W = np.abs(type_scores)  # Ensure non-negative weights
    type_totals = W.sum(axis=0)  # (J,)
    type_totals = np.where(type_totals > 0, type_totals, 1.0)
    theta = (W.T @ county_priors) / type_totals  # (J,)
    return theta
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/prediction/test_forecast_engine.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/prediction/forecast_engine.py tests/prediction/test_forecast_engine.py
git commit -m "feat: compute_theta_prior — type-level priors from county baselines"
```

---

### Task 2: θ_national Estimation

**Files:**
- Create: `src/prediction/national_environment.py`
- Create: `tests/prediction/test_national_environment.py`

Estimate θ_national from all polls across all races via regularized regression.

- [ ] **Step 1: Write failing tests**

```python
# tests/prediction/test_national_environment.py
import numpy as np
import pytest

from src.prediction.national_environment import estimate_theta_national


def test_no_polls_returns_prior():
    """With zero polls, θ_national should equal θ_prior."""
    J = 5
    theta_prior = np.array([0.5, 0.4, 0.6, 0.45, 0.55])
    theta = estimate_theta_national(
        W_polls=np.empty((0, J)),
        y_polls=np.empty(0),
        sigma_polls=np.empty(0),
        theta_prior=theta_prior,
        lam=1.0,
    )
    np.testing.assert_allclose(theta, theta_prior)


def test_single_poll_moves_toward_observation():
    """One poll should pull θ_national toward the observed value."""
    J = 2
    theta_prior = np.array([0.5, 0.5])
    W = np.array([[0.7, 0.3]])  # Poll mostly observes type 0
    y = np.array([0.6])  # Poll says 60% Dem
    sigma = np.array([0.02])
    theta = estimate_theta_national(W, y, sigma, theta_prior, lam=1.0)
    # Type 0 should move toward 0.6 more than type 1
    assert theta[0] > theta_prior[0]


def test_many_polls_dominate_prior():
    """With many precise polls, θ_national should be data-driven."""
    J = 2
    theta_prior = np.array([0.5, 0.5])
    n_polls = 100
    W = np.tile([0.6, 0.4], (n_polls, 1))
    y = np.full(n_polls, 0.55)
    sigma = np.full(n_polls, 0.01)  # Very precise
    theta = estimate_theta_national(W, y, sigma, theta_prior, lam=0.1)
    # W · θ should be close to 0.55
    pred = W[0] @ theta
    assert abs(pred - 0.55) < 0.02


def test_lambda_controls_prior_weight():
    """Higher λ should keep θ_national closer to θ_prior."""
    J = 3
    theta_prior = np.array([0.5, 0.5, 0.5])
    W = np.array([[1.0, 0.0, 0.0]])
    y = np.array([0.7])
    sigma = np.array([0.02])

    theta_low_lam = estimate_theta_national(W, y, sigma, theta_prior, lam=0.01)
    theta_high_lam = estimate_theta_national(W, y, sigma, theta_prior, lam=100.0)

    # Low λ: θ moves more toward data
    assert abs(theta_low_lam[0] - 0.7) < abs(theta_high_lam[0] - 0.7)
    # High λ: θ stays closer to prior
    assert abs(theta_high_lam[0] - 0.5) < abs(theta_low_lam[0] - 0.5)


def test_output_shape():
    J = 10
    theta_prior = np.random.rand(J)
    W = np.random.rand(5, J)
    W = W / W.sum(axis=1, keepdims=True)
    y = np.random.rand(5) * 0.3 + 0.35
    sigma = np.full(5, 0.03)
    theta = estimate_theta_national(W, y, sigma, theta_prior, lam=1.0)
    assert theta.shape == (J,)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/prediction/test_national_environment.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement estimate_theta_national**

```python
# src/prediction/national_environment.py
"""Estimate θ_national — citizen sentiment per type from all polls.

Solves: minimize Σ_p (1/σ_p²)(y_p - W_p·θ)² + λ·||θ - θ_prior||²
This is weighted Ridge regression with prior centering.
Closed-form: θ = (WᵀΣ⁻¹W + λI)⁻¹(WᵀΣ⁻¹y + λ·θ_prior)
"""

import numpy as np


def estimate_theta_national(
    W_polls: np.ndarray,      # (n_polls, J) — type composition per poll
    y_polls: np.ndarray,      # (n_polls,) — observed Dem shares
    sigma_polls: np.ndarray,  # (n_polls,) — poll noise (std dev)
    theta_prior: np.ndarray,  # (J,) — model prior
    lam: float = 1.0,         # regularization strength (prior trust)
) -> np.ndarray:
    """Estimate national citizen sentiment θ_national from pooled polls.

    With zero polls, returns θ_prior exactly.
    With many precise polls, converges to data-driven estimate.
    λ controls the trade-off.
    """
    J = len(theta_prior)

    if len(y_polls) == 0:
        return theta_prior.copy()

    # Inverse-variance weights: (n_polls,)
    inv_var = 1.0 / (sigma_polls ** 2)

    # Weighted normal equations
    # A = WᵀΣ⁻¹W + λI
    # b = WᵀΣ⁻¹y + λ·θ_prior
    W_weighted = W_polls * inv_var[:, np.newaxis]  # (n_polls, J) weighted
    A = W_weighted.T @ W_polls + lam * np.eye(J)
    b = W_weighted.T @ y_polls + lam * theta_prior

    theta = np.linalg.solve(A, b)
    return theta
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/prediction/test_national_environment.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/prediction/national_environment.py tests/prediction/test_national_environment.py
git commit -m "feat: estimate_theta_national — citizen sentiment from pooled polls"
```

---

### Task 3: δ_race Estimation

**Files:**
- Create: `src/prediction/candidate_effects.py`
- Create: `tests/prediction/test_candidate_effects.py`

Estimate per-race candidate effects as residuals from θ_national.

- [ ] **Step 1: Write failing tests**

```python
# tests/prediction/test_candidate_effects.py
import numpy as np
import pytest

from src.prediction.candidate_effects import estimate_delta_race


def test_no_polls_returns_zeros():
    """With zero polls, δ_race should be all zeros."""
    J = 5
    delta = estimate_delta_race(
        W_polls=np.empty((0, J)),
        residuals=np.empty(0),
        sigma_polls=np.empty(0),
        J=J,
        mu=1.0,
    )
    np.testing.assert_allclose(delta, np.zeros(J))


def test_single_poll_small_delta():
    """One poll with strong regularization should produce small δ."""
    J = 3
    W = np.array([[0.5, 0.3, 0.2]])
    residuals = np.array([0.05])  # Race is 5pp above national
    sigma = np.array([0.03])
    delta = estimate_delta_race(W, residuals, sigma, J, mu=10.0)
    assert np.all(np.abs(delta) < 0.05)  # Strong regularization keeps δ small


def test_many_polls_larger_delta():
    """Many polls with weak regularization should produce δ close to residual."""
    J = 2
    n = 50
    W = np.tile([0.6, 0.4], (n, 1))
    residuals = np.full(n, 0.03)  # Consistent 3pp above national
    sigma = np.full(n, 0.02)
    delta = estimate_delta_race(W, residuals, sigma, J, mu=0.01)
    pred_residual = W[0] @ delta
    assert abs(pred_residual - 0.03) < 0.01


def test_mu_controls_shrinkage():
    """Higher μ should shrink δ toward zero."""
    J = 3
    W = np.array([[0.5, 0.3, 0.2]] * 5)
    residuals = np.full(5, 0.05)
    sigma = np.full(5, 0.02)

    delta_low_mu = estimate_delta_race(W, residuals, sigma, J, mu=0.01)
    delta_high_mu = estimate_delta_race(W, residuals, sigma, J, mu=100.0)

    assert np.linalg.norm(delta_high_mu) < np.linalg.norm(delta_low_mu)


def test_output_shape():
    J = 10
    W = np.random.rand(3, J)
    W = W / W.sum(axis=1, keepdims=True)
    residuals = np.random.randn(3) * 0.02
    sigma = np.full(3, 0.03)
    delta = estimate_delta_race(W, residuals, sigma, J, mu=1.0)
    assert delta.shape == (J,)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/prediction/test_candidate_effects.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement estimate_delta_race**

```python
# src/prediction/candidate_effects.py
"""Estimate δ_race — candidate effects per type for a specific race.

Solves: minimize Σ_p (1/σ_p²)(r_p - W_p·δ)² + μ·||δ||²
where r_p = y_p - W_p·θ_national (residual from national sentiment).

δ represents how this specific candidate matchup deviates from the
generic national environment for each type.
"""

import numpy as np


def estimate_delta_race(
    W_polls: np.ndarray,      # (n_polls, J) — type composition per poll
    residuals: np.ndarray,    # (n_polls,) — y_p - W_p · θ_national
    sigma_polls: np.ndarray,  # (n_polls,) — poll noise
    J: int,                   # number of types
    mu: float = 1.0,          # regularization toward zero
) -> np.ndarray:
    """Estimate candidate effect δ for a single race.

    With zero polls, returns zeros (no candidate effect detected).
    With many polls, captures the race-specific deviation from national sentiment.
    μ controls shrinkage toward zero.
    """
    if len(residuals) == 0:
        return np.zeros(J)

    inv_var = 1.0 / (sigma_polls ** 2)

    # Weighted normal equations (same structure as θ_national but centered at 0)
    W_weighted = W_polls * inv_var[:, np.newaxis]
    A = W_weighted.T @ W_polls + mu * np.eye(J)
    b = W_weighted.T @ residuals  # No prior term — centered at zero

    delta = np.linalg.solve(A, b)
    return delta
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/prediction/test_candidate_effects.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/prediction/candidate_effects.py tests/prediction/test_candidate_effects.py
git commit -m "feat: estimate_delta_race — per-race candidate effects from residuals"
```

---

### Task 4: Forecast Engine Orchestration

**Files:**
- Modify: `src/prediction/forecast_engine.py`
- Modify: `tests/prediction/test_forecast_engine.py`

Wire θ_prior → θ_national → δ_race → county predictions into a single engine.

- [ ] **Step 1: Write failing integration tests**

```python
# Add to tests/prediction/test_forecast_engine.py

from src.prediction.forecast_engine import (
    compute_theta_prior,
    build_W_state,
    run_forecast,
    ForecastResult,
)


def test_build_W_state():
    """W for a state should be vote-weighted mean of county type memberships."""
    J = 3
    type_scores = np.array([
        [0.8, 0.1, 0.1],  # county 0 (state A)
        [0.2, 0.7, 0.1],  # county 1 (state A)
        [0.1, 0.1, 0.8],  # county 2 (state B)
    ])
    states = ["A", "A", "B"]
    votes = np.array([1000, 500, 800])
    W = build_W_state("A", type_scores, states, votes)
    # Vote-weighted: (1000*[0.8,0.1,0.1] + 500*[0.2,0.7,0.1]) / 1500
    expected = (1000 * np.array([0.8, 0.1, 0.1]) + 500 * np.array([0.2, 0.7, 0.1])) / 1500
    np.testing.assert_allclose(W, expected, atol=1e-6)


def test_run_forecast_no_polls():
    """With no polls, national and local modes should return prior-based predictions."""
    J = 3
    n_counties = 4
    type_scores = np.eye(J + 1, J)[:n_counties]  # Simple membership
    county_priors = np.array([0.6, 0.4, 0.5, 0.45])
    states = ["A", "A", "B", "B"]
    votes = np.array([100, 200, 150, 250])
    polls = {}  # No polls for any race
    races = ["2026 A Senate", "2026 B Governor"]

    result = run_forecast(
        type_scores=type_scores,
        county_priors=county_priors,
        states=states,
        county_votes=votes,
        polls_by_race=polls,
        races=races,
        lam=1.0,
        mu=1.0,
    )
    assert "2026 A Senate" in result
    assert result["2026 A Senate"].theta_national is not None
    assert result["2026 A Senate"].delta_race is not None
    np.testing.assert_allclose(result["2026 A Senate"].delta_race, np.zeros(J))


def test_run_forecast_with_polls():
    """With polls, local mode predictions should differ from national mode."""
    J = 2
    type_scores = np.array([
        [0.8, 0.2],
        [0.3, 0.7],
    ])
    county_priors = np.array([0.55, 0.45])
    states = ["A", "A"]
    votes = np.array([100, 100])
    polls = {
        "2026 A Senate": [
            {"dem_share": 0.60, "n_sample": 800, "state": "A"},
        ],
    }
    races = ["2026 A Senate"]

    result = run_forecast(
        type_scores=type_scores,
        county_priors=county_priors,
        states=states,
        county_votes=votes,
        polls_by_race=polls,
        races=races,
        lam=1.0,
        mu=1.0,
    )
    r = result["2026 A Senate"]
    # National and local should differ (δ ≠ 0)
    national_preds = type_scores @ r.theta_national
    local_preds = type_scores @ (r.theta_national + r.delta_race)
    assert not np.allclose(national_preds, local_preds)


class TestForecastResult:
    def test_has_both_modes(self):
        """ForecastResult should expose national and local county predictions."""
        J = 2
        result = ForecastResult(
            theta_prior=np.array([0.5, 0.5]),
            theta_national=np.array([0.52, 0.48]),
            delta_race=np.array([0.01, -0.01]),
            county_preds_national=np.array([0.50, 0.49]),
            county_preds_local=np.array([0.51, 0.48]),
            n_polls=3,
        )
        assert result.n_polls == 3
        assert len(result.county_preds_national) == 2
        assert len(result.county_preds_local) == 2
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/prediction/test_forecast_engine.py -v`
Expected: FAIL (new functions not implemented)

- [ ] **Step 3: Implement the orchestrator**

```python
# Add to src/prediction/forecast_engine.py

from dataclasses import dataclass

from src.prediction.national_environment import estimate_theta_national
from src.prediction.candidate_effects import estimate_delta_race


@dataclass
class ForecastResult:
    """Result for a single race."""
    theta_prior: np.ndarray          # (J,)
    theta_national: np.ndarray       # (J,)
    delta_race: np.ndarray           # (J,)
    county_preds_national: np.ndarray  # (n_counties,) — θ_national mode
    county_preds_local: np.ndarray     # (n_counties,) — θ_national + δ mode
    n_polls: int


def build_W_state(
    state: str,
    type_scores: np.ndarray,  # (n_counties, J)
    states: list[str],
    county_votes: np.ndarray,  # (n_counties,)
) -> np.ndarray:
    """Build W vector for a state: vote-weighted mean of county type memberships."""
    mask = np.array([s == state for s in states])
    if not mask.any():
        J = type_scores.shape[1]
        return np.ones(J) / J  # Uniform fallback

    state_scores = np.abs(type_scores[mask])
    state_votes = county_votes[mask]

    if state_votes.sum() > 0:
        weights = state_votes / state_votes.sum()
        W = (state_scores * weights[:, np.newaxis]).sum(axis=0)
    else:
        W = state_scores.mean(axis=0)

    W_sum = W.sum()
    return W / W_sum if W_sum > 0 else np.ones(type_scores.shape[1]) / type_scores.shape[1]


def _build_poll_arrays(
    polls_by_race: dict[str, list[dict]],
    type_scores: np.ndarray,
    states: list[str],
    county_votes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build W, y, sigma arrays from all polls across all races.

    Returns: (W_all, y_all, sigma_all, race_labels)
    """
    W_rows, y_vals, sigma_vals, race_labels = [], [], [], []

    for race_id, polls in polls_by_race.items():
        for p in polls:
            state = p["state"]
            dem_share = p["dem_share"]
            n_sample = p["n_sample"]

            W_row = build_W_state(state, type_scores, states, county_votes)
            sigma = np.sqrt(dem_share * (1 - dem_share) / max(n_sample, 1))

            W_rows.append(W_row)
            y_vals.append(dem_share)
            sigma_vals.append(max(sigma, 1e-6))  # Floor to avoid div-by-zero
            race_labels.append(race_id)

    J = type_scores.shape[1]
    if not W_rows:
        return np.empty((0, J)), np.empty(0), np.empty(0), []

    return (
        np.array(W_rows),
        np.array(y_vals),
        np.array(sigma_vals),
        race_labels,
    )


def run_forecast(
    type_scores: np.ndarray,      # (n_counties, J)
    county_priors: np.ndarray,    # (n_counties,)
    states: list[str],            # (n_counties,) state per county
    county_votes: np.ndarray,     # (n_counties,) votes per county
    polls_by_race: dict[str, list[dict]],  # race_id -> list of poll dicts
    races: list[str],             # all race IDs to forecast
    lam: float = 1.0,            # θ_national regularization
    mu: float = 1.0,             # δ_race regularization
    generic_ballot_shift: float = 0.0,
) -> dict[str, ForecastResult]:
    """Run the full hierarchical forecast for all races.

    1. Compute θ_prior from county priors
    2. Estimate θ_national from all polls pooled
    3. For each race, estimate δ_race from residuals
    4. Produce county predictions in both modes
    """
    J = type_scores.shape[1]

    # Apply generic ballot shift to county priors
    adjusted_priors = county_priors + generic_ballot_shift

    # Step 1: θ_prior
    theta_prior = compute_theta_prior(type_scores, adjusted_priors)

    # Step 2: Build poll arrays and estimate θ_national
    W_all, y_all, sigma_all, race_labels = _build_poll_arrays(
        polls_by_race, type_scores, states, county_votes,
    )
    theta_national = estimate_theta_national(W_all, y_all, sigma_all, theta_prior, lam)

    # Step 3 & 4: Per-race δ and predictions
    results = {}
    for race_id in races:
        # Get this race's polls
        race_polls = polls_by_race.get(race_id, [])
        n_polls = len(race_polls)

        if n_polls > 0:
            # Build race-specific W and residuals
            race_W, race_y, race_sigma, _ = _build_poll_arrays(
                {race_id: race_polls}, type_scores, states, county_votes,
            )
            residuals = race_y - race_W @ theta_national
            delta = estimate_delta_race(race_W, residuals, race_sigma, J, mu)
        else:
            delta = np.zeros(J)

        # County predictions
        county_preds_national = type_scores @ theta_national
        county_preds_local = type_scores @ (theta_national + delta)

        results[race_id] = ForecastResult(
            theta_prior=theta_prior,
            theta_national=theta_national,
            delta_race=delta,
            county_preds_national=county_preds_national,
            county_preds_local=county_preds_local,
            n_polls=n_polls,
        )

    return results
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/prediction/test_forecast_engine.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/prediction/forecast_engine.py tests/prediction/test_forecast_engine.py
git commit -m "feat: forecast engine orchestrates θ_prior → θ_national → δ_race → county preds"
```

---

### Task 5: Wire Forecast Engine into Prediction Pipeline

**Files:**
- Modify: `src/prediction/predict_2026_types.py`

Replace the per-race `predict_race()` loop with the forecast engine.

- [ ] **Step 1: Read the current run() function**

Read `src/prediction/predict_2026_types.py` focusing on the `run()` function (~line 363-534). Understand how it loads data, builds poll aggregation, and calls `predict_race()`.

- [ ] **Step 2: Add forecast engine as an alternative path**

Keep the existing `predict_race()` function intact for now (backward compatibility). Add a new flag to `run()`:

```python
# At the top of run(), add parameter:
def run(use_new_engine: bool = True, ...):

# After loading type_scores, county_priors, polls, etc., add:
if use_new_engine:
    from src.prediction.forecast_engine import run_forecast
    from src.assembly.define_races import load_races, race_ids

    # Build polls_by_race dict from poll_agg DataFrame
    polls_by_race = {}
    if poll_agg is not None and len(poll_agg) > 0:
        for race, group in poll_agg.groupby("race"):
            if race.startswith("2026 Generic Ballot"):
                continue
            polls_by_race[race] = [
                {
                    "dem_share": float(row["dem_share"]),
                    "n_sample": int(row["n_sample"]),
                    "state": str(row["state"]),
                }
                for _, row in group.iterrows()
                if row.get("geo_level", "state") == "state"
            ]

    all_race_ids = race_ids(2026)

    results = run_forecast(
        type_scores=type_scores,
        county_priors=county_priors,
        states=states,
        county_votes=county_votes,
        polls_by_race=polls_by_race,
        races=all_race_ids,
        lam=1.0,   # TODO: learned from calibration (Plan C)
        mu=1.0,    # TODO: learned from calibration (Plan C)
        generic_ballot_shift=gb_info.shift,
    )

    # Convert ForecastResult → DataFrame rows (both modes)
    for race_id, fr in results.items():
        for mode in ("national", "local"):
            preds = fr.county_preds_national if mode == "national" else fr.county_preds_local
            row = pd.DataFrame({
                "county_fips": county_fips,
                "state": states,
                "county_name": county_names,
                "pred_dem_share": preds,
                "race": race_id,
                "forecast_mode": mode,
            })
            all_predictions.append(row)

    # Also add baseline (no polls, no GB shift — pure model prior)
    theta_baseline = compute_theta_prior(type_scores, county_priors)
    baseline_preds = type_scores @ theta_baseline
    baseline_row = pd.DataFrame({
        "county_fips": county_fips,
        "state": states,
        "county_name": county_names,
        "pred_dem_share": baseline_preds,
        "race": "baseline",
        "forecast_mode": "national",
    })
    all_predictions.append(baseline_row)
```

- [ ] **Step 3: Update DuckDB schema for forecast_mode column**

Add `forecast_mode` column to predictions table in `build_database.py`:

```python
# In the CREATE TABLE statement for predictions, add:
forecast_mode  VARCHAR NOT NULL DEFAULT 'local',
# Update PRIMARY KEY to include mode:
PRIMARY KEY (county_fips, race, version_id, forecast_mode)
```

- [ ] **Step 4: Run full pipeline and test**

```bash
uv run python -m src.prediction.predict_2026_types
uv run python src/db/build_database.py --reset
uv run pytest tests/ -q --tb=short
```

- [ ] **Step 5: Commit**

```bash
git add src/prediction/predict_2026_types.py src/db/build_database.py
git commit -m "feat: wire forecast engine into prediction pipeline with dual-mode output"
```

---

### Task 6: API Support for Forecast Mode Toggle

**Files:**
- Modify: `api/routers/forecast.py`

Add `mode` query parameter to forecast endpoints.

- [ ] **Step 1: Add mode parameter to /forecast endpoint**

```python
@router.get("/forecast")
async def get_forecast(
    race: str | None = None,
    state: str | None = None,
    mode: str = "local",  # "national" or "local"
):
    # Add forecast_mode filter to query:
    # WHERE forecast_mode = ?
```

- [ ] **Step 2: Add mode to /forecast/race/{slug} endpoint**

```python
@router.get("/forecast/race/{slug}")
async def get_race_detail(slug: str, mode: str = "local"):
    # Pass mode through to prediction query
    # Return both modes' state_pred in response for comparison
```

- [ ] **Step 3: Extend RaceDetail response model**

```python
class RaceDetail(BaseModel):
    # ... existing fields ...
    forecast_mode: str                    # "national" or "local"
    state_pred_national: float | None     # θ_national mode prediction
    state_pred_local: float | None        # θ_national + δ mode prediction
    candidate_effect_margin: float | None # Difference (local - national)
    n_polls: int
```

- [ ] **Step 4: Run API tests**

Run: `uv run pytest tests/ -k "forecast" -v`

- [ ] **Step 5: Commit**

```bash
git add api/routers/forecast.py
git commit -m "feat: API supports mode=national|local query param for forecast toggle"
```

---

### Task 7: Frontend Forecast Toggle

**Files:**
- Modify: `web/app/forecast/[slug]/page.tsx`

Add the segmented control toggle between National Environment and Local Polling.

- [ ] **Step 1: Create ForecastToggle component**

```typescript
// web/components/ForecastToggle.tsx
"use client";

import { useState } from "react";

interface ForecastToggleProps {
  hasPolls: boolean;
  nPolls: number;
  statePredNational: number | null;
  statePredLocal: number | null;
  onModeChange: (mode: "national" | "local") => void;
}

export default function ForecastToggle({
  hasPolls,
  nPolls,
  statePredNational,
  statePredLocal,
  onModeChange,
}: ForecastToggleProps) {
  const [mode, setMode] = useState<"national" | "local">(
    hasPolls ? "local" : "national"
  );

  const handleChange = (newMode: "national" | "local") => {
    setMode(newMode);
    onModeChange(newMode);
  };

  const margin = statePredLocal && statePredNational
    ? statePredLocal - statePredNational
    : null;

  return (
    <div className="forecast-toggle" role="radiogroup" aria-label="Forecast mode">
      <button
        role="radio"
        aria-checked={mode === "national"}
        className={`toggle-btn ${mode === "national" ? "active" : ""}`}
        onClick={() => handleChange("national")}
      >
        National Environment
      </button>
      <button
        role="radio"
        aria-checked={mode === "local"}
        className={`toggle-btn ${mode === "local" ? "active" : ""} ${!hasPolls ? "disabled" : ""}`}
        onClick={() => hasPolls && handleChange("local")}
        disabled={!hasPolls}
        title={hasPolls ? undefined : "No polls available for this race"}
      >
        Local Polling
      </button>
      <p className="toggle-description">
        {mode === "national"
          ? "Based on national political environment — no race-specific polling applied."
          : `Based on national environment + ${nPolls} poll${nPolls !== 1 ? "s" : ""} for this race.`}
      </p>
      {margin !== null && hasPolls && (
        <p className="toggle-delta">
          Local polling shifts this race {margin > 0 ? "D" : "R"}+
          {Math.abs(margin * 100).toFixed(1)} from national environment.
        </p>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Wire toggle into race detail page**

The race detail page fetches both modes' data and re-renders predictions on toggle.

- [ ] **Step 3: Add CSS for toggle component**

```css
/* Add to globals.css */
.forecast-toggle {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin: 1rem 0;
  align-items: center;
}

.toggle-btn {
  padding: 0.5rem 1rem;
  border: 1px solid var(--color-border);
  background: var(--color-bg);
  color: var(--color-text);
  cursor: pointer;
  border-radius: 4px;
  font-size: 0.875rem;
}

.toggle-btn.active {
  background: var(--color-primary);
  color: white;
  border-color: var(--color-primary);
}

.toggle-btn.disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.toggle-description {
  width: 100%;
  font-size: 0.8rem;
  color: var(--color-text-muted);
  margin: 0.25rem 0 0;
}

.toggle-delta {
  width: 100%;
  font-size: 0.8rem;
  font-weight: 600;
  margin: 0.25rem 0 0;
}
```

- [ ] **Step 4: Build and test**

```bash
cd /home/hayden/projects/wethervane/web
npm run build
```

- [ ] **Step 5: Commit**

```bash
git add web/components/ForecastToggle.tsx web/app/forecast/\[slug\]/page.tsx web/app/globals.css
git commit -m "feat: frontend forecast toggle — national environment vs local polling"
```

---

### Task 8: End-to-End Verification

**Files:** None new — execution task.

- [ ] **Step 1: Run full pipeline**

```bash
cd /home/hayden/projects/wethervane
uv run python -m src.prediction.predict_2026_types
uv run python src/db/build_database.py --reset
```

- [ ] **Step 2: Run full test suite**

```bash
uv run pytest tests/ -q --tb=short
```

- [ ] **Step 3: Verify API responses**

```bash
# National mode
curl -s "http://localhost:8002/api/v1/forecast/race/2026-fl-senate?mode=national" | python3 -m json.tool | head -20
# Local mode (default)
curl -s "http://localhost:8002/api/v1/forecast/race/2026-fl-senate?mode=local" | python3 -m json.tool | head -20
# Unpolled race (should only have national)
curl -s "http://localhost:8002/api/v1/forecast/race/2026-wy-senate?mode=national" | python3 -m json.tool | head -20
```

- [ ] **Step 4: Rebuild and deploy frontend**

```bash
cd web && npm run build
cp -r public/ .next/standalone/public/
cp -r .next/static/ .next/standalone/.next/static/
sudo systemctl restart wethervane-api.service wethervane-frontend.service
```

- [ ] **Step 5: Spot-check in browser**

Visit:
- `https://wethervane.hhaines.duckdns.org/forecast/2026-fl-senate` — toggle should work
- `https://wethervane.hhaines.duckdns.org/forecast/2026-wy-senate` — "Local Polling" grayed out
- `https://wethervane.hhaines.duckdns.org/forecast` — all ~83 races should appear

- [ ] **Step 6: Commit and push**

```bash
cd /home/hayden/projects/wethervane
git add -A
git commit -m "feat: national environment model — complete with frontend toggle"
git push
```

---

## Validation Checklist

- [ ] θ_prior computed correctly (type-level weighted mean of county priors)
- [ ] θ_national with 0 polls returns θ_prior exactly
- [ ] θ_national with many polls converges to data
- [ ] δ_race with 0 polls returns zeros
- [ ] δ_race with polls captures candidate effect
- [ ] λ and μ affect regularization as expected
- [ ] Both forecast modes stored in DuckDB
- [ ] API serves both modes via query param
- [ ] Frontend toggle works for polled races
- [ ] Frontend grays out "Local Polling" for unpolled races
- [ ] Candidate effect delta annotation shown
- [ ] Full test suite passes
