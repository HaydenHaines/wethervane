# Rich Poll Ingestion Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire poll quality weighting into the forecast engine and build tiered W vector construction with LV/RV type-screen adjustments and non-weighted dimension inference.

**Architecture:** Two layers that compose independently. Layer 1 wires the existing `poll_weighting.py` pipeline into `forecast_engine.py` so polls get quality-adjusted σ values. Layer 2 replaces generic state-level W vectors with poll-specific W vectors that account for LV/RV screening and non-weighted demographic dimensions (religion, urbanicity, income). Both layers feed into the existing Bayesian update math unchanged.

**Tech Stack:** Python, NumPy, pandas. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-29-rich-poll-ingestion-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `src/prediction/poll_enrichment.py` | Tiered W vector dispatch + Tier 3 adjustments (LV/RV screen, non-weighted dimensions). ~150 lines. |
| `src/prediction/propensity_model.py` | Config-driven propensity scoring per type for LV/RV screen modeling. ~60 lines. |
| `data/config/poll_method_adjustments.json` | All tunable parameters: propensity coefficients, LV/RV factors, method reach profiles, dimension sets. |
| `tests/prediction/test_poll_enrichment.py` | Unit tests for all tiers of W vector construction. |
| `tests/prediction/test_propensity_model.py` | Unit tests for propensity scoring. |

### Modified Files

| File | Change |
|------|--------|
| `src/prediction/forecast_engine.py` | Add `prepare_polls()` function; modify `_build_poll_arrays()` to accept a `w_builder` callable; add `w_vector_mode` param to `run_forecast()`. |
| `tests/prediction/test_forecast_engine.py` | Add tests for `prepare_polls()` and W vector integration. |

---

### Task 1: Config File + Propensity Model

**Files:**
- Create: `data/config/poll_method_adjustments.json`
- Create: `src/prediction/propensity_model.py`
- Create: `tests/prediction/test_propensity_model.py`

- [ ] **Step 1: Create the config file**

```bash
mkdir -p data/config
```

Write `data/config/poll_method_adjustments.json`:

```json
{
  "lv_propensity_coefficients": {
    "median_age": 0.3,
    "pct_owner_occupied": 0.4,
    "pct_bachelors_plus": 0.3
  },
  "lv_downweight_factor": 0.5,
  "rv_downweight_factor": 0.8,
  "method_reach_profiles": {
    "online_panel": {"log_pop_density_shift": 0.05},
    "phone_ivr": {},
    "phone_live": {},
    "unknown": {}
  },
  "w_vector_dimensions": {
    "core": ["evangelical_share", "catholic_share", "mainline_share"],
    "full": [
      "evangelical_share", "catholic_share", "mainline_share",
      "log_pop_density", "median_hh_income", "pct_owner_occupied"
    ]
  }
}
```

- [ ] **Step 2: Write failing tests for propensity model**

Write `tests/prediction/test_propensity_model.py`:

```python
"""Tests for LV/RV propensity scoring."""

import numpy as np
import pandas as pd
import pytest

from src.prediction.propensity_model import compute_propensity_scores, load_config


def _make_type_profiles(n_types: int = 5) -> pd.DataFrame:
    """Create synthetic type profiles with known propensity-correlated fields."""
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "type_id": range(n_types),
        "median_age": rng.uniform(25, 65, n_types),
        "pct_owner_occupied": rng.uniform(0.3, 0.8, n_types),
        "pct_bachelors_plus": rng.uniform(0.1, 0.6, n_types),
        "evangelical_share": rng.uniform(0.05, 0.5, n_types),
    })


class TestLoadConfig:
    def test_loads_from_default_path(self):
        cfg = load_config()
        assert "lv_propensity_coefficients" in cfg
        assert "lv_downweight_factor" in cfg
        assert "w_vector_dimensions" in cfg

    def test_coefficients_sum_to_one(self):
        cfg = load_config()
        coeffs = cfg["lv_propensity_coefficients"]
        total = sum(v for k, v in coeffs.items() if not k.startswith("_"))
        assert abs(total - 1.0) < 1e-6


class TestComputePropensity:
    def test_output_shape(self):
        tp = _make_type_profiles(10)
        scores = compute_propensity_scores(tp)
        assert scores.shape == (10,)

    def test_scores_in_unit_interval(self):
        tp = _make_type_profiles(20)
        scores = compute_propensity_scores(tp)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_older_homeowner_educated_scores_higher(self):
        """Types with high age, homeownership, and education should have higher propensity."""
        tp = pd.DataFrame({
            "type_id": [0, 1],
            "median_age": [60.0, 25.0],
            "pct_owner_occupied": [0.8, 0.3],
            "pct_bachelors_plus": [0.5, 0.1],
        })
        scores = compute_propensity_scores(tp)
        assert scores[0] > scores[1]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/prediction/test_propensity_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.prediction.propensity_model'`

- [ ] **Step 4: Implement propensity model**

Write `src/prediction/propensity_model.py`:

```python
"""LV/RV type-screen propensity scoring.

Computes a propensity score per electoral type based on demographic proxies
for voter turnout. Used by poll_enrichment.py to model which types an LV
screen systematically includes/excludes.

The model is a config-driven linear combination — not a trained model.
Coefficients are from political science literature on voter turnout correlates.
All tunable parameters live in data/config/poll_method_adjustments.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "data" / "config" / "poll_method_adjustments.json"


def load_config(path: Path | str | None = None) -> dict:
    """Load poll method adjustment configuration."""
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    with p.open() as f:
        return json.load(f)


def compute_propensity_scores(
    type_profiles: pd.DataFrame,
    config: dict | None = None,
) -> np.ndarray:
    """Compute turnout propensity score per type.

    Returns array of shape (n_types,) with values in [0, 1].
    Higher = more likely to pass an LV screen.
    """
    if config is None:
        config = load_config()

    coefficients = config["lv_propensity_coefficients"]
    fields = [k for k in coefficients if not k.startswith("_")]
    weights = np.array([coefficients[k] for k in fields])

    # Extract and normalize each field to [0, 1]
    values = np.zeros((len(type_profiles), len(fields)))
    for i, field in enumerate(fields):
        col = type_profiles[field].values.astype(float)
        col_min, col_max = col.min(), col.max()
        if col_max > col_min:
            values[:, i] = (col - col_min) / (col_max - col_min)
        else:
            values[:, i] = 0.5

    # Weighted linear combination → [0, 1]
    raw_scores = values @ weights
    s_min, s_max = raw_scores.min(), raw_scores.max()
    if s_max > s_min:
        return (raw_scores - s_min) / (s_max - s_min)
    return np.full(len(type_profiles), 0.5)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/prediction/test_propensity_model.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add data/config/poll_method_adjustments.json src/prediction/propensity_model.py tests/prediction/test_propensity_model.py
git commit -m "feat: propensity model + config for poll method adjustments

Config-driven linear combination of type demographics (median_age,
pct_owner_occupied, pct_bachelors_plus) produces a [0,1] propensity
score per type. Used downstream by poll enrichment to model LV/RV
type-screen effects."
```

---

### Task 2: Poll Enrichment Module (Tiered W Vectors)

**Files:**
- Create: `src/prediction/poll_enrichment.py`
- Create: `tests/prediction/test_poll_enrichment.py`

- [ ] **Step 1: Write failing tests**

Write `tests/prediction/test_poll_enrichment.py`:

```python
"""Tests for tiered W vector construction."""

import numpy as np
import pandas as pd
import pytest

from src.prediction.poll_enrichment import (
    build_W_poll,
    build_W_with_adjustments,
    build_W_from_crosstabs,
    build_W_from_raw_sample,
    parse_methodology,
)


def _make_type_profiles(j: int = 5) -> pd.DataFrame:
    """Synthetic type profiles with varying demographics."""
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "type_id": range(j),
        "median_age": np.linspace(25, 65, j),
        "pct_owner_occupied": np.linspace(0.3, 0.8, j),
        "pct_bachelors_plus": np.linspace(0.1, 0.6, j),
        "evangelical_share": np.linspace(0.05, 0.5, j),
        "catholic_share": np.linspace(0.4, 0.1, j),
        "mainline_share": np.linspace(0.1, 0.3, j),
        "log_pop_density": np.linspace(-0.5, 0.5, j),
        "median_hh_income": np.linspace(30000, 100000, j),
    })


def _make_state_type_weights(j: int = 5) -> np.ndarray:
    """State-level type weights (vote-weighted presence in state)."""
    w = np.array([0.3, 0.2, 0.2, 0.2, 0.1])[:j]
    return w / w.sum()


class TestTierDispatch:
    def test_tier1_when_raw_data_present(self):
        tp = _make_type_profiles()
        stw = _make_state_type_weights()
        raw = {"pct_black": 0.33, "evangelical_share": 0.25}
        W = build_W_poll(
            poll={"state": "GA", "dem_share": 0.53, "n_sample": 600},
            type_profiles=tp,
            state_type_weights=stw,
            raw_sample_demographics=raw,
        )
        assert W.shape == (5,)
        assert abs(W.sum() - 1.0) < 1e-6

    def test_tier2_when_crosstabs_present(self):
        tp = _make_type_profiles()
        stw = _make_state_type_weights()
        xt = [
            {"demographic_group": "race", "group_value": "black",
             "pct_of_sample": 0.33, "dem_share": 0.90},
            {"demographic_group": "race", "group_value": "white",
             "pct_of_sample": 0.55, "dem_share": 0.40},
        ]
        result = build_W_poll(
            poll={"state": "GA", "dem_share": 0.53, "n_sample": 600},
            type_profiles=tp,
            state_type_weights=stw,
            poll_crosstabs=xt,
        )
        # Tier 2 returns multiple observation rows
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(r["W"].shape == (5,) for r in result)

    def test_tier3_when_topline_only(self):
        tp = _make_type_profiles()
        stw = _make_state_type_weights()
        W = build_W_poll(
            poll={"state": "GA", "dem_share": 0.53, "n_sample": 600},
            type_profiles=tp,
            state_type_weights=stw,
        )
        assert W.shape == (5,)
        assert abs(W.sum() - 1.0) < 1e-6


class TestTier3Adjustments:
    def test_lv_downweights_low_propensity(self):
        """LV poll should give less weight to low-propensity types than no-screen poll."""
        tp = _make_type_profiles()
        stw = _make_state_type_weights()

        W_lv = build_W_with_adjustments(
            poll={"state": "GA", "dem_share": 0.53, "n_sample": 600,
                  "methodology": "LV"},
            type_profiles=tp,
            state_type_weights=stw,
        )
        W_none = build_W_with_adjustments(
            poll={"state": "GA", "dem_share": 0.53, "n_sample": 600,
                  "methodology": ""},
            type_profiles=tp,
            state_type_weights=stw,
        )
        # No-methodology should return state weights unchanged
        np.testing.assert_allclose(W_none, stw, atol=1e-6)

        # LV should differ from state weights
        assert not np.allclose(W_lv, stw, atol=1e-3)

        # Type 0 has lowest propensity (youngest, lowest homeownership, lowest education)
        # LV should downweight it relative to state weights
        assert W_lv[0] < stw[0]

    def test_core_vs_full_mode(self):
        """Full mode uses more dimensions than core mode."""
        tp = _make_type_profiles()
        stw = _make_state_type_weights()
        poll = {"state": "GA", "dem_share": 0.53, "n_sample": 600,
                "methodology": "LV"}

        W_core = build_W_with_adjustments(poll, tp, stw, w_vector_mode="core")
        W_full = build_W_with_adjustments(poll, tp, stw, w_vector_mode="full")

        # Both should be valid W vectors
        assert abs(W_core.sum() - 1.0) < 1e-6
        assert abs(W_full.sum() - 1.0) < 1e-6

        # They should differ (full uses more dimensions for adjustment)
        assert not np.allclose(W_core, W_full, atol=1e-6)

    def test_no_methodology_returns_state_weights(self):
        """No methodology info → no adjustment → return state_type_weights."""
        tp = _make_type_profiles()
        stw = _make_state_type_weights()
        W = build_W_with_adjustments(
            poll={"state": "GA", "dem_share": 0.53, "n_sample": 600,
                  "methodology": ""},
            type_profiles=tp,
            state_type_weights=stw,
        )
        np.testing.assert_allclose(W, stw, atol=1e-6)


class TestParseMethodology:
    def test_lv_from_notes(self):
        assert parse_methodology("D=34.0% R=53.0%; LV; src=wikipedia") == "LV"

    def test_rv_from_notes(self):
        assert parse_methodology("D=34.0% R=53.0%; RV; src=wikipedia") == "RV"

    def test_no_method(self):
        assert parse_methodology("D=34.0% R=53.0%; src=wikipedia") == ""

    def test_empty_notes(self):
        assert parse_methodology("") == ""

    def test_none_notes(self):
        assert parse_methodology(None) == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/prediction/test_poll_enrichment.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.prediction.poll_enrichment'`

- [ ] **Step 3: Implement poll enrichment**

Write `src/prediction/poll_enrichment.py`:

```python
"""Tiered W vector construction for poll-specific type composition.

Three tiers, each with a defined off-ramp:
  Tier 1: Raw unweighted sample demographics → direct type mapping
  Tier 2: Weighted topline + crosstabs → per-group type mapping
  Tier 3: Weighted topline only → LV/RV screen + non-weighted dimensions

See docs/superpowers/specs/2026-03-29-rich-poll-ingestion-design.md
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.prediction.propensity_model import compute_propensity_scores, load_config


# ---------------------------------------------------------------------------
# Methodology parsing
# ---------------------------------------------------------------------------

_LV_PATTERN = re.compile(r"\bLV\b", re.IGNORECASE)
_RV_PATTERN = re.compile(r"\bRV\b", re.IGNORECASE)


def parse_methodology(notes: str | None) -> str:
    """Extract LV/RV methodology from poll notes string.

    Returns "LV", "RV", or "" (unknown).
    """
    if not notes:
        return ""
    if _LV_PATTERN.search(notes):
        return "LV"
    if _RV_PATTERN.search(notes):
        return "RV"
    return ""


# ---------------------------------------------------------------------------
# Tier 3: Adjusted state-level W
# ---------------------------------------------------------------------------

def build_W_with_adjustments(
    poll: dict,
    type_profiles: pd.DataFrame,
    state_type_weights: np.ndarray,
    w_vector_mode: str = "core",
    config: dict | None = None,
) -> np.ndarray:
    """Tier 3: Adjust state-level W for LV/RV screen and non-weighted dimensions.

    With no methodology info, returns state_type_weights unchanged.
    """
    if config is None:
        config = load_config()

    J = len(state_type_weights)
    W = state_type_weights.copy().astype(float)

    methodology = poll.get("methodology", "")
    if not methodology:
        methodology = parse_methodology(poll.get("notes", ""))

    # Adjustment 1: LV/RV type screen
    if methodology in ("LV", "RV"):
        propensity = compute_propensity_scores(type_profiles, config)
        if methodology == "LV":
            factor = config.get("lv_downweight_factor", 0.5)
        else:
            factor = config.get("rv_downweight_factor", 0.8)
        # Scale W by propensity: high-propensity types keep weight,
        # low-propensity types get downweighted.
        # adjustment = factor + (1 - factor) * propensity
        # At propensity=1: adjustment=1.0 (no change)
        # At propensity=0: adjustment=factor (downweighted)
        adjustment = factor + (1.0 - factor) * propensity
        W = W * adjustment

    # Adjustment 2: Non-weighted dimensions (religion, urbanicity, income)
    # Only applied when methodology is known (otherwise we have no signal)
    if methodology:
        dims = config.get("w_vector_dimensions", {}).get(w_vector_mode, [])
        if dims:
            # Compute type diversity on non-weighted dimensions.
            # Types that are extreme on these dimensions get slightly adjusted
            # based on polling method reach profiles.
            method_key = _infer_method_type(poll)
            reach = config.get("method_reach_profiles", {}).get(method_key, {})
            for dim in dims:
                shift = reach.get(f"{dim}_shift", 0.0)
                if shift != 0.0 and dim in type_profiles.columns:
                    col = type_profiles[dim].values.astype(float)
                    col_norm = col - col.mean()
                    col_range = col.max() - col.min()
                    if col_range > 0:
                        col_norm = col_norm / col_range
                    W = W * (1.0 + shift * col_norm)

    # Normalize
    W_sum = W.sum()
    if W_sum > 0:
        W = W / W_sum
    else:
        W = np.ones(J) / J
    return W


def _infer_method_type(poll: dict) -> str:
    """Infer polling method from notes or pollster name."""
    notes = (poll.get("notes", "") or "").lower()
    if "online" in notes or "panel" in notes:
        return "online_panel"
    if "ivr" in notes:
        return "phone_ivr"
    if "live" in notes or "phone" in notes:
        return "phone_live"
    return "unknown"


# ---------------------------------------------------------------------------
# Tier 2: Crosstab-based W (interface — populated when data arrives)
# ---------------------------------------------------------------------------

def build_W_from_crosstabs(
    poll: dict,
    crosstabs: list[dict],
    type_profiles: pd.DataFrame,
    state_type_weights: np.ndarray,
) -> list[dict]:
    """Tier 2: Map crosstab groups to types, return multiple observations.

    Each crosstab group becomes a separate observation with its own W vector
    and y value (dem_share for that sub-group).

    Returns list of {"W": np.ndarray, "y": float, "sigma": float} dicts.
    """
    J = len(state_type_weights)
    observations = []

    for xt in crosstabs:
        dem_share = xt.get("dem_share")
        pct_of_sample = xt.get("pct_of_sample", 0.0)
        n_sample = poll.get("n_sample", 600)

        if dem_share is None or pct_of_sample <= 0:
            continue

        # Sub-group effective sample size
        sub_n = max(int(n_sample * pct_of_sample), 1)
        sigma = np.sqrt(dem_share * (1 - dem_share) / sub_n)

        # Map this demographic group to type weights via profile similarity
        group = xt.get("demographic_group", "")
        value = xt.get("group_value", "")
        W = _map_demographic_to_types(group, value, type_profiles, state_type_weights)

        observations.append({"W": W, "y": dem_share, "sigma": max(sigma, 1e-6)})

    if not observations:
        # Fallback: single topline observation with state weights
        dem_share = poll["dem_share"]
        n_sample = poll.get("n_sample", 600)
        sigma = np.sqrt(dem_share * (1 - dem_share) / max(n_sample, 1))
        return [{"W": state_type_weights.copy(), "y": dem_share, "sigma": max(sigma, 1e-6)}]

    return observations


def _map_demographic_to_types(
    group: str,
    value: str,
    type_profiles: pd.DataFrame,
    state_type_weights: np.ndarray,
) -> np.ndarray:
    """Map a demographic group/value to type weights via profile similarity.

    E.g., group="race", value="black" → upweight types with high pct_black.
    """
    J = len(state_type_weights)

    # Mapping from crosstab group/value to type profile column
    demo_col_map = {
        ("race", "black"): "pct_black",
        ("race", "white"): "pct_white_nh",
        ("race", "hispanic"): "pct_hispanic",
        ("race", "asian"): "pct_asian",
        ("education", "college"): "pct_bachelors_plus",
        ("education", "college_plus"): "pct_bachelors_plus",
        ("religion", "evangelical"): "evangelical_share",
    }

    col = demo_col_map.get((group.lower(), value.lower()))
    if col is None or col not in type_profiles.columns:
        return state_type_weights.copy()

    # Weight types by their demographic concentration
    type_vals = type_profiles[col].values.astype(float)
    W = state_type_weights * type_vals
    W_sum = W.sum()
    return W / W_sum if W_sum > 0 else np.ones(J) / J


# ---------------------------------------------------------------------------
# Tier 1: Raw unweighted sample (interface — populated when data arrives)
# ---------------------------------------------------------------------------

def build_W_from_raw_sample(
    poll: dict,
    raw_demographics: dict[str, float],
    type_profiles: pd.DataFrame,
    state_type_weights: np.ndarray,
) -> np.ndarray:
    """Tier 1: Map raw unweighted sample demographics directly to type weights.

    Computes demographic similarity between the raw sample composition
    and each type's demographic profile, gated by state presence.
    """
    J = len(state_type_weights)

    # Available dimensions: intersection of raw_demographics keys and type_profiles columns
    dims = [k for k in raw_demographics if k in type_profiles.columns]
    if not dims:
        return state_type_weights.copy()

    # Inverse squared distance similarity
    diffs_sq = np.zeros(J)
    for dim in dims:
        poll_val = raw_demographics[dim]
        type_vals = type_profiles[dim].values.astype(float)
        col_range = type_vals.max() - type_vals.min()
        if col_range > 0:
            diffs_sq += ((poll_val - type_vals) / col_range) ** 2

    distance = np.sqrt(diffs_sq / len(dims))
    similarity = 1.0 / (1.0 + distance * 5.0)  # scale factor

    W = similarity * state_type_weights
    W_sum = W.sum()
    return W / W_sum if W_sum > 0 else np.ones(J) / J


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def build_W_poll(
    poll: dict,
    type_profiles: pd.DataFrame,
    state_type_weights: np.ndarray,
    poll_crosstabs: list[dict] | None = None,
    raw_sample_demographics: dict[str, float] | None = None,
    w_vector_mode: str = "core",
) -> np.ndarray | list[dict]:
    """Construct poll-specific W vector at the best available tier.

    Returns:
      - Tier 1 & 3: np.ndarray of shape (J,) — single W vector
      - Tier 2: list of {"W": ndarray, "y": float, "sigma": float} dicts
    """
    if raw_sample_demographics is not None:
        return build_W_from_raw_sample(
            poll, raw_sample_demographics, type_profiles, state_type_weights,
        )

    if poll_crosstabs is not None:
        return build_W_from_crosstabs(
            poll, poll_crosstabs, type_profiles, state_type_weights,
        )

    return build_W_with_adjustments(
        poll, type_profiles, state_type_weights, w_vector_mode=w_vector_mode,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/prediction/test_poll_enrichment.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/prediction/poll_enrichment.py tests/prediction/test_poll_enrichment.py
git commit -m "feat: tiered W vector construction for poll enrichment

Three tiers with off-ramps:
- Tier 1: raw unweighted sample → direct type mapping
- Tier 2: crosstabs → per-group observations (multiple W rows per poll)
- Tier 3: topline only → LV/RV propensity screen + non-weighted dimensions

Tier 3 is the active path for current data. Tiers 1-2 are interfaces
ready for when crosstab data arrives."
```

---

### Task 3: Wire Quality Weighting into Forecast Engine (Layer 1)

**Files:**
- Modify: `src/prediction/forecast_engine.py`
- Modify: `tests/prediction/test_forecast_engine.py`

- [ ] **Step 1: Write failing test for prepare_polls**

Add to `tests/prediction/test_forecast_engine.py`:

```python
from src.prediction.forecast_engine import prepare_polls


class TestPreparePolls:
    def test_returns_adjusted_dicts(self):
        """prepare_polls should return dicts with adjusted dem_share and n_sample."""
        raw = {
            "2026 GA Senate": [
                {"dem_share": 0.53, "n_sample": 600, "state": "GA",
                 "date": "2026-03-01", "pollster": "Emerson College",
                 "notes": "LV"},
            ]
        }
        result = prepare_polls(raw, reference_date="2026-03-29")
        assert "2026 GA Senate" in result
        polls = result["2026 GA Senate"]
        assert len(polls) == 1
        p = polls[0]
        # Should still have required fields
        assert "dem_share" in p
        assert "n_sample" in p
        assert "state" in p
        # n_sample should be reduced by time decay (28 days with 30-day half-life)
        assert p["n_sample"] < 600

    def test_preserves_state_and_notes(self):
        """Metadata fields should survive the transformation."""
        raw = {
            "2026 GA Senate": [
                {"dem_share": 0.53, "n_sample": 600, "state": "GA",
                 "date": "2026-03-15", "pollster": "TestPollster",
                 "notes": "RV; src=test"},
            ]
        }
        result = prepare_polls(raw, reference_date="2026-03-29")
        p = result["2026 GA Senate"][0]
        assert p["state"] == "GA"
        assert "notes" in p

    def test_empty_input(self):
        result = prepare_polls({}, reference_date="2026-03-29")
        assert result == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/prediction/test_forecast_engine.py::TestPreparePolls -v`
Expected: FAIL — `ImportError: cannot import name 'prepare_polls'`

- [ ] **Step 3: Implement prepare_polls in forecast_engine.py**

Add to `src/prediction/forecast_engine.py` after the imports:

```python
from src.propagation.propagate_polls import PollObservation
from src.propagation.poll_weighting import apply_all_weights


def prepare_polls(
    polls_by_race: dict[str, list[dict]],
    reference_date: str,
    half_life_days: float = 30.0,
) -> dict[str, list[dict]]:
    """Apply quality weighting to raw poll dicts.

    Converts dicts → PollObservation → apply_all_weights → back to dicts.
    Returns polls with adjusted dem_share (house effects) and n_sample
    (time decay, pollster grade, pre-primary discount).
    """
    if not polls_by_race:
        return {}

    # Flatten all polls, keeping race labels
    all_obs: list[PollObservation] = []
    all_notes: list[str] = []
    race_labels: list[str] = []

    for race_id, polls in polls_by_race.items():
        for p in polls:
            obs = PollObservation(
                geography=p.get("state", ""),
                dem_share=p["dem_share"],
                n_sample=int(p["n_sample"]),
                race=race_id,
                date=p.get("date", ""),
                pollster=p.get("pollster", ""),
                geo_level=p.get("geo_level", "state"),
            )
            all_obs.append(obs)
            all_notes.append(p.get("notes", ""))
            race_labels.append(race_id)

    # Apply all quality adjustments
    weighted = apply_all_weights(
        all_obs,
        reference_date=reference_date,
        half_life_days=half_life_days,
        poll_notes=all_notes,
    )

    # Reconstruct dicts grouped by race
    result: dict[str, list[dict]] = {}
    for obs, notes, race_id in zip(weighted, all_notes, race_labels):
        d = {
            "dem_share": obs.dem_share,
            "n_sample": obs.n_sample,
            "state": obs.geography,
            "date": obs.date,
            "pollster": obs.pollster,
            "notes": notes,
        }
        result.setdefault(race_id, []).append(d)

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/prediction/test_forecast_engine.py -v`
Expected: All tests PASS (existing + new)

- [ ] **Step 5: Commit**

```bash
git add src/prediction/forecast_engine.py tests/prediction/test_forecast_engine.py
git commit -m "feat: wire poll quality weighting into forecast engine

prepare_polls() converts raw poll dicts through the existing
apply_all_weights pipeline (house effects, time decay, pollster
grade, pre-primary discount) and returns quality-adjusted dicts."
```

---

### Task 4: Integrate W Vectors + Quality into run_forecast

**Files:**
- Modify: `src/prediction/forecast_engine.py`
- Modify: `tests/prediction/test_forecast_engine.py`

- [ ] **Step 1: Write failing test for enriched _build_poll_arrays**

Add to `tests/prediction/test_forecast_engine.py`:

```python
class TestEnrichedForecast:
    def test_w_vector_mode_parameter(self):
        """run_forecast should accept w_vector_mode parameter."""
        J = 3
        n = 6
        type_scores = np.random.RandomState(42).rand(n, J)
        type_scores = type_scores / type_scores.sum(axis=1, keepdims=True)
        county_priors = np.full(n, 0.5)
        states = ["GA"] * 3 + ["FL"] * 3
        county_votes = np.ones(n)

        polls = {"2026 GA Senate": [
            {"dem_share": 0.53, "n_sample": 600, "state": "GA",
             "date": "2026-03-01", "pollster": "Test", "notes": "LV"},
        ]}

        result = run_forecast(
            type_scores=type_scores,
            county_priors=county_priors,
            states=states,
            county_votes=county_votes,
            polls_by_race=polls,
            races=["2026 GA Senate"],
            w_vector_mode="core",
        )
        assert "2026 GA Senate" in result
        assert result["2026 GA Senate"].n_polls == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/prediction/test_forecast_engine.py::TestEnrichedForecast -v`
Expected: FAIL — `TypeError: run_forecast() got an unexpected keyword argument 'w_vector_mode'`

- [ ] **Step 3: Modify _build_poll_arrays and run_forecast**

In `src/prediction/forecast_engine.py`, modify `_build_poll_arrays` to accept an optional `w_builder` callable, and modify `run_forecast` to add `w_vector_mode` and `reference_date` parameters:

Replace the existing `_build_poll_arrays` function:

```python
def _build_poll_arrays(
    polls_by_race: dict[str, list[dict]],
    type_scores: np.ndarray,
    states: list[str],
    county_votes: np.ndarray,
    w_builder: callable | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build W, y, sigma arrays from all polls across all races.

    When w_builder is provided, it is called for each poll to produce
    a poll-specific W vector (or list of observation dicts for Tier 2).
    When w_builder is None, falls back to build_W_state (current behavior).

    Returns: (W_all, y_all, sigma_all, race_labels)
    """
    W_rows: list[np.ndarray] = []
    y_vals: list[float] = []
    sigma_vals: list[float] = []
    race_labels: list[str] = []

    for race_id, polls in polls_by_race.items():
        for p in polls:
            state = p["state"]
            dem_share = p["dem_share"]
            n_sample = p["n_sample"]

            if w_builder is not None:
                result = w_builder(p)
                if isinstance(result, list):
                    # Tier 2: multiple observations per poll
                    for obs in result:
                        W_rows.append(obs["W"])
                        y_vals.append(obs["y"])
                        sigma_vals.append(obs["sigma"])
                        race_labels.append(race_id)
                    continue
                else:
                    W_row = result
            else:
                W_row = build_W_state(state, type_scores, states, county_votes)

            sigma = np.sqrt(dem_share * (1 - dem_share) / max(n_sample, 1))

            W_rows.append(W_row)
            y_vals.append(dem_share)
            sigma_vals.append(max(sigma, 1e-6))
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
```

Update `run_forecast` signature and body:

```python
def run_forecast(
    type_scores: np.ndarray,
    county_priors: np.ndarray,
    states: list[str],
    county_votes: np.ndarray,
    polls_by_race: dict[str, list[dict]],
    races: list[str],
    lam: float = 1.0,
    mu: float = 1.0,
    generic_ballot_shift: float = 0.0,
    w_vector_mode: str = "core",
    reference_date: str | None = None,
    type_profiles: pd.DataFrame | None = None,
) -> dict[str, ForecastResult]:
    """Run the full hierarchical forecast for all races.

    1. Compute θ_prior from county priors
    2. Apply poll quality weighting (if reference_date provided)
    3. Estimate θ_national from all polls pooled
    4. For each race, estimate δ_race from residuals
    5. Produce county predictions in both modes
    """
    J = type_scores.shape[1]

    # Apply generic ballot shift to county priors
    adjusted_priors = county_priors + generic_ballot_shift

    # Step 1: θ_prior
    theta_prior = compute_theta_prior(type_scores, adjusted_priors)

    # Step 1.5: Apply poll quality weighting
    working_polls = polls_by_race
    if reference_date:
        working_polls = prepare_polls(polls_by_race, reference_date)

    # Step 1.6: Build W vector builder if type_profiles available
    w_builder = None
    if type_profiles is not None:
        from src.prediction.poll_enrichment import build_W_poll

        # Precompute state-level type weights for W vector construction
        state_type_weight_cache: dict[str, np.ndarray] = {}

        def _w_builder(poll: dict) -> np.ndarray | list[dict]:
            st = poll["state"]
            if st not in state_type_weight_cache:
                state_type_weight_cache[st] = build_W_state(
                    st, type_scores, states, county_votes,
                )
            return build_W_poll(
                poll=poll,
                type_profiles=type_profiles,
                state_type_weights=state_type_weight_cache[st],
                w_vector_mode=w_vector_mode,
            )

        w_builder = _w_builder

    # Step 2: Build poll arrays and estimate θ_national
    W_all, y_all, sigma_all, race_labels = _build_poll_arrays(
        working_polls, type_scores, states, county_votes,
        w_builder=w_builder,
    )
    theta_national = estimate_theta_national(W_all, y_all, sigma_all, theta_prior, lam)

    # Step 3 & 4: Per-race δ and predictions
    results: dict[str, ForecastResult] = {}
    for race_id in races:
        race_polls = working_polls.get(race_id, [])
        n_polls = len(race_polls)

        if n_polls > 0:
            race_W, race_y, race_sigma, _ = _build_poll_arrays(
                {race_id: race_polls}, type_scores, states, county_votes,
                w_builder=w_builder,
            )
            residuals = race_y - race_W @ theta_national
            delta = estimate_delta_race(race_W, residuals, race_sigma, J, mu)
        else:
            delta = np.zeros(J)

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

Add `import pandas as pd` to the imports at the top of the file.

- [ ] **Step 4: Run all tests**

Run: `uv run pytest tests/prediction/test_forecast_engine.py -v`
Expected: All tests PASS (existing tests still work because new params have defaults)

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest --tb=short -q`
Expected: 2,497+ pass, 0 fail

- [ ] **Step 6: Commit**

```bash
git add src/prediction/forecast_engine.py tests/prediction/test_forecast_engine.py
git commit -m "feat: integrate poll quality + enriched W vectors into forecast engine

run_forecast() now accepts w_vector_mode, reference_date, and
type_profiles parameters. When provided:
- Polls are quality-weighted via prepare_polls (house effects,
  time decay, pollster grade)
- W vectors use poll-specific enrichment (LV/RV screen,
  non-weighted dimension adjustment)

All parameters have backwards-compatible defaults — existing
callers are unaffected."
```

---

### Task 5: Wire into predict_2026_types.py + Validation

**Files:**
- Modify: `src/prediction/predict_2026_types.py`

- [ ] **Step 1: Pass type_profiles and w_vector_mode to run_forecast**

In `src/prediction/predict_2026_types.py`, find the `run_forecast()` call (around line 498) and add the new parameters:

```python
    # Load type profiles for W vector enrichment
    type_profiles_path = PROJECT_ROOT / "data" / "communities" / "type_profiles.parquet"
    type_profiles_df = None
    if type_profiles_path.exists():
        type_profiles_df = pd.read_parquet(type_profiles_path)
        log.info("Loaded type profiles for poll enrichment: %d types", len(type_profiles_df))

    # Run the hierarchical forecast: θ_prior → θ_national → δ_race
    forecast_results = run_forecast(
        type_scores=type_scores,
        county_priors=county_prior_values,
        states=states,
        county_votes=county_votes,
        polls_by_race=polls_by_race,
        races=all_race_ids,
        lam=1.0,
        mu=1.0,
        w_vector_mode="core",  # TODO: compare core vs full, set winner
        reference_date=str(date.today()),
        type_profiles=type_profiles_df,
    )
```

Add `from datetime import date` to imports if not already present.

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest --tb=short -q`
Expected: 2,497+ pass

- [ ] **Step 3: Commit**

```bash
git add src/prediction/predict_2026_types.py
git commit -m "feat: wire poll enrichment into prediction pipeline

predict_2026_types.run() now passes type_profiles and reference_date
to run_forecast(), enabling poll quality weighting and enriched W
vectors in the production prediction pipeline."
```

---

### Task 6: Comparison Script (core vs full modes)

**Files:**
- Create: `scripts/compare_w_vector_modes.py`

- [ ] **Step 1: Create comparison script**

Write `scripts/compare_w_vector_modes.py`:

```python
"""Compare W vector modes: core vs full vs baseline (no enrichment).

Runs the forecast pipeline three times and reports prediction differences.
Usage: uv run python scripts/compare_w_vector_modes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datetime import date
from src.prediction.forecast_engine import run_forecast


def main():
    # Load shared inputs
    ta_df = pd.read_parquet(PROJECT_ROOT / "data" / "communities" / "type_assignments.parquet")
    county_fips = ta_df["county_fips"].astype(str).str.zfill(5).tolist()
    score_cols = sorted([c for c in ta_df.columns if c.endswith("_score")])
    type_scores = ta_df[score_cols].values

    # County priors
    from src.prediction.predict_2026_types import compute_county_priors
    county_priors = compute_county_priors(county_fips)
    ridge_path = PROJECT_ROOT / "data" / "models" / "ridge_model" / "ridge_county_priors.parquet"
    if ridge_path.exists():
        rdf = pd.read_parquet(ridge_path)
        rdf["county_fips"] = rdf["county_fips"].astype(str).str.zfill(5)
        rmap = dict(zip(rdf["county_fips"], rdf["ridge_pred_dem_share"]))
        for i, f in enumerate(county_fips):
            if f in rmap:
                county_priors[i] = rmap[f]

    states = [f[:2] for f in county_fips]  # simplified
    from src.prediction.predict_2026_types import _STATE_FIPS_TO_ABBR
    states = [_STATE_FIPS_TO_ABBR.get(f[:2], "??") for f in county_fips]
    county_votes = np.ones(len(county_fips))

    # Type profiles
    tp = pd.read_parquet(PROJECT_ROOT / "data" / "communities" / "type_profiles.parquet")

    # Load polls
    polls_csv = pd.read_csv(PROJECT_ROOT / "data" / "polls" / "polls_2026.csv")
    polls_by_race: dict[str, list[dict]] = {}
    for _, row in polls_csv.iterrows():
        race = row["race"]
        if "Senate" not in race:
            continue
        polls_by_race.setdefault(race, []).append({
            "dem_share": float(row["dem_share"]),
            "n_sample": int(row["n_sample"]),
            "state": str(row["geography"]),
            "date": str(row.get("date", "")),
            "pollster": str(row.get("pollster", "")),
            "notes": str(row.get("notes", "")),
        })

    races = list(polls_by_race.keys())
    ref_date = str(date.today())

    modes = {
        "baseline": {"type_profiles": None, "w_vector_mode": "core", "reference_date": None},
        "core": {"type_profiles": tp, "w_vector_mode": "core", "reference_date": ref_date},
        "full": {"type_profiles": tp, "w_vector_mode": "full", "reference_date": ref_date},
    }

    results = {}
    for mode_name, kwargs in modes.items():
        fr = run_forecast(
            type_scores=type_scores,
            county_priors=county_priors,
            states=states,
            county_votes=county_votes,
            polls_by_race=polls_by_race,
            races=races,
            **kwargs,
        )
        # State-level predictions (simple mean across state counties)
        state_preds = {}
        for race_id, res in fr.items():
            st = race_id.split()[1]  # "2026 GA Senate" → "GA"
            mask = np.array([s == st for s in states])
            if mask.any():
                state_preds[race_id] = float(res.county_preds_local[mask].mean())
        results[mode_name] = state_preds

    # Compare
    print(f"\n{'Race':<25s} {'Baseline':>10s} {'Core':>10s} {'Full':>10s} {'Δ core':>8s} {'Δ full':>8s}")
    print("-" * 75)
    for race in sorted(races):
        bl = results["baseline"].get(race, 0.5)
        co = results["core"].get(race, 0.5)
        fu = results["full"].get(race, 0.5)
        d_co = (co - bl) * 100
        d_fu = (fu - bl) * 100
        print(f"{race:<25s} {bl:>10.4f} {co:>10.4f} {fu:>10.4f} {d_co:>+7.2f}pp {d_fu:>+7.2f}pp")

    avg_shift_core = np.mean([abs(results["core"][r] - results["baseline"][r]) * 100
                               for r in races if r in results["baseline"] and r in results["core"]])
    avg_shift_full = np.mean([abs(results["full"][r] - results["baseline"][r]) * 100
                               for r in races if r in results["baseline"] and r in results["full"]])
    print(f"\nAvg |shift| core: {avg_shift_core:.2f}pp")
    print(f"Avg |shift| full: {avg_shift_full:.2f}pp")
    print(f"\nValidation criterion: shifts should average < 2pp")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the comparison**

Run: `uv run python scripts/compare_w_vector_modes.py`
Expected: Table showing per-race predictions for baseline/core/full with shift magnitudes. Shifts should be small (< 2pp average).

- [ ] **Step 3: Commit**

```bash
git add scripts/compare_w_vector_modes.py
git commit -m "feat: comparison script for W vector modes (core vs full vs baseline)"
```

---

### Task 7: TODO Documentation

**Files:**
- Modify: `docs/TODO-autonomous-improvements.md`

- [ ] **Step 1: Add poll enrichment TODOs**

Append to `docs/TODO-autonomous-improvements.md`:

```markdown

---

## Priority 4 — Rich Poll Ingestion (Future Work)

These items build on the Tier 3 poll enrichment shipped in S249. See spec: `docs/superpowers/specs/2026-03-29-rich-poll-ingestion-design.md`.

- [ ] **TODO-POLL-1: Crosstab Scraping Pipeline** — Per-pollster integrations for extracting demographic breakdowns from original poll releases (PDFs, pollster websites). Priority targets: Emerson College, Cygnal, Trafalgar, Quantus, TIPP Insights. No known structured API or aggregator exists. Enables Tier 2 W vectors — the biggest information gain. Each pollster is a separate parser.

- [ ] **TODO-POLL-2: Undersampled Group Identification** — Compare poll's inferred/actual demographic coverage against the *political diversity within demographic groups* in the polled state. Core insight: demographic representation ≠ type representation. A poll weighted to 33% Black in GA can still miss that Atlanta Black voters (Type 29) behave differently from rural SW GA Black voters (Type 50). Output: per-type σ inflation for underrepresented types + API reporting of coverage gaps. Sample-size-aware (n=300 misses far more tracts than n=10000).

- [ ] **TODO-POLL-3: House Effects as Type Signal** — Persistent house effects may reflect which types a pollster systematically reaches, not pollster bias. Trafalgar's R-lean may mean they reach rural evangelical types others miss. Future work: decompose house effects into type-reach profiles per pollster. Requires TODO-POLL-1 data to validate. Risk: double-counting if we both correct dem_share AND infer type composition from house effects. Resolution: replace house effect correction with type-reach inference once validated.

- [ ] **TODO-MODEL-GA: State Prediction Tuning** — The forecast engine's county reconstruction (`type_scores @ theta`) can diverge from Ridge county priors (e.g., GA: Ridge R+2.1 vs engine D+10.4). This is a known consequence of type-level estimation projecting back to county level. Not a bug per se — the engine's hierarchical decomposition is the intended architecture — but predictions need validation against county-level ground truth. Tune λ/μ regularization and validate state-level aggregates against historical results.
```

- [ ] **Step 2: Commit**

```bash
git add docs/TODO-autonomous-improvements.md
git commit -m "docs: add poll enrichment TODOs (crosstabs, undersampling, house effects, GA tuning)"
```

---

### Task 8: Final Integration Test

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest --tb=short -q
```

Expected: 2,497+ pass, 0 fail. New tests from Tasks 1-4 should add ~17 tests.

- [ ] **Step 2: Run comparison script**

```bash
uv run python scripts/compare_w_vector_modes.py
```

Review output: shifts should be < 2pp average. If core and full modes produce nearly identical results, default to core (fewer dimensions = less noise). If full shows measurably different (and defensible) shifts, note it for future investigation.

- [ ] **Step 3: Final commit with any adjustments**

```bash
git add -A
git commit -m "feat: rich poll ingestion model complete

Layer 1: Poll quality weighting wired into forecast engine
Layer 2: Tiered W vectors (Tier 3 active: LV/RV screen + non-weighted dims)
Comparison script: core vs full vs baseline mode evaluation
TODOs documented for crosstabs, undersampling, house effects"
```
