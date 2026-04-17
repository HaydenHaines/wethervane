"""Candidate CTOV prior adjustments for 2026 predictions.

Applies historical candidate overperformance patterns (CTOV vectors) as
type-level prior adjustments. Unlike race_adjustments (which applies a uniform
state-level shift), CTOV adjustments are per-type: Lindsey Graham's evangelical
overperformance shifts different counties by different amounts depending on
their type composition.

Math for each county c in the race's state:
    adjusted_prior[c] += sum(W[c, j] * ctov[j] for j in range(J))

where W[c, j] is county c's soft membership in type j, and ctov[j] is the
candidate's (possibly averaged) CTOV for type j.

CTOV is already in dem_share units:
  - Positive CTOV → D overperformed (raises dem_share prior)
  - Negative CTOV → R overperformed (lowers dem_share prior)
No party-based sign flip needed — the sign encodes the direction.

For multi-race candidates, CTOV is averaged across races, with recency
weighting (most recent race gets weight 2, others weight 1).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# CEC threshold: only apply CTOV for candidates with cross-election consistency
# above this value. CEC < 0.3 means the signal is too noisy to trust.
# For single-race candidates, CEC is not available — apply their CTOV anyway
# since we have no reason to doubt it (just less evidence).
_CEC_THRESHOLD = 0.3


def load_ctov_adjustments(
    crosswalk_path: str | Path | None = None,
    ctov_path: str | Path | None = None,
    badges_path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Load CTOV adjustments for all 2026 races with candidate data.

    Returns a mapping from race_id → CTOV vector (shape J) to apply as a
    type-level prior shift. Only races with candidates who have historical
    CTOV data are included.

    For races with two candidates who both have CTOV data, the adjustment
    is the incumbent's CTOV only (challenger CTOV is less relevant for
    prior adjustment — it would double-count).

    Parameters
    ----------
    crosswalk_path : path-like, optional
        Path to candidate_2026_crosswalk.json. Defaults to data/sabermetrics/.
    ctov_path : path-like, optional
        Path to candidate_ctov.parquet. Defaults to data/sabermetrics/.
    badges_path : path-like, optional
        Path to candidate_badges.json (for CEC lookup). Defaults to data/sabermetrics/.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from race_id → CTOV vector of shape (J,).
    """
    saber_dir = PROJECT_ROOT / "data" / "sabermetrics"
    xwalk_p = Path(crosswalk_path) if crosswalk_path else saber_dir / "candidate_2026_crosswalk.json"
    ctov_p = Path(ctov_path) if ctov_path else saber_dir / "candidate_ctov.parquet"
    badges_p = Path(badges_path) if badges_path else saber_dir / "candidate_badges.json"

    if not xwalk_p.exists() or not ctov_p.exists():
        log.warning("CTOV data files not found; skipping candidate adjustments")
        return {}

    with open(xwalk_p) as f:
        crosswalk = json.load(f)

    ctov_df = pd.read_parquet(ctov_p)
    ctov_cols = sorted([c for c in ctov_df.columns if c.startswith("ctov_type_")])
    J = len(ctov_cols)

    # Load CEC data for quality gating
    cec_by_person: dict[str, float | None] = {}
    if badges_p.exists():
        with open(badges_p) as f:
            badges = json.load(f)
        for pid, data in badges.items():
            cec_by_person[pid] = data.get("cec")

    adjustments: dict[str, np.ndarray] = {}

    for race_id, parties in crosswalk.get("races", {}).items():
        # Find the first candidate with historical data.
        # Priority: incumbent party's candidate, then challenger.
        best_bioguide = None
        best_n_races = 0

        for party, candidates in parties.items():
            for cand in candidates:
                bio = cand.get("bioguide_id")
                n_races = cand.get("historical_races", 0)
                if bio and n_races > 0 and n_races > best_n_races:
                    best_bioguide = bio
                    best_n_races = n_races

        if best_bioguide is None:
            continue

        # CEC quality gate for multi-race candidates
        cec = cec_by_person.get(best_bioguide)
        if best_n_races > 1 and cec is not None and cec < _CEC_THRESHOLD:
            log.info(
                "Skipping CTOV for %s: CEC=%.3f < %.1f threshold",
                race_id, cec, _CEC_THRESHOLD,
            )
            continue

        # Compute (recency-weighted) career CTOV
        person_rows = ctov_df[ctov_df["person_id"] == best_bioguide].sort_values("year")
        if person_rows.empty:
            continue

        ctov_matrix = person_rows[ctov_cols].values  # shape: (n_races, J)
        if len(person_rows) == 1:
            ctov_vec = ctov_matrix[0]
        else:
            # Recency weighting: most recent race gets weight 2, others get 1
            weights = np.ones(len(person_rows))
            weights[-1] = 2.0
            weights /= weights.sum()
            ctov_vec = (ctov_matrix.T @ weights)  # weighted average

        adjustments[race_id] = ctov_vec
        log.info(
            "CTOV adjustment for %s: candidate=%s, n_races=%d, "
            "ctov_range=[%.4f, %.4f]",
            race_id,
            best_bioguide,
            best_n_races,
            ctov_vec.min(),
            ctov_vec.max(),
        )

    log.info("Loaded CTOV adjustments for %d races", len(adjustments))
    return adjustments


# CTOV is historical residual signal — apply conservatively to future priors.
# Scale factor of 0.3 reflects uncertainty: the candidate's past overperformance
# won't fully repeat in a new election (different opponent, environment, etc.).
# Max shift of 5pp per county prevents extreme distortion from high-membership types.
CTOV_SCALE = 0.3
CTOV_MAX_SHIFT = 0.05  # 5pp cap per county


def apply_ctov_adjustment(
    county_priors: np.ndarray,
    type_scores: np.ndarray,
    ctov_vec: np.ndarray,
    state_mask: np.ndarray,
    scale: float = CTOV_SCALE,
    max_shift: float = CTOV_MAX_SHIFT,
) -> np.ndarray:
    """Apply a type-level CTOV adjustment to county priors for one race.

    For each county c in the state:
        raw_shift[c] = sum(W[c, j] * ctov[j] for j in range(J))
        shift[c] = clip(raw_shift[c] * scale, -max_shift, +max_shift)
        adjusted[c] = prior[c] + shift[c]

    The scale factor and shift cap prevent extreme CTOV values in
    high-membership types from distorting predictions. Without these,
    a county with 93% membership in a type where the candidate has
    -0.25 CTOV would shift by -23pp — too much for a prior.

    Parameters
    ----------
    county_priors : ndarray of shape (N,)
        All county priors (will be copied; original not modified).
    type_scores : ndarray of shape (N, J)
        Soft type membership for all counties.
    ctov_vec : ndarray of shape (J,)
        CTOV vector for the candidate.
    state_mask : ndarray of shape (N,) bool
        Which counties belong to this race's state.
    scale : float
        Multiplicative scale for CTOV shift (default 0.3).
    max_shift : float
        Max absolute shift per county in dem_share units (default 0.05 = 5pp).

    Returns
    -------
    ndarray of shape (N,)
        Adjusted priors with CTOV applied to the race's state.
    """
    adjusted = county_priors.copy()
    # Type-level shift: each county gets shifted by its type-weighted CTOV
    raw_shift = type_scores[state_mask] @ ctov_vec
    capped_shift = np.clip(raw_shift * scale, -max_shift, max_shift)
    adjusted[state_mask] += capped_shift
    return adjusted
