"""House effect (partisan bias) correction for polls.

Corrects each poll's dem_share for systematic over/under-estimation
of Democrats by each pollster.  Uses a priority cascade:
  1. Empirical bias from pollster_accuracy.json (mean_error_pp from 2022 actuals)
  2. Silver Bulletin "House Effect" column (mean-reverted, percentage points)
  3. 538 ``bias_ppm`` column (same sign convention)
  4. 0.0 (no correction) for unknown pollsters

Positive house_effect means the pollster overestimates Democrats,
so subtracting it moves the share toward the true value.

The empirical source (Priority 1) takes precedence because it is derived
directly from observed 2022 election results, giving it grounding in
measured behavior rather than prior assessments.
"""

from __future__ import annotations

import json
import logging
import re
from copy import copy
from pathlib import Path

from src.propagation.propagate_polls import PollObservation

log = logging.getLogger(__name__)

# Clamp corrected dem_share to this range so it stays a valid proportion.
_HE_DEM_SHARE_MIN: float = 0.01
_HE_DEM_SHARE_MAX: float = 0.99

# Minimum number of polls a pollster must have in the accuracy data to
# use empirical bias.  Pollsters with very few polls have unreliable bias
# estimates and are better handled by external sources.
_EMPIRICAL_MIN_POLLS: int = 2

# Default path to the pollster accuracy JSON (relative to project root).
_DEFAULT_ACCURACY_PATH: Path = (
    Path(__file__).parents[2] / "data" / "experiments" / "pollster_accuracy.json"
)

# Cached house effect lookups: None = not yet loaded.
_SB_HOUSE_EFFECTS: dict[str, float] | None = None
_538_BIAS: dict[str, float] | None = None
_EMPIRICAL_BIAS: dict[str, float] | None = None


# ---------------------------------------------------------------------------
# Empirical bias loading
# ---------------------------------------------------------------------------


def _tokenize_name(name: str) -> set[str]:
    """Split a pollster name into lowercase word tokens for fuzzy matching."""
    return set(re.findall(r"[a-z0-9]+", name.lower()))


def load_empirical_house_effects(
    accuracy_path: Path | None = None,
    min_polls: int = _EMPIRICAL_MIN_POLLS,
) -> dict[str, float]:
    """Load empirical pollster bias (house effects) from pollster_accuracy.json.

    Each pollster's ``mean_error_pp`` field is their systematic partisan bias
    measured against 2022 actual results:
      - Positive mean_error_pp → pollster over-predicts Democrats.
      - Negative mean_error_pp → pollster under-predicts Democrats (Republican lean).

    Pollsters with fewer than ``min_polls`` polls in the accuracy data are
    excluded — their bias estimates are too noisy to be useful.  These pollsters
    fall through to Silver Bulletin / 538 in the priority cascade.

    Parameters
    ----------
    accuracy_path:
        Path to pollster_accuracy.json.  Defaults to
        ``data/experiments/pollster_accuracy.json`` relative to project root.
    min_polls:
        Minimum number of 2022 polls required to trust the empirical bias.

    Returns
    -------
    dict[str, float]
        Maps pollster name (as stored in the JSON) → bias in percentage points.
    """
    path = accuracy_path or _DEFAULT_ACCURACY_PATH
    if not path.exists():
        log.debug("Pollster accuracy file not found: %s", path)
        return {}

    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        log.warning("Failed to load pollster accuracy file: %s", exc)
        return {}

    result: dict[str, float] = {}
    for entry in data.get("pollsters", []):
        name = entry.get("pollster", "").strip()
        bias = entry.get("mean_error_pp")
        n_polls = entry.get("n_polls", 0)
        if name and isinstance(bias, (int, float)) and n_polls >= min_polls:
            result[name] = float(bias)

    log.debug("Loaded %d empirical house effects from %s", len(result), path)
    return result


def _lookup_empirical_bias(
    pollster_name: str,
    empirical_bias: dict[str, float],
) -> float | None:
    """Look up a pollster's empirical bias using exact then fuzzy matching.

    Returns the bias in percentage points, or None if not found.

    Fuzzy matching uses a Jaccard-like token similarity threshold of 0.5,
    consistent with the poll_quality.py fuzzy matching approach.
    """
    if not empirical_bias:
        return None

    # Exact match (case-sensitive — names come from the same source as polls)
    if pollster_name in empirical_bias:
        return empirical_bias[pollster_name]

    # Case-insensitive exact match
    lower_name = pollster_name.lower()
    for key, val in empirical_bias.items():
        if key.lower() == lower_name:
            return val

    # Fuzzy token match — same algorithm as poll_quality._fuzzy_match_pollster
    _FUZZY_THRESHOLD = 0.5
    query_tokens = _tokenize_name(pollster_name)
    if not query_tokens:
        return None

    best_score = 0.0
    best_val: float | None = None
    for key, val in empirical_bias.items():
        key_tokens = _tokenize_name(key)
        if not key_tokens:
            continue
        intersection = len(query_tokens & key_tokens)
        union = len(query_tokens | key_tokens)
        score = intersection / union if union > 0 else 0.0
        if score > best_score:
            best_score = score
            best_val = val

    if best_score >= _FUZZY_THRESHOLD:
        return best_val
    return None


def _ensure_house_effect_caches() -> None:
    """Populate module-level house effect caches on first call."""
    global _SB_HOUSE_EFFECTS, _538_BIAS, _EMPIRICAL_BIAS

    if _EMPIRICAL_BIAS is None:
        _EMPIRICAL_BIAS = load_empirical_house_effects()

    if _SB_HOUSE_EFFECTS is None:
        try:
            from src.assembly.silver_bulletin_ratings import load_pollster_house_effects
            _SB_HOUSE_EFFECTS = load_pollster_house_effects()
            log.debug("Loaded %d Silver Bulletin house effects", len(_SB_HOUSE_EFFECTS))
        except FileNotFoundError:
            log.debug("Silver Bulletin XLSX not found; SB house effects unavailable")
            _SB_HOUSE_EFFECTS = {}
        except Exception as exc:  # pragma: no cover
            log.warning("Failed to load Silver Bulletin house effects: %s", exc)
            _SB_HOUSE_EFFECTS = {}

    if _538_BIAS is None:
        try:
            from src.assembly.silver_bulletin_ratings import load_538_bias
            _538_BIAS = load_538_bias()
            log.debug("Loaded %d 538 bias estimates", len(_538_BIAS))
        except FileNotFoundError:
            log.debug("538 pollster-ratings CSV not found; 538 bias unavailable")
            _538_BIAS = {}
        except Exception as exc:  # pragma: no cover
            log.warning("Failed to load 538 bias estimates: %s", exc)
            _538_BIAS = {}


def _lookup_house_effect(
    pollster_name: str,
    sb_house_effects: dict[str, float],
    bias_538: dict[str, float],
    empirical_bias: dict[str, float] | None = None,
) -> tuple[float, str]:
    """Return (house_effect_pp, source) for a pollster using priority order.

    Priority:
      0. Empirical bias from pollster_accuracy.json (most trusted — measured
         against 2022 actual results).
      1. Silver Bulletin "House Effect" (mean-reverted, pp) — exact then fuzzy.
      2. 538 ``bias_ppm`` — exact match only.
      3. 0.0 (no correction) for unknown pollsters.

    Returns the source string for logging:
      'empirical', 'sb_exact', 'sb_fuzzy', '538', 'none'.
    """
    # 0. Empirical bias — derived from observed 2022 actual outcomes.
    #    This is the highest-priority source because it is measured, not assessed.
    if empirical_bias:
        val = _lookup_empirical_bias(pollster_name, empirical_bias)
        if val is not None:
            return val, "empirical"

    # 1. Silver Bulletin — delegate fuzzy matching to the silver_bulletin module
    if sb_house_effects:
        try:
            from src.assembly.silver_bulletin_ratings import _name_similarity, _normalize
        except ImportError:  # pragma: no cover
            pass
        else:
            # Exact match
            if pollster_name in sb_house_effects:
                return sb_house_effects[pollster_name], "sb_exact"

            # Normalised / fuzzy match
            norm_query = _normalize(pollster_name)
            norm_sb = {_normalize(k): v for k, v in sb_house_effects.items()}
            if norm_query in norm_sb:
                return norm_sb[norm_query], "sb_exact"

            # Best Jaccard fuzzy match (threshold 0.4 matches get_pollster_quality)
            _FUZZY_THRESHOLD = 0.4
            best_val: float | None = None
            best_sim = 0.0
            for norm_key, val in norm_sb.items():
                sim = _name_similarity(norm_query, norm_key)
                if sim > best_sim:
                    best_sim = sim
                    best_val = val
            if best_sim >= _FUZZY_THRESHOLD and best_val is not None:
                return best_val, "sb_fuzzy"

    # 2. 538 bias — exact name match only (no fuzzy; 538 names differ more)
    if bias_538 and pollster_name in bias_538:
        return bias_538[pollster_name], "538"

    return 0.0, "none"


def apply_house_effect_correction(
    polls: list[PollObservation],
    sb_house_effects: dict[str, float] | None = None,
    bias_538: dict[str, float] | None = None,
    empirical_bias: dict[str, float] | None = None,
) -> list[PollObservation]:
    """Correct each poll's dem_share for pollster partisan bias (house effects).

    For each poll, looks up the pollster's house effect using priority order:
      1. Empirical bias from pollster_accuracy.json (measured vs 2022 actuals)
      2. Silver Bulletin "House Effect" (mean-reverted, pp)
      3. 538 ``bias_ppm``
      4. 0.0 (no correction)

    Adjustment: corrected_dem_share = dem_share - house_effect / 100
      Positive house_effect means pollster overestimates Democrats, so subtracting
      moves the share toward the true value.

    Result is clamped to [_HE_DEM_SHARE_MIN, _HE_DEM_SHARE_MAX] to keep it
    a valid proportion.  The original ``dem_share`` is preserved on the returned
    PollObservation; ``dem_share`` is updated to the corrected value.

    Parameters
    ----------
    polls:
        List of PollObservation objects to correct.
    sb_house_effects:
        Optional pre-loaded Silver Bulletin house effects dict.  If None,
        loads from the module-level cache (populated from XLSX on first call).
    bias_538:
        Optional pre-loaded 538 bias dict.  If None, loads from the module-level
        cache (populated from CSV on first call).
    empirical_bias:
        Optional pre-loaded empirical bias dict from pollster_accuracy.json.
        If None, loads from the module-level cache (populated from JSON on
        first call).  Empirical bias takes precedence over all external sources
        because it is derived directly from observed 2022 election results.

    Returns
    -------
    list[PollObservation]
        New PollObservation copies with dem_share corrected for house effects.
        The uncorrected value is accessible via poll.raw_dem_share if needed
        (the field is not currently on PollObservation; original logged here).
    """
    _ensure_house_effect_caches()
    he_dict = sb_house_effects if sb_house_effects is not None else (_SB_HOUSE_EFFECTS or {})
    b538_dict = bias_538 if bias_538 is not None else (_538_BIAS or {})
    emp_dict = empirical_bias if empirical_bias is not None else (_EMPIRICAL_BIAS or {})

    result: list[PollObservation] = []
    for poll in polls:
        new_poll = copy(poll)
        if not poll.pollster:
            result.append(new_poll)
            continue

        house_effect_pp, source = _lookup_house_effect(
            poll.pollster, he_dict, b538_dict, emp_dict
        )

        if source == "none":
            log.debug("House effect: pollster %r not found, no correction applied", poll.pollster)
        else:
            log.debug(
                "House effect: pollster %r -> %.3f pp (%s); dem_share %.4f -> %.4f",
                poll.pollster,
                house_effect_pp,
                source,
                poll.dem_share,
                poll.dem_share - house_effect_pp / 100.0,
            )

        correction = house_effect_pp / 100.0
        corrected = poll.dem_share - correction
        corrected = max(_HE_DEM_SHARE_MIN, min(_HE_DEM_SHARE_MAX, corrected))
        new_poll.dem_share = corrected
        result.append(new_poll)

    return result


def reset_house_effect_cache() -> None:
    """Reset house effect caches (useful in tests that inject custom data)."""
    global _SB_HOUSE_EFFECTS, _538_BIAS, _EMPIRICAL_BIAS
    _SB_HOUSE_EFFECTS = None
    _538_BIAS = None
    _EMPIRICAL_BIAS = None
