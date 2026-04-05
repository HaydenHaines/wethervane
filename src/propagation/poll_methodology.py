"""Methodology-based poll quality weighting.

Phone polls are empirically more accurate than online-only polls.  IVR
(robocall) polls have the worst track record in recent cycles; live-phone
interviews are the gold standard.  This module applies a multiplier to each
poll's effective sample size based on its methodology tag.

The multipliers are stored in ``data/config/prediction_params.json`` under
``poll_weighting.methodology_weights``, so they can be updated without
touching code.  The constants below serve as in-code fallbacks when the
config key is absent.

Multiplier rationale (empirical basis from published pollster accuracy
research and 538 retrospectives):
  - phone:   1.15x — live-phone interviews have consistently lower house
             effects and better demographic targeting than other modes.
  - mixed:   1.05x — IVR+online or phone+online blends perform better than
             pure online but below live-phone.
  - online:  0.90x — online panels have larger house effects and panel-
             conditioning biases in recent cycles.
  - IVR:     0.85x — robocall-only polls have higher refusal rates and
             systematically underrepresent cell-phone-only households.
  - unknown: 1.00x — neutral; no information to adjust.

These multipliers are applied multiplicatively on top of the existing RMSE /
Silver Bulletin / grade-based quality weight.  A poll that has already been
downweighted by a poor pollster grade will be further adjusted by methodology.
"""

from __future__ import annotations

from copy import copy
from pathlib import Path

from src.propagation.propagate_polls import PollObservation

# ---------------------------------------------------------------------------
# Default multipliers (fallback when not loaded from config)
# ---------------------------------------------------------------------------

# Allowed methodology values (from scripts/tag_pollster_methodology.py)
_VALID_METHODOLOGIES: frozenset[str] = frozenset({"phone", "online", "IVR", "mixed", "unknown"})

# Default multiplier table.
# Keys are the exact strings written by the methodology tagger.
# Unknown/unrecognized strings fall back to 1.0 (neutral).
_DEFAULT_METHODOLOGY_WEIGHTS: dict[str, float] = {
    "phone": 1.15,
    "mixed": 1.05,
    "online": 0.90,
    "IVR": 0.85,
    "unknown": 1.00,
}

# Multiplier applied when a poll has no methodology tag at all.
# This is distinct from "unknown" (tagged but unresolved) — it covers polls
# from CSV rows where the methodology column was absent or blank.
_MISSING_METHODOLOGY_MULTIPLIER: float = 1.00


def methodology_to_multiplier(
    methodology: str | None,
    weights: dict[str, float] | None = None,
) -> float:
    """Return a quality multiplier for a given methodology string.

    Parameters
    ----------
    methodology:
        Methodology tag string (e.g. "phone", "online", "IVR", "mixed",
        "unknown").  None or empty string returns the missing-methodology
        neutral multiplier (1.0).
    weights:
        Custom multiplier table.  When None, uses ``_DEFAULT_METHODOLOGY_WEIGHTS``.
        Unrecognized methodology values fall back to 1.0 (neutral).

    Returns
    -------
    float
        Multiplier in approximately [0.5, 1.5].  The exact range depends on
        the weights table but defaults keep values between 0.85 and 1.15.
    """
    if not methodology:
        return _MISSING_METHODOLOGY_MULTIPLIER
    table = weights if weights is not None else _DEFAULT_METHODOLOGY_WEIGHTS
    # Neutral fallback for any methodology not in the table (e.g. future values)
    return table.get(methodology, 1.0)


def apply_methodology_weights(
    polls: list[PollObservation],
    methodologies: list[str | None],
    weights: dict[str, float] | None = None,
) -> list[PollObservation]:
    """Apply methodology-based quality multipliers to a list of polls.

    The multiplier is applied to ``n_sample`` multiplicatively, stacking on
    top of any quality weight already applied (RMSE / Silver Bulletin / grade).

    Parameters
    ----------
    polls:
        List of ``PollObservation`` objects.  Must be the same length as
        ``methodologies``.
    methodologies:
        Parallel list of methodology strings for each poll.  None or empty
        string is treated as missing methodology (neutral 1.0 multiplier).
    weights:
        Custom multiplier table.  When None, uses ``_DEFAULT_METHODOLOGY_WEIGHTS``.

    Returns
    -------
    list[PollObservation]
        New ``PollObservation`` copies with adjusted ``n_sample``.  All other
        fields are unchanged.

    Raises
    ------
    ValueError
        If ``polls`` and ``methodologies`` have different lengths.
    """
    if len(polls) != len(methodologies):
        raise ValueError(
            f"polls and methodologies must have the same length; "
            f"got {len(polls)} polls and {len(methodologies)} methodology entries"
        )

    result: list[PollObservation] = []
    for poll, method in zip(polls, methodologies):
        multiplier = methodology_to_multiplier(method, weights)
        n_effective = int(max(1, round(poll.n_sample * multiplier)))
        new_poll = copy(poll)
        new_poll.n_sample = n_effective
        result.append(new_poll)

    return result


def load_methodology_weights(params_path: Path) -> dict[str, float]:
    """Load methodology weight multipliers from ``prediction_params.json``.

    Returns ``_DEFAULT_METHODOLOGY_WEIGHTS`` when the file is missing, the
    ``poll_weighting.methodology_weights`` key is absent, or the JSON is
    malformed.  This ensures the pipeline degrades gracefully.

    Parameters
    ----------
    params_path:
        Absolute path to ``data/config/prediction_params.json``.
    """
    if not params_path.exists():
        return dict(_DEFAULT_METHODOLOGY_WEIGHTS)

    import json
    try:
        with params_path.open(encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return dict(_DEFAULT_METHODOLOGY_WEIGHTS)

    pw = data.get("poll_weighting", {})
    mw = pw.get("methodology_weights")
    if not isinstance(mw, dict):
        return dict(_DEFAULT_METHODOLOGY_WEIGHTS)

    # Merge config values over defaults so any missing keys still work
    merged = dict(_DEFAULT_METHODOLOGY_WEIGHTS)
    for k, v in mw.items():
        try:
            merged[k] = float(v)
        except (TypeError, ValueError):
            pass  # Skip malformed entries; keep default

    return merged
