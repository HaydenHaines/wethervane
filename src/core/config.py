"""Central configuration loader for the WetherVane pipeline.

Reads config/model.yaml. All pipeline scripts should import from here
rather than hardcoding constants.

Usage:
    from src.core import config
    states = config.STATES          # {"FL": "12", "GA": "13", "AL": "01"}
    pres_pairs = config.PRES_PAIRS  # [("00", "04"), ...]
"""
from __future__ import annotations

from pathlib import Path
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _ROOT / "config" / "model.yaml"


def load(path: Path | None = None) -> dict:
    """Load and return the raw config dict."""
    p = path or _DEFAULT_CONFIG
    with open(p) as f:
        return yaml.safe_load(f)


# ── Eagerly resolved constants (import these directly) ──────────────────────

_cfg = load()

STATES: dict[str, str] = _cfg["geography"]["state_fips"]          # abbr → fips prefix
STATE_ABBR: dict[str, str] = {v: k for k, v in STATES.items()}   # fips prefix → abbr
AL_FIPS_PREFIX: str = _cfg["geography"]["al_fips_prefix"]

PRES_YEARS: list[int] = _cfg["election"]["presidential_years"]
GOV_YEARS: list[int] = _cfg["election"]["governor_years"]
SENATE_YEARS: list[int] = _cfg["election"]["senate_years"]

PRES_PAIRS: list[tuple[str, str]] = [
    (str(a)[-2:].zfill(2), str(b)[-2:].zfill(2))
    for a, b in _cfg["election"]["presidential_pairs"]
]
GOV_PAIRS: list[tuple[str, str]] = [
    (str(a)[-2:].zfill(2), str(b)[-2:].zfill(2))
    for a, b in _cfg["election"]["governor_pairs"]
]
SENATE_PAIRS: list[tuple[str, str]] = [
    (str(a)[-2:].zfill(2), str(b)[-2:].zfill(2))
    for a, b in _cfg["election"]["senate_pairs"]
]
HOLDOUT_PRES_PAIRS: list[tuple[str, str]] = [
    (str(a)[-2:].zfill(2), str(b)[-2:].zfill(2))
    for a, b in _cfg["election"]["holdout_pairs"]["presidential"]
]

VOTE_SHARE_TYPE: str = _cfg["election"]["vote_share_type"]   # "total" or "twoparty"
SHIFT_TYPE: str = _cfg["election"]["shift_type"]             # "logodds" or "raw"
LOGODDS_EPSILON: float = _cfg["election"]["logodds_epsilon"]

PRES_FILES: dict[str, str] = _cfg["data"]["presidential_files"]
GOV_FILES: dict[str, str] = _cfg["data"]["governor_files"]
SENATE_FILES: dict[str, str] = _cfg["data"]["senate_files"]
SPINE_FILE: str = _cfg["data"]["spine_file"]
