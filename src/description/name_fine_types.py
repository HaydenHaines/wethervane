"""Fine type naming logic.

Generates human-readable 2–4 word names for individual electoral types
from their demographic z-score signatures.  Names are deterministic,
unique, and derived purely from data (no LLM calls).

Algorithm overview
------------------
1. Compute national population-weighted z-scores for all types.
2. For each type, scan _VOCAB and collect 2–3 descriptive tokens.
3. Assemble a 2-word base name: "Token1 Token2".
4. Disambiguate duplicates by promoting a 3rd or 4th token from extended vocab.
5. Any remaining duplicates try a political-lean suffix before ordinal fallback.

Electoral lean helpers (_compute_type_lean, _lean_label) live here because they
read shift data files and are specific to fine type disambiguation.
The core disambiguation algorithm lives in naming_scoring._disambiguate.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.description.naming_scoring import (
    _disambiguate,
    _top_tokens,
    compute_zscores,
)
from src.description.naming_vocab import _VOCAB

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
TYPE_PROFILES_PATH = _ROOT / "data" / "communities" / "type_profiles.parquet"
COUNTY_TYPE_ASSIGNMENTS_PATH = (
    _ROOT / "data" / "communities" / "county_type_assignments_full.parquet"
)


def _super_fallback_word(super_id: int) -> str:
    """Return a short fallback word for a super-type when vocab tokens are scarce.

    These are broad characterizations used only when vocab is exhausted.
    They are NOT the canonical super-type names (those come from name_super_types()).
    """
    _FALLBACKS = {
        0: "Heartland",
        1: "Diverse",
        2: "Mainstream",
        3: "Border",
        4: "Sunbelt",
    }
    return _FALLBACKS.get(super_id, "Mixed")


def name_types(
    profiles: pd.DataFrame | None = None,
    county_assignments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate descriptive display names for all fine electoral types.

    Names are 2–4 token strings derived from national population-weighted
    z-scores.  No state prefix is included — types now span the whole country
    so a state label would be misleading.

    Parameters
    ----------
    profiles:
        type_profiles DataFrame (55 rows × demographic columns).
        If None, loaded from ``data/communities/type_profiles.parquet``.
    county_assignments:
        county_type_assignments_full DataFrame; used to map each fine type
        to its super-type.  If None, loaded from disk.

    Returns
    -------
    DataFrame with columns ``[type_id, display_name]``, one row per type,
    sorted by ``type_id``.  All display names are unique.

    Side effects
    ------------
    When *profiles* is None (loading from disk), the ``display_name`` column
    is persisted back into ``type_profiles.parquet``.
    """
    save_to_disk = profiles is None

    if profiles is None:
        profiles = pd.read_parquet(TYPE_PROFILES_PATH)
    if county_assignments is None and COUNTY_TYPE_ASSIGNMENTS_PATH.exists():
        county_assignments = pd.read_parquet(COUNTY_TYPE_ASSIGNMENTS_PATH)

    # ---- Compute national z-scores ------------------------------------------
    # Use ALL types together as the reference population (national model).
    # This replaces the old within-state z-score approach.
    unique_features: list[str] = []
    seen_f: set[str] = set()
    for feat, *_ in _VOCAB:
        if feat not in seen_f:
            unique_features.append(feat)
            seen_f.add(feat)

    z_df = compute_zscores(profiles, unique_features, weight_col="pop_total")
    feat_cols = [c for c in z_df.columns if c != "type_id"]

    # ---- Determine each type's super-type -----------------------------------
    type_to_super: dict[int, int] = {}
    if county_assignments is not None and "super_type" in county_assignments.columns:
        mapping = (
            county_assignments.groupby("dominant_type")["super_type"]
            .agg(lambda x: int(x.mode().iloc[0]))
            .reset_index()
        )
        type_to_super = {
            int(r["dominant_type"]): int(r["super_type"])
            for _, r in mapping.iterrows()
        }

    # ---- Collect base token lists for every type ----------------------------
    # Collect up to 3 tokens for initial naming (token-only, no state prefix).
    _all_tokens: dict[int, list[str]] = {}
    raw_names: dict[int, str] = {}

    for _, row in profiles.iterrows():
        tid = int(row["type_id"])
        z_sub = z_df.loc[z_df["type_id"] == tid, feat_cols]
        if z_sub.empty:
            raw_names[tid] = f"Type {tid}"
            _all_tokens[tid] = []
            continue

        z_row = z_sub.iloc[0]
        # Collect up to 3 tokens from national z-scores
        tokens = _top_tokens(z_row, _VOCAB, n=3, raw_row=row)
        _all_tokens[tid] = tokens

        # If fewer than 2 tokens, use super-type as fallback context
        super_id = type_to_super.get(tid, -1)

        if len(tokens) >= 2:
            raw_names[tid] = f"{tokens[0]} {tokens[1]}"
        elif len(tokens) == 1:
            # One demographic token + super-type fallback word
            fallback = _super_fallback_word(super_id)
            raw_names[tid] = f"{tokens[0]} {fallback}"
        else:
            # No tokens at all — use super-type description
            fallback = _super_fallback_word(super_id)
            raw_names[tid] = f"Mixed {fallback}"

    # ---- Pre-promote additional tokens to resolve duplicate names -----------
    # Strategy: iteratively promote vocab tokens into names until each name
    # is unique or we hit the 4-word budget.
    #
    # For each duplicate group, scan the extended token list to find the first
    # position where types diverge.  Sub-groups that remain tied after a split
    # are recursively split at subsequent token positions.

    # Collect extended tokens (up to 8) for all types
    _ext_tokens: dict[int, list[str]] = {}
    for _, row in profiles.iterrows():
        tid = int(row["type_id"])
        z_sub = z_df.loc[z_df["type_id"] == tid, feat_cols]
        if z_sub.empty:
            _ext_tokens[tid] = _all_tokens.get(tid, [])
            continue
        z_row = z_sub.iloc[0]
        _ext_tokens[tid] = _top_tokens(z_row, _VOCAB, n=8, raw_row=row)

    def _try_promote_group(tids_in: list[int], current_names: dict[int, str]) -> None:
        """Recursively promote tokens to resolve a single duplicate group.

        Strategy:
        1. Prefer a position that makes ALL types in the group unique (full split).
        2. If no full split exists within the 4-word budget, use the first
           partial split and recurse on remaining sub-groups.
        3. For sub-groups already at 4 words, try REPLACING the last token
           with a more differentiating token that creates full uniqueness.
        """
        # Group the types by their current name
        by_name: dict[str, list[int]] = defaultdict(list)
        for tid in tids_in:
            by_name[current_names[tid]].append(tid)

        for name, group_tids in by_name.items():
            if len(group_tids) < 2:
                continue
            n_words = len(name.split())
            name_parts = name.split()
            start_pos = len(name_parts)    # next token index to try

            max_pos = max(len(_ext_tokens.get(t, [])) for t in group_tids)

            if n_words >= 4:
                # Can't append — try replacing the last token instead.
                name_prefix = " ".join(name.split()[:-1])  # drop last word
                replace_start = start_pos - 1  # re-try last position
                for pos in range(replace_start, max_pos):
                    candidate = {
                        tid: (_ext_tokens[tid][pos] if len(_ext_tokens[tid]) > pos else "")
                        for tid in group_tids
                    }
                    non_empty = [v for v in candidate.values() if v]
                    if len(set(non_empty)) == len(group_tids):
                        # Replace last token with a differentiating one
                        for tid in group_tids:
                            tok = candidate[tid]
                            if tok:
                                current_names[tid] = f"{name_prefix} {tok}"
                        return
                continue  # no replacement found; leave for DISAMBIG

            # Budget available: scan for best position
            # Prefer fully-unique over partially-unique splits
            first_partial_pos: int | None = None
            for pos in range(start_pos, max_pos):
                candidate = {
                    tid: (_ext_tokens[tid][pos] if len(_ext_tokens[tid]) > pos else "")
                    for tid in group_tids
                }
                non_empty = [v for v in candidate.values() if v]
                n_unique = len(set(non_empty))
                if n_unique < 2:
                    continue
                if n_unique == len(group_tids):
                    # Fully unique — use this position directly
                    for tid in group_tids:
                        tok = candidate[tid]
                        if tok:
                            current_names[tid] = f"{name} {tok}"
                    return  # all resolved, no recursion needed
                # Partially differentiating — remember the first such pos
                if first_partial_pos is None:
                    first_partial_pos = pos

            # No fully-unique position found — use the first partial split
            # and recurse on each sub-group
            if first_partial_pos is not None:
                pos = first_partial_pos
                candidate = {
                    tid: (_ext_tokens[tid][pos] if len(_ext_tokens[tid]) > pos else "")
                    for tid in group_tids
                }
                for tid in group_tids:
                    tok = candidate[tid]
                    if tok:
                        current_names[tid] = f"{name} {tok}"
                # Recurse to split any remaining sub-group duplicates
                _try_promote_group(group_tids, current_names)

    # Run up to 3 passes of pre-promotion (each pass may create new sub-groups)
    for _pre_pass in range(3):
        dupe_counts: Counter[str] = Counter(raw_names.values())
        dup_names = [n for n, c in dupe_counts.items() if c >= 2]
        if not dup_names:
            break
        for name in dup_names:
            tids_with_name = [t for t, n in raw_names.items() if n == name]
            _try_promote_group(tids_with_name, raw_names)

    # ---- Disambiguation -----------------------------------------------------
    display_names = _disambiguate(raw_names, profiles, z_df, feat_cols)

    # ---- Build result -------------------------------------------------------
    result = pd.DataFrame(
        [{"type_id": tid, "display_name": name}
         for tid, name in sorted(display_names.items())]
    )

    # ---- Persist ------------------------------------------------------------
    if save_to_disk and TYPE_PROFILES_PATH.exists():
        profiles_on_disk = pd.read_parquet(TYPE_PROFILES_PATH)
        name_map = dict(zip(result["type_id"], result["display_name"]))
        profiles_on_disk["display_name"] = profiles_on_disk["type_id"].map(name_map)
        profiles_on_disk.to_parquet(TYPE_PROFILES_PATH, index=False)
        log.info("Saved display_name to %s", TYPE_PROFILES_PATH)

    return result
