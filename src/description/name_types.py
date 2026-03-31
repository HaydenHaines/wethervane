"""Type naming logic — split across submodules, re-exported from here for backwards compatibility.

Algorithm overview
------------------
1. Compute national population-weighted z-scores for all demographic features.
   (All types, national reference — NOT within-state.)
2. For each type, scan an ordered vocabulary and collect 2-3 descriptive tokens.
3. Assemble a 2-word base name: "Token1 Token2".
4. Disambiguate duplicates by promoting a 3rd or 4th token from extended vocab.
5. Any remaining duplicates try a political-lean suffix before ordinal fallback.

Submodule responsibilities
--------------------------
- naming_vocab.py     — vocabulary tables, threshold constants, phrase dictionaries
- naming_scoring.py   — z-score computation, _get_label, _top_tokens
- name_fine_types.py  — fine type naming (name_types function)
- name_super_types.py — super-type naming (name_super_types function)

Usage (CLI)::

    python -m src.description.name_types
"""
from __future__ import annotations

import logging

from src.description.name_fine_types import name_types
from src.description.name_super_types import name_super_types
from src.description.naming_scoring import compute_zscores

__all__ = ["compute_zscores", "name_types", "name_super_types"]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Step 1: name fine types
    result = name_types()
    print(f"\nGenerated {len(result)} type names:\n")
    for _, row in result.iterrows():
        print(f"  Type {int(row.type_id):2d}: {row.display_name}")
    n_unique = result["display_name"].nunique()
    print(f"\n{n_unique}/{len(result)} unique names.")
    if n_unique < len(result):
        dupes = result[result["display_name"].duplicated(keep=False)]
        print("\nDUPLICATES:")
        print(dupes.to_string())

    # Step 2: name super-types
    print("\n--- Super-type names ---")
    super_result = name_super_types(fine_type_names=result)
    for _, row in super_result.iterrows():
        print(f"  Super {int(row.super_type_id)}: {row.display_name}")


if __name__ == "__main__":
    main()
