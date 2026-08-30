"""Compatibility wrapper for the Cygnal crosstab parser.

The active implementation lives in ``parse_cygnal_report.py``. This module
preserves the documented ``scripts/parse_cygnal_crosstab.py`` entrypoint
without duplicating parser logic.
"""

from __future__ import annotations

from parse_cygnal_report import (
    extract_text,
    main,
    parse_cygnal_report,
    parse_cygnal_text,
    parse_demographic_vote_shares,
    parse_header,
    parse_sample_composition,
    two_party_dem_share,
    update_polls_csv,
)

__all__ = [
    "extract_text",
    "main",
    "parse_cygnal_report",
    "parse_cygnal_text",
    "parse_demographic_vote_shares",
    "parse_header",
    "parse_sample_composition",
    "two_party_dem_share",
    "update_polls_csv",
]


if __name__ == "__main__":
    main()
