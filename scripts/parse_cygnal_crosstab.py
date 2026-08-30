"""Compatibility shim for the legacy Cygnal crosstab module path.

The canonical public Cygnal parser lives at
``scripts/parse_cygnal_report.py:parse_cygnal_report``. This module stays on
disk only to redirect older imports and CLI invocations to that entrypoint.
"""

from __future__ import annotations

from pathlib import Path

from parse_cygnal_report import main as canonical_main
from parse_cygnal_report import parse_cygnal_report as canonical_parse_cygnal_report

__all__ = [
    "main",
    "parse_cygnal_report",
]


def parse_cygnal_report(path: str | Path) -> dict[str, object]:
    """Redirect legacy callers to the canonical Cygnal parser entrypoint."""
    return canonical_parse_cygnal_report(path)


def main() -> None:
    """Redirect legacy CLI usage to the canonical parser CLI."""
    canonical_main()


if __name__ == "__main__":
    main()
