"""Type validation orchestrator — imports from submodules for backwards compatibility.

The validation logic has been split into focused modules:
- type_coherence.py    — within/between type variance ratio
- type_stability.py    — subspace angle across time windows
- holdout_accuracy.py  — holdout Pearson r and RMSE metrics (all variants)
- validation_report.py — report generation and I/O orchestration

This module re-exports everything so existing callers continue to work without
any import changes.

Usage:
    python -m src.validation.validate_types
"""
from __future__ import annotations

import logging

from src.validation.holdout_accuracy import (  # noqa: F401
    RMSE_FLAG_THRESHOLD,
    holdout_accuracy,
    holdout_accuracy_county_prior,
    holdout_accuracy_county_prior_loo,
    holdout_accuracy_ridge,
    holdout_accuracy_ridge_augmented,
    rmse_by_super_type,
)
from src.validation.type_coherence import type_coherence  # noqa: F401
from src.validation.type_stability import type_stability  # noqa: F401
from src.validation.validation_report import generate_type_validation_report  # noqa: F401


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Type validation report")
    parser.add_argument("--shifts", default="data/shifts/county_shifts_multiyear.parquet")
    parser.add_argument("--assignments", default="data/communities/type_assignments.parquet")
    parser.add_argument("--covariance", default="data/covariance/type_covariance.parquet")
    parser.add_argument("--profiles", default="data/communities/type_profiles.parquet")
    parser.add_argument("--min-year", type=int, default=2008,
                        help="Min start year for training shifts (default: 2008, matching type discovery)")
    args = parser.parse_args()

    generate_type_validation_report(
        shift_parquet_path=args.shifts,
        type_assignments_path=args.assignments,
        type_covariance_path=args.covariance,
        type_profiles_path=args.profiles,
        min_year=args.min_year,
    )


if __name__ == "__main__":
    main()
