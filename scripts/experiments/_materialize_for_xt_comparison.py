"""One-shot materializer: export type artifacts from DuckDB to parquet.

The `scripts/experiments/compare_xt_impact.py` script expects parquet files
under `data/communities/` and `data/covariance/`, but those directories are
gitignored on disk.  All data lives in `data/wethervane.duckdb`.

This script exports:
  data/communities/type_assignments.parquet — county_fips + type_*_score cols
  data/communities/type_profiles.parquet    — per-type demographic profile
  data/covariance/type_covariance.parquet   — J x J covariance matrix

It's intentionally one-shot / experiment-local and is not wired into the
production build pipeline.  Don't use for predict_2026 — use the canonical
build scripts there.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "wethervane.duckdb"


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)

    # --- type_scores: long → wide (county_fips × type_id) ---
    ts = con.execute("""
        select county_fips, type_id, score
        from type_scores
    """).fetchdf()
    n_types = int(ts["type_id"].max()) + 1
    print(f"Loaded {len(ts)} type_scores rows, J={n_types}")

    # Pivot to wide. Column names: type_{id}_score.
    wide = ts.pivot(index="county_fips", columns="type_id", values="score").fillna(0.0)
    wide.columns = [f"type_{i}_score" for i in wide.columns]
    wide = wide.reset_index()
    wide["county_fips"] = wide["county_fips"].astype(str).str.zfill(5)
    # Row-normalize to sum to 1 (handles any float drift).
    score_cols = [c for c in wide.columns if c.endswith("_score")]
    row_sums = wide[score_cols].sum(axis=1)
    wide[score_cols] = wide[score_cols].div(row_sums.where(row_sums > 0, 1.0), axis=0)

    out_ta = PROJECT_ROOT / "data" / "communities" / "type_assignments.parquet"
    out_ta.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(out_ta, index=False)
    print(f"  Wrote {out_ta} ({len(wide)} counties)")

    # --- type_covariance: long → J x J matrix ---
    cov = con.execute("""
        select type_i, type_j, value
        from type_covariance
    """).fetchdf()
    cov_mat = np.zeros((n_types, n_types), dtype=float)
    for _, row in cov.iterrows():
        cov_mat[int(row["type_i"]), int(row["type_j"])] = float(row["value"])
    cov_df = pd.DataFrame(cov_mat)
    out_cov = PROJECT_ROOT / "data" / "covariance" / "type_covariance.parquet"
    out_cov.parent.mkdir(parents=True, exist_ok=True)
    cov_df.to_parquet(out_cov, index=False)
    print(f"  Wrote {out_cov} ({cov_mat.shape})")

    # --- type_profiles: from `types` table ---
    # The forecast engine expects columns matching poll_enrichment._map_demographic_to_types:
    #   pct_bachelors_plus, pct_white_nh, pct_black, pct_hispanic, pct_asian,
    #   evangelical_share, median_age, log_pop_density
    tp = con.execute("""
        select type_id,
               pct_bachelors_plus,
               pct_white_nh,
               pct_black,
               pct_hispanic,
               pct_asian,
               evangelical_share,
               median_age,
               log_pop_density,
               pct_owner_occupied,
               median_hh_income,
               log_median_hh_income
        from types
        order by type_id
    """).fetchdf()
    out_tp = PROJECT_ROOT / "data" / "communities" / "type_profiles.parquet"
    out_tp.parent.mkdir(parents=True, exist_ok=True)
    tp.to_parquet(out_tp, index=False)
    print(f"  Wrote {out_tp} ({len(tp)} types)")

    con.close()


if __name__ == "__main__":
    main()
