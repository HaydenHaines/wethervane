"""Compare two experiment runs via clustering agreement metrics.

Usage:
    python -m src.experiments.compare_runs --run-a <dir> --run-b <dir>
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def compare_runs(run_a_dir: Path, run_b_dir: Path) -> dict:
    """Compare two experiment runs.

    Loads assignments from both directories and computes:
    - Adjusted Rand Index (ARI)
    - Normalized Mutual Information (NMI)
    - Type correspondence matrix (crosstab of dominant types)

    Parameters
    ----------
    run_a_dir : Path
        Directory of experiment run A (must contain assignments.parquet).
    run_b_dir : Path
        Directory of experiment run B (must contain assignments.parquet).

    Returns
    -------
    dict with keys: ari, nmi, correspondence (as nested dict)
    """
    df_a = pd.read_parquet(run_a_dir / "assignments.parquet")
    df_b = pd.read_parquet(run_b_dir / "assignments.parquet")

    labels_a = df_a["dominant_type"].values
    labels_b = df_b["dominant_type"].values

    ari = adjusted_rand_score(labels_a, labels_b)
    nmi = normalized_mutual_info_score(labels_a, labels_b)

    # Build correspondence matrix
    correspondence = pd.crosstab(
        pd.Series(labels_a, name="run_a"),
        pd.Series(labels_b, name="run_b"),
    )

    return {
        "ari": float(ari),
        "nmi": float(nmi),
        "correspondence": correspondence.to_dict(),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare two experiment runs")
    parser.add_argument("--run-a", required=True, help="Path to run A directory")
    parser.add_argument("--run-b", required=True, help="Path to run B directory")
    parser.add_argument("--output", default=None, help="Output JSON path (optional)")
    args = parser.parse_args()

    result = compare_runs(Path(args.run_a), Path(args.run_b))

    print(f"ARI:  {result['ari']:.4f}")
    print(f"NMI:  {result['nmi']:.4f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
