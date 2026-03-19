"""Data contract definitions for Bedrock's two-layer community model.

These define the canonical column names and schemas for model outputs.
All pipeline scripts must produce outputs matching these contracts.
"""
from __future__ import annotations
from dataclasses import dataclass, field


# ── Layer 1: Community assignments ───────────────────────────────────────────

# Required columns in any Layer 1 community assignment output
LAYER1_REQUIRED_COLS = ["county_fips", "community_id"]

# Optional but standard columns
LAYER1_STANDARD_COLS = [
    "county_fips",      # str, 5-char zero-padded FIPS
    "community_id",     # int, 0-indexed community label from HAC
    "state_abbr",       # str, 'FL'/'GA'/'AL'
    "n_counties",       # int, size of community (filled in by describe step)
]


@dataclass
class Layer1Output:
    """Standard output from the community discovery step (Layer 1).

    Produced by: src/discovery/run_county_clustering.py
    Consumed by: src/models/type_classifier.py, src/db/build_database.py,
                 API, visualization
    """
    model_id: str
    k: int
    shift_type: str         # "logodds" or "raw"
    vote_share_type: str    # "total" or "twoparty"
    training_dims: int
    holdout_r: float        # validation holdout Pearson r
    assignment_file: str    # path to the parquet with county_fips, community_id


# ── Layer 2: Type assignments ─────────────────────────────────────────────────

# Required columns in any Layer 2 type assignment output
# Wide format: one row per community, one column per type
# type_weight_0, type_weight_1, ..., type_weight_{J-1}
LAYER2_REQUIRED_COLS = ["community_id", "dominant_type_id"]
# Plus: type_weight_{j} for j in range(J)


@dataclass
class Layer2Output:
    """Standard output from the type classification step (Layer 2).

    Produced by: src/models/type_classifier.py
    Consumed by: src/db/build_database.py, API, visualization
    """
    model_id: str
    j: int              # number of types
    communities_file: str   # path to parquet with type_weight_{j} columns
    type_profiles_file: str  # path to parquet: type_id, mean shift vector, description


# ── Naming conventions ────────────────────────────────────────────────────────

def layer1_output_path(model_id: str, base_dir: str = "data/models") -> str:
    return f"{base_dir}/versions/{model_id}/layer1_assignments.parquet"

def layer2_output_path(model_id: str, base_dir: str = "data/models") -> str:
    return f"{base_dir}/versions/{model_id}/layer2_type_assignments.parquet"

def type_profiles_path(model_id: str, base_dir: str = "data/models") -> str:
    return f"{base_dir}/versions/{model_id}/type_profiles.parquet"
