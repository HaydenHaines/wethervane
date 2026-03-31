"""Vocabulary tables and linguistic configuration for type naming.

All lookup tables, phrase dictionaries, and category labels live here.
No computation — just data.  Every naming module imports from this file.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Z-score thresholds
# ---------------------------------------------------------------------------
# These values are empirically tuned for the national 55-type model.
# Z_HIGH fires on rare/strongly-distinctive features (top ~15% of types).
# Z_MOD fires on moderately-distinctive features (top ~30% of types).
# Z_LOW is reserved for disambiguation only — low enough to split near-ties.
Z_HIGH = 1.2
Z_MOD = 0.6
Z_LOW = 0.3

# ---------------------------------------------------------------------------
# Minimum absolute values for race/ethnicity labels
# ---------------------------------------------------------------------------
# Race z-scores can fire even at relatively low absolute shares when the
# cross-type variance is compressed.  These floors prevent misleading labels.
# E.g. "Majority-Black" must mean the Black population is genuinely the majority,
# not just above-average relative to other (predominantly white) types.
_MIN_ABS: dict[str, float] = {
    "pct_black": 0.15,
    "pct_hispanic": 0.10,
    "pct_asian": 0.05,
}

# Per-label absolute minimums — override _MIN_ABS for specific positive labels.
# Key: (feature, positive_label) → minimum absolute value required.
_MIN_ABS_PER_LABEL: dict[tuple[str, str], float] = {
    # "Majority-Black" must mean the Black population is genuinely the majority.
    ("pct_black", "Majority-Black"): 0.50,
    # "Asian" label requires at least 10% Asian (not just z-score elevated)
    ("pct_asian", "Asian"): 0.10,
}

# ---------------------------------------------------------------------------
# Primary vocabulary — ordered by specificity (rare demographics first).
# Each entry: (feature, threshold, positive_label, negative_label)
# Labels are hyphenated single tokens so word counting works.
# Uses NATIONAL population-weighted z-scores to reflect genuine distinctiveness
# across the full 55-type national population.
#
# Ordering rationale:
# 1. Rare racial/ethnic composition distinguishes Black Belt, Hispanic, Asian types
# 2. Urbanicity — Urban vs Deep-Rural is the biggest structural split
# 3. Religion — evangelical/Catholic/secular is highly differentiating
# 4. Age — distinguishes retirement communities from young rural
# 5. Income — differentiates remaining types before education
# 6. Migration — growth vs decline
# 7. Education / professional class — last because many rural types share low-edu
# ---------------------------------------------------------------------------
_VOCAB: list[tuple[str, float, str, str]] = [
    # Rare racial/ethnic composition (with absolute minimums enforced).
    # Three-level pct_black system:
    #   z >= 3.0 (~>45% Black)  → "Majority-Black"  (extreme Black Belt core)
    #   z >= 1.2 (~>30% Black)  → "Black-Belt"       (traditional Black Belt)
    #   z >= 0.6 (~>20% Black)  → "Black-Belt"       (elevated minority share)
    # Using 3.0 keeps "Majority-Black" to ~3 types (the most extreme),
    # leaving "Black-Belt" for the moderate-to-high tier.
    ("pct_black",               3.0,    "Majority-Black", ""),
    ("pct_hispanic",            Z_HIGH, "Hispanic",       ""),
    ("pct_asian",               Z_HIGH, "Asian",          ""),
    ("pct_black",               Z_MOD,  "Black-Belt",     ""),
    ("pct_hispanic",            Z_MOD,  "Hispanic",       ""),

    # Urbanicity / density / transit
    ("log_pop_density",         Z_HIGH, "Urban",        "Deep-Rural"),
    ("pct_transit",             Z_HIGH, "Transit-Hub",  ""),
    ("log_pop_density",         Z_MOD,  "Suburban",     "Rural"),
    ("pct_transit",             Z_MOD,  "Transit",      ""),

    # Religion — ordered to maximise differentiation:
    # 1. Evangelical Z_HIGH: identifies the dominant rural Protestant bloc
    # 2. Catholic Z_HIGH: coastal/Hispanic types
    # 3. Mainline Z_HIGH: identifies specific mainline-dominant communities
    # 4. religious_adherence_rate: "Devout" (very high adherence) fires early;
    #    negative = "Low-Church" (NOT "Secular" — avoids contradiction with
    #    "Evangelical" since a community can be ethnically evangelical but
    #    have lower formal church participation).
    # 5. religious_adherence_rate Z_MOD: Churched/Unchurched
    # 6. black_protestant_share: placed AFTER adherence so that within-the-
    #    evangelical-blob types get "Low-Church"/"Unchurched" first, leaving
    #    "Black-Church" as a 4th token for further disambiguation.
    ("evangelical_share",       Z_HIGH, "Evangelical",  ""),
    ("catholic_share",          Z_HIGH, "Catholic",     ""),
    ("mainline_share",          Z_HIGH, "Mainline",     ""),
    ("religious_adherence_rate", Z_HIGH, "Devout",      ""),
    ("evangelical_share",       Z_MOD,  "Evangelical",  ""),
    ("catholic_share",          Z_MOD,  "Catholic",     ""),
    ("religious_adherence_rate", Z_MOD,  "Churched",    "Unchurched"),
    ("black_protestant_share",  Z_HIGH, "Black-Church", ""),

    # Age — retirement communities and young growth areas.
    # Before income so that rural types like "Deep-Rural Evangelical Retiree"
    # vs "Deep-Rural Evangelical Younger" are distinguished early.
    ("median_age",              Z_HIGH, "Retiree",      "Young"),
    ("median_age",              Z_MOD,  "Older",        "Younger"),

    # Income — after religion+age so the remaining types haven't already diverged
    ("median_hh_income",        Z_HIGH, "Affluent",     "Low-Income"),
    ("avg_inflow_income",       Z_HIGH, "Wealth-Magnet", ""),
    ("median_hh_income",        Z_MOD,  "Mid-Income",   "Blue-Collar"),

    # Migration / growth
    ("net_migration_rate",      Z_HIGH, "High-Growth",  "Declining"),
    ("inflow_outflow_ratio",    Z_HIGH, "Inflow",       "Outflow"),
    ("migration_diversity",     Z_HIGH, "Cosmopolitan", ""),
    ("net_migration_rate",      Z_MOD,  "Growing",      "Shrinking"),

    # Education / professional class — last to avoid Working-Class domination
    ("pct_bachelors_plus",      Z_HIGH, "College",      ""),
    ("pct_graduate",            Z_HIGH, "Grad-Degree",  ""),
    ("pct_management",          Z_HIGH, "Professional", ""),
    ("pct_bachelors_plus",      Z_MOD,  "Educated",     "Working-Class"),

    # Work / commute
    ("pct_wfh",                 Z_HIGH, "Remote-Work",  ""),
    ("pct_car",                 Z_MOD,  "Auto-Commute", ""),

    # Homeownership
    ("pct_owner_occupied",      Z_HIGH, "Homeowner",    "Renter"),

    # Land area (sprawling counties)
    ("land_area_sq_mi",         Z_HIGH, "Sprawling",    "Compact"),
]

# Feature families — only one token per family per type.
# Multiple vocab entries for the same feature family compete; the first that fires wins.
_FAMILY: dict[str, str] = {
    "pct_graduate":             "education",
    "pct_bachelors_plus":       "education",
    "pct_management":           "professional",
    "log_pop_density":          "density",
    "pop_per_sq_mi":            "density",
    "pct_wfh":                  "work",
    "pct_car":                  "commute",
    "evangelical_share":        "religion_ev",
    "catholic_share":           "religion_cat",
    "mainline_share":           "religion_ml",
    "black_protestant_share":   "religion_bp",
    "religious_adherence_rate":  "religion_adh",
    "congregations_per_1000":   "religion_cong",
    "pct_black":                "race_black",
    "pct_hispanic":             "race_hisp",
    "pct_asian":                "race_asian",
    "pct_white_nh":             "race_white",
    "median_hh_income":         "income",
    "avg_inflow_income":        "income_in",
    "median_age":               "age",
    "net_migration_rate":       "migration",
    "inflow_outflow_ratio":     "migration_io",
    "migration_diversity":      "migration_div",
    "pct_owner_occupied":       "ownership",
    "pct_transit":              "transit",
    "land_area_sq_mi":          "area",
    "pop_total":                "population",
    "n_counties":               "breadth",
}

# ---------------------------------------------------------------------------
# Super-type vocabulary
# ---------------------------------------------------------------------------
# Same structure as _VOCAB but tuned for aggregate profiles.
# Since super-types aggregate many fine types, z-scores are compressed — use
# lower thresholds so more features fire.
_SUPER_VOCAB: list[tuple[str, float, str, str]] = [
    # Race / ethnicity
    ("pct_black",               0.8,    "Black-Belt",    ""),
    ("pct_hispanic",            0.8,    "Hispanic",      ""),
    ("pct_asian",               0.8,    "Asian-Pacific", ""),
    # Urbanicity
    ("log_pop_density",         0.8,    "Urban",         "Rural"),
    ("log_pop_density",         0.4,    "Suburban",      "Exurban"),
    # Religion
    ("evangelical_share",       0.8,    "Evangelical",   ""),
    ("catholic_share",          0.8,    "Catholic",      ""),
    # Age
    ("median_age",              0.8,    "Retiree",       "Young"),
    # Income / education
    ("median_hh_income",        0.8,    "Affluent",      "Low-Income"),
    ("pct_bachelors_plus",      0.8,    "College",       ""),
    # Migration
    ("net_migration_rate",      0.8,    "High-Growth",   "Declining"),
    # Professional / WFH
    ("pct_wfh",                 0.8,    "Remote-Work",   ""),
    ("pct_management",          0.8,    "Professional",  ""),
]

# ---------------------------------------------------------------------------
# Disambiguation vocabulary
# ---------------------------------------------------------------------------
# Extended vocab for 4th-token disambiguation — different features from primary.
# Ordered to try the most differentiating features first.
# Public-facing labels — no "Blacker"/"Whiter" or relative ethnicity framing.
_DISAMBIG_VOCAB: list[tuple[str, float, str, str]] = [
    # Age is the single best differentiator within Rural/Deep-Rural groups
    ("median_age",              Z_LOW,  "Older",        "Younger"),
    # Migration: growth vs decline cleaves stagnant from boom rural types
    ("net_migration_rate",      Z_LOW,  "Growing",      "Shrinking"),
    ("inflow_outflow_ratio",    Z_LOW,  "Inflow",       "Outflow"),
    # Religion: evangelical intensity splits Deep-Rural types cleanly
    ("evangelical_share",       Z_LOW,  "Evangelical",  ""),
    ("catholic_share",          Z_LOW,  "Catholic",     ""),
    ("mainline_share",          Z_LOW,  "Mainline",     ""),
    ("black_protestant_share",  Z_LOW,  "Black-Church", ""),
    ("religious_adherence_rate", Z_LOW, "Devout",       "Unchurched"),
    # Income inflow: wealth magnet vs not
    ("avg_inflow_income",       Z_LOW,  "Wealth-Magnet", ""),
    ("median_hh_income",        Z_LOW,  "Higher-Inc",   "Lower-Inc"),
    # Education / professional
    ("pct_management",          Z_LOW,  "White-Collar", "Blue-Collar"),
    ("pct_bachelors_plus",      Z_LOW,  "Educated",     ""),
    # Minority composition — public-friendly framing
    ("pct_black",               Z_LOW,  "Minority-Mix", "Anglo-Majority"),
    ("pct_hispanic",            Z_LOW,  "Latino-Mix",   ""),
    # Housing / ownership
    ("pct_owner_occupied",      Z_LOW,  "Homeowner",    "Renter"),
    # Migration diversity
    ("migration_diversity",     Z_LOW,  "Cosmopolitan", "Insular"),
    # Spatial / density
    ("land_area_sq_mi",         Z_LOW,  "Sprawling",    "Compact"),
    ("log_pop_density",         Z_LOW,  "Denser",       "Sparser"),
    # Commute patterns
    ("pct_car",                 Z_LOW,  "Auto-Commute", ""),
    ("pct_wfh",                 Z_LOW,  "Remote-Work",  ""),
    ("pct_transit",             Z_LOW,  "Transit",      ""),
    # Type breadth
    ("congregations_per_1000",  Z_LOW,  "Many-Churches", ""),
    ("pop_total",               Z_LOW,  "Large",        "Small"),
    ("n_counties",              Z_LOW,  "Broad",        "Concentrated"),
]

# ---------------------------------------------------------------------------
# FIPS → state abbreviation lookup
# ---------------------------------------------------------------------------
# Complete mapping for all 50 states + DC.
# Source: US Census Bureau FIPS state codes.
_FIPS_TO_STATE: dict[int, str] = {
    1: "AL",
    2: "AK",
    4: "AZ",
    5: "AR",
    6: "CA",
    8: "CO",
    9: "CT",
    10: "DE",
    11: "DC",
    12: "FL",
    13: "GA",
    15: "HI",
    16: "ID",
    17: "IL",
    18: "IN",
    19: "IA",
    20: "KS",
    21: "KY",
    22: "LA",
    23: "ME",
    24: "MD",
    25: "MA",
    26: "MI",
    27: "MN",
    28: "MS",
    29: "MO",
    30: "MT",
    31: "NE",
    32: "NV",
    33: "NH",
    34: "NJ",
    35: "NM",
    36: "NY",
    37: "NC",
    38: "ND",
    39: "OH",
    40: "OK",
    41: "OR",
    42: "PA",
    44: "RI",
    45: "SC",
    46: "SD",
    47: "TN",
    48: "TX",
    49: "UT",
    50: "VT",
    51: "VA",
    53: "WA",
    54: "WV",
    55: "WI",
    56: "WY",
}
