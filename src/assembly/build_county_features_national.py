"""Build a consolidated national county features file joining all assembled data sources.

Reads:
    data/assembled/county_acs_features.parquet         (3,144 counties × 15 cols)
    data/assembled/county_rcms_features.parquet        (3,141 counties × 8 cols)
    data/assembled/county_qcew_features.parquet        (12,768 rows county_fips×year × 12 cols)
    data/assembled/county_health_features.parquet      (3,143 counties × 33+ cols, expanded CHR)
    data/assembled/county_migration_features.parquet   (3,127 counties × 5 cols)
    data/assembled/county_urbanicity_features.parquet  (3,135 counties × 4 cols)
    data/assembled/county_sci_features.parquet         (3,200+ counties × 5 cols)
    data/assembled/county_broadband_features.parquet   (3,100+ counties × 6 cols)
    data/assembled/county_bea_features.parquet         (3,149 counties × 5 cols)
    data/assembled/county_cdc_mortality_features.parquet (3,078 counties × 9 cols)
    data/assembled/county_covid_features.parquet       (3,208 counties × 5 cols)
    data/assembled/va_disability_features.parquet      (3,100+ counties × 4 cols)
    data/assembled/usda_typology_features.parquet      (3,100+ counties × 14 cols)
    data/assembled/transportation_features.parquet     (3,100+ counties × 8 cols)

Outputs:
    data/assembled/county_features_national.parquet  (3,100+ counties × ~90 cols)

Broadband features (from build_acs_broadband_features.py; ACS B28002):
    pct_broadband, pct_no_internet, pct_satellite, pct_cable_fiber, broadband_gap

Derived ACS features (from build_county_acs_features.py):
    pct_white_nh, pct_black, pct_asian, pct_hispanic
    median_age, median_hh_income, log_median_hh_income
    pct_bachelors_plus, pct_graduate
    pct_owner_occupied, pct_wfh, pct_transit, pct_management
    pop_total

RCMS features (joined on county_fips):
    evangelical_share, mainline_share, catholic_share, black_protestant_share
    congregations_per_1000, religious_adherence_rate

QCEW features (industry mix, aggregated to 2023; joined on county_fips):
    manufacturing_share, government_share, healthcare_share, retail_share,
    construction_share, finance_share, hospitality_share, industry_diversity_hhi
    Note: top_industry (categorical) and avg_annual_pay (ACS overlap) are dropped.

CHR features (County Health Rankings; joined on county_fips):
    premature_death_rate, adult_smoking_pct, adult_obesity_pct,
    excessive_drinking_pct, uninsured_pct, primary_care_physicians_rate,
    mental_health_providers_rate, children_in_poverty_pct, insufficient_sleep_pct,
    physical_inactivity_pct, severe_housing_problems_pct, drive_alone_pct,
    life_expectancy, diabetes_prevalence_pct, poor_mental_health_days
    Note: median_household_income, high_school_completion_pct, some_college_pct
    are dropped (overlap with ACS).

Counties missing RCMS/QCEW/CHR data are imputed with state-level medians for
each feature, consistent with the strategy in build_features.py.

FIPS format: string "01001" throughout (zero-padded, 5 characters).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
ACS_PATH = PROJECT_ROOT / "data" / "assembled" / "county_acs_features.parquet"
RCMS_PATH = PROJECT_ROOT / "data" / "assembled" / "county_rcms_features.parquet"
QCEW_PATH = PROJECT_ROOT / "data" / "assembled" / "county_qcew_features.parquet"
CHR_PATH = PROJECT_ROOT / "data" / "assembled" / "county_health_features.parquet"
MIGRATION_PATH = PROJECT_ROOT / "data" / "assembled" / "county_migration_features.parquet"
URBANICITY_PATH = PROJECT_ROOT / "data" / "assembled" / "county_urbanicity_features.parquet"
SCI_PATH = PROJECT_ROOT / "data" / "assembled" / "county_sci_features.parquet"
BROADBAND_PATH = PROJECT_ROOT / "data" / "assembled" / "county_broadband_features.parquet"
BEA_PATH = PROJECT_ROOT / "data" / "assembled" / "county_bea_features.parquet"
CDC_MORTALITY_PATH = PROJECT_ROOT / "data" / "assembled" / "county_cdc_mortality_features.parquet"
COVID_PATH = PROJECT_ROOT / "data" / "assembled" / "county_covid_features.parquet"
VA_PATH = PROJECT_ROOT / "data" / "assembled" / "va_disability_features.parquet"
USDA_PATH = PROJECT_ROOT / "data" / "assembled" / "usda_typology_features.parquet"
TRANSPORT_PATH = PROJECT_ROOT / "data" / "assembled" / "transportation_features.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "assembled" / "county_features_national.parquet"

RCMS_FEATURE_COLS = [
    "evangelical_share",
    "mainline_share",
    "catholic_share",
    "black_protestant_share",
    "congregations_per_1000",
    "religious_adherence_rate",
]

# QCEW industry-mix features (top_industry dropped: categorical; avg_annual_pay: ACS overlap)
QCEW_FEATURE_COLS = [
    "manufacturing_share",
    "government_share",
    "healthcare_share",
    "retail_share",
    "construction_share",
    "finance_share",
    "hospitality_share",
    "industry_diversity_hhi",
]

# Migration features (IRS SOI inflow/outflow; derived from county_migration_features.parquet)
MIGRATION_FEATURE_COLS = [
    "net_migration_rate",
    "avg_inflow_income",
    "migration_diversity",
    "inflow_outflow_ratio",
]

# Urbanicity features (Census Gazetteer + ACS pop_total)
URBANICITY_FEATURE_COLS = [
    "log_pop_density",
    "land_area_sq_mi",
    "pop_per_sq_mi",
]

# SCI features (Facebook Social Connectedness Index)
SCI_FEATURE_COLS = [
    "network_diversity",
    "pct_sci_instate",
    "sci_top5_mean_dem_share",
    "sci_geographic_reach",
]

# CHR features (median_household_income, high_school_completion_pct, some_college_pct
# dropped: overlap with ACS; state_abbr and data_year are metadata, not features)
CHR_FEATURE_COLS = [
    "premature_death_rate",
    "adult_smoking_pct",
    "adult_obesity_pct",
    "excessive_drinking_pct",
    "uninsured_pct",
    "primary_care_physicians_rate",
    "mental_health_providers_rate",
    "children_in_poverty_pct",
    "insufficient_sleep_pct",
    "physical_inactivity_pct",
    "severe_housing_problems_pct",
    "drive_alone_pct",
    "life_expectancy",
    "diabetes_prevalence_pct",
    "poor_mental_health_days",
]

# Expanded CHR features (from chr_2024 raw data; separate list to handle
# graceful fallback if CHR hasn't been rebuilt with expanded measures yet)
CHR_EXPANDED_COLS = [
    "drug_overdose_deaths_rate",
    "suicide_rate",
    "firearm_fatalities_rate",
    "food_insecurity_pct",
    "social_associations_rate",
    "voter_turnout_pct",
    "disconnected_youth_pct",
    "residential_segregation",
    "homeownership_pct",
    "census_participation_pct",
    "free_reduced_lunch_pct",
    "alcohol_impaired_driving_deaths_pct",
    "injury_deaths_rate",
    "single_parent_households_pct",
    "homicide_rate",
]

# Broadband / internet access features (ACS B28002)
BROADBAND_FEATURE_COLS = [
    "pct_broadband",
    "pct_no_internet",
    "pct_satellite",
    "pct_cable_fiber",
    "broadband_gap",
]

# BEA income/GDP features (Bureau of Economic Analysis)
BEA_FEATURE_COLS = [
    "pci",
    "pci_growth",
    "gdp_per_capita",
    "gdp_growth",
]

# CDC mortality features (exclude reserved all-NaN cols: heart_disease_rate, cancer_rate, suicide_rate)
CDC_MORTALITY_FEATURE_COLS = [
    "drug_overdose_rate",
    "despair_death_rate",
    "covid_death_rate",
    "allcause_age_adj_rate",
    "excess_mortality_ratio",
]

# COVID vaccination features
COVID_FEATURE_COLS = [
    "vax_complete_pct",
    "vax_booster_pct",
    "vax_dose1_pct",
]

# VA disability features
VA_FEATURE_COLS = [
    "va_disability_per_1000",
    "va_disability_pct_100rated",
    "va_disability_pct_young",
]

# USDA economic typology features (binary flags + ordinal)
USDA_FEATURE_COLS = [
    "High_Farming_2025",
    "High_Mining_2025",
    "High_Manufacturing_2025",
    "High_Government_2025",
    "High_Recreation_2025",
    "Nonspecialized_2025",
    "Low_PostSecondary_Ed_2025",
    "Low_Employment_2025",
    "Population_Loss_2025",
    "Housing_Stress_2025",
    "Retirement_Destination_2025",
    "Persistent_Poverty_1721",
    "Industry_Dependence_2025",
]

# DOT transportation features (aggregated from tract-level)
TRANSPORT_FEATURE_COLS = [
    "transport_pop_density",
    "transport_job_density",
    "transport_intersection_density",
    "transport_pct_local_roads",
    "transport_broadband",
    "transport_dead_end_proportion",
    "transport_circuity_avg",
]


def _impute_state_medians(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Impute NaN values in cols using state-level medians (state derived from county_fips).

    Falls back to national median for counties whose entire state has NaN values
    (e.g., Connecticut 2022 planning regions which have no RCMS match because
    RCMS still uses the pre-2022 county structure).
    """
    df = df.copy()
    state_fips = df["county_fips"].str[:2]

    for col in cols:
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue

        state_medians = df.groupby(state_fips)[col].median()
        mask = df[col].isna()
        df.loc[mask, col] = state_fips[mask].map(state_medians)

        # Fall back to national median for counties where state median is also NaN
        still_missing = df[col].isna()
        if still_missing.any():
            national_median = df[col].median()
            df.loc[still_missing, col] = national_median
            log.info(
                "  %-35s  %d NaN → state imputed, then %d → national median (%.3f)",
                col,
                n_missing,
                still_missing.sum(),
                national_median,
            )
        else:
            n_remaining = df[col].isna().sum()
            log.info(
                "  %-35s  %d NaN → imputed %d, remaining %d",
                col,
                n_missing,
                n_missing - n_remaining,
                n_remaining,
            )

    return df


def build_national_features(
    acs: pd.DataFrame,
    rcms: pd.DataFrame,
    qcew: pd.DataFrame | None = None,
    chr_df: pd.DataFrame | None = None,
    migration: pd.DataFrame | None = None,
    urbanicity: pd.DataFrame | None = None,
    sci: pd.DataFrame | None = None,
    broadband: pd.DataFrame | None = None,
    bea: pd.DataFrame | None = None,
    cdc_mortality: pd.DataFrame | None = None,
    covid: pd.DataFrame | None = None,
    va: pd.DataFrame | None = None,
    usda: pd.DataFrame | None = None,
    transport: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join all assembled county feature sources into a single feature matrix.

    All sources are left-joined onto the ACS spine (which defines the county set).
    Missing values are imputed with state-level medians, falling back to national
    median for counties in states with no data for a given feature.

    Returns
    -------
    DataFrame with county_fips, pop_total, and all available feature columns.
    """
    # Validate FIPS format
    if not acs["county_fips"].str.len().eq(5).all():
        raise ValueError("ACS county_fips must be 5-char zero-padded strings")
    if not rcms["county_fips"].str.len().eq(5).all():
        raise ValueError("RCMS county_fips must be 5-char zero-padded strings")

    # Left join: keep all ACS counties, bring in RCMS where available
    rcms_cols = ["county_fips"] + RCMS_FEATURE_COLS
    merged = acs.merge(rcms[rcms_cols], on="county_fips", how="left")

    n_missing_rcms = merged[RCMS_FEATURE_COLS[0]].isna().sum()
    if n_missing_rcms > 0:
        log.info(
            "%d counties lack RCMS data — imputing with state-level medians",
            n_missing_rcms,
        )
        merged = _impute_state_medians(merged, RCMS_FEATURE_COLS)

    # ── QCEW industry features ───────────────────────────────────────────────
    if qcew is not None:
        if not qcew["county_fips"].str.len().eq(5).all():
            raise ValueError("QCEW county_fips must be 5-char zero-padded strings")

        # Aggregate to latest available year (2023)
        latest_year = qcew["year"].max()
        log.info("Aggregating QCEW to latest year: %d", latest_year)
        qcew_latest = qcew[qcew["year"] == latest_year][["county_fips"] + QCEW_FEATURE_COLS].copy()

        merged = merged.merge(qcew_latest, on="county_fips", how="left")

        n_missing_qcew = merged[QCEW_FEATURE_COLS[0]].isna().sum()
        if n_missing_qcew > 0:
            log.info(
                "%d counties lack QCEW data — imputing with state-level medians",
                n_missing_qcew,
            )
            merged = _impute_state_medians(merged, QCEW_FEATURE_COLS)

    # ── CHR health features ──────────────────────────────────────────────────
    if chr_df is not None:
        if not chr_df["county_fips"].str.len().eq(5).all():
            raise ValueError("CHR county_fips must be 5-char zero-padded strings")

        chr_cols = ["county_fips"] + CHR_FEATURE_COLS
        merged = merged.merge(chr_df[chr_cols], on="county_fips", how="left")

        n_missing_chr = merged[CHR_FEATURE_COLS[0]].isna().sum()
        if n_missing_chr > 0:
            log.info(
                "%d counties lack CHR data — imputing with state-level medians",
                n_missing_chr,
            )
            merged = _impute_state_medians(merged, CHR_FEATURE_COLS)

    # ── Migration features ───────────────────────────────────────────────────
    if migration is not None:
        if not migration["county_fips"].str.len().eq(5).all():
            raise ValueError("Migration county_fips must be 5-char zero-padded strings")

        mig_cols = ["county_fips"] + MIGRATION_FEATURE_COLS
        merged = merged.merge(migration[mig_cols], on="county_fips", how="left")

        n_missing_mig = merged[MIGRATION_FEATURE_COLS[0]].isna().sum()
        if n_missing_mig > 0:
            log.info(
                "%d counties lack migration data — imputing with state-level medians",
                n_missing_mig,
            )
            merged = _impute_state_medians(merged, MIGRATION_FEATURE_COLS)

    # ── Urbanicity features ──────────────────────────────────────────────────
    if urbanicity is not None:
        if not urbanicity["county_fips"].str.len().eq(5).all():
            raise ValueError("Urbanicity county_fips must be 5-char zero-padded strings")

        urb_cols = ["county_fips"] + URBANICITY_FEATURE_COLS
        merged = merged.merge(urbanicity[urb_cols], on="county_fips", how="left")

        n_missing_urb = merged[URBANICITY_FEATURE_COLS[0]].isna().sum()
        if n_missing_urb > 0:
            log.info(
                "%d counties lack urbanicity data — imputing with state-level medians",
                n_missing_urb,
            )
            merged = _impute_state_medians(merged, URBANICITY_FEATURE_COLS)

    # ── SCI social connectedness features ────────────────────────────────────
    if sci is not None:
        if not sci["county_fips"].str.len().eq(5).all():
            raise ValueError("SCI county_fips must be 5-char zero-padded strings")

        sci_cols = ["county_fips"] + SCI_FEATURE_COLS
        merged = merged.merge(sci[sci_cols], on="county_fips", how="left")

        n_missing_sci = merged[SCI_FEATURE_COLS[0]].isna().sum()
        if n_missing_sci > 0:
            log.info(
                "%d counties lack SCI data — imputing with state-level medians",
                n_missing_sci,
            )
            merged = _impute_state_medians(merged, SCI_FEATURE_COLS)

    # ── Broadband / internet access features ────────────────────────────────
    if broadband is not None:
        if not broadband["county_fips"].str.len().eq(5).all():
            raise ValueError("Broadband county_fips must be 5-char zero-padded strings")

        bb_cols = ["county_fips"] + BROADBAND_FEATURE_COLS
        merged = merged.merge(broadband[bb_cols], on="county_fips", how="left")

        n_missing_bb = merged[BROADBAND_FEATURE_COLS[0]].isna().sum()
        if n_missing_bb > 0:
            log.info(
                "%d counties lack broadband data — imputing with state-level medians",
                n_missing_bb,
            )
            merged = _impute_state_medians(merged, BROADBAND_FEATURE_COLS)

    # ── BEA income/GDP features ─────────────────────────────────────────────
    if bea is not None:
        if not bea["county_fips"].str.len().eq(5).all():
            raise ValueError("BEA county_fips must be 5-char zero-padded strings")

        bea_cols = ["county_fips"] + BEA_FEATURE_COLS
        available_bea = [c for c in bea_cols if c in bea.columns]
        merged = merged.merge(bea[available_bea], on="county_fips", how="left")

        actual_bea_feats = [c for c in BEA_FEATURE_COLS if c in merged.columns]
        if actual_bea_feats:
            n_missing = merged[actual_bea_feats[0]].isna().sum()
            if n_missing > 0:
                log.info(
                    "%d counties lack BEA data — imputing with state-level medians",
                    n_missing,
                )
                merged = _impute_state_medians(merged, actual_bea_feats)

    # ── CDC mortality features ──────────────────────────────────────────────
    if cdc_mortality is not None:
        if not cdc_mortality["county_fips"].str.len().eq(5).all():
            raise ValueError("CDC mortality county_fips must be 5-char zero-padded strings")

        cdc_cols = ["county_fips"] + CDC_MORTALITY_FEATURE_COLS
        available_cdc = [c for c in cdc_cols if c in cdc_mortality.columns]
        merged = merged.merge(cdc_mortality[available_cdc], on="county_fips", how="left")

        actual_cdc_feats = [c for c in CDC_MORTALITY_FEATURE_COLS if c in merged.columns]
        if actual_cdc_feats:
            n_missing = merged[actual_cdc_feats[0]].isna().sum()
            if n_missing > 0:
                log.info(
                    "%d counties lack CDC mortality data — imputing with state-level medians",
                    n_missing,
                )
                merged = _impute_state_medians(merged, actual_cdc_feats)

    # ── COVID vaccination features ──────────────────────────────────────────
    if covid is not None:
        if not covid["county_fips"].str.len().eq(5).all():
            raise ValueError("COVID county_fips must be 5-char zero-padded strings")

        covid_cols = ["county_fips"] + COVID_FEATURE_COLS
        available_covid = [c for c in covid_cols if c in covid.columns]
        merged = merged.merge(covid[available_covid], on="county_fips", how="left")

        actual_covid_feats = [c for c in COVID_FEATURE_COLS if c in merged.columns]
        if actual_covid_feats:
            n_missing = merged[actual_covid_feats[0]].isna().sum()
            if n_missing > 0:
                log.info(
                    "%d counties lack COVID vaccination data — imputing with state-level medians",
                    n_missing,
                )
                merged = _impute_state_medians(merged, actual_covid_feats)

    # ── VA disability features ──────────────────────────────────────────────
    if va is not None:
        if not va["county_fips"].str.len().eq(5).all():
            raise ValueError("VA county_fips must be 5-char zero-padded strings")

        va_cols = ["county_fips"] + VA_FEATURE_COLS
        available_va = [c for c in va_cols if c in va.columns]
        merged = merged.merge(va[available_va], on="county_fips", how="left")

        actual_va_feats = [c for c in VA_FEATURE_COLS if c in merged.columns]
        if actual_va_feats:
            n_missing = merged[actual_va_feats[0]].isna().sum()
            if n_missing > 0:
                log.info(
                    "%d counties lack VA disability data — imputing with state-level medians",
                    n_missing,
                )
                merged = _impute_state_medians(merged, actual_va_feats)

    # ── USDA economic typology features ─────────────────────────────────────
    if usda is not None:
        if not usda["county_fips"].str.len().eq(5).all():
            raise ValueError("USDA county_fips must be 5-char zero-padded strings")

        usda_cols = ["county_fips"] + USDA_FEATURE_COLS
        available_usda = [c for c in usda_cols if c in usda.columns]
        merged = merged.merge(usda[available_usda], on="county_fips", how="left")

        # USDA flags: fill missing with 0 (absence of flag = not classified)
        actual_usda_feats = [c for c in USDA_FEATURE_COLS if c in merged.columns]
        for col in actual_usda_feats:
            merged[col] = merged[col].fillna(0)

    # ── DOT transportation features ─────────────────────────────────────────
    if transport is not None:
        if not transport["county_fips"].str.len().eq(5).all():
            raise ValueError("Transport county_fips must be 5-char zero-padded strings")

        tr_cols = ["county_fips"] + TRANSPORT_FEATURE_COLS
        available_tr = [c for c in tr_cols if c in transport.columns]
        merged = merged.merge(transport[available_tr], on="county_fips", how="left")

        actual_tr_feats = [c for c in TRANSPORT_FEATURE_COLS if c in merged.columns]
        if actual_tr_feats:
            n_missing = merged[actual_tr_feats[0]].isna().sum()
            if n_missing > 0:
                log.info(
                    "%d counties lack transportation data — imputing with state-level medians",
                    n_missing,
                )
                merged = _impute_state_medians(merged, actual_tr_feats)

    # ── Expanded CHR features (if available in the assembled CHR file) ──────
    # These come from the same chr_df but are separate columns added by the
    # expanded fetch. Only merge if they exist in the input.
    if chr_df is not None:
        available_expanded = [c for c in CHR_EXPANDED_COLS if c in chr_df.columns]
        if available_expanded:
            exp_cols = ["county_fips"] + available_expanded
            # Avoid duplicate merge — only merge columns not already in merged
            new_cols = [c for c in available_expanded if c not in merged.columns]
            if new_cols:
                merged = merged.merge(
                    chr_df[["county_fips"] + new_cols], on="county_fips", how="left"
                )
                n_missing = merged[new_cols[0]].isna().sum()
                if n_missing > 0:
                    log.info(
                        "%d counties lack expanded CHR data — imputing with state-level medians",
                        n_missing,
                    )
                    merged = _impute_state_medians(merged, new_cols)
                log.info("Added %d expanded CHR features", len(new_cols))

    return merged.reset_index(drop=True)


def main() -> None:
    log.info("Loading ACS county features from %s", ACS_PATH)
    acs = pd.read_parquet(ACS_PATH)
    log.info("  %d counties × %d cols", len(acs), len(acs.columns))

    log.info("Loading RCMS county features from %s", RCMS_PATH)
    rcms = pd.read_parquet(RCMS_PATH)
    log.info("  %d counties × %d cols", len(rcms), len(rcms.columns))

    # Load QCEW (industry) features if available
    qcew: pd.DataFrame | None = None
    if QCEW_PATH.exists():
        log.info("Loading QCEW county features from %s", QCEW_PATH)
        qcew = pd.read_parquet(QCEW_PATH)
        log.info("  %d rows × %d cols (county × year)", len(qcew), len(qcew.columns))
    else:
        log.warning("QCEW features not found at %s — skipping", QCEW_PATH)

    # Load CHR (County Health Rankings) features if available
    chr_df: pd.DataFrame | None = None
    if CHR_PATH.exists():
        log.info("Loading CHR county features from %s", CHR_PATH)
        chr_df = pd.read_parquet(CHR_PATH)
        log.info("  %d counties × %d cols", len(chr_df), len(chr_df.columns))
    else:
        log.warning("CHR features not found at %s — skipping", CHR_PATH)

    # Load Migration features if available
    migration: pd.DataFrame | None = None
    if MIGRATION_PATH.exists():
        log.info("Loading Migration county features from %s", MIGRATION_PATH)
        migration = pd.read_parquet(MIGRATION_PATH)
        log.info("  %d counties × %d cols", len(migration), len(migration.columns))
    else:
        log.warning("Migration features not found at %s — skipping", MIGRATION_PATH)

    # Load Urbanicity features if available
    urbanicity: pd.DataFrame | None = None
    if URBANICITY_PATH.exists():
        log.info("Loading Urbanicity county features from %s", URBANICITY_PATH)
        urbanicity = pd.read_parquet(URBANICITY_PATH)
        log.info("  %d counties × %d cols", len(urbanicity), len(urbanicity.columns))
    else:
        log.warning("Urbanicity features not found at %s — skipping", URBANICITY_PATH)

    # Load SCI features if available
    sci: pd.DataFrame | None = None
    if SCI_PATH.exists():
        log.info("Loading SCI county features from %s", SCI_PATH)
        sci = pd.read_parquet(SCI_PATH)
        log.info("  %d counties × %d cols", len(sci), len(sci.columns))
    else:
        log.warning("SCI features not found at %s — skipping", SCI_PATH)

    # Load Broadband features if available
    broadband: pd.DataFrame | None = None
    if BROADBAND_PATH.exists():
        log.info("Loading Broadband county features from %s", BROADBAND_PATH)
        broadband = pd.read_parquet(BROADBAND_PATH)
        log.info("  %d counties × %d cols", len(broadband), len(broadband.columns))
    else:
        log.warning("Broadband features not found at %s — skipping", BROADBAND_PATH)

    # Load BEA income/GDP features if available
    bea: pd.DataFrame | None = None
    if BEA_PATH.exists():
        log.info("Loading BEA county features from %s", BEA_PATH)
        bea = pd.read_parquet(BEA_PATH)
        log.info("  %d counties × %d cols", len(bea), len(bea.columns))
    else:
        log.warning("BEA features not found at %s — skipping", BEA_PATH)

    # Load CDC mortality features if available
    cdc_mortality: pd.DataFrame | None = None
    if CDC_MORTALITY_PATH.exists():
        log.info("Loading CDC mortality features from %s", CDC_MORTALITY_PATH)
        cdc_mortality = pd.read_parquet(CDC_MORTALITY_PATH)
        log.info("  %d counties × %d cols", len(cdc_mortality), len(cdc_mortality.columns))
    else:
        log.warning("CDC mortality features not found at %s — skipping", CDC_MORTALITY_PATH)

    # Load COVID vaccination features if available
    covid: pd.DataFrame | None = None
    if COVID_PATH.exists():
        log.info("Loading COVID vaccination features from %s", COVID_PATH)
        covid = pd.read_parquet(COVID_PATH)
        log.info("  %d counties × %d cols", len(covid), len(covid.columns))
    else:
        log.warning("COVID features not found at %s — skipping", COVID_PATH)

    # Load VA disability features if available
    va: pd.DataFrame | None = None
    if VA_PATH.exists():
        log.info("Loading VA disability features from %s", VA_PATH)
        va = pd.read_parquet(VA_PATH)
        log.info("  %d counties × %d cols", len(va), len(va.columns))
    else:
        log.warning("VA disability features not found at %s — skipping", VA_PATH)

    # Load USDA typology features if available
    usda: pd.DataFrame | None = None
    if USDA_PATH.exists():
        log.info("Loading USDA typology features from %s", USDA_PATH)
        usda = pd.read_parquet(USDA_PATH)
        log.info("  %d counties × %d cols", len(usda), len(usda.columns))
    else:
        log.warning("USDA typology features not found at %s — skipping", USDA_PATH)

    # Load transportation features if available
    transport: pd.DataFrame | None = None
    if TRANSPORT_PATH.exists():
        log.info("Loading transportation features from %s", TRANSPORT_PATH)
        transport = pd.read_parquet(TRANSPORT_PATH)
        log.info("  %d counties × %d cols", len(transport), len(transport.columns))
    else:
        log.warning("Transportation features not found at %s — skipping", TRANSPORT_PATH)

    features = build_national_features(
        acs, rcms, qcew=qcew, chr_df=chr_df,
        migration=migration, urbanicity=urbanicity, sci=sci, broadband=broadband,
        bea=bea, cdc_mortality=cdc_mortality, covid=covid,
        va=va, usda=usda, transport=transport,
    )

    n_na_any = features.isnull().any(axis=1).sum()
    log.info(
        "Built %d county feature rows × %d columns | %d counties with ≥1 NaN",
        len(features),
        len(features.columns),
        n_na_any,
    )

    if n_na_any > 0:
        log.warning("Columns with remaining NaN:")
        for col in features.columns:
            n = features[col].isna().sum()
            if n > 0:
                log.warning("  %-35s  %d NaN", col, n)

    # Sanity check: FIPS format
    bad_fips = features[~features["county_fips"].str.match(r"^\d{5}$")]
    if len(bad_fips):
        log.error("%d rows with malformed county_fips: %s", len(bad_fips), bad_fips["county_fips"].tolist()[:5])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)
    log.info("Saved → %s", OUTPUT_PATH)

    # Summary stats for key features
    log.info("\nFeature summary (all counties):")
    summary_cols = ["pct_white_nh", "pct_black", "pct_hispanic", "pct_bachelors_plus",
                    "median_hh_income", "evangelical_share", "catholic_share",
                    "pct_broadband", "pct_no_internet"]
    for col in summary_cols:
        if col not in features.columns:
            continue
        q1, med, q3 = features[col].quantile([0.25, 0.5, 0.75])
        log.info("  %-30s  Q1=%.3f  median=%.3f  Q3=%.3f", col, q1, med, q3)


if __name__ == "__main__":
    main()
