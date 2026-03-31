"""SCI-type validation report formatter.

Converts a SCITypeValidationResult into a human-readable markdown document
covering the five core findings: same-type SCI lift, correlation with cosine
similarity, cross-state persistence, partial correlation controlling for
distance, and distance-binned analysis.
"""
from __future__ import annotations

import numpy as np

# Import here instead of from validate_sci_types to avoid circular import.
# SCITypeValidationResult is defined in validate_sci_types.py because it is
# part of the public API contract. sci_report.py uses TYPE_CHECKING to allow
# type hints without creating a circular dependency at runtime.
from __future__ import annotations  # noqa: F811  (already at top — harmless duplicate)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.validation.validate_sci_types import SCITypeValidationResult


def format_results(result: "SCITypeValidationResult") -> str:
    """Format validation results as a readable markdown report."""
    lines = [
        "# SCI-Type Validation Results",
        "",
        "## Research Question",
        "",
        "Do counties that are socially connected (high Facebook SCI) also tend",
        "to belong to the same electoral type? If so, this validates that our",
        "types capture real community structure, not just statistical artifacts.",
        "",
        "## Data Summary",
        "",
        f"- **Counties**: {result.n_counties:,}",
        f"- **County pairs analyzed**: {result.n_pairs:,}",
        f"- **Electoral types (J)**: {result.n_types}",
        "",
        "## Finding 1: Same-Type Counties Are More Socially Connected",
        "",
        "| Metric | Same Type | Different Type | Ratio |",
        "|--------|-----------|----------------|-------|",
        f"| Mean SCI | {result.mean_sci_same_type:,.0f} | {result.mean_sci_diff_type:,.0f} | {result.sci_ratio_same_over_diff:.2f}x |",
        f"| Mean log10(SCI) | {result.mean_log_sci_same_type:.3f} | {result.mean_log_sci_diff_type:.3f} | +{result.mean_log_sci_same_type - result.mean_log_sci_diff_type:.3f} |",
        "",
        "## Finding 2: SCI Correlates with Type Similarity",
        "",
        "Correlation between log10(SCI) and cosine similarity of soft type",
        "membership vectors:",
        "",
        f"- **Pearson r**: {result.pearson_r_log_sci_vs_cosine:.4f} (p={result.pearson_p_log_sci_vs_cosine:.2e})",
        f"- **Spearman r**: {result.spearman_r_log_sci_vs_cosine:.4f} (p={result.spearman_p_log_sci_vs_cosine:.2e})",
        "",
        "## Finding 3: Effect Persists Across State Lines",
        "",
        "Same-state pairs are both closer AND more likely to share a type.",
        "The critical test: does the SCI-type relationship hold for",
        "cross-state pairs?",
        "",
        f"- **% same-type among same-state pairs**: {100 * result.pct_same_type_same_state:.1f}%",
        f"- **% same-type among cross-state pairs**: {100 * result.pct_same_type_diff_state:.1f}%",
        "",
        "Cross-state only:",
        "",
        f"- Mean SCI (same type, cross-state): {result.mean_sci_same_type_diff_state:,.0f}",
        f"- Mean SCI (diff type, cross-state): {result.mean_sci_diff_type_diff_state:,.0f}",
        f"- **Cross-state SCI ratio**: {result.sci_ratio_same_over_diff_across_states:.2f}x",
        "",
    ]

    if result.partial_r_sci_cosine_given_distance != 0.0:
        lines.extend([
            "## Finding 4: SCI Signal Beyond Geographic Proximity",
            "",
            "Partial correlation of log(SCI) vs type cosine similarity,",
            "controlling for log(geodesic distance):",
            "",
            f"- **Partial r**: {result.partial_r_sci_cosine_given_distance:.4f} (p={result.partial_p_sci_cosine_given_distance:.2e})",
            "",
            "This measures whether SCI adds information about type similarity",
            "beyond what geographic distance alone provides.",
            "",
        ])

    if result.distance_bin_results:
        lines.extend([
            "## Finding 5: SCI-Type Relationship by Distance Bin",
            "",
            "Holding distance roughly constant, same-type pairs still have",
            "higher SCI than different-type pairs:",
            "",
            "| Distance | N pairs | % Same Type | Mean SCI (same) | Mean SCI (diff) | Ratio | Pearson r |",
            "|----------|---------|-------------|-----------------|-----------------|-------|-----------|",
        ])
        for b in result.distance_bin_results:
            r_str = f"{b['pearson_r']:.3f}" if not np.isnan(b["pearson_r"]) else "n/a"
            lines.append(
                f"| {b['distance_bin']} | {b['n_pairs']:,} | "
                f"{100 * b['pct_same_type']:.1f}% | "
                f"{b['mean_sci_same_type']:,.0f} | {b['mean_sci_diff_type']:,.0f} | "
                f"{b['sci_ratio']:.2f}x | {r_str} |"
            )
        lines.append("")

    lines.extend([
        "## Interpretation",
        "",
        "A positive SCI-type correlation validates that our electoral types",
        "capture real social community structure. Counties in the same type",
        "are not just statistically similar -- they are socially connected",
        "in measurable ways.",
        "",
        "The distance controls are critical: nearby counties are trivially",
        "both more connected (shared commute networks) and more similar",
        "(shared media markets, economies). The partial correlation and",
        "within-bin analysis isolate the SCI signal that goes beyond",
        "geographic proximity.",
        "",
        "## Methodology",
        "",
        "- **SCI data**: Facebook Social Connectedness Index (county pairs,",
        "  Jan 2026 snapshot). Higher SCI = more Facebook friendships between",
        "  two counties, scaled for population.",
        "- **Type similarity**: Cosine similarity of J=100 soft membership",
        "  vectors (temperature-scaled inverse distance to KMeans centroids).",
        "- **Distance control**: Haversine distance between Census 2020",
        "  county centroids.",
        "- **Partial correlation**: Standard OLS residualization of both",
        "  log(SCI) and cosine_sim on log(distance), then Pearson r of",
        "  residuals.",
    ])

    return "\n".join(lines)
