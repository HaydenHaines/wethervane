"""
Build a blended-color GeoJSON for community type visualization.

Each census tract gets a pre-computed RGBA fill color that is the
membership-weighted average of the 7 community type colors. This produces
the "dominant community builds a polygon, blends where communities overlap"
effect:

  - Pure-dominant tracts show that community's color strongly
  - Mixed tracts show intermediate blends
  - High-entropy tracts (c7 generic baseline) appear muted gray

Community color palette (chosen to be visually distinct, non-political):
  c1 White rural homeowner    → sienna brown    (160,  82,  45)
  c2 Black urban              → cobalt blue     ( 30, 100, 200)
  c3 Knowledge worker         → emerald         ( 46, 160, 110)
  c4 Asian                    → violet          (128,  72, 176)
  c5 Working-class homeowner  → burnt orange    (210, 120,  40)
  c6 Hispanic low-income      → crimson         (200,  60,  60)
  c7 Generic suburban         → slate gray      (150, 150, 155)

Alpha encoding:
  Base alpha = 220 (slightly transparent).
  Applied uniformly — mixed tracts are not penalized with lower alpha
  because their blended color already communicates heterogeneity.

Output:
  data/viz/tract_memberships_k7_blended.geojson
  (same geometry as tract_memberships_k7.geojson, lightweight color properties added)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
VIZ_DIR = PROJECT_ROOT / "data" / "viz"

COMP_COLS = [f"c{k}" for k in range(1, 8)]

# Community colors as (R, G, B) tuples — perceptually distinct, accessible
COMMUNITY_COLORS = {
    "c1": (160,  82,  45),   # sienna — White rural homeowner
    "c2": ( 30, 100, 200),   # cobalt — Black urban
    "c3": ( 46, 160, 110),   # emerald — Knowledge worker
    "c4": (128,  72, 176),   # violet — Asian
    "c5": (210, 120,  40),   # burnt orange — Working-class homeowner
    "c6": (200,  60,  60),   # crimson — Hispanic low-income
    "c7": (150, 150, 155),   # slate gray — Generic suburban baseline
}

LABELS = {
    "c1": "White rural homeowner",
    "c2": "Black urban",
    "c3": "Knowledge worker",
    "c4": "Asian",
    "c5": "Working-class homeowner",
    "c6": "Hispanic low-income",
    "c7": "Generic suburban baseline",
}

BASE_ALPHA = 220
MAJORITY_THRESHOLD = 0.50  # above this → pure dominant color


def compute_blended_color(weights: dict[str, float]) -> list[int]:
    """
    Majority → solid color; plurality → top-2 blend.

    If the dominant community exceeds MAJORITY_THRESHOLD (50%), the tract
    renders as that community's pure color.  Otherwise the top-2 communities
    are renormalized and blended, capturing genuine transition zones without
    muddying areas that merely lack a supermajority.

    Returns [R, G, B, A] list suitable for deck.gl getFillColor.
    """
    sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    top2 = [(comp, w) for comp, w in sorted_items[:2] if w > 0]

    if not top2:
        return [128, 128, 128, BASE_ALPHA]

    dominant_comp, dominant_w = top2[0]

    if dominant_w >= MAJORITY_THRESHOLD:
        # Majority: render as pure community color
        r, g, b = COMMUNITY_COLORS[dominant_comp]
        return [r, g, b, BASE_ALPHA]

    # Plurality: blend top-2 only, renormalized to sum to 1
    total = sum(w for _, w in top2)
    r, g, b = 0.0, 0.0, 0.0
    for comp, w in top2:
        color = COMMUNITY_COLORS[comp]
        frac = w / total
        r += frac * color[0]
        g += frac * color[1]
        b += frac * color[2]

    return [int(round(r)), int(round(g)), int(round(b)), BASE_ALPHA]


def build_blended_geojson(source_path: Path, output_path: Path) -> None:
    log.info("Loading source GeoJSON: %s", source_path)
    with open(source_path) as f:
        geojson = json.load(f)

    features = geojson["features"]
    log.info("Processing %d features...", len(features))

    n_uninhabited = 0
    for feature in features:
        props = feature["properties"]

        if props.get("is_uninhabited", False):
            # Uninhabited tracts: transparent black
            props["fill_color"] = [0, 0, 0, 0]
            props["dominant_label"] = "uninhabited"
            n_uninhabited += 1
            continue

        weights = {c: float(props.get(c, 0.0)) for c in COMP_COLS}
        props["fill_color"] = compute_blended_color(weights)

        # Dominant community label for tooltip
        dominant = max(weights, key=weights.get)
        props["dominant_label"] = LABELS[dominant]
        props["dominant_weight"] = round(weights[dominant], 3)

        # Formatted weight summary for tooltip
        weight_lines = []
        for comp in sorted(weights, key=weights.get, reverse=True):
            if weights[comp] > 0.05:  # only show >5% components
                weight_lines.append(f"{comp}: {weights[comp]:.1%}")
        props["weight_summary"] = " | ".join(weight_lines)

    log.info("Processed: %d inhabited, %d uninhabited", len(features) - n_uninhabited, n_uninhabited)

    with open(output_path, "w") as f:
        json.dump(geojson, f)

    size_mb = output_path.stat().st_size / 1e6
    log.info("Saved blended GeoJSON: %s (%.1f MB)", output_path, size_mb)


def main() -> None:
    source = VIZ_DIR / "tract_memberships_k7.geojson"
    output = VIZ_DIR / "tract_memberships_k7_blended.geojson"

    if not source.exists():
        log.error("Source not found: %s", source)
        log.error("Run src/viz/build_tract_geojson.py first to generate the source GeoJSON.")
        return

    build_blended_geojson(source, output)

    # Print color legend
    print("\nCommunity color legend:")
    for comp, color in COMMUNITY_COLORS.items():
        hex_color = "#{:02x}{:02x}{:02x}".format(*color)
        print(f"  {comp}  {hex_color}  rgb{color}  {LABELS[comp]}")


if __name__ == "__main__":
    main()
