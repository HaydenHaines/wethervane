/**
 * Map palette — re-exports the canonical super-type color palette from
 * lib/config/palette.ts so map components can import from a local path
 * without duplicating the definitions.
 *
 * Also re-exports the choropleth interpolation function used for
 * partisan-lean coloring.
 */

export {
  SUPER_TYPE_COLORS as PALETTE,
  getSuperTypeColor as getColorForSuperType,
  dustyInkChoropleth,
  DUSTY_INK,
  RATING_COLORS,
} from "@/lib/config/palette";
