import { RATING_COLORS, RATING_LABELS } from "./config/palette.ts";
import type { Rating } from "./types.ts";

export {
  DUSTY_INK,
  RATING_COLORS,
  RATING_LABELS,
  dustyInkChoropleth,
  marginToRating,
} from "./config/palette.ts";

export type { Rating } from "./types.ts";

export function ratingColor(rating: Rating): string {
  return RATING_COLORS[rating];
}

export function ratingLabel(rating: Rating): string {
  return RATING_LABELS[rating];
}
