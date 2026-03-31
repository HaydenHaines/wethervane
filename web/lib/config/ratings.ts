/**
 * Canonical rating key definitions — must match the `rating` field values
 * returned by the API's /senate/overview and /forecast/* endpoints.
 *
 * If the API ever changes its rating vocabulary, update this file and all
 * consuming components will pick up the change automatically.
 */

export const RATING_KEYS = [
  "safe_d",
  "likely_d",
  "lean_d",
  "tossup",
  "lean_r",
  "likely_r",
  "safe_r",
] as const;

export type RatingKey = (typeof RATING_KEYS)[number];

/**
 * Ratings that represent genuinely competitive races.
 *
 * "Key races" are tossups and lean seats — the races where the model's
 * posterior uncertainty is highest and polling has the most predictive value.
 *
 * Typed as Set<string> so it can be used directly with .has() against API
 * string fields without requiring a cast at every call site.
 */
export const KEY_RATING_SET: Set<string> = new Set<RatingKey>(["tossup", "lean_d", "lean_r"]);
