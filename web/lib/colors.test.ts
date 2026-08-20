import assert from "node:assert/strict";
import test from "node:test";

import {
  DUSTY_INK as LEGACY_DUSTY_INK,
  marginToRating as legacyMarginToRating,
  ratingColor,
  ratingLabel,
} from "./colors.ts";
import { DUSTY_INK, RATING_COLORS, RATING_LABELS, marginToRating } from "./config/palette.ts";
import type { Rating } from "./types.ts";

const RATING_CASES: Array<{ demShare: number; rating: Rating }> = [
  { demShare: 0.20, rating: "safe_r" },
  { demShare: 0.41, rating: "likely_r" },
  { demShare: 0.45, rating: "lean_r" },
  { demShare: 0.50, rating: "tossup" },
  { demShare: 0.55, rating: "lean_d" },
  { demShare: 0.59, rating: "likely_d" },
  { demShare: 0.70, rating: "safe_d" },
];

test("legacy colors shim re-exports the canonical dusty ink palette", () => {
  assert.strictEqual(LEGACY_DUSTY_INK, DUSTY_INK);
  assert.deepEqual(LEGACY_DUSTY_INK, DUSTY_INK);
});

test("legacy rating helpers read canonical labels and colors", () => {
  for (const { rating } of RATING_CASES) {
    assert.equal(ratingColor(rating), RATING_COLORS[rating]);
    assert.equal(ratingLabel(rating), RATING_LABELS[rating]);
  }
});

test("legacy marginToRating matches the canonical thresholds", () => {
  for (const { demShare, rating } of RATING_CASES) {
    assert.equal(marginToRating(demShare), rating);
    assert.equal(legacyMarginToRating(demShare), rating);
    assert.equal(legacyMarginToRating(demShare), marginToRating(demShare));
  }
});
