import test from "node:test";
import assert from "node:assert/strict";
import { parseHouseLiveNavigation, houseLiveSelectionUrl } from "../lib/house-live-navigation.ts";
import { makeHouseLivePublication } from "./fixtures/house-live-publication.mjs";

test("current model, district and comparison links roundtrip without a saved-view feature", () => {
  const publication = makeHouseLivePublication();
  const selection = { raceId: publication.view.races[0].id, comparisonId: publication.catalog.entries[0].id };
  const url = houseLiveSelectionUrl(publication, selection);
  assert.match(url, /^\/\?chamber=house&edition=current&run=/);
  assert.deepEqual(parseHouseLiveNavigation(new URL(url, "https://example.invalid").search),
    { status: "valid", selection: { runId: publication.view.run.id, ...selection } });
  assert.deepEqual(parseHouseLiveNavigation("?chamber=house"),
    { status: "valid", selection: { runId: null, raceId: null, comparisonId: null } });
});

test("ambiguous, historical or actual-winner selections cannot be silently treated as current", () => {
  for (const search of ["?run=a&run=b", "?chamber=senate", "?edition=historical", "?year=2018", "?view=actual", "?run=", "?run=../private", "?race=2018:TX-23:house:regular", "?run=a&compare=a", "?compare=a&compare=b"])
    assert.equal(parseHouseLiveNavigation(search).status, "invalid", search);
});

test("generated links use only districts and comparison models in the displayed collection", () => {
  const publication = makeHouseLivePublication();
  assert.throws(() => houseLiveSelectionUrl(publication, { raceId: "2026:ZZ-99:house:regular" }), /district/);
  assert.throws(() => houseLiveSelectionUrl(publication, { comparisonId: "unknown" }), /comparison/);
  assert.throws(() => houseLiveSelectionUrl(publication, { comparisonId: publication.view.run.id }), /comparison/);
});
