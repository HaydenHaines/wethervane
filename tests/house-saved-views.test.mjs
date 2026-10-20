import assert from "node:assert/strict";
import test from "node:test";
import {
  HOUSE_SAVED_VIEWS_STORAGE_KEY,
  MAX_HOUSE_SAVED_VIEWS,
  createHouseSavedView,
  parseHouseSavedViews,
  serializeHouseSavedViews,
  houseSavedViewAvailability,
  verifyHouseSavedView,
} from "../lib/house-saved-views.ts";
import { fixtureSha256, makeHousePublication } from "./fixtures/house.mjs";

function fixture(options = {}) {
  const publication = makeHousePublication(options);
  const result = publication.result;
  const selection = {
    raceId: result.races.find((race) => race.district.districtKey === "CA-01")
      .id,
    mapView: "forecast",
  };
  const view = createHouseSavedView("My House view", publication, selection, {
    id: "local-house-view-1",
    savedAt: "2026-10-20T12:00:00.000Z",
  });
  const catalog = {
    schemaVersion: 1,
    defaultRunId: result.run.id,
    selectionNote: "Synthetic test catalog.",
    runs: [
      {
        ...result.run,
        resultSha256: publication.resultSha256,
        raceCount: result.races.length,
        available: true,
        unavailableReason: null,
      },
    ],
  };
  return { publication, selection, view, catalog };
}
const stored = (views) => JSON.stringify({ schemaVersion: 1, views });

function comparisonFixture(mainPublication) {
  const comparison = structuredClone(mainPublication);
  comparison.result.run.id = "synthetic-house-comparison-2022-16";
  comparison.result.run.configurationId = "synthetic-house-comparison-config";
  comparison.result.run.configurationSha256 = fixtureSha256(
    comparison.result.run.configurationId,
  );
  comparison.result.run.label = "Synthetic comparison model";
  comparison.result.metadata.configurationId = comparison.result.run.configurationId;
  comparison.resultSha256 = fixtureSha256(comparison.result);
  return comparison;
}

function catalogRun(publication, overrides = {}) {
  return {
    ...publication.result.run,
    resultSha256: publication.resultSha256,
    raceCount: publication.result.races.length,
    available: true,
    unavailableReason: null,
    ...overrides,
  };
}

test("versioned local records round-trip exact result, map, configuration and selection bindings", () => {
  const { view, publication, selection, catalog } = fixture();
  assert.equal(
    HOUSE_SAVED_VIEWS_STORAGE_KEY,
    "ballotmancy.house.saved-views.v1",
  );
  assert.deepEqual(parseHouseSavedViews(serializeHouseSavedViews([view])), [
    view,
  ]);
  assert.equal(serializeHouseSavedViews([view]), stored([view]));
  assert.equal("comparison" in view, false);
  assert.equal(
    view.configurationSha256,
    publication.result.run.configurationSha256,
  );
  assert.equal(view.resultSha256, publication.resultSha256);
  assert.equal(view.year, 2022);
  assert.equal(view.mapVintage, "house_election_map_2022");
  assert.deepEqual(houseSavedViewAvailability(view, catalog), {
    status: "catalog_compatible",
    reason: null,
    verification: "catalog_only",
  });
  assert.deepEqual(verifyHouseSavedView(view, publication), {
    status: "verified",
    reason: null,
    verification: "exact_result",
    selection,
  });
});

test("same-year comparisons round-trip and restore only with the exact second result", () => {
  const { publication, selection, view, catalog } = fixture();
  const comparison = comparisonFixture(publication);
  const compared = createHouseSavedView(
    "My House comparison",
    publication,
    { ...selection, comparison },
    { id: "local-house-comparison-1", savedAt: view.savedAt },
  );
  assert.deepEqual(compared.comparison, {
    runId: comparison.result.run.id,
    resultSha256: comparison.resultSha256,
    configurationSha256: comparison.result.run.configurationSha256,
    year: comparison.result.run.year,
    mapVintage: comparison.result.metadata.mapVintage,
    runLabel: comparison.result.run.label,
  });
  assert.deepEqual(parseHouseSavedViews(serializeHouseSavedViews([compared])), [
    compared,
  ]);
  assert.equal(verifyHouseSavedView(compared, publication).status, "unavailable");
  assert.equal(
    verifyHouseSavedView(compared, publication, comparison).status,
    "verified",
  );
  const bothCatalogRuns = {
    ...catalog,
    runs: [catalogRun(publication), catalogRun(comparison)],
  };
  assert.equal(
    houseSavedViewAvailability(compared, bothCatalogRuns).status,
    "catalog_compatible",
  );
});

test("a saved comparison fails closed when its second loaded result is missing or changed", () => {
  const { publication, selection, view } = fixture();
  const comparison = comparisonFixture(publication);
  const compared = createHouseSavedView(
    "Pinned comparison",
    publication,
    { ...selection, comparison },
    { id: "pinned-comparison", savedAt: view.savedAt },
  );
  assert.match(
    verifyHouseSavedView(compared, publication).reason,
    /exact second result to be loaded/,
  );
  const changedRunId = structuredClone(comparison);
  changedRunId.result.run.id = "synthetic-house-replacement-2022-16";
  const changedResultHash = {
    ...comparison,
    resultSha256: "a".repeat(64),
  };
  const changedConfiguration = structuredClone(comparison);
  changedConfiguration.result.run.configurationSha256 = "b".repeat(64);
  const otherYear = makeHousePublication({ year: 2018 });
  const otherMap = structuredClone(comparison);
  otherMap.result.metadata.mapVintage = "house_election_map_2018";
  for (const race of otherMap.result.races)
    race.district.mapVintage = "house_election_map_2018";
  for (const changed of [
    changedRunId,
    changedResultHash,
    changedConfiguration,
    otherYear,
    otherMap,
  ]) {
    assert.equal(verifyHouseSavedView(compared, publication, changed).status, "unavailable");
  }
});

test("comparison catalog bindings detect a withdrawn or changed reference result", () => {
  const { publication, selection, view, catalog } = fixture();
  const comparison = comparisonFixture(publication);
  const compared = createHouseSavedView(
    "Catalog-bound comparison",
    publication,
    { ...selection, comparison },
    { id: "catalog-comparison", savedAt: view.savedAt },
  );
  const main = catalogRun(publication);
  for (const reference of [
    null,
    catalogRun(comparison, {
      available: false,
      resultSha256: null,
      unavailableReason: "Comparison result withdrawn for review.",
    }),
    catalogRun(comparison, { resultSha256: "c".repeat(64) }),
    catalogRun(comparison, { configurationSha256: "d".repeat(64) }),
    catalogRun(comparison, { year: 2018 }),
  ]) {
    const status = houseSavedViewAvailability(compared, {
      ...catalog,
      runs: reference ? [main, reference] : [main],
    });
    assert.equal(status.status, "unavailable");
    assert.ok(status.reason);
  }
  assert.match(
    houseSavedViewAvailability(compared, {
      ...catalog,
      runs: [main, catalogRun(comparison, {
        available: false,
        resultSha256: null,
        unavailableReason: "Comparison result withdrawn for review.",
      })],
    }).reason,
    /withdrawn for review/,
  );
});

test("missing and unavailable catalog runs remain stored with explicit preflight reasons", () => {
  const { view, catalog } = fixture();
  for (const altered of [
    null,
    { ...catalog, runs: [] },
    {
      ...catalog,
      runs: [
        {
          ...catalog.runs[0],
          available: false,
          resultSha256: null,
          unavailableReason: "Source release is held for review.",
        },
      ],
    },
  ]) {
    const parsed = parseHouseSavedViews(stored([view]));
    assert.deepEqual(parsed, [view]);
    const status = houseSavedViewAvailability(parsed[0], altered);
    assert.equal(status.status, "unavailable");
    assert.ok(status.reason);
    assert.equal(status.verification, "catalog_only");
    assert.deepEqual(parseHouseSavedViews(serializeHouseSavedViews(parsed)), [
      view,
    ]);
  }
  assert.match(
    houseSavedViewAvailability(view, {
      ...catalog,
      runs: [
        {
          ...catalog.runs[0],
          available: false,
          unavailableReason: "Source release is held for review.",
        },
      ],
    }).reason,
    /held for review/,
  );
});

test("changed fingerprints, configuration or year never silently rebind a stale saved view", () => {
  const { view, catalog, publication } = fixture();
  for (const changes of [
    { resultSha256: "0".repeat(64) },
    { configurationSha256: "1".repeat(64) },
    { year: 2018 },
  ]) {
    const altered = { ...catalog, runs: [{ ...catalog.runs[0], ...changes }] };
    assert.equal(
      houseSavedViewAvailability(view, altered).status,
      "unavailable",
    );
  }
  for (const change of [
    (v) => (v.resultSha256 = "0".repeat(64)),
    (v) => (v.result.run.configurationSha256 = "0".repeat(64)),
    (v) => (v.result.run.id = "different-house-run"),
  ]) {
    const loaded = structuredClone(publication);
    change(loaded);
    const result = verifyHouseSavedView(view, loaded);
    assert.equal(result.status, "unavailable");
    assert.ok(result.reason);
    assert.equal("selection" in result, false);
  }
  assert.deepEqual(parseHouseSavedViews(stored([view])), [view]);
});

test("catalog compatibility is not race or map verification", () => {
  const { view, catalog, publication } = fixture();
  for (const changed of [
    { ...view, mapVintage: "house_election_map_2018" },
    { ...view, raceId: "2022:CA-99:house:regular" },
  ]) {
    assert.equal(
      houseSavedViewAvailability(changed, catalog).status,
      "catalog_compatible",
    );
    const exact = verifyHouseSavedView(changed, publication);
    assert.equal(exact.status, "unavailable");
    assert.equal(exact.verification, "exact_result");
    assert.equal("selection" in exact, false);
    assert.deepEqual(parseHouseSavedViews(stored([changed])), [changed]);
  }
});

test("restoration verifies dated district and eligible race but does not require a map", () => {
  const { view, publication } = fixture();
  assert.equal(
    verifyHouseSavedView(view, { ...publication, geography: null }).status,
    "verified",
  );
  const changed = structuredClone(publication);
  changed.result.races.find(
    (race) => race.id === view.raceId,
  ).district.mapVintage = "house_election_map_2018";
  assert.equal(verifyHouseSavedView(view, changed).status, "unavailable");
  const excluded = structuredClone(publication);
  excluded.result.coverage.find((row) => row.raceId === view.raceId).eligible =
    false;
  assert.equal(verifyHouseSavedView(view, excluded).status, "unavailable");
  assert.equal(
    verifyHouseSavedView(view, {
      result: {},
      resultSha256: publication.resultSha256,
    }).status,
    "unavailable",
  );
});

test("actual-winner selection requires historical mode and preserves an unknown observed winner", () => {
  const { publication, selection, view, catalog } = fixture();
  const actual = createHouseSavedView(
    "Actual winners",
    publication,
    { ...selection, mapView: "actual" },
    { id: "actual-view", savedAt: view.savedAt },
  );
  assert.equal(verifyHouseSavedView(actual, publication).status, "verified");
  const absent = structuredClone(publication);
  delete absent.result.races.find((race) => race.id === view.raceId).observed;
  assert.equal(verifyHouseSavedView(actual, absent).status, "verified");
  const unknown = createHouseSavedView("Unknown actual winner", absent, {
    ...selection,
    mapView: "actual",
  });
  assert.equal(unknown.mapView, "actual");
  assert.equal(
    verifyHouseSavedView(unknown, absent).selection.mapView,
    "actual",
  );
  assert.equal(
    absent.result.races.find((race) => race.id === view.raceId).observed,
    undefined,
  );
  const current = structuredClone(publication);
  current.result.run.historical = false;
  assert.equal(verifyHouseSavedView(actual, current).status, "unavailable");
  assert.equal(
    houseSavedViewAvailability(actual, {
      ...catalog,
      runs: [{ ...catalog.runs[0], historical: false }],
    }).status,
    "unavailable",
  );
});

test("malformed persisted data is safe, bounded and cannot add unknown fields", () => {
  const { view } = fixture();
  for (const input of [
    null,
    undefined,
    42,
    "{broken",
    "null",
    "[]",
    JSON.stringify({ schemaVersion: 2, views: [view] }),
    " ".repeat(128001),
  ])
    assert.deepEqual(parseHouseSavedViews(input), []);
  for (const changes of [
    { savedAt: "not-a-date" },
    { savedAt: "2026-02-30T00:00:00.000Z" },
    { year: 2020 },
    { mapView: "other" },
    { resultSha256: "wrong" },
    { label: "bad\nname" },
    { id: "" },
    { raceId: "2022:CA-1:house:regular" },
    { comparison: null },
    { comparison: { runId: "invalid" } },
    { comparison: { runId: "second", resultSha256: "bad" } },
    {
      comparison: {
        runId: "second",
        resultSha256: "e".repeat(64),
        configurationSha256: "f".repeat(64),
        year: 2018,
        mapVintage: view.mapVintage,
      },
    },
    {
      comparison: {
        runId: "second",
        resultSha256: "e".repeat(64),
        configurationSha256: "f".repeat(64),
        year: view.year,
        mapVintage: "house_election_map_2018",
      },
    },
  ])
    assert.deepEqual(
      parseHouseSavedViews(stored([{ ...view, ...changes }])),
      [],
    );
  const canonical = parseHouseSavedViews(
    stored([{ ...view, unknown: "discard", selection: { raceId: "wrong" } }]),
  );
  assert.deepEqual(canonical, [view]);
  assert.equal("unknown" in canonical[0], false);
  assert.deepEqual(
    parseHouseSavedViews(stored([{ ...view, comparison: { runId: "bad" } }])),
    [],
  );
  assert.equal(
    parseHouseSavedViews(stored([view, { ...view, label: "collision" }]))
      .length,
    1,
  );
  const many = Array.from({ length: 100 }, (_, i) => ({
    ...view,
    id: `view-${i}`,
  }));
  assert.equal(
    parseHouseSavedViews(stored(many)).length,
    MAX_HOUSE_SAVED_VIEWS,
  );
  assert.equal(
    parseHouseSavedViews(serializeHouseSavedViews(many)).length,
    MAX_HOUSE_SAVED_VIEWS,
  );
  assert.throws(() => serializeHouseSavedViews(null), TypeError);
});

test("create sanitizes display labels without changing scientific identity and rejects invalid current selections", () => {
  const { publication, selection, view } = fixture();
  const named = createHouseSavedView(
    "  Custom\nview  ",
    publication,
    selection,
    { id: "label-test", savedAt: view.savedAt },
  );
  assert.equal(named.label, "Custom view");
  assert.equal(named.runLabel, publication.result.run.label);
  const long = createHouseSavedView(
    "x".repeat(79) + "😀",
    publication,
    selection,
    { id: "unicode-test", savedAt: view.savedAt },
  );
  assert.equal(long.label, "x".repeat(79));
  assert.equal(
    createHouseSavedView(" ", publication, selection).label,
    "Saved House view",
  );
  assert.throws(
    () =>
      createHouseSavedView("bad", publication, {
        ...selection,
        raceId: "2022:CA-99:house:regular",
      }),
    /exact saved race/,
  );
  assert.throws(
    () =>
      createHouseSavedView("bad", publication, {
        ...selection,
        mapView: "other",
      }),
    TypeError,
  );
  const comparison = comparisonFixture(publication);
  assert.throws(
    () => createHouseSavedView("bad", publication, {
      ...selection,
      comparison: makeHousePublication({ year: 2018 }),
    }),
    /same election year and map edition/,
  );
  assert.throws(
    () => createHouseSavedView("bad", publication, {
      ...selection,
      comparison: { ...comparison, resultSha256: "bad" },
    }),
    /comparison House result identity is invalid/,
  );
  assert.throws(
    () => createHouseSavedView("bad", publication, {
      ...selection,
      comparison: publication,
    }),
    /different saved result/,
  );
  assert.throws(
    () =>
      createHouseSavedView(
        "bad",
        { ...publication, resultSha256: "bad" },
        selection,
      ),
    TypeError,
  );
});

test("helpers do not mutate caller records or infer identity from recorded model labels", () => {
  const { publication, selection, catalog, view } = fixture();
  const snapshot = JSON.stringify({ publication, selection, catalog, view });
  const labelled = { ...view, runLabel: "Old model name" };
  assert.equal(
    houseSavedViewAvailability(labelled, catalog).status,
    "catalog_compatible",
  );
  assert.equal(verifyHouseSavedView(labelled, publication).status, "verified");
  parseHouseSavedViews(serializeHouseSavedViews([view]));
  createHouseSavedView("copy", publication, selection);
  assert.equal(
    JSON.stringify({ publication, selection, catalog, view }),
    snapshot,
  );
});
