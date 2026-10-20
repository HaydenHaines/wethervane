import assert from "node:assert/strict";
import test from "node:test";
import {
  joinDistrictGeography,
  parseDistrictKey,
  districtIdentityKey,
} from "../lib/district-geography.ts";

const identity = { cycle: 2022, mapVintage: "house_election_map_2022" };
const sourceSha256 = "a".repeat(64);
const feature = (
  districtKey,
  geometry = {
    type: "Polygon",
    coordinates: [
      [
        [-101, 38],
        [-100, 38],
        [-100, 39],
        [-101, 39],
        [-101, 38],
      ],
    ],
  },
) => ({ type: "Feature", properties: { districtKey }, geometry });
const entry = (districtKey, id = `2022:${districtKey}:house:regular`) => ({
  districtKey,
  race: {
    id,
    state: districtKey.slice(0, 2),
    seatId: districtKey,
    electionLabel: "Regular election",
    outcomes: [
      {
        id: "D",
        name: "Candidate",
        party: "D",
        kind: "candidate",
        members: [
          { id: "candidate", name: "Candidate", party: "D", sourceUrls: [] },
        ],
        winProbability: 1,
      },
    ],
    notes: [],
  },
});
const geography = (features, patch = {}) => ({
  metadata: { ...identity, admission: "verified", sourceSha256, ...patch },
  geojson: { type: "FeatureCollection", features },
});

test("exact district identity joins deterministically regardless of input order", () => {
  const races = [entry("NY-02"), entry("AK-00"), entry("NY-01")];
  const geo = geography([feature("NY-01"), feature("NY-02"), feature("AK-00")]);
  const before = JSON.stringify({ races, geo });
  const joined = joinDistrictGeography(races, identity, geo);
  assert.equal(joined.status, "ready");
  assert.deepEqual(
    joined.matches.map((match) => match.identity.districtKey),
    ["AK-00", "NY-01", "NY-02"],
  );
  assert.deepEqual(joined.matches[0].identity, {
    ...identity,
    districtKey: "AK-00",
  });
  assert.equal(
    joined.matches[0].identityKey,
    JSON.stringify([2022, "house_election_map_2022", "AK-00"]),
  );
  assert.deepEqual(
    joinDistrictGeography([...races].reverse(), identity, {
      ...geo,
      geojson: {
        ...geo.geojson,
        features: [...geo.geojson.features].reverse(),
      },
    }),
    joined,
  );
  assert.equal(
    JSON.stringify({ races, geo }),
    before,
    "the adapter does not mutate source races or geometry",
  );
});

test("at-large 00 is preserved and never aliased to district 01", () => {
  assert.deepEqual(parseDistrictKey("AK-00"), { state: "AK", district: "00" });
  const joined = joinDistrictGeography(
    [entry("AK-00")],
    identity,
    geography([feature("AK-01")]),
  );
  assert.equal(joined.status, "partial");
  assert.equal(joined.matches.length, 0);
  assert.deepEqual(joined.missingDistrictKeys, ["AK-00"]);
  assert.deepEqual(joined.extraDistrictKeys, ["AK-01"]);
  for (const bad of [
    "AK-0",
    "AK-1",
    "ak-00",
    "AK-AL",
    " AK-00",
    "AK-00 ",
    "0200",
    0,
    null,
  ])
    assert.equal(parseDistrictKey(bad), null);
});

test("current geography cannot silently stand in for a historical cycle", () => {
  const result = joinDistrictGeography(
    [entry("NY-01")],
    identity,
    geography([feature("NY-01")], { cycle: 2026 }),
  );
  assert.equal(result.status, "invalid");
  assert.equal(result.matches.length, 0);
  assert.ok(result.diagnostics.some((d) => d.code === "cycle_mismatch"));
});

test("matching cycles require the exact map edition, not a similar publisher label", () => {
  for (const mapVintage of [
    "house_election_map_2024",
    "house_election_map_2022_revised",
    "HOUSE_ELECTION_MAP_2022",
  ]) {
    const result = joinDistrictGeography(
      [entry("NY-01")],
      identity,
      geography([feature("NY-01")], { mapVintage }),
    );
    assert.equal(result.status, "invalid");
    assert.ok(result.diagnostics.some((d) => d.code === "vintage_mismatch"));
  }
});

test("publisher identity alone never admits unverified geometry", () => {
  const result = joinDistrictGeography(
    [entry("NY-01")],
    identity,
    geography([feature("NY-01")], { admission: "unverified" }),
  );
  assert.equal(result.status, "unavailable");
  assert.equal(result.matches.length, 0);
  assert.ok(result.diagnostics.some((d) => d.code === "unverified_geometry"));
});

test("a verified geometry source must also carry its actual source fingerprint", () => {
  for (const digest of [
    "",
    undefined,
    "publisher-map-2022",
    "g".repeat(64),
    "a".repeat(63),
  ]) {
    const result = joinDistrictGeography(
      [entry("NY-01")],
      identity,
      geography([feature("NY-01")], { sourceSha256: digest }),
    );
    assert.equal(result.status, "unavailable");
    assert.ok(
      result.diagnostics.some((d) => d.code === "missing_source_digest"),
    );
  }
});

test("colliding race district keys are rejected rather than choosing a regular or special election", () => {
  const result = joinDistrictGeography(
    [entry("NY-01"), entry("NY-01", "2022:NY-01:house:special")],
    identity,
    geography([feature("NY-01")]),
  );
  assert.equal(result.status, "invalid");
  assert.equal(result.matches.length, 0);
  assert.ok(result.diagnostics.some((d) => d.code === "duplicate_race_key"));
});

test("colliding feature keys are rejected rather than silently dissolving geometry", () => {
  const result = joinDistrictGeography(
    [entry("NY-01")],
    identity,
    geography([feature("NY-01"), feature("NY-01")]),
  );
  assert.equal(result.status, "invalid");
  assert.ok(result.diagnostics.some((d) => d.code === "duplicate_feature_key"));
});

test("duplicate race IDs and cross-state/cross-cycle district identities are explicit failures", () => {
  const sameId = "2022:NY-01:house:regular";
  const duplicate = joinDistrictGeography(
    [entry("NY-01", sameId), entry("NY-02", sameId)],
    identity,
    geography([feature("NY-01"), feature("NY-02")]),
  );
  assert.equal(duplicate.status, "invalid");
  assert.ok(duplicate.diagnostics.some((d) => d.code === "duplicate_race_id"));
  for (const id of [
    "2024:NY-01:house:regular",
    "2022:NY-02:house:regular",
    "2022:NY-01:senate:regular",
    "2022:NY-01:house:regular:extra",
  ]) {
    const result = joinDistrictGeography(
      [entry("NY-01", id)],
      identity,
      geography([feature("NY-01")]),
    );
    assert.equal(result.status, "invalid");
    assert.ok(
      result.diagnostics.some((d) => d.code === "race_identity_mismatch"),
    );
  }
  const wrongState = entry("NY-01");
  wrongState.race.state = "CA";
  assert.equal(
    joinDistrictGeography([wrongState], identity, geography([feature("NY-01")]))
      .status,
    "invalid",
  );
});

test("partial coverage stays visible with explicit missing and uncolored district lists", () => {
  const result = joinDistrictGeography(
    [entry("NY-01"), entry("NY-02")],
    identity,
    geography([feature("NY-01"), feature("CA-01")]),
  );
  assert.equal(result.status, "partial");
  assert.deepEqual(
    result.matches.map((m) => m.identity.districtKey),
    ["NY-01"],
  );
  assert.deepEqual(result.missingDistrictKeys, ["NY-02"]);
  assert.deepEqual(result.extraDistrictKeys, ["CA-01"]);
  assert.deepEqual(
    result.unmatchedFeatures.map((f) => f.properties.districtKey),
    ["CA-01"],
  );
});

test("invalid geometry and noncanonical feature identifiers cannot acquire race colors", () => {
  const malformed = [
    null,
    { type: "Point", coordinates: [-100, 39] },
    {
      type: "Polygon",
      coordinates: [
        [
          [-100, 39],
          [-99, 39],
          [-99, 40],
        ],
      ],
    },
    {
      type: "Polygon",
      coordinates: [
        [
          [-100, 39],
          [-99, 39],
          [-99, 95],
          [-100, 39],
        ],
      ],
    },
    {
      type: "Polygon",
      coordinates: [
        [
          [NaN, 39],
          [-99, 39],
          [-99, 40],
          [NaN, 39],
        ],
      ],
    },
  ];
  for (const geometry of malformed) {
    const result = joinDistrictGeography(
      [entry("NY-01")],
      identity,
      geography([feature("NY-01", geometry)]),
    );
    assert.equal(result.status, "invalid");
    assert.ok(result.diagnostics.some((d) => d.code === "invalid_geometry"));
  }
  assert.equal(
    joinDistrictGeography(
      [entry("NY-01")],
      identity,
      geography([feature("NY-1")]),
    ).status,
    "invalid",
  );
});

test("MultiPolygon districts, empty collections and complete data-free geometry are explicit", () => {
  const polygon = feature("AK-00").geometry.coordinates;
  const multi = feature("AK-00", {
    type: "MultiPolygon",
    coordinates: [polygon, polygon],
  });
  assert.equal(
    joinDistrictGeography([entry("AK-00")], identity, geography([multi]))
      .status,
    "ready",
  );
  assert.equal(
    joinDistrictGeography([], identity, geography([])).status,
    "partial",
  );
  const noData = joinDistrictGeography(
    [],
    identity,
    geography([feature("AK-00")]),
  );
  assert.equal(noData.status, "partial");
  assert.equal(noData.matches.length, 0);
  assert.equal(noData.unmatchedFeatures.length, 1);
});

test("identity tuple keys distinguish geography editions and cannot collide on separators", () => {
  assert.notEqual(
    districtIdentityKey(identity, "NY-01"),
    districtIdentityKey({ ...identity, cycle: 2024 }, "NY-01"),
  );
  assert.notEqual(
    districtIdentityKey(identity, "NY-01"),
    districtIdentityKey(
      { ...identity, mapVintage: "house_election_map_2022:revision" },
      "NY-01",
    ),
  );
  assert.notEqual(
    districtIdentityKey({ cycle: 2022, mapVintage: "a:b" }, "c"),
    districtIdentityKey({ cycle: 2022, mapVintage: "a" }, "b:c"),
  );
});
