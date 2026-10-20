import assert from "node:assert/strict";
import test from "node:test";
import { CHAMBERS, districtIdentityKey, validateChamberReadiness } from "../lib/chamber.ts";

const digest = "a".repeat(64);
const mapVintage = "synthetic-test-map";
const cycle = 2042;

function pointMass(seats, size) {
  const pmf = Array(size + 1).fill(0);
  pmf[seats] = 1;
  return { expectedSeats: seats, seatQuantiles: { p05: seats, p50: seats, p95: seats }, pmf };
}

function scenario(size, d, r, retained = { D: 0, R: 0, U: 0 }) {
  const threshold = size === 435 ? { D: 218, R: 218 } : { D: 51, R: 50 };
  return {
    id: "test-scenario", label: "Synthetic test only", description: "Not a public forecast",
    thresholds: threshold, retainedSeats: retained,
    control: {
      description: "Synthetic joint fixture",
      democraticMajorityProbability: d >= threshold.D ? 1 : 0,
      republicanMajorityProbability: r >= threshold.R ? 1 : 0,
      unresolvedOrganizingProbability: d < threshold.D && r < threshold.R ? 1 : 0,
    },
    seats: { D: pointMass(d, size), R: pointMass(r, size), U: pointMass(size - d - r, size) },
  };
}

function forecast(count = 10, full = false) {
  const races = Array.from({ length: count }, (_, index) => {
    // Synthetic identifiers occupy five test state-like namespaces; no public data is written.
    const state = ["AA", "AB", "AC", "AD", "AE"][Math.floor(index / 99)];
    const districtKey = `${state}-${String(index % 99 + 1).padStart(2, "0")}`;
    return {
      id: `${cycle}:${districtKey}:house:regular`, seatId: districtKey, state,
      district: { cycle, mapVintage, districtKey },
      outcomes: [{ id: "D", winProbability: index < 218 ? 1 : 0 }, { id: "R", winProbability: index < 218 ? 0 : 1 }],
    };
  });
  const jointDraws = { id: "local-draw-identifiers", configurationId: "synthetic-config", cycle, sourceSha256: digest, simulationCount: 20, raceIds: races.map((race) => race.id) };
  return {
    metadata: {
      chamber: "house", chamber_size: 435, majority_threshold: 218,
      scope: full ? "full_chamber" : "eligible_race_subset", full_chamber: full,
      cycle, mapVintage, configurationId: "synthetic-config",
    },
    expectedRaceIds: races.map((race) => race.id), races,
    geography: { kind: "district", cycle, mapVintage, admission: "verified", sourceSha256: digest, keys: races.map((race) => race.district.districtKey) },
    jointDraws,
  };
}

function addFullProof(input) {
  const size = input.metadata.chamber_size;
  input.chamberProof = {
    sourceSha256: "b".repeat(64), jointDrawsId: input.jointDraws.id,
    jointArtifactSha256: input.jointDraws.sourceSha256,
    configurationId: input.jointDraws.configurationId, cycle: input.metadata.cycle,
    expectedSeatIds: input.races.map((race) => race.seatId), retainedSeatIds: [],
    scenarios: [scenario(size, 218, 217)],
  };
  return input;
}

test("chamber metadata distinguishes the 100-seat Senate from the 435-seat House", () => {
  assert.equal(CHAMBERS.senate.chamberSize, 100);
  assert.equal(CHAMBERS.senate.majorityThreshold, 51);
  assert.equal(CHAMBERS.house.chamberSize, 435);
  assert.equal(CHAMBERS.house.majorityThreshold, 218);
  const wrong = forecast();
  wrong.metadata.chamber_size = 100;
  wrong.metadata.majority_threshold = 51;
  assert.equal(validateChamberReadiness(wrong).status, "invalid");
  assert.equal(validateChamberReadiness(null).canShowControl, false);
});

test("metadata without a House forecast stays unavailable rather than becoming a neutral forecast", () => {
  const input = forecast(0);
  delete input.metadata.mapVintage;
  const result = validateChamberReadiness(input);
  assert.equal(result.status, "unavailable");
  assert.equal(result.canShowRaces, false);
  assert.equal(result.canShowMap, false);
  assert.equal(result.canShowControl, false);
  assert.equal(result.canShowTipping, false);
});

test("an admitted eligible subset can display race maps and a race ribbon, never chamber control", () => {
  const input = forecast(10);
  const result = validateChamberReadiness(input);
  assert.equal(result.status, "race_subset");
  assert.equal(result.canShowRaces, true);
  assert.equal(result.canShowMap, true);
  assert.equal(result.canShowRaceRibbon, true);
  assert.equal(result.canShowControl, false);
  assert.equal(result.canShowTipping, false);
  assert.equal(result.coverage.availableRaces, 10);
  assert.equal(result.coverage.chamberSize, 435);
  assert.ok(result.issues.some((issue) => issue.code === "subset_has_no_control"));
});

test("even 435 aligned race draws do not turn a declared subset into a full chamber", () => {
  const input = addFullProof(forecast(435));
  const result = validateChamberReadiness(input);
  assert.equal(result.canShowRaceRibbon, true);
  assert.equal(result.canShowControl, false);
  assert.equal(result.status, "race_subset");
});

test("a full-chamber flag and common cycle draws are insufficient without full seat proof", () => {
  const input = forecast(435, true);
  const result = validateChamberReadiness(input);
  assert.equal(result.status, "races_only");
  assert.equal(result.canShowMap, true);
  assert.equal(result.canShowControl, false);
  assert.ok(result.issues.some((issue) => issue.code === "full_chamber_proof_missing"));
});

test("complete 435-seat accounting with joint-bound distributions permits control at 218", () => {
  const input = addFullProof(forecast(435, true));
  const result = validateChamberReadiness(input);
  assert.equal(result.status, "full_chamber");
  assert.equal(result.canShowControl, true);
  assert.equal(result.canShowTipping, false);
  assert.deepEqual(result.issues, []);
  input.chamberProof.scenarios[0].thresholds.D = 51;
  assert.equal(validateChamberReadiness(input).canShowControl, false);
});

test("missing races, missing seats, and 100-seat PMFs cannot qualify a full House forecast", () => {
  for (const change of [
    (input) => input.races.pop(),
    (input) => input.chamberProof.expectedSeatIds.pop(),
    (input) => input.chamberProof.scenarios[0].seats.D = pointMass(51, 100),
    (input) => input.chamberProof.scenarios[0].control.democraticMajorityProbability = 0.5,
  ]) {
    const input = addFullProof(forecast(435, true));
    change(input);
    assert.equal(validateChamberReadiness(input).canShowControl, false);
  }
  const incomplete = forecast(10);
  incomplete.expectedRaceIds.push(`${cycle}:AF-01:house:regular`);
  const result = validateChamberReadiness(incomplete);
  assert.equal(result.canShowRaces, true);
  assert.deepEqual(result.coverage.missingRaceIds, [`${cycle}:AF-01:house:regular`]);
});

test("district identities preserve map vintage, cycle, explicit at-large zero, and formatting", () => {
  const base = { cycle, mapVintage, districtKey: "ND-00" };
  assert.equal(districtIdentityKey(base), JSON.stringify([cycle, mapVintage, "ND-00"]));
  assert.notEqual(districtIdentityKey(base), districtIdentityKey({ ...base, cycle: cycle - 2 }));
  assert.notEqual(districtIdentityKey(base), districtIdentityKey({ ...base, mapVintage: "another-map" }));
  assert.equal(districtIdentityKey({ ...base, districtKey: "ND-0" }), null);
  assert.equal(districtIdentityKey({ ...base, districtKey: "ND-01 " }), null);
  const input = forecast();
  input.races[0].district.mapVintage = "other-map";
  assert.equal(validateChamberReadiness(input).status, "invalid");
});

test("a publisher map label does not admit geometry or enable cross-vintage joins", () => {
  for (const change of [
    (input) => input.geography.admission = "unverified",
    (input) => delete input.geography.sourceSha256,
    (input) => input.geography.sourceSha256 = "publisher-map-label",
    (input) => input.geography.mapVintage = "other-map",
    (input) => input.geography.cycle--,
    (input) => input.geography.keys.push(input.geography.keys[0]),
  ]) {
    const input = forecast();
    change(input);
    const result = validateChamberReadiness(input);
    assert.equal(result.canShowRaces, true);
    assert.equal(result.canShowMap, false);
  }
  const missing = forecast();
  missing.geography.keys.pop();
  assert.deepEqual(validateChamberReadiness(missing).coverage.missingGeographyIds, [missing.races.at(-1).district.districtKey]);
});

test("invalid probabilities, unexpected race IDs, and duplicated district identities fail closed", () => {
  for (const change of [
    (input) => input.races[0].outcomes[0].winProbability = NaN,
    (input) => input.races[0].outcomes[0].winProbability = 0.5,
    (input) => input.races[0].outcomes[0].winProbability = -1,
    (input) => input.expectedRaceIds.pop(),
    (input) => input.races[1].district = { ...input.races[0].district },
    (input) => input.races[0].id = "wrong-cycle-or-district",
  ]) {
    const input = forecast();
    change(input);
    const result = validateChamberReadiness(input);
    assert.equal(result.status, "invalid");
    assert.equal(result.canShowRaces, false);
    assert.equal(result.canShowControl, false);
  }
});

test("identical draw labels from different configuration-cycle artifacts cannot be paired", () => {
  for (const change of [
    (input) => input.chamberProof.configurationId = "another-configuration",
    (input) => input.chamberProof.cycle--,
    (input) => input.chamberProof.jointArtifactSha256 = "c".repeat(64),
    (input) => input.jointDraws.configurationId = "another-configuration",
    (input) => input.jointDraws.raceIds.pop(),
  ]) {
    const input = addFullProof(forecast(435, true));
    change(input);
    assert.equal(validateChamberReadiness(input).canShowControl, false);
  }
});

test("tipping display needs its own complete joint-bound probabilities", () => {
  const input = addFullProof(forecast(435, true));
  const vector = Object.fromEntries(input.races.map((race, index) => [race.id, index === 217 ? 1 : 0]));
  input.tippingProof = {
    sourceSha256: "d".repeat(64), jointDrawsId: input.jointDraws.id,
    jointArtifactSha256: input.jointDraws.sourceSha256,
    configurationId: input.jointDraws.configurationId, cycle,
    method: "synthetic-full-chamber-rank-fixture",
    scenarios: [{ id: "test-scenario", frequencies: { D: vector, R: { ...vector } } }],
  };
  assert.equal(validateChamberReadiness(input).canShowTipping, true);
  delete input.tippingProof.scenarios[0].frequencies.R[input.races[0].id];
  const result = validateChamberReadiness(input);
  assert.equal(result.canShowControl, true);
  assert.equal(result.canShowTipping, false);
});

test("the same full-chamber validator handles 35 Senate races plus 65 retained seats", () => {
  const input = forecast(35, true);
  input.metadata = { chamber: "senate", chamber_size: 100, majority_threshold: 51, scope: "full_chamber", full_chamber: true, cycle, configurationId: "synthetic-config" };
  input.geography = { kind: "state", keys: [...new Set(input.races.map((race) => race.state))] };
  const retained = Array.from({ length: 65 }, (_, index) => `retained:${index}`);
  input.chamberProof = {
    sourceSha256: "e".repeat(64), jointDrawsId: input.jointDraws.id,
    jointArtifactSha256: input.jointDraws.sourceSha256,
    configurationId: input.jointDraws.configurationId, cycle,
    expectedSeatIds: [...input.races.map((race) => race.seatId), ...retained], retainedSeatIds: retained,
    scenarios: [scenario(100, 51, 49, { D: 16, R: 49, U: 0 })],
  };
  const before = JSON.stringify(input);
  const result = validateChamberReadiness(input);
  assert.equal(result.canShowControl, true);
  assert.equal(result.coverage.chamberSize, 100);
  assert.equal(JSON.stringify(input), before);
});
