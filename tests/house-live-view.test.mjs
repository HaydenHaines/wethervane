import assert from "node:assert/strict";
import test from "node:test";
import { houseLiveView } from "../lib/house-live-view.ts";
import { makeHouseLiveCatalog, makeHouseLiveResult } from "./fixtures/house-live-fixtures.mjs";

function selection(result = makeHouseLiveResult()) {
  // Only exercise the projection's precondition; this invented label is not
  // publication admission or evidence that the synthetic fixture was reviewed.
  result.qualification.publication_status = "reviewed";
  const catalog = makeHouseLiveCatalog();
  const entry = catalog.entries.find((item) => item.id === result.id);
  assert.ok(entry, "synthetic catalog should have an entry for this saved configuration");
  return { result, entry };
}

function deepFreeze(value) {
  if (value && typeof value === "object" && !Object.isFrozen(value)) {
    Object.freeze(value);
    for (const child of Object.values(value)) deepFreeze(child);
  }
  return value;
}

test("projection preserves grouped outcomes without allocating their probabilities to member candidates", () => {
  const { result, entry } = selection();
  const groupedSource = result.races.find((race) => race.outcomes.some((outcome) => outcome.kind === "group"));
  const view = houseLiveView(result, entry);
  const groupedView = view.races.find((race) => race.id === groupedSource.race_id);
  const group = groupedView.outcomes.find((outcome) => outcome.kind === "group");

  assert.equal(group.winProbability, 0.5);
  assert.equal(group.party, "D");
  assert.equal(group.members.length, 2);
  assert.ok(group.members.every((member) => !("winProbability" in member)));
  assert.ok(group.members.every((member) => !("share" in member)));
  assert.ok(groupedView.outcomes.every((outcome) => !("share" in outcome)));
  assert.deepEqual(group.members.map((member) => member.id), groupedSource.outcomes[0].member_candidate_ids);
});

test("projection retains independent O identity and keeps conditional pair evidence separate from group win probability", () => {
  const { result, entry } = selection();
  const oSource = result.races.find((race) => race.outcomes.some((outcome) => outcome.party === "O"));
  const pairSource = result.races.find((race) => race.conditional_shares !== null);
  const view = houseLiveView(result, entry);
  const oRace = view.races.find((race) => race.id === oSource.race_id);
  const oOutcome = oRace.outcomes.find((outcome) => outcome.party === "O");
  const oCandidate = oRace.candidates.find((candidate) => candidate.aggregationParty === "O");
  const pairRace = view.races.find((race) => race.id === pairSource.race_id);
  const dOutcome = pairRace.outcomes.find((outcome) => outcome.party === "D");

  assert.equal(oOutcome.party, "O");
  assert.equal(oOutcome.kind, "candidate");
  assert.equal(oCandidate.partyLabel, "Independent");
  assert.equal(oCandidate.aggregationParty, "O");
  assert.equal(oOutcome.members[0].party, "O");
  assert.equal(oOutcome.members[0].id, oCandidate.id);
  assert.equal(pairRace.conditionalPair.firstCandidateId, pairSource.conditional_shares.first_candidate_id);
  assert.equal(pairRace.conditionalPair.denominator, pairSource.conditional_shares.denominator);
  assert.equal(pairRace.conditionalPair.firstExceedsHalf, 0.45);
  assert.equal(dOutcome.winProbability, 0.5);
  assert.notEqual(pairRace.conditionalPair.firstExceedsHalf, dOutcome.winProbability);
  assert.ok(pairRace.outcomes.every((outcome) => !("share" in outcome)));
});

test("native O seat distribution projects to unassigned U, with no O caucus added", () => {
  const { result, entry } = selection();
  const view = houseLiveView(result, entry);
  assert.equal(view.chamber.seats.U.expectedSeats, result.chamber.seats.O.expected);
  assert.deepEqual(view.chamber.seats.U.pmf, result.chamber.seats.O.pmf);
  assert.equal(view.chamber.seats.D.expectedSeats, result.chamber.seats.D.expected);
  assert.equal(view.chamber.seats.R.expectedSeats, result.chamber.seats.R.expected);
  assert.deepEqual(Object.keys(view.chamber.seats).sort(), ["D", "R", "U"]);
  assert.deepEqual(view.chamber.retainedSeats, { D: 0, R: 0, U: 0 });
  assert.equal(view.chamber.thresholds.D, 218);
  assert.equal(view.chamber.thresholds.R, 218);
});

test("unqualified chamber and tipping stay unavailable in the projected view", () => {
  const { result, entry } = selection();
  result.chamber = null;
  result.qualification.limitations.push("Chamber calculation is unqualified for this synthetic result.");
  const view = houseLiveView(result, entry);
  assert.equal(view.chamber, null);
  assert.equal(view.chamberPrecision, null);
  assert.equal(view.tipping, null);
});

test("projection exposes only approved source and artifact metadata, not native paths or private fields", () => {
  const { result, entry } = selection();
  const view = houseLiveView(result, entry);
  const forbidden = new Set(["path", "source_ids", "source_receipt", "geography", "provenance", "configurationSha256"]);
  const inspect = (value) => {
    if (!value || typeof value !== "object") return;
    for (const [key, child] of Object.entries(value)) {
      assert.ok(!forbidden.has(key), `unexpected native/private field leaked: ${key}`);
      inspect(child);
    }
  };
  inspect(view);
  assert.deepEqual(Object.keys(view.sources[0]).sort(), ["id", "label", "retrieved_at", "role", "sha256", "url"]);
  assert.equal(view.jointDraws.sha256, result.joint_draws.sha256);
  assert.equal(view.geographySha256, result.geography.sha256);
  assert.deepEqual(view.qualification.evidenceHashes, result.qualification.evidence.map((pin) => pin.sha256));
  assert.equal(view.run.resultSha256, entry.result.sha256);
});

test("projection copies optional capture notes and keeps native source details private", () => {
  const { result, entry } = selection();
  result.provenance[0].capture_kind = "rendered_primary_source";
  result.provenance[0].limitations = ["Invented captured text, not original PDF bytes."];
  result.provenance[0].privateCapturePath = "/private/original.pdf";
  const view = houseLiveView(result, entry);
  assert.deepEqual(view.sources[0].limitations, result.provenance[0].limitations);
  assert.equal(view.sources[0].capture_kind, "rendered_primary_source");
  assert.equal("privateCapturePath" in view.sources[0], false);
  view.sources[0].limitations.push("Display-only note.");
  assert.equal(result.provenance[0].limitations.length, 1);
});

test("projection rejects a mismatched run, mismatched configuration, or unreviewed publication status", () => {
  const { result, entry } = selection();
  assert.throws(() => houseLiveView(result, { ...entry, id: "another-run" }), /does not match/);
  assert.throws(() => houseLiveView(result, {
    ...entry,
    configuration: { ...entry.configuration, polling: "none" },
  }), /does not match/);

  const unreviewed = structuredClone(result);
  unreviewed.qualification.publication_status = "not_for_publication";
  assert.throws(() => houseLiveView(unreviewed, entry), /reviewed publication status/);
});

test("projection copies data and does not mutate or share mutable nested structures with the native result", () => {
  const { result, entry } = selection();
  deepFreeze(result);
  const before = JSON.stringify(result);
  const view = houseLiveView(result, entry);

  view.races[0].notes.push("display-only note");
  view.races[0].candidates[0].sourceUrls.push("https://example.invalid/display-only");
  view.races[0].outcomes[0].members[0].name = "Display-only name";
  view.chamber.seats.U.pmf[0] = 1;
  view.sources[0].label = "Display-only source label";
  view.qualification.limitations.push("Display-only limitation");

  assert.equal(JSON.stringify(result), before);
});
