import test from "node:test";
import assert from "node:assert/strict";
import { join } from "node:path";
import { loadComponentStudies } from "../lib/component-study-server.ts";
import { componentStudySlices, studyRaces, studyShareError, comparableStudies, parseStudySavedViews } from "../lib/component-study-view.ts";
import { deriveRaceScoreboard } from "../lib/scoreboard.ts";

const { catalog, results } = await loadComponentStudies(join(process.cwd(), "data"));
test("race-only scores reproduce observed study outcomes without creating chamber estimates", () => {
  for (const [i, expected] of [[0, [24, .23568285202083333, .0437300169296305]], [1, [27, .14526426055208336, .028500811324451217]]]) {
    const snapshot = JSON.stringify(results[i]);
    const races = studyRaces(results[i]), score = deriveRaceScoreboard(races);
    assert.equal(score.coverage.scored, 30);
    assert.equal(score.favorites.correct, expected[0]);
    assert.ok(Math.abs(score.brier.mean - expected[1]) < 1e-12);
    assert.ok(Math.abs(studyShareError(results[i]) - expected[2]) < 1e-12);
    assert.deepEqual(score.seats, []); assert.equal(score.control, null);
    assert.ok(races.every(r => r.outcomes.every(o => !o.share)));
    assert.equal(JSON.stringify(results[i]), snapshot);
  }
});
test("study comparison requires matching outcomes, dates, denominator and coverage", () => {
  assert.equal(comparableStudies(...results), true);
  for (const mutate of [b => b.races.pop(), b => b.races[0].candidates[0].name = "Someone else", b => b.races[0].observed.democraticVotes++, b => b.informationCutoff = "2018-10-24T00:00:00Z", b => b.races[0].candidates[0].share.denominator = "All votes"]) {
    const altered = structuredClone(results[1]); mutate(altered);
    assert.equal(comparableStudies(results[0], altered), false);
  }
});
test("seven concurrent providers occupy three roles; unused slices and unsupported alternatives stay disabled", () => {
  const slices = componentStudySlices(catalog, { studyId: results[0].id, selections: results[0].selections });
  assert.equal(slices.length, 8); assert.equal(slices.filter(s => !s.controls.length).length, 5);
  assert.equal(slices.flatMap(s => s.controls).length, 7);
  const controls = Object.fromEntries(slices.flatMap(s => s.controls).map(c => [c.id, c]));
  assert.equal(controls.candidate.options.find(o => o.id === "off").availability, "requires_computation");
  assert.equal(controls.context.value, "off");
  assert.equal(controls.context.options.find(o => o.id === "party_midterm").availability, "unavailable");
  assert.equal(controls.poll.toggle.offValue, "identity");
  assert.equal(controls.race.required, true); assert.equal(controls.cycle.required, true);
});
test("saved study views reject stale results and malformed browser data", () => {
  const entry = catalog.studies[0], valid = { name: "Example", studyId: entry.id, sha256: entry.sha256, raceId: results[0].races[0].id, compare: true };
  assert.deepEqual(parseStudySavedViews(JSON.stringify([valid]), catalog), [valid]);
  for (const patch of [{ sha256: "0".repeat(64) }, { studyId: "unavailable" }, { name: " " }, { raceId: 42 }, { compare: "yes" }])
    assert.deepEqual(parseStudySavedViews(JSON.stringify([{ ...valid, ...patch }]), catalog), []);
  assert.deepEqual(parseStudySavedViews("broken", catalog), []);
});
