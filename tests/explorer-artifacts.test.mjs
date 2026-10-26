import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { validateExplorerBundle } from "../lib/explorer-state.ts";

const root = path.resolve(import.meta.dirname, "..");
const registry = JSON.parse(
  fs.readFileSync(path.join(root, "public/data/explorer/registry.json")),
);
const load = (entry) =>
  JSON.parse(fs.readFileSync(path.join(root, "public", entry.href)));
const close = (actual, expected, epsilon = 1e-9) =>
  assert.ok(Math.abs(actual - expected) < epsilon, `${actual} != ${expected}`);

test("every selectable run has verified bytes, coherent probabilities, and complete coverage", () => {
  assert.equal(registry.runs.filter((run) => run.year === 2026).length, 16);
  assert.equal(registry.runs.filter((run) => run.year === 2022).length, 16);
  assert.equal(registry.runs.filter((run) => run.year === 2018).length, 16);
  for (const entry of registry.runs) {
    const raw = fs.readFileSync(path.join(root, "public", entry.href));
    assert.equal(
      createHash("sha256").update(raw).digest("hex"),
      entry.sha256,
      entry.id,
    );
    const bundle = JSON.parse(raw);
    assert.equal(validateExplorerBundle(bundle, entry), true, entry.id);
    assert.equal(bundle.races.length, 35, entry.id);
    assert.equal(bundle.scenarios.length, 3, entry.id);
    for (const hash of Object.values(bundle.provenance.sources))
      assert.match(hash, /^[a-f0-9]{64}$/);
    if (entry.historical) {
      assert.equal(entry.strictAsOf, false);
      assert.equal(entry.horizonDays, 14);
      assert.equal(bundle.tipping, null);
      assert.ok(bundle.races.every((race) => race.observed));
      assert.ok(
        bundle.races
          .flatMap((race) => race.outcomes)
          .every((outcome) => !outcome.share),
      );
    } else {
      assert.equal(
        entry.components.polling,
        entry.id === "current-fundamentals" || entry.id.endsWith("-f")
          ? "none"
          : "standard",
      );
      assert.ok(bundle.tipping);
      assert.ok(bundle.races.every((race) => race.outcomes.length === 2));
    }
  }
});

test("historical scores agree with public probability distributions and fixed observed results", () => {
  for (const entry of registry.runs.filter((run) => run.historical)) {
    const bundle = load(entry);
    const actualD = entry.year === 2018 ? 47 : 51;
    const actualControl = entry.year === 2018 ? "R" : "D";
    for (const scenario of bundle.scenarios) {
      const score = scenario.performance;
      assert.deepEqual(score.observedSeats, {
        D: actualD,
        R: 100 - actualD,
        U: 0,
      });
      assert.equal(score.observedControl, actualControl);
      assert.deepEqual(
        scenario.thresholds,
        entry.year === 2018 ? { D: 51, R: 50 } : { D: 50, R: 51 },
      );
      for (const [party, key] of [
        ["D", "dSeatCrps"],
        ["R", "rSeatCrps"],
      ]) {
        let cdf = 0;
        let crps = 0;
        for (let seats = 0; seats < 100; seats++) {
          cdf += scenario.seats[party].pmf[seats];
          crps += (cdf - Number(seats >= score.observedSeats[party])) ** 2;
        }
        close(score[key], crps);
      }
      const probs = {
        D: scenario.control.democraticMajorityProbability,
        R: scenario.control.republicanMajorityProbability,
        U: scenario.control.unresolvedOrganizingProbability,
      };
      const brier = Object.entries(probs).reduce(
        (s, [party, p]) => s + (p - Number(party === actualControl)) ** 2,
        0,
      );
      close(score.controlBrier, brier);
      if (probs[actualControl] === 0) assert.equal(score.controlLogScore, null);
      else close(score.controlLogScore, -Math.log(probs[actualControl]));
      for (const interval of score.intervals) {
        const actual = score.observedSeats[interval.party];
        assert.equal(
          interval.covered,
          actual >= interval.lower && actual <= interval.upper,
        );
      }
    }
  }
});

test("historical party groups never manufacture individual candidate probabilities", () => {
  const history2018 = load(
    registry.runs.find((run) => run.id === "history-2018-FP-1"),
  );
  const california = history2018.races.find((race) => race.state === "CA");
  const democrats = california.outcomes.find((outcome) => outcome.id === "D");
  assert.equal(democrats.kind, "group");
  assert.equal(democrats.members.length, 2);
  assert.equal(democrats.winProbability, 1);
  assert.ok(
    democrats.members.every((candidate) => !("winProbability" in candidate)),
  );
  assert.match(california.observed.winnerName, /Feinstein/i);
  assert.equal(
    history2018.races.filter((race) => race.state === "MS").length,
    2,
  );
  const history2022 = load(
    registry.runs.find((run) => run.id === "history-2022-FP-1"),
  );
  const alaskaR = history2022.races
    .find((race) => race.state === "AK")
    .outcomes.find((outcome) => outcome.id === "R");
  assert.equal(alaskaR.kind, "group");
  assert.ok(alaskaR.members.length > 1);
  assert.equal(
    history2022.races.filter((race) => race.state === "OK").length,
    2,
  );
});

test("the original October 20 forecast and tipping frequencies remain exactly preserved", () => {
  const original = JSON.parse(
    fs.readFileSync(path.join(root, "public/data/forecast.json")),
  );
  const tipping = JSON.parse(
    fs.readFileSync(path.join(root, "public/data/tipping-points.json")),
  );
  for (const [id, key] of [
    ["current-polling", "polling"],
    ["current-fundamentals", "fundamentals"],
  ]) {
    const bundle = id === "current-polling"
      ? JSON.parse(fs.readFileSync(path.join(root, "public/data/explorer/editions/2026-10-20/current-polling.json")))
      : load(registry.runs.find((run) => run.id === id));
    assert.deepEqual(bundle.tipping, tipping.models[key]);
    for (const race of bundle.races) {
      const previous = original.models[key].races.find(
        (candidate) => candidate.id === race.id,
      );
      assert.equal(
        race.outcomes[0].winProbability,
        previous.firstWinProbability,
      );
      assert.equal(
        race.outcomes[1].winProbability,
        previous.secondWinProbability,
      );
      assert.equal(race.outcomes[0].share.mean, previous.firstShareMean);
    }
  }
});
