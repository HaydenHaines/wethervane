import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { createHash } from "node:crypto";
import path from "node:path";
import test from "node:test";
import {
  HOUSE_COMPARISON_SOURCE_PINS,
  parseHouseComparisonManifest,
  houseComparisonAvailability,
  verifyHouseComparison,
  summarizeHouseComparison,
  summarizeHouseResult,
  houseComparisonCanonicalJson,
  houseComparisonDigest,
} from "../lib/house-comparison.ts";
import { buildHouseComparisonManifest } from "../scripts/build_house_comparisons.mjs";
import { makeHousePublication } from "./fixtures/house.mjs";

const read = (file) => JSON.parse(readFileSync(file, "utf8"));
const manifest = read("data/house-comparisons.json");
const published = read("data/house/publication.json");
const catalog = read(path.join("data/house", published.catalog.file));
const load = (id) => {
  const entry = catalog.runs.find((run) => run.id === id);
  return {
    result: read(path.join("data/house", entry.result.file)),
    resultSha256: entry.resultSha256,
  };
};
const fixed = (year) => load(`house_history_refreshed_v1-${year}-h14`);
const context = (year) => load(`house_relative_context_v1-${year}-h14`);
const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-12, `${a} != ${b}`);

test("source-bound static manifest admits only the reviewed semantics and has no private paths", () => {
  assert.ok(parseHouseComparisonManifest(manifest));
  assert.equal(manifest.groups.length, 4);
  assert.equal(
    manifest.groups.reduce((n, g) => n + g.runs.length, 0),
    16,
  );
  assert.equal(manifest.nominalHorizonDays, 14);
  assert.equal(
    manifest.denominator,
    "Democratic and Republican candidate votes",
  );
  assert.deepEqual(manifest.sourcePins, HOUSE_COMPARISON_SOURCE_PINS);
  assert.doesNotMatch(
    JSON.stringify(manifest),
    /\/home\/|research\/|\.npz|file:\/\//,
  );
  for (const change of [
    (m) => (m.targetId = "some-other-target"),
    (m) => (m.nominalHorizonDays = 7),
    (m) => (m.sourcePins.reviewReceipt = "0".repeat(64)),
    (m) => m.groups[0].raceIds.pop(),
    (m) => (m.groups[0].runs[1].runId = m.groups[0].runs[0].runId),
    (m) => (m.groups[0].runs[0].planSha256 = "0".repeat(64)),
  ]) {
    const changed = structuredClone(manifest);
    change(changed);
    assert.equal(parseHouseComparisonManifest(changed), null);
  }
  for (const unknown of [null, [], {}, "invalid", { schemaVersion: 1 }])
    assert.equal(parseHouseComparisonManifest(unknown), null);
});

test("preflight rejects self, other years and unknown choices without claiming payload verification", () => {
  assert.equal(
    houseComparisonAvailability(
      manifest,
      fixed(2024).result.run.id,
      context(2024).result.run.id,
    ).status,
    "eligible",
  );
  assert.match(
    houseComparisonAvailability(
      manifest,
      fixed(2024).result.run.id,
      fixed(2024).result.run.id,
    ).reason,
    /different model/,
  );
  assert.equal(
    houseComparisonAvailability(
      manifest,
      fixed(2024).result.run.id,
      context(2018).result.run.id,
    ).status,
    "unavailable",
  );
  assert.equal(
    houseComparisonAvailability(
      manifest,
      "not-published",
      context(2024).result.run.id,
    ).status,
    "unavailable",
  );
});

test("all 48 distinct ordered same-year pairs verify real immutable support, candidates and truth", async () => {
  let pairs = 0;
  for (const group of manifest.groups) {
    for (const first of group.runs)
      for (const second of group.runs) {
        if (first.runId === second.runId) continue;
        const verification = await verifyHouseComparison(
          manifest,
          load(first.runId),
          load(second.runId),
        );
        assert.equal(verification.status, "comparable", verification.reason);
        assert.equal(verification.group.year, group.year);
        const summary = summarizeHouseComparison(verification);
        assert.equal(summary.races.length, group.raceCount);
        assert.equal(summary.primary.observed, group.raceCount);
        assert.equal(summary.reference.observed, group.raceCount);
        for (const race of summary.races)
          for (const candidate of race.candidates)
            close(
              candidate.difference,
              candidate.primaryProbability - candidate.referenceProbability,
            );
        pairs++;
      }
  }
  assert.equal(pairs, 48);
});

test("actual 2024 scores retain proper categorical scale and signed primary-minus-reference change", async () => {
  const verification = await verifyHouseComparison(
    manifest,
    context(2024),
    fixed(2024),
  );
  assert.equal(verification.status, "comparable");
  const summary = summarizeHouseComparison(verification);
  assert.deepEqual(
    { ...summary.primary, brier: 0 },
    { observed: 375, decisive: 375, correct: 366, ties: 0, brier: 0 },
  );
  assert.deepEqual(
    { ...summary.reference, brier: 0 },
    { observed: 375, decisive: 375, correct: 365, ties: 0, brier: 0 },
  );
  close(summary.primary.brier, 0.035402029118333345);
  close(summary.reference.brier, 0.04520387457833331);
  close(
    summary.brierDifference,
    summary.primary.brier - summary.reference.brier,
  );
  assert.ok(summary.brierDifference < 0);
  assert.ok(
    Math.abs(summary.reference.brier - 0.022601937289166662) > 1e-3,
    "Do not substitute the source's binary 0–1 score.",
  );
  const reverse = summarizeHouseComparison(
    await verifyHouseComparison(manifest, fixed(2024), context(2024)),
  );
  close(reverse.brierDifference, -summary.brierDifference);
});

test("claimed file hash cannot conceal changed probabilities, configuration, cohort, map, truth or source", async () => {
  for (const mutate of [
    (p) => (p.resultSha256 = "0".repeat(64)),
    (p) => (p.result.run.configurationSha256 = "0".repeat(64)),
    (p) => (p.result.run.informationMode = "strict_asof"),
    (p) => (p.result.jointDraws.sourceSha256 = "0".repeat(64)),
    (p) => {
      p.result.races[0].outcomes[0].winProbability = 0.6;
      p.result.races[0].outcomes[1].winProbability = 0.4;
    },
    (p) => {
      p.result.metadata.mapVintage = "other-edition";
      for (const r of p.result.races) r.district.mapVintage = "other-edition";
    },
    (p) => {
      const race = p.result.races.pop();
      p.result.coverage.find((row) => row.raceId === race.id).eligible = false;
      p.result.coverage.find((row) => row.raceId === race.id).exclusionReason =
        "Changed support";
    },
    (p) => {
      const r = p.result.races[0],
        other = r.outcomes.find((o) => o.id !== r.observed.outcomeId);
      r.observed = {
        ...r.observed,
        outcomeId: other.id,
        winnerName: other.name,
        winnerParty: other.party,
      };
    },
    (p) => {
      const c = p.result.races[0].outcomes[0];
      c.name = "Different identity";
      c.members[0].name = c.name;
    },
  ]) {
    const changed = context(2024);
    mutate(changed);
    const result = await verifyHouseComparison(manifest, changed, fixed(2024));
    assert.equal(result.status, "unavailable");
    assert.ok(result.reason);
  }
});

test("cross-year and partially observed comparisons cannot fall back to intersection scores", async () => {
  assert.equal(
    (await verifyHouseComparison(manifest, fixed(2024), context(2018))).status,
    "unavailable",
  );
  const missing = context(2024);
  delete missing.result.races[0].observed;
  assert.equal(
    (await verifyHouseComparison(manifest, missing, fixed(2024))).status,
    "unavailable",
  );
});

test("verification snapshots caller data and never couples the independently seeded draw identities", async () => {
  const first = context(2024),
    second = fixed(2024),
    before = JSON.stringify([first, second]);
  const verified = await verifyHouseComparison(manifest, first, second);
  assert.equal(verified.status, "comparable");
  assert.notEqual(
    verified.primary.result.jointDraws.sourceSha256,
    verified.reference.result.jointDraws.sourceSha256,
  );
  assert.equal(JSON.stringify([first, second]), before);
  const expected = summarizeHouseComparison(verified).primary.brier;
  first.result.races[0].outcomes[0].winProbability = 0.5;
  assert.equal(summarizeHouseComparison(verified).primary.brier, expected);
});

test("standalone score preserves tied favorites and missing observed outcomes without changing probability mass", () => {
  const p = makeHousePublication();
  p.result.races[0].outcomes.forEach((o) => (o.winProbability = 0.5));
  delete p.result.races[1].observed;
  const summary = summarizeHouseResult(p.result);
  assert.equal(summary.observed, 15);
  assert.equal(summary.decisive, 14);
  assert.equal(summary.ties, 1);
  assert.ok(summary.brier >= 0 && summary.brier <= 2);
  p.result.races.forEach((r) => delete r.observed);
  assert.equal(summarizeHouseResult(p.result).brier, null);
  assert.throws(() => summarizeHouseResult({}), TypeError);
});

test("canonical object hashes are deterministic and distinct from source file-byte hashes", async () => {
  const value = { b: 2, a: { d: 4, c: 3 } };
  assert.equal(
    await houseComparisonDigest(value),
    createHash("sha256")
      .update(houseComparisonCanonicalJson(value))
      .digest("hex"),
  );
  assert.equal(
    houseComparisonCanonicalJson(value),
    houseComparisonCanonicalJson({ a: { c: 3, d: 4 }, b: 2 }),
  );
  assert.throws(() => houseComparisonCanonicalJson({ value: Infinity }));
});

const sourceRoot =
  process.env.BALLOTMANCY_HOUSE_RESEARCH_ROOT ??
  "/home/hayden/projects/ballotmancy-house-benchmark";
test(
  "source-bound generator exactly reproduces the reviewed manifest",
  {
    skip: !existsSync(
      path.join(sourceRoot, "research/house-benchmark-20261020/HANDOFF.md"),
    ),
  },
  async () => {
    const reproduced = await buildHouseComparisonManifest({
      sourceRoot,
      publicationDirectory: "data/house",
    });
    assert.deepEqual(reproduced, manifest);
  },
);
