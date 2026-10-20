import assert from "node:assert/strict";
import test from "node:test";
import {
  readFileSync,
  existsSync,
  mkdtempSync,
  mkdirSync,
  writeFileSync,
  rmSync,
} from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import {
  parseHouseScoreEvidenceManifest,
  verifyHouseScoreEvidence,
  HOUSE_SCORE_EVIDENCE_ENTRIES_SHA256,
} from "../lib/house-score-evidence.ts";
import { houseComparisonDigest } from "../lib/house-comparison.ts";
import {
  buildHouseScoreEvidence,
  validateHouseScoreRows,
  parseCsv,
} from "../scripts/build_house_score_evidence.mjs";
const read = (p) => JSON.parse(readFileSync(p, "utf8"));
const manifest = read("data/house-score-evidence.json");
const publication = read("data/house/publication.json");
const catalog = read(path.join("data/house", publication.catalog.file));
const load = (id) => {
  const e = catalog.runs.find((r) => r.id === id);
  return {
    result: read(path.join("data/house", e.result.file)),
    resultSha256: e.resultSha256,
  };
};
const fixed = () => load("house_history_refreshed_v1-2024-h14");
const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-12, `${a} != ${b}`);
const sourceRoot = "/home/hayden/projects/ballotmancy-house-benchmark";
const base = path.join(sourceRoot, "research/house-benchmark-20261020");
const hasSource = existsSync(base);

test("manifest is summary-only with exact metric units, source pins and immutable entries", async () => {
  assert.ok(parseHouseScoreEvidenceManifest(manifest));
  assert.equal(manifest.entries.length, 16);
  assert.equal(
    await houseComparisonDigest(manifest.entries),
    HOUSE_SCORE_EVIDENCE_ENTRIES_SHA256,
  );
  assert.doesNotMatch(
    JSON.stringify(manifest),
    /\/home\/|research\/|\.npz|file:\/\//,
  );
  for (const entry of manifest.entries) {
    assert.equal(entry.units, "share_fraction");
    assert.equal(entry.intervalLevel, 0.9);
    assert.equal(entry.total, entry.raceIds.length);
    assert.equal(entry.scope, "eligible_race_subset");
    assert.equal("actual_dem_share" in entry, false);
  }
  for (const unknown of [null, {}, [], { ...manifest, id: "unreviewed" }])
    assert.equal(parseHouseScoreEvidenceManifest(unknown), null);
  for (const mutate of [
    (m) => m.entries.push(m.entries[0]),
    (m) => (m.entries[0].intervalLevel = 0.8),
    (m) => (m.entries[0].units = "margin_points"),
    (m) => (m.entries[0].covered = m.entries[0].total + 1),
    (m) => (m.entries[0].crps = NaN),
    (m) => m.entries[0].raceIds.pop(),
    (m) => (m.sourcePins.scoreReceipt = "0".repeat(64)),
  ]) {
    const m = structuredClone(manifest);
    mutate(m);
    assert.equal(parseHouseScoreEvidenceManifest(m), null);
  }
});
test("all sixteen exact public results verify and Fixed2024 uses share error, not margin error", async () => {
  for (const entry of manifest.entries) {
    const v = await verifyHouseScoreEvidence(manifest, load(entry.runId));
    assert.equal(v.status, "verified");
    assert.deepEqual(v.summary, entry);
  }
  const { summary } = await verifyHouseScoreEvidence(manifest, fixed());
  assert.equal(summary.total, 375);
  assert.equal(summary.covered, 367);
  close(summary.meanAbsoluteError, 0.021172829884947055);
  close(summary.meanIntervalWidth, 0.19147057264919518);
  close(summary.crps, 0.01911193185910748);
  close(summary.intervalScore, 0.19485721784279228);
  assert.ok(summary.meanAbsoluteError < 0.03);
});
test("stale publication, altered predictions, candidate/truth changes and altered evidence fail closed", async () => {
  for (const mutate of [
    (p) => (p.resultSha256 = "0".repeat(64)),
    (p) => (p.result.run.configurationSha256 = "0".repeat(64)),
    (p) => (p.result.races[0].outcomes[0].share.mean += 0.001),
    (p) => (p.result.races[0].outcomes[0].name = "Another candidate"),
    (p) => (p.result.races[0].observed = null),
    (p) => (p.result.metadata.mapVintage = "another-map"),
    (p) => p.result.races.pop(),
  ]) {
    const p = fixed();
    mutate(p);
    assert.equal(
      (await verifyHouseScoreEvidence(manifest, p)).status,
      "unavailable",
    );
  }
  const altered = structuredClone(manifest);
  altered.entries[0].meanAbsoluteError += 0.001;
  assert.equal(
    (await verifyHouseScoreEvidence(altered, fixed())).status,
    "unavailable",
  );
  assert.equal(
    (
      await verifyHouseScoreEvidence(manifest, {
        result: null,
        resultSha256: "",
      })
    ).status,
    "unavailable",
  );
});
test("verification snapshots caller data before asynchronous digest work", async () => {
  const m = structuredClone(manifest),
    p = fixed();
  const promise = verifyHouseScoreEvidence(m, p);
  m.entries[0].crps = 1;
  p.result.races[0].outcomes[0].share.mean = 0;
  const v = await promise;
  assert.equal(v.status, "verified");
  assert.notEqual(v.summary.crps, 1);
});
test("CSV reader preserves commas, quotes and newlines in identities; rejects truncated records", () => {
  assert.deepEqual(
    parseCsv('id,name\r\n1,"Smith, Jane"\r\n2,"A ""quoted""\nname"\r\n'),
    [
      { id: "1", name: "Smith, Jane" },
      { id: "2", name: 'A "quoted"\nname' },
    ],
  );
  assert.throws(() => parseCsv('id,name\n1,"broken'));
  assert.throws(() => parseCsv("id,name\n1"));
});
test(
  "source replay reproduces summary bytes without reconstructing posterior arrays",
  { skip: !hasSource },
  async () => {
    assert.deepEqual(
      await buildHouseScoreEvidence({
        sourceRoot,
        publicationDirectory: path.resolve("data/house"),
      }),
      manifest,
    );
  },
);
test(
  "saved rows reject missing, duplicate, changed labels, changed candidates and score definitions",
  { skip: !hasSource },
  () => {
    const result = fixed().result;
    const all = parseCsv(
      readFileSync(
        path.join(base, "final-scores-v2/per-race-scores.csv"),
        "utf8",
      ),
    );
    const selected = all.filter(
      (r) =>
        r.configuration === result.run.configurationId && +r.cycle === 2024,
    );
    const labels = parseCsv(
      readFileSync(path.join(base, "panel-v6/outcomes-SEPARATE.csv"), "utf8"),
    );
    const features = parseCsv(
      readFileSync(path.join(base, "panel-v6/features.csv"), "utf8"),
    );
    const run = (rows = selected, l = labels, f = features) =>
      validateHouseScoreRows(result, rows, l, f);
    assert.equal(run().covered, 367);
    for (const mutate of [
      (r) => r.pop(),
      (r) => r.push({ ...r[0] }),
      (r) => (r[0].mean = "0.123"),
      (r) => (r[0].absolute_error = "0"),
      (r) => (r[0].interval_score = "0"),
      (r) => (r[0].coverage = r[0].coverage === "True" ? "False" : "True"),
      (r) => (r[0].multiclass_brier = r[0].binary_brier),
      (r) => (r[0].actual_dem_share = ".5"),
      (r) => (r[0].configuration = "unreviewed"),
    ]) {
      const rows = structuredClone(selected);
      mutate(rows);
      assert.throws(() => run(rows));
    }
    const changedLabels = structuredClone(labels);
    changedLabels.find(
      (r) => r.race_id === result.races[0].id,
    ).official_dem_candidate = "Different name";
    assert.throws(() => run(selected, changedLabels));
    const changedFeatures = structuredClone(features);
    changedFeatures.find((r) => r.race_id === result.races[0].id).map_vintage =
      "new-map";
    assert.throws(() => run(selected, labels, changedFeatures));
  },
);
test(
  "generator rejects changed pinned score receipt before loading result or numerical arrays",
  { skip: !hasSource },
  async () => {
    const temporary = mkdtempSync(path.join(tmpdir(), "house-score-pins-"));
    try {
      const target = path.join(
        temporary,
        "research/house-benchmark-20261020/final-scores-v2",
      );
      mkdirSync(target, { recursive: true });
      writeFileSync(path.join(target, "receipt.json"), "{}");
      await assert.rejects(
        buildHouseScoreEvidence({
          sourceRoot: temporary,
          publicationDirectory: path.resolve("data/house"),
        }),
        /Source pin changed/,
      );
    } finally {
      rmSync(temporary, { recursive: true, force: true });
    }
  },
);

test(
  "saved central-90 coverage includes exact endpoints and rejects an outside outcome",
  { skip: !hasSource },
  () => {
    const all = parseCsv(
      readFileSync(
        path.join(base, "final-scores-v2/per-race-scores.csv"),
        "utf8",
      ),
    );
    const labels = parseCsv(
      readFileSync(path.join(base, "panel-v6/outcomes-SEPARATE.csv"), "utf8"),
    );
    const features = parseCsv(
      readFileSync(path.join(base, "panel-v6/features.csv"), "utf8"),
    );
    const result = fixed().result;
    result.races = [result.races[0]];
    const race = result.races[0];
    const original = all.find(
      (r) =>
        r.configuration === result.run.configurationId && r.race_id === race.id,
    );
    const label = labels.find((r) => r.race_id === race.id),
      feature = features.find((r) => r.race_id === race.id);
    for (const actual of [
      +original.lower,
      +original.upper,
      +original.lower - 0.0001,
    ]) {
      const row = { ...original },
        truth = { ...label };
      truth.actual_dem_share = String(actual);
      truth.dem_votes = String(actual * 100000);
      truth.rep_votes = String((1 - actual) * 100000);
      row.actual_dem_share = String(actual);
      row.absolute_error = String(Math.abs(+row.mean - actual));
      row.squared_error = String((+row.mean - actual) ** 2);
      row.coverage =
        actual >= +row.lower && actual <= +row.upper ? "True" : "False";
      row.interval_score = String(
        +row.interval_width +
          20 *
            (Math.max(+row.lower - actual, 0) +
              Math.max(actual - +row.upper, 0)),
      );
      const d = race.outcomes.find((o) => o.party === "D");
      race.observed.outcomeId = (
        actual > 0.5 ? d : race.outcomes.find((o) => o.party === "R")
      ).id;
      row.binary_brier = String((d.winProbability - Number(actual > 0.5)) ** 2);
      row.multiclass_brier = String(2 * +row.binary_brier);
      const summary = validateHouseScoreRows(result, [row], [truth], [feature]);
      assert.equal(summary.covered, Number(row.coverage === "True"));
    }
  },
);
