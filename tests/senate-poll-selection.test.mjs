import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";
import { senatePollReceiptFor } from "../lib/senate-poll-policy.ts";

const root = new URL("../", import.meta.url);
const readJson = (url) => JSON.parse(readFileSync(url, "utf8"));
const receipt = readJson(new URL("data/polling-selection-20261026.json", root));
const oldReceipt = readJson(new URL("data/polling-selection-20261020.json", root));
const registry = readJson(new URL("public/data/explorer/registry.json", root));
const byId = new Map(receipt.records.map((row) => [row.id, row]));
const record = (id) => {
  assert.ok(byId.has(id), `Missing release ${id}`);
  return byId.get(id);
};

test("October 26 receipt accounts for the reviewed question set and date/provider gaps", () => {
  assert.equal(receipt.schemaVersion, 2);
  assert.equal(receipt.cutoffAt, "2026-10-26T15:03:57.212405+00:00");
  assert.equal(receipt.records.length, 187);
  assert.equal(byId.size, 187);
  assert.equal(receipt.records.filter((row) => row.status === "included").length, 131);
  assert.equal(receipt.records.filter((row) => row.status === "excluded").length, 56);
  assert.equal(receipt.unknownPublicationDates, 22);
  assert.equal(
    new Set(receipt.records.filter((row) => row.status === "included").map((row) => row.raceId)).size,
    25,
  );

  const included = receipt.records.filter((row) => row.status === "included");
  const unknownDates = included.filter((row) => row.publicationDate === null);
  assert.equal(unknownDates.length, 22);
  for (const row of unknownDates) {
    assert.equal(row.publicationDateVerified, false, row.id);
    assert.ok(row.availableBy, `${row.id} must retain its availability bound`);
    assert.ok(Date.parse(row.availableBy) <= Date.parse(receipt.cutoffAt), row.id);
  }

  const providerUnknown = included.filter((row) => row.actualFieldworkProvider === null);
  assert.equal(providerUnknown.length, 113);
  for (const row of providerUnknown) {
    assert.match(row.providerIdentityStatus, /proxy_actual_subcontractor_unknown/, row.id);
  }
  for (const row of included.filter((item) => item.actualFieldworkProvider !== null)) {
    assert.ok(row.actualFieldworkProvider.trim(), row.id);
    assert.doesNotMatch(row.providerIdentityStatus, /proxy_actual_subcontractor_unknown/, row.id);
  }
});

test("question sample size stays separate from overall survey N and all source URLs are safe", () => {
  const akFinal = record("dfp-ak-2026-10-08-2026-10-16:Q18");
  assert.equal(akFinal.questionSampleSize, 671);
  assert.equal(akFinal.nominalSampleSize, 694);
  assert.equal(akFinal.nominalSampleUniverse, "Likely voters");

  const akInsider = record("insideradvantage-ak-2026-10-09-2026-10-10:ballot");
  assert.equal(akInsider.questionSampleSize, null);
  assert.equal(akInsider.nominalSampleSize, 1200);
  assert.equal(akInsider.nominalSampleUniverse, "Likely voters");
  assert.equal(
    receipt.records.filter((row) => row.status === "included" && row.questionSampleSize === null && row.nominalSampleSize !== null).length,
    77,
  );

  for (const row of receipt.records) {
    const url = new URL(row.sourceUrl);
    assert.ok(["http:", "https:"].includes(url.protocol), row.id);
    assert.ok(url.hostname && !url.username && !url.password, row.id);
    assert.match(row.raceId, /^2026:[A-Z]{2}:senate:(regular|special)$/);
    assert.ok(row.pollster.trim(), row.id);
    assert.ok(row.reason.length > 20, row.id);
    assert.ok(row.questionUniverse.length > 5, row.id);
    assert.ok(row.questionKind.length > 5, row.id);
  }
});

test("same-survey alternatives and source/date holds retain distinct reviewed reasons", () => {
  for (const id of [
    "dfp-ak-2026-10-08-2026-10-16:Q15",
    "dfp-ak-2026-10-08-2026-10-16:Q16",
    "dfp-ak-2026-10-08-2026-10-16:Q19",
  ]) {
    const row = record(id);
    assert.equal(row.status, "excluded");
    assert.match(row.reason, /final ranked-choice preferences/);
    assert.ok(row.preferredSurveyQuestions.some((question) =>
      question.id === "dfp-ak-2026-10-08-2026-10-16:Q18" && question.included,
    ));
  }

  const dateHold = record("insideradvantage-sc-2026-09-28-2026-09-29:ballot");
  assert.equal(dateHold.status, "excluded");
  assert.match(dateHold.reason, /publication date has not been verified/);
  const sourceHold = record("concord-public-opinion-partners-sd-2026-10-06-2026-10-08:ballot");
  assert.equal(sourceHold.status, "excluded");
  assert.match(sourceHold.reason, /original release was unavailable/);
});

test("all ten refreshed polling runs require their exact registry and bundle fingerprints", () => {
  const pollingRuns = registry.runs.filter(
    (run) => run.year === 2026 && !run.historical && run.components.polling === "standard",
  );
  assert.equal(pollingRuns.length, 10);
  assert.deepEqual(Object.keys(receipt.applicableRuns).sort(), pollingRuns.map((run) => run.id).sort());

  const counts = new Map();
  for (const row of receipt.records.filter((item) => item.status === "included"))
    counts.set(row.raceId, (counts.get(row.raceId) || 0) + 1);

  for (const run of pollingRuns) {
    const expectedHash = receipt.applicableRuns[run.id];
    assert.equal(expectedHash, run.sha256, run.id);
    const bytes = readFileSync(new URL(`public${run.href}`, root));
    assert.equal(createHash("sha256").update(bytes).digest("hex"), expectedHash, run.id);
    const bundle = JSON.parse(bytes);
    assert.equal(bundle.run.id, run.id);
    assert.equal(bundle.run.cutoffAt, receipt.cutoffAt);
    assert.equal(bundle.races.length, 35);
    for (const race of bundle.races)
      assert.equal(race.pollCount, counts.get(race.id) || 0, `${run.id}: ${race.id}`);
    assert.equal(bundle.races.reduce((sum, race) => sum + race.pollCount, 0), 131, run.id);
  }
});

test("a wrong fingerprint cannot borrow either edition's receipt", () => {
  assert.equal(receipt.applicableRuns["current-polling"].length, 64);
  assert.notEqual(
    receipt.applicableRuns["current-polling"],
    oldReceipt.applicableRuns["current-polling"],
    "refreshed and October 20 artifacts must not share an applicability fingerprint",
  );
  const current = readJson(new URL("public/data/explorer/current-polling.json", root));
  const previous = readJson(new URL("public/data/explorer/editions/2026-10-20/current-polling.json", root));
  const receipts = [receipt, oldReceipt];
  assert.equal(senatePollReceiptFor(receipts, current, "0".repeat(64)), null);
  assert.equal(senatePollReceiptFor(receipts, current, receipt.applicableRuns["current-polling"]), receipt);
  assert.equal(senatePollReceiptFor(receipts, previous, oldReceipt.applicableRuns["current-polling"]), oldReceipt);
  assert.equal(senatePollReceiptFor(receipts, current, oldReceipt.applicableRuns["current-polling"]), null);
  const wrongCount = structuredClone(current); wrongCount.races[0].pollCount++;
  assert.equal(senatePollReceiptFor(receipts, wrongCount, receipt.applicableRuns["current-polling"]), null);
});
