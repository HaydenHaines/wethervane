import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import test from "node:test";

const root = new URL("../", import.meta.url);
const receipt = JSON.parse(readFileSync(new URL("data/polling-selection-20261020.json", root)));
const registry = JSON.parse(readFileSync(new URL("public/data/explorer/registry.json", root)));
const byId = new Map(receipt.records.map(row => [row.id, row]));
const record = id => {
  assert.ok(byId.has(id), `Missing release ${id}`);
  return byId.get(id);
};

test("receipt preserves 63 questions, 49 included and 14 explicitly excluded", () => {
  assert.equal(receipt.schemaVersion, 1);
  assert.equal(receipt.cutoffAt, "2026-10-20T16:21:44Z");
  assert.equal(receipt.records.length, 63);
  assert.equal(byId.size, 63);
  assert.equal(receipt.records.filter(row => row.status === "included").length, 49);
  assert.equal(receipt.records.filter(row => row.status === "excluded").length, 14);
  assert.equal(receipt.records.filter(row => row.reason.startsWith("A newer sample")).length, 5);
  for (const row of receipt.records) {
    assert.match(row.raceId, /^2026:[A-Z]{2}:senate:(regular|special)$/);
    assert.equal(row.state, row.raceId.split(":")[1]);
    assert.ok(row.reason.length > 20);
    assert.ok(row.pollster.trim());
    assert.ok(["included", "excluded"].includes(row.status));
    assert.equal(new URL(row.sourceUrl).protocol, "https:");
    for (const key of ["fieldStart", "fieldEnd", "publicationDate"]) {
      assert.match(row[key], /^2026-\d{2}-\d{2}$/);
      assert.ok(row[key] <= receipt.cutoffAt.slice(0, 10));
    }
    assert.ok(row.fieldStart <= row.fieldEnd);
    assert.ok(row.questionKind.length > 5);
    assert.ok(!["LV", "RV", "voters_unspecified"].includes(row.questionUniverse));
  }
});

test("shared sample alternatives never inherit the selected question's status", () => {
  for (const [sample, selected, excluded] of [
    ["advanced-targeting-research-id-2026-09-13-2026-09-16", "three_candidate", "two_candidate"],
    ["targoz-market-research-tn-2026-08-15-2026-08-26", "senate_LV", "senate_total"],
    ["co-efficient-oh-2026-10-05-2026-10-06", "Q6", "Q7"],
    ["co-efficient-ia-2026-10-15-2026-10-19", "Q8", "Q9"],
    ["co-efficient-ne-2026-09-28-2026-09-30", "Q6", "Q7"],
  ]) {
    assert.equal(record(`${sample}:${selected}`).status, "included");
    assert.equal(record(`${sample}:${excluded}`).status, "excluded");
    assert.match(record(`${sample}:${excluded}`).reason, /same survey/);
  }
  assert.equal(record("targoz-market-research-tn-2026-08-15-2026-08-26:senate_LV").questionUniverse, "Likely voters");
  assert.equal(record("targoz-market-research-tn-2026-08-15-2026-08-26:senate_total").questionUniverse, "Registered voters");
  assert.match(record("public-policy-polling-il-2026-09-30-2026-10-01:ballot").questionUniverse, /unspecified by the publisher/);
});

test("holds and ranked-choice target exclusions retain their substantive reasons", () => {
  const sc = record("insideradvantage-sc-2026-09-28-2026-09-29:ballot");
  assert.equal(sc.status, "excluded");
  assert.match(sc.reason, /publication date was provisional/);
  const sd = record("concord-public-opinion-partners-sd-2026-10-06-2026-10-08:ballot");
  assert.equal(sd.status, "excluded");
  assert.match(sd.reason, /original release was unavailable/);
  for (const question of ["Q6", "Q7"]) {
    const row = record(`co-efficient-ak-2026-09-14-2026-09-17:${question}`);
    assert.equal(row.status, "excluded");
    assert.match(row.reason, /does not measure the elimination and transfer sequence/);
  }
});

test("exact artifact fingerprints gate all five polling runs and every race count", () => {
  assert.deepEqual(Object.keys(receipt.applicableRuns).sort(), ["current-cycle-high", "current-cycle-low", "current-error-high", "current-error-low", "current-polling"]);
  const included = new Map();
  for (const row of receipt.records.filter(row => row.status === "included")) included.set(row.raceId, (included.get(row.raceId) || 0) + 1);
  for (const [runId, digest] of Object.entries(receipt.applicableRuns)) {
    const entry = registry.runs.find(run => run.id === runId);
    assert.equal(digest, entry.sha256);
    const bytes = readFileSync(new URL(`public${entry.href}`, root));
    assert.equal(createHash("sha256").update(bytes).digest("hex"), digest);
    const bundle = JSON.parse(bytes);
    assert.equal(bundle.run.cutoffAt, receipt.cutoffAt);
    assert.equal(bundle.races.length, 35);
    for (const race of bundle.races) assert.equal(race.pollCount, included.get(race.id) || 0, `${runId}: ${race.id}`);
    assert.equal(bundle.races.reduce((total, race) => total + race.pollCount, 0), 49);
  }
  assert.ok(!("current-fundamentals" in receipt.applicableRuns));
});

test("sanitized provenance contains hashes and no source paths or raw source content", () => {
  for (const [key, digest] of Object.entries(receipt.sourceHashes)) {
    assert.match(key, /^[a-zA-Z0-9:-]+$/);
    assert.match(digest, /^[a-f0-9]{64}$/);
  }
  assert.match(receipt.sourceHashes.selectionCode, /^43f1f564/);
  assert.ok(Object.keys(receipt.sourceHashes).some(key => key.startsWith("acquiredEvidence")));
  const serialized = JSON.stringify(receipt);
  assert.ok(!serialized.includes("/home/"));
  assert.ok(!serialized.includes("source_files"));
  assert.ok(!serialized.includes("qualifiers"));
  assert.ok(!serialized.includes("source-snapshots"));
});

test("exporter rejects unexplained exclusions and changed dependent-question selection", () => {
  const script = `
import importlib.util, json
spec = importlib.util.spec_from_file_location('selection', 'scripts/export_poll_selection.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
cases = [
 ({'release_id':'new-unreviewed-question','sample_id':'new-sample','qualifiers':[], 'admission_status':'primary_documented_qualified_aggregate_only'}, {}),
 ({'release_id':'advanced-targeting-research-id-2026-09-13-2026-09-16:two_candidate', 'sample_id':'shared', 'qualifiers':['choose three-candidate test once'], 'admission_status':'primary_documented_qualified_aggregate_only'}, {'shared':{'question_id':'wrong-alternative'}}),
 ({'release_id':'held','sample_id':'sample','qualifiers':[], 'admission_status':'held_publication_date_not_verified'}, {}),
]
rejected = []
for row, selected in cases:
 try: module.unselected_reason(row, selected)
 except ValueError: rejected.append(True)
 else: rejected.append(False)
print(json.dumps(rejected))
`;
  const output = execFileSync("python3", ["-c", script], { cwd: fileURLToPath(root), encoding: "utf8" });
  assert.deepEqual(JSON.parse(output), [true, true, true]);
});
