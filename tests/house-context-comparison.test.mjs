import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdtemp, mkdir, readFile, writeFile, rm, symlink } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { importHouseContextComparison } from "../scripts/import_house_context_comparison.mjs";
import { matchHouseContextComparison, projectHouseContextComparisonCase } from "../lib/house-context-comparison.ts";

const root = resolve("data/house-live");
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
const clone = value => structuredClone(value);
const native = JSON.parse(await readFile(join(root, "results/house-current-polling.json"), "utf8"));
const publicationPin = { path: "publication.json", sha256: hash(await readFile(join(root, "publication.json"))) };
const catalogPin = { path: "catalog.json", sha256: hash(await readFile(join(root, "catalog.json"))) };
const resultPin = { path: "results/house-current-polling.json", sha256: hash(await readFile(join(root, "results/house-current-polling.json"))) };
const baselineResultPin = { path: "results/house-current-fundamentals.json",
  sha256: hash(await readFile(join(root, "results/house-current-fundamentals.json"))) };
const baselineNative = JSON.parse(await readFile(join(root, baselineResultPin.path), "utf8"));
const put = async (dir, name, value) => {
  const bytes = Buffer.from(`${JSON.stringify(value, null, 2)}\n`);
  await writeFile(join(dir, name), bytes);
  return { path: name, sha256: hash(bytes) };
};
const caseFor = (district, rating, favored) => {
  const race = native.races.find(r => r.district_key === district);
  const pair = race.conditional_shares;
  return { district_key: district, race_id: race.race_id, cook_rating: rating,
    cook_favored_party: favored, current_model_id: native.id,
    forecast_issued_at: native.issued_at, forecast_cutoff: native.forecast_cutoff,
    baseline_issued_at: native.issued_at,
    party_win_probability: Object.fromEntries(["D", "R", "O"].map(p => [p, race.outcomes.find(o => o.party === p)?.p_win ?? 0])),
    delta_party_win_probability: { D: 0, R: 0, O: 0 }, poll_count: race.poll_count,
    baseline_poll_count: race.poll_count, candidate_records_changed: false,
    principal_pair_changed: false,
    ordered_principal_ids: pair ? [pair.first_candidate_id, pair.second_candidate_id] : [null, null],
    baseline_ordered_principal_ids: pair ? [pair.first_candidate_id, pair.second_candidate_id] : [null, null],
    conditional_pair_denominator: pair?.denominator ?? null, first_candidate_pair_share: pair?.mean ?? null,
    rule_family: race.rule_family, baseline_source_finding: "Invented test-only finding.",
    interpretation: "Invented test-only interpretation; no publication authorization." };
};

async function setup(t) {
  const base = await mkdtemp(join(tmpdir(), "house-context-test-"));
  t.after(() => rm(base, { recursive: true, force: true }));
  const owner = join(base, "owner"), operator = join(base, "operator"), policyRoot = join(base, "source-policy");
  await Promise.all([owner, operator, policyRoot].map(path => mkdir(path)));
  const ready = await put(owner, "ready.json", { schema: "HouseLiveReadyPacketV1",
    status: "scientific_packet_ready_for_website_review", publication_qualified: false,
    catalog: { path: "native/catalog.json", sha256: catalogPin.sha256 } });
  const receipt = await put(owner, "receipt.json", { schema: "HouseReviewedCookComparisonV2",
    status: "research_triage_not_forecast_evaluation", publication_qualified: false,
    current_ready_packet: ready, baseline_ready_packet: ready, cook_snapshot: "Synthetic date",
    input_pins: [catalogPin, resultPin] });
  const cases = [caseFor("CO-04", "Lean R", "R"), caseFor("CA-06", "Solid D", "D")];
  const handoff = await put(owner, "handoff.json", { schema: "HouseCookPriorityCasesHandoffV1",
    status: "private_research_preparation_not_website_admission", publication_qualified: false,
    cycle: 2026, map_vintage: "house_election_map_2026", self_comparison: true,
    current_ready_packet: ready, baseline_ready_packet: ready, comparison_receipt: receipt,
    cook_snapshot: "Synthetic date", source_url: "https://example.test/invented-ratings",
    cases, input_pins: [], contract: { cook_scale: "ordinal_categories_without_assigned_probabilities",
      missing: "null_is_inapplicable_not_zero" } });
  const handoffReview = await put(owner, "review.json", { schema: "HouseCookPriorityCasesHandoffReviewV1",
    status: "passed_for_private_frontend_preparation", publication_qualified: false,
    handoff: { path: handoff.path, sha256: handoff.sha256 } });
  const edition = { publication: publicationPin, catalog: catalogPin, result: resultPin,
    readyPacket: ready, runId: native.id };
  const operatorValue = { schema: "HouseContextExcerptOperatorApprovalV1",
    status: "approved_for_offline_normalization", scope: "private_selected_excerpts_only",
    handoffSha256: handoff.sha256, handoffReviewSha256: handoffReview.sha256,
    comparisonReceiptSha256: receipt.sha256, currentReadyPacket: ready, baselineReadyPacket: ready,
    selectedDistricts: ["CO-04", "CA-06"], publicationQualified: false,
    currentEdition: edition, baselineEdition: edition };
  const policyValue = { schema: "HouseContextExcerptSourcePolicyV1",
    status: "approved_for_private_excerpts", distribution: "private_research_only",
    publicDisplayApproved: false, handoffSha256: handoff.sha256,
    sourceUrl: "https://example.test/invented-ratings", cookSnapshot: "Synthetic date",
    permittedDistricts: ["CO-04", "CA-06"] };
  const operatorApproval = await put(operator, "approval.json", operatorValue);
  const sourcePolicyApproval = await put(policyRoot, "source-policy.json", policyValue);
  return { base, owner, operator, policyRoot, ready, receipt, handoff, handoffReview,
    operatorApproval, sourcePolicyApproval, operatorValue, policyValue, cases,
    options: { ownerRoot: owner, currentSiteRoot: root, baselineSiteRoot: root,
      operatorRoot: operator, sourcePolicyRoot: policyRoot, outputDir: join(base, "stage"),
      handoff, handoffReview, operatorApproval, sourcePolicyApproval } };
}

test("offline importer keeps selected ordinal excerpts private and binds both exact editions", async t => {
  const f = await setup(t);
  const { catalog, manifest } = await importHouseContextComparison(f.options);
  assert.equal(catalog.publicationQualified, false);
  assert.equal(manifest.externalActivation, false);
  assert.equal(catalog.cases.length, 2);
  assert.equal(catalog.cases[0].current.winnerGroups.R?.kind, "group");
  assert.equal(catalog.cases[0].current.winnerGroups.R?.memberCandidateIds.length, 2);
  assert.equal(catalog.cases[1].current.winnerGroups.R, null);
  assert.equal(catalog.cases[1].current.pair?.secondCandidateId,
    native.races.find(r => r.district_key === "CA-06").conditional_shares.second_candidate_id);
  assert.equal(catalog.cases[1].current.candidates.find(c => c.id === catalog.cases[1].current.pair.secondCandidateId).aggregationParty, "O");
  assert.equal(catalog.cases[1].cook.scale, "ordinal_no_probability");
  assert.equal(catalog.cases[0].pairComparison.status, "comparable");
  assert.equal(catalog.cases[0].pairComparison.firstShareMeanDelta, 0);
  assert.equal(matchHouseContextComparison(catalog, catalog.currentEdition, catalog.baselineEdition), catalog);
  assert.equal(matchHouseContextComparison(catalog,
    { ...catalog.currentEdition, resultSha256: "f".repeat(64) }, catalog.baselineEdition), null);
  assert.equal(matchHouseContextComparison(catalog,
    catalog.currentEdition, { ...catalog.baselineEdition, publicationSha256: "f".repeat(64) }), null);
  assert.equal(matchHouseContextComparison({ ...catalog, sourcePins: {} },
    catalog.currentEdition, catalog.baselineEdition), null);
  assert.equal(matchHouseContextComparison({ ...catalog, cases: [catalog.cases[0], catalog.cases[0]] },
    catalog.currentEdition, catalog.baselineEdition), null);
});

test("approval roots cannot contain site inputs or escape through a symlink", async t => {
  const f = await setup(t);
  await assert.rejects(importHouseContextComparison({ ...f.options, operatorRoot: resolve("data"),
    outputDir: join(f.base, "nested-stage") }), /Approval trust roots/);
  const link = join(f.base, "operator-link");
  await symlink(resolve("data"), link);
  await assert.rejects(importHouseContextComparison({ ...f.options, operatorRoot: link,
    outputDir: join(f.base, "symlink-stage") }), /Approval trust roots/);
  await assert.rejects(importHouseContextComparison({ ...f.options, sourcePolicyRoot: f.base,
    outputDir: join(f.base, "parent-stage") }), /Approval trust roots/);
});

test("distinct current and baseline editions require independently mapped baseline run IDs", async t => {
  const f = await setup(t);
  const old = JSON.parse(await readFile(join(f.owner, f.handoff.path), "utf8"));
  const baselineReady = await put(f.owner, "baseline-ready.json", { schema: "HouseLiveReadyPacketV1",
    status: "scientific_packet_ready_for_website_review", publication_qualified: false,
    catalog: { path: "native/catalog.json", sha256: catalogPin.sha256 },
    syntheticDistinctEdition: true });
  const receipt = await put(f.owner, "distinct-receipt.json", { schema: "HouseReviewedCookComparisonV2",
    status: "research_triage_not_forecast_evaluation", publication_qualified: false,
    current_ready_packet: f.ready, baseline_ready_packet: baselineReady,
    cook_snapshot: "Synthetic date", input_pins: [catalogPin, resultPin, baselineResultPin] });
  const selected = caseFor("CA-06", "Solid D", "D");
  const currentRace = native.races.find(r => r.district_key === "CA-06");
  const baselineRace = baselineNative.races.find(r => r.district_key === "CA-06");
  selected.baseline_issued_at = baselineNative.issued_at;
  selected.baseline_poll_count = baselineRace.poll_count;
  selected.candidate_records_changed = JSON.stringify(currentRace.candidates) !== JSON.stringify(baselineRace.candidates);
  selected.principal_pair_changed = JSON.stringify(currentRace.conditional_shares && [currentRace.conditional_shares.first_candidate_id,
    currentRace.conditional_shares.second_candidate_id, currentRace.conditional_shares.denominator]) !==
    JSON.stringify(baselineRace.conditional_shares && [baselineRace.conditional_shares.first_candidate_id,
      baselineRace.conditional_shares.second_candidate_id, baselineRace.conditional_shares.denominator]);
  selected.baseline_ordered_principal_ids = baselineRace.conditional_shares ?
    [baselineRace.conditional_shares.first_candidate_id, baselineRace.conditional_shares.second_candidate_id] : [null, null];
  selected.delta_party_win_probability = Object.fromEntries(["D", "R", "O"].map(p => [p,
    (currentRace.outcomes.find(o => o.party === p)?.p_win ?? 0) -
    (baselineRace.outcomes.find(o => o.party === p)?.p_win ?? 0)]));
  const handoff = await put(f.owner, "distinct-handoff.json", { ...old, self_comparison: false,
    current_ready_packet: f.ready, baseline_ready_packet: baselineReady,
    comparison_receipt: receipt, cases: [selected] });
  const handoffReview = await put(f.owner, "distinct-review.json", { schema: "HouseCookPriorityCasesHandoffReviewV1",
    status: "passed_for_private_frontend_preparation", publication_qualified: false,
    handoff: { path: handoff.path, sha256: handoff.sha256 } });
  const approval = { ...f.operatorValue, handoffSha256: handoff.sha256,
    handoffReviewSha256: handoffReview.sha256, comparisonReceiptSha256: receipt.sha256,
    baselineReadyPacket: baselineReady, selectedDistricts: ["CA-06"],
    baselineEdition: { ...f.operatorValue.baselineEdition, readyPacket: baselineReady,
      result: baselineResultPin, runId: baselineNative.id } };
  const sourcePolicy = { ...f.policyValue, handoffSha256: handoff.sha256,
    permittedDistricts: ["CA-06"] };
  const noMap = await put(f.operator, "distinct-no-map.json", approval);
  const sourcePolicyApproval = await put(f.policyRoot, "distinct-policy.json", sourcePolicy);
  const request = { ...f.options, handoff, handoffReview, operatorApproval: noMap,
    sourcePolicyApproval, outputDir: join(f.base, "distinct-stage") };
  await assert.rejects(importHouseContextComparison(request), /baseline run mapping/);
  const mapped = await put(f.operator, "distinct-mapped.json", { ...approval,
    baselineModelByDistrict: { "CA-06": baselineNative.id } });
  const { catalog } = await importHouseContextComparison({ ...request, operatorApproval: mapped });
  assert.equal(catalog.selfComparison, false);
  assert.equal(catalog.currentEdition.resultSha256, resultPin.sha256);
  assert.equal(catalog.baselineEdition.resultSha256, baselineResultPin.sha256);
  assert.equal(catalog.cases[0].baseline.pollCount, baselineRace.poll_count);
  assert.equal(catalog.cases[0].cook.scale, "ordinal_no_probability");
  assert.equal(matchHouseContextComparison(catalog, catalog.currentEdition,
    { ...catalog.baselineEdition, resultSha256: resultPin.sha256 }), null);
});

test("offline import rejects missing source permission, stale approval and mixed publication/result", async t => {
  const f = await setup(t);
  const noPolicy = await put(f.policyRoot, "denied.json", { ...f.policyValue, publicDisplayApproved: true });
  await assert.rejects(importHouseContextComparison({ ...f.options, sourcePolicyApproval: noPolicy,
    outputDir: join(f.base, "stage-denied") }), /source-policy permission|Public use/);
  const stale = await put(f.operator, "stale.json", { ...f.operatorValue, handoffSha256: "f".repeat(64) });
  await assert.rejects(importHouseContextComparison({ ...f.options, operatorApproval: stale,
    outputDir: join(f.base, "stage-stale") }), /operator approval/);
  const mixed = await put(f.operator, "mixed.json", { ...f.operatorValue,
    currentEdition: { ...f.operatorValue.currentEdition, result: { ...resultPin, sha256: "f".repeat(64) } } });
  await assert.rejects(importHouseContextComparison({ ...f.options, operatorApproval: mixed,
    outputDir: join(f.base, "stage-mixed") }), /Artifact SHA differs|publication/);
});

test("changed ordered pair or candidate support suppresses numerical comparisons", () => {
  const raw = clone(caseFor("CA-06", "Solid D", "D"));
  const prior = clone(native);
  const race = prior.races.find(r => r.district_key === "CA-06");
  [race.conditional_shares.first_candidate_id, race.conditional_shares.second_candidate_id] =
    [race.conditional_shares.second_candidate_id, race.conditional_shares.first_candidate_id];
  raw.principal_pair_changed = true;
  raw.baseline_ordered_principal_ids.reverse();
  const pairChanged = projectHouseContextComparisonCase(raw, native, prior);
  assert.deepEqual(pairChanged.pairComparison, { status: "incompatible",
    reason: "The ordered named pair or denominator changed between editions." });
  assert.throws(() => projectHouseContextComparisonCase({ ...raw, principal_pair_changed: false }, native, prior), /named-pair identity/);
  const support = clone(native);
  support.races.find(r => r.district_key === "CA-06").candidates[0].party_label = "Changed candidate record";
  const supportRaw = { ...caseFor("CA-06", "Solid D", "D"), candidate_records_changed: true };
  const changed = projectHouseContextComparisonCase(supportRaw, native, support);
  assert.equal(changed.winnerGroupDelta.D, null);
  assert.match(changed.winnerGroupDeltaReason, /Candidate records changed/);
  assert.equal(changed.pairComparison.status, "incompatible");
});

test("inapplicable pair remains null and D/O candidate groups remain distinct", () => {
  const raw = caseFor("CA-06", "Solid D", "D");
  const a = clone(native), b = clone(native);
  a.races.find(r => r.district_key === "CA-06").conditional_shares = null;
  b.races.find(r => r.district_key === "CA-06").conditional_shares = null;
  raw.ordered_principal_ids = [null, null];
  raw.baseline_ordered_principal_ids = [null, null];
  raw.conditional_pair_denominator = null;
  raw.first_candidate_pair_share = null;
  const result = projectHouseContextComparisonCase(raw, a, b);
  assert.equal(result.current.pair, null);
  assert.equal(result.baseline.pair, null);
  assert.equal(result.pairComparison.status, "incompatible");
  assert.equal(result.current.winnerGroups.R, null);
  assert.equal(result.current.winnerGroups.O?.kind, "candidate");
});

const ownerRoot = resolve("..", "ballotmancy-house-benchmark");
const actualHandoffPath = join(ownerRoot, "research/house-disagreement-20261026/comparison-handoff-v2.json");
test("local old-release handoff projects seven private cases with synthetic test approvals only",
  { skip: !existsSync(actualHandoffPath) }, async t => {
    const base = await mkdtemp(join(tmpdir(), "house-context-owner-probe-"));
    t.after(() => rm(base, { recursive: true, force: true }));
    const operator = join(base, "operator"), policyRoot = join(base, "source-policy");
    await Promise.all([operator, policyRoot].map(path => mkdir(path)));
    const handoff = { path: "research/house-disagreement-20261026/comparison-handoff-v2.json",
      sha256: hash(await readFile(actualHandoffPath)) };
    const handoffReview = { path: "research/house-disagreement-20261026/comparison-handoff-root-review-v1.json",
      sha256: hash(await readFile(join(ownerRoot, "research/house-disagreement-20261026/comparison-handoff-root-review-v1.json"))) };
    const raw = JSON.parse(await readFile(actualHandoffPath, "utf8"));
    const selectedDistricts = raw.cases.map(c => c.district_key);
    const edition = { publication: publicationPin, catalog: catalogPin, result: resultPin,
      readyPacket: raw.current_ready_packet, runId: native.id };
    const operatorApproval = await put(operator, "invented-approval.json", {
      schema: "HouseContextExcerptOperatorApprovalV1", status: "approved_for_offline_normalization",
      scope: "private_selected_excerpts_only", handoffSha256: handoff.sha256,
      handoffReviewSha256: handoffReview.sha256,
      comparisonReceiptSha256: raw.comparison_receipt.sha256,
      currentReadyPacket: raw.current_ready_packet, baselineReadyPacket: raw.baseline_ready_packet,
      selectedDistricts, publicationQualified: false, currentEdition: edition, baselineEdition: edition,
      testOnlyInventedApproval: true });
    const sourcePolicyApproval = await put(policyRoot, "invented-policy.json", {
      schema: "HouseContextExcerptSourcePolicyV1", status: "approved_for_private_excerpts",
      distribution: "private_research_only", publicDisplayApproved: false,
      handoffSha256: handoff.sha256, sourceUrl: raw.source_url,
      cookSnapshot: raw.cook_snapshot, permittedDistricts: selectedDistricts,
      testOnlyInventedApproval: true });
    const { catalog } = await importHouseContextComparison({ ownerRoot,
      currentSiteRoot: root, baselineSiteRoot: root, operatorRoot: operator,
      sourcePolicyRoot: policyRoot, outputDir: join(base, "stage"),
      handoff, handoffReview, operatorApproval, sourcePolicyApproval });
    assert.equal(catalog.cases.length, 7);
    assert.equal(catalog.selfComparison, true);
    assert.equal(catalog.cases.find(c => c.districtKey === "AK-00").ruleFamily, "ranked_choice");
    assert.equal(catalog.cases.find(c => c.districtKey === "CO-04").current.winnerGroups.R.kind, "group");
    assert.equal(catalog.cases.find(c => c.districtKey === "CA-06").current.winnerGroups.R, null);
    assert.equal(catalog.cases.every(c => c.winnerGroupDelta.D === 0 || c.winnerGroupDelta.D === null), true);
  });
