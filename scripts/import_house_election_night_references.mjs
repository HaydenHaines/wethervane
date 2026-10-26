/** Offline metadata-only normalization. The caller supplies an independently
 * trusted operator review pin; producer statuses are not publication admission.
 * NPZ/NPY bytes are NOT rehashed: their exact pins and scoped paths are bound to
 * the admitted independent audit. This does not establish forecast accuracy.
 */
import { createHash } from "node:crypto";
import { mkdir, readFile, realpath, rm, stat, writeFile } from "node:fs/promises";
import { basename, dirname, isAbsolute, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { parseHouseLiveCatalog } from "../lib/house-live-catalog.ts";
import { validateHouseLiveCollection } from "../lib/house-live-collection.ts";
import { parseHouseLiveResult } from "../lib/house-live-result.ts";
import { parseHouseLivePublicationQualification } from "../lib/house-live-native-qualification.ts";

const need = (test, message) => { if (!test) throw Error(message); };
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
const bytesOf = value => Buffer.from(`${JSON.stringify(value, null, 2)}\n`);
const canonical = v => Array.isArray(v) ? v.map(canonical) : v && typeof v === "object" ?
  Object.fromEntries(Object.keys(v).sort().map(k => [k, canonical(v[k])])) : v;
const same = (a, b) => JSON.stringify(canonical(a)) === JSON.stringify(canonical(b));
const object = x => !!x && typeof x === "object" && !Array.isArray(x);
const text = x => typeof x === "string" && x.trim() === x && x.length > 0 && x.length <= 10000;
const date = x => text(x) && Number.isFinite(Date.parse(x));
const inside = (root, path) => { const p = relative(root, path); return !p || (!isAbsolute(p) && p !== ".." && !p.startsWith(`..${sep}`)); };
const pin = p => object(p) && Object.keys(p).length === 2 && typeof p.path === "string" &&
  /^[A-Za-z0-9][A-Za-z0-9._/-]*$/.test(p.path) && p.path.split("/").every(v => v && v !== "." && v !== "..") &&
  /^[a-f0-9]{64}$/.test(p.sha256);
const eqPin = (a, b) => pin(a) && pin(b) && a.path === b.path && a.sha256 === b.sha256;
const contains = (list, p) => Array.isArray(list) && list.some(v => eqPin(v, p));
const positive = x => Number.isSafeInteger(x) && x > 0;
const finite = (x, lo, hi) => Number.isFinite(x) && x >= lo && x <= hi;
const notes = x => Array.isArray(x) && x.length > 0 && x.every(text);
const metricIds = ["mean_signed_pair_margin_error_pp", "mean_absolute_pair_margin_error_pp", "central90_pair_interval_coverage_count", "modal_group_miss_count"];
export const HOUSE_EXPECTED_ERROR_MODAL_TIE_RULE = "first maximum in declared D, R, O ordering; exact native probability ties recorded";

async function scoped(root, p) {
  need(pin(p), "Invalid relative artifact pin");
  const path = await realpath(resolve(root, p.path));
  need(path !== root && inside(root, path) && (await stat(path)).isFile(), "Artifact escapes explicit root or is not a file");
  return path;
}

function reader() {
  const authenticated = new Map(), delegated = new Map();
  async function bytes(root, p) {
    const path = await scoped(root, p);
    const key = `${path}:${p.sha256}`;
    if (authenticated.has(key)) return authenticated.get(key);
    need((await stat(path)).size <= 64 * 1024 * 1024, "Metadata exceeds offline import size limit");
    const b = await readFile(path);
    need(hash(b) === p.sha256, `Artifact fingerprint mismatch: ${p.path}`);
    authenticated.set(key, b);
    return b;
  }
  async function json(root, p) { return JSON.parse((await bytes(root, p)).toString("utf8")); }
  async function dependency(root, p) {
    // Raw arrays are trusted ONLY through the separately admitted exact audit.
    if (/\.(npz|npy)$/.test(p?.path ?? "")) {
      const path = await scoped(root, p);
      delegated.set(`${path}:${p.sha256}`, p);
    } else await bytes(root, p);
  }
  return { bytes, json, dependency, authenticated, delegated };
}

function validateSummaries(s, rows) {
  need(object(s) && same(Object.keys(s).sort(), [...metricIds].sort()), "Exact four joint summary metrics required");
  for (const [i, id] of metricIds.entries()) {
    const m = s[id], max = i === 2 ? 412 : i === 3 ? 435 : 200, min = i === 0 ? -200 : 0;
    need(object(m) && finite(m.mean, min, max) && finite(m.sd, 0, 1000) && finite(m.mean_mcse, 0, 1000) &&
      finite(m.rank_rhat, Number.MIN_VALUE, 1000) && object(m.quantiles) &&
      ["p05", "p50", "p95"].every(k => finite(m.quantiles[k], min, max)) &&
      m.quantiles.p05 <= m.quantiles.p50 && m.quantiles.p50 <= m.quantiles.p95 &&
      m.units === (i < 2 ? "percentage_points" : "count") && m.quantile_convention === (i < 2 ? "linear" : "inverse_cdf"),
    "Malformed metric-specific joint summary");
    if (i >= 2) {
      need(Array.isArray(m.pmf) && m.pmf.length === max + 1 && m.pmf.every(p => finite(p, 0, 1)) &&
        Math.abs(m.pmf.reduce((a, b) => a + b, 0) - 1) < 1e-8 &&
        Math.abs(m.pmf.reduce((a, p, k) => a + p * k, 0) - m.mean) < 1e-7 &&
        Object.values(m.quantiles).every(Number.isInteger), "Invalid count distribution");
      const counts = m.pmf.map(p => Math.round(p * rows));
      need(counts.every((count, k) => Math.abs(m.pmf[k] - count / rows) <= 1e-12) &&
        counts.reduce((a, b) => a + b, 0) === rows, "Count PMF does not describe original integer rows");
      const mean = counts.reduce((sum, count, k) => sum + k * count, 0) / rows;
      const sd = Math.sqrt(counts.reduce((sum, count, k) => sum + (k - mean) ** 2 * count, 0) / rows);
      need(Math.abs(mean - m.mean) <= 1e-7 && Math.abs(sd - m.sd) <= 1e-7,
        "Count PMF mean or population standard deviation differs");
      for (const [q, key] of [[.05, "p05"], [.5, "p50"], [.95, "p95"]]) {
        let cumulative = 0;
        const saved = counts.findIndex(n => (cumulative += n) >= Math.ceil(q * rows));
        need(saved === m.quantiles[key], "Count PMF inverse-CDF quantile differs");
      }
    } else need(m.pmf === undefined, "Continuous summary cannot carry a count PMF");
  }
  need(Math.abs(s.mean_signed_pair_margin_error_pp.mean) <= 1e-7, "Own-bank mean signed error must be zero");
}

function validateRaceSummaries(s, native) {
  const ids = native.races.map(r => r.race_id), pairs = native.races.filter(r => r.conditional_shares);
  need(pairs.length === 412 && same(Object.keys(s.race_expected_errors).sort(), [...ids].sort()) &&
    Array.isArray(s.modal_calls) && s.modal_calls.length === 435 &&
    same(s.draw_layout.ordered_pair_race_ids, pairs.map(r => r.race_id)) &&
    same(s.draw_layout.pair_mask_in_winner_order, native.races.map(r => !!r.conditional_shares)), "Pair cohort, mask or race identities differ");
  let expectedMisses = 0, perRaceError = 0;
  native.races.forEach((r, i) => {
    const row = s.race_expected_errors[r.race_id], p = r.conditional_shares;
    need(object(row), "Missing per-race row");
    if (p) {
      need(row.availability === "available" && row.first_candidate_id === p.first_candidate_id &&
        row.second_candidate_id === p.second_candidate_id && row.denominator === p.denominator &&
        finite(row.expected_absolute_margin_error_pp, 0, 200) &&
        finite(row.point_margin_pp, -100, 100) && Math.abs(row.point_margin_pp - (200 * p.mean - 100)) < 1e-9 &&
        same(row.qualifications, p.qualifications), "Named pair, denominator or native point differs");
      perRaceError += row.expected_absolute_margin_error_pp;
    } else need(row.availability === "inapplicable" && row.expected_absolute_margin_error_pp === null && text(row.reason) &&
      row.first_candidate_id === undefined && row.second_candidate_id === undefined, "Inapplicable pair row must remain unavailable");
    const maximum = Math.max(...r.outcomes.map(o => o.p_win));
    const tied = ["D", "R", "O"].filter(id => r.outcomes.some(o => o.outcome_id === id && o.p_win === maximum));
    const chosen = r.outcomes.find(o => o.outcome_id === tied[0]);
    const call = s.modal_calls[i];
    need(object(call) && call.race_id === r.race_id && call.outcome_id === chosen.outcome_id &&
      call.party_or_group === chosen.party && call.kind === chosen.kind &&
      same(call.member_candidate_ids, chosen.member_candidate_ids) &&
      same(call.tied_modal_outcomes, tied.length > 1 ? tied : []), "Modal call or D/R/O tie identity differs");
    expectedMisses += 1 - maximum;
  });
  need(Math.abs(s.joint_summaries.modal_group_miss_count.mean - expectedMisses) < 1e-7 &&
    Math.abs(s.joint_summaries.mean_absolute_pair_margin_error_pp.mean - perRaceError / 412) < 1e-7,
  "Summary expectation differs from frozen race summaries");
}

/** Normalize one complete, explicitly reviewed dated16 collection. All roots
 * and the trusted operator review digest must come from the operator, not from
 * the scientific producer. No network, executable driver, fit or archive decode.
 */
export async function normalizeHouseElectionNightReferences({ sourceRoot, collectionRoot, nativeRoot,
  catalog: catalogPin, receipt: receiptPin, nativeCatalog: nativePin, readyPacket: readyPin,
  reviewRoot, review: reviewPin } = {}) {
  need([sourceRoot, collectionRoot, nativeRoot, reviewRoot].every(text), "Explicit producer, collection, native and operator roots required");
  const [root, collection, nativeDirectory, operator] = await Promise.all([sourceRoot, collectionRoot, nativeRoot, reviewRoot].map(p => realpath(p)));
  need(inside(root, collection) && inside(root, nativeDirectory) && !inside(root, operator) && !inside(operator, root),
    "Producer roots must be scoped separately from operator review root");
  const rd = reader();
  const globalPin = async (base, p) => ({ path: relative(root, await scoped(base, p)).split(sep).join("/"), sha256: p.sha256 });
  const globalCatalog = await globalPin(collection, catalogPin), globalNative = await globalPin(nativeDirectory, nativePin);
  const [review, catalog, receipt, rawNative, ready] = await Promise.all([
    rd.json(operator, reviewPin), rd.json(collection, catalogPin), rd.json(collection, receiptPin),
    rd.json(nativeDirectory, nativePin), rd.json(root, readyPin),
  ]);
  need(review.schema === "HouseElectionNightReferenceImportReviewV1" && review.scope === "offline_reference_normalization" &&
    review.status === "approved" && review.collectionSha256 === catalogPin.sha256 && review.collectionReceiptSha256 === receiptPin.sha256 &&
    review.nativeCatalogSha256 === nativePin.sha256 && review.readyPacketSha256 === readyPin.sha256 && /^[a-f0-9]{64}$/.test(review.auditorSha256) &&
    Array.isArray(review.results) && review.results.length === 16 && new Set(review.results.map(r => r.runId)).size === 16,
  "Trusted operator review does not admit this exact collection");
  const nativeCatalog = parseHouseLiveCatalog(rawNative);
  need(nativeCatalog, "Invalid complete16 native catalog");
  need(catalog.schema === "HouseExpectedErrorCatalogV1" && catalog.status === "independently_reconciled_model_implied_summaries" &&
    catalog.cycle === 2026 && catalog.default_id === nativeCatalog.default_id && date(catalog.issued_at) &&
    catalog.evaluation_lock === false && catalog.website_review_required === true &&
    [catalog.interpretation, catalog.update_contract, catalog.joint_scope].every(text) &&
    eqPin(catalog.native_catalog, globalNative) && eqPin(catalog.ready_packet, readyPin) &&
    Array.isArray(catalog.entries) && catalog.entries.length === 16 && new Set(catalog.entries.map(e => e.id)).size === 16,
  "Invalid or mixed-edition expected-error catalog");
  need(receipt.schema === "HouseExpectedErrorCollectionReceiptV1" && receipt.status === "complete_independently_reconciled_collection" &&
    receipt.configurations === 16 && receipt.publication_qualified === false && contains(receipt.outputs, globalCatalog) && receipt.outputs.length === 2 &&
    contains(receipt.input_pins, globalNative) && contains(receipt.input_pins, readyPin), "Collection receipt does not bind exact inputs/outputs");
  need(ready.schema === "HouseLiveReadyPacketV1" && ready.status === "scientific_packet_ready_for_website_review" &&
    ready.publication_qualified === false && eqPin(ready.catalog, globalNative), "Wrong native ready packet");
  const planPin = receipt.outputs.find(p => !eqPin(p, globalCatalog));
  const plan = await rd.json(root, planPin);
  need(plan.schema === "HouseExpectedErrorCollectionPlanV1" && plan.publication_qualified === false &&
    eqPin(plan.native_catalog, globalNative) && eqPin(plan.ready_packet, readyPin) && eqPin(plan.native_review, ready.native_review) &&
    same(plan.configurations, nativeCatalog.entries.map(e => e.id)) && Array.isArray(plan.drivers) && plan.drivers.length > 0,
  "Collection plan links differ");
  const [science, numerical, qualification] = await Promise.all([
    rd.json(root, ready.scientific_review), rd.json(root, ready.native_review), rd.json(nativeDirectory, nativeCatalog.qualification),
  ]);
  need(science.schema === "HouseLiveScientificCollectionReviewV1" && science.status === "passed" &&
    science.selection_sha256 === plan.selection?.sha256 && contains(science.input_pins, plan.selection) &&
    Array.isArray(science.configurations) && science.configurations.length === 16 &&
    numerical.schema === "HouseLiveNativeNumericalReviewV1" && numerical.status === "passed" &&
    contains(numerical.input_pins, globalNative) && contains(numerical.input_pins, plan.selection) &&
    Array.isArray(numerical.results) && numerical.results.length === 16 && parseHouseLivePublicationQualification(qualification), "Native scientific/numerical qualification absent");
  await rd.bytes(nativeDirectory, nativeCatalog.source_policy);
  await rd.bytes(nativeDirectory, qualification.numerical_review);
  need(qualification.numerical_review.sha256 === ready.scientific_review.sha256, "Qualification scientific review differs");
  for (const p of [...receipt.input_pins, ...receipt.outputs, ...plan.drivers]) await rd.dependency(root, p);
  const results = [], entries = [];
  const catalogDirectory = dirname(await scoped(collection, catalogPin));
  for (const n of nativeCatalog.entries) {
    const e = catalog.entries.find(e => e.id === n.id), approval = review.results.find(r => r.runId === n.id);
    need(e && approval && same(e.configuration, n.configuration), "Missing or mismatched catalog configuration");
    const resultGlobal = await globalPin(nativeDirectory, n.result);
    const sidecarGlobal = await globalPin(catalogDirectory, e.sidecar), receiptGlobal = await globalPin(catalogDirectory, e.receipt);
    const auditGlobal = await globalPin(catalogDirectory, e.independent_audit);
    need(eqPin(e.native_result, resultGlobal) && approval.resultSha256 === n.result.sha256 &&
      approval.sidecarSha256 === e.sidecar.sha256 && approval.sidecarReceiptSha256 === e.receipt.sha256 &&
      approval.independentAuditSha256 === e.independent_audit.sha256, "Operator admission or native result edition differs");
    for (const p of [resultGlobal, sidecarGlobal, receiptGlobal, auditGlobal]) need(contains(receipt.input_pins, p), "Collection receipt omits entry evidence");
    const [rawResult, s, sr, a] = await Promise.all([rd.json(nativeDirectory, n.result), rd.json(catalogDirectory, e.sidecar),
      rd.json(catalogDirectory, e.receipt), rd.json(catalogDirectory, e.independent_audit)]);
    const native = parseHouseLiveResult(rawResult);
    need(native && native.id === n.id && same(native.configuration, n.configuration) &&
      native.qualification.publication_status === "reviewed", "Invalid or unreviewed native result/configuration");
    results.push(native);
    await rd.dependency(nativeDirectory, { path: native.joint_draws.path, sha256: native.joint_draws.sha256 });
    need(s.schema === "HouseExpectedErrorSidecarV1" && s.status === "derived_pending_independent_review" &&
      s.edition_kind === "dated_forecast_not_election_night_lock" && s.run_id === n.id && same(s.configuration, n.configuration) &&
      s.forecast_issued_at === native.issued_at && s.forecast_cutoff === native.forecast_cutoff && date(s.issued_at) &&
      Date.parse(s.issued_at) >= Date.parse(native.issued_at) &&
      eqPin(s.native_result, resultGlobal) && eqPin(s.catalog, globalNative) && eqPin(s.ready_packet, readyPin) &&
      eqPin(s.selection, plan.selection) && eqPin(s.native_numerical_review, ready.native_review) &&
      s.scientific_review?.sha256 === ready.scientific_review.sha256 && eqPin(s.qualification, await globalPin(nativeDirectory, nativeCatalog.qualification)) &&
      eqPin(e.source_receipt, s.source_receipt) && eqPin(e.joint_draws, s.joint_draws) && s.joint_draws?.sha256 === native.joint_draws.sha256,
    "Sidecar/native/source edition links differ");
    const d = s.draw_layout;
    need(object(d) && positive(d.rows) && positive(d.chains) && positive(d.draws_per_chain) &&
      Number.isSafeInteger(d.chains * d.draws_per_chain) && d.chains * d.draws_per_chain === d.rows &&
      d.rows === e.rows && d.chains === e.chains && d.rows === native.joint_draws.count && d.chains === native.joint_draws.chain_count &&
      same(d.ordered_winner_race_ids, native.joint_draws.ordered_race_ids) && d.weights === "equal_weight_per_original_joint_row" &&
      e.winner_races === 435 && e.named_pair_races === 412 && e.pair_inapplicable_races === 23,
    "Draw rows/chains/order/weights or cohort counts differ");
    need(object(s.definitions) && s.definitions.modal_tie_rule === HOUSE_EXPECTED_ERROR_MODAL_TIE_RULE &&
      text(s.definitions.modal_miss) && text(s.definitions.expected_absolute_error) &&
      text(s.precision_scope) && notes(s.qualifications) && notes(s.native_limitations), "Tie or metric-specific precision qualifications missing");
    need(sr.schema === "HouseExpectedErrorSidecarReceiptV1" && sr.status === "derived_pending_independent_review" &&
      Array.isArray(sr.input_pins) && Array.isArray(sr.outputs) && sr.outputs.length === 2 &&
      contains(sr.outputs, sidecarGlobal) && contains(sr.outputs, s.joint_metric_archive), "Sidecar receipt output links differ");
    for (const key of ["native_result", "catalog", "selection", "scientific_review", "native_numerical_review", "qualification",
      "source_receipt", "source_plan", "conditional_shares", "joint_draws", "ready_packet", "builder"]) {
      need(contains(sr.input_pins, s[key]), `Sidecar receipt omits ${key}`);
      await rd.dependency(root, s[key]);
    }
    for (const p of [...sr.input_pins, ...sr.outputs, ...(s.selected_reviews ?? [])]) await rd.dependency(root, p);
    const [sourceReceipt, sourcePlan] = await Promise.all([rd.json(root, s.source_receipt), rd.json(root, s.source_plan)]);
    need(["HouseLivePredictionReceiptV1", "HouseLivePredictionReceiptV2"].includes(sourceReceipt.schema) &&
      sourceReceipt.status === "passed" && same(sourceReceipt.configuration, { id: n.id, ...n.configuration }) &&
      eqPin(sourceReceipt.plan, s.source_plan) && contains(sourceReceipt.outputs, s.joint_draws) &&
      contains(sourceReceipt.outputs, s.conditional_shares) &&
      ["HouseLivePredictionPlanV1", "HouseLiveCombinedPredictionPlanV1", "HouseLiveCombinedPredictionPlanV2"].includes(sourcePlan.schema) &&
      sourcePlan.pilot === false && same(sourcePlan.configuration, sourceReceipt.configuration) &&
      sourcePlan.source_2026_outcomes_loaded === false &&
      sourcePlan.settings?.chains === d.chains && sourcePlan.settings?.draws === d.draws_per_chain &&
      same(d.source_runs, sourcePlan.source_runs ?? null),
    "Original prediction receipt/plan does not bind exact configuration, bank or layout");
    if (sourceReceipt.schema === "HouseLivePredictionReceiptV2") {
      need(sourcePlan.schema === "HouseLiveCombinedPredictionPlanV2" &&
        sourcePlan.conditional_storage?.schema === "ConditionalSegmentsManifestV1" &&
        sourcePlan.conditional_storage.weights === "equal_weight_per_original_joint_row" &&
        eqPin(sourcePlan.conditional_storage.manifest, s.conditional_shares) &&
        same(s.conditional_storage, sourcePlan.conditional_storage) && basename(s.conditional_shares.path) === "conditional-shares.index.json",
      "V2 selected conditional manifest or weights differ");
    } else need(sourcePlan.schema !== "HouseLiveCombinedPredictionPlanV2" &&
      basename(s.conditional_shares.path) === "conditional-shares.npz" &&
      (s.conditional_storage === undefined || same(s.conditional_storage, { schema: "LegacyConditionalNPZV1", weights: "equal_weight_per_original_joint_row" })),
    "V1 selected conditional archive differs");
    need(a.schema === "ExpectedErrorSidecarIndependentAuditV1" && a.status === "passed" &&
      eqPin(a.sidecar, sidecarGlobal) && eqPin(a.receipt, receiptGlobal) && a.auditor?.sha256 === review.auditorSha256 &&
      same(a.configuration, n.configuration) && a.rows === d.rows && a.chains === d.chains && a.draws_per_chain === d.draws_per_chain &&
      a.winner_races === 435 && a.named_pair_races === 412 && a.pair_inapplicable_races === 23 &&
      finite(a.max_abs_per_race_expected_error_diff_pp, 0, 1e-8) && object(a.max_abs_metric_vector_diff) &&
      metricIds.every(k => finite(a.max_abs_metric_vector_diff[k], 0, 1e-8)), "Independent audit does not reconcile this sidecar/receipt/layout");
    await rd.bytes(root, a.auditor);
    need(contains(plan.drivers, a.auditor) && contains(plan.drivers, s.builder), "Auditor/builder absent from collection driver pins");
    const sciRow = science.configurations.filter(r => r.configuration?.id === n.id);
    const numRow = numerical.results.filter(r => r.id === n.id);
    need(sciRow.length === 1 && same(sciRow[0].configuration, { id: n.id, ...n.configuration }) &&
      eqPin(sciRow[0].receipt, s.source_receipt) && sciRow[0].draws === d.rows && sciRow[0].chains === d.chains &&
      numRow.length === 1 && numRow[0].status === "reconciled" && numRow[0].races === 435 && numRow[0].conditional_pairs === 412,
    "Source scientific/native numerical review does not cover exact configuration");
    need(contains(numerical.input_pins, resultGlobal) && contains(numerical.input_pins, sciRow[0].independent_audit),
      "Native numerical review omits result or selected source audit");
    await rd.bytes(root, sciRow[0].independent_audit);
    need(contains(s.selected_reviews, sciRow[0].independent_audit) &&
      s.selected_reviews.every(p => contains(sr.input_pins, p)), "Selected source review does not bind source science/receipt");
    need(a.exact_modal_call_tie_count === s.modal_calls.filter(c => c.tied_modal_outcomes?.length > 1).length,
      "Independent audit tie count differs");
    validateSummaries(s.joint_summaries, d.rows);
    need(object(s.race_expected_errors), "Missing race expected errors");
    validateRaceSummaries(s, native);
    const sourcePins = { collectionSha256: catalogPin.sha256, collectionReceiptSha256: receiptPin.sha256,
      nativeCatalogSha256: nativePin.sha256, readyPacketSha256: readyPin.sha256, referenceSha256: e.sidecar.sha256,
      sidecarReceiptSha256: e.receipt.sha256, reviewSha256: e.independent_audit.sha256, auditorSha256: a.auditor.sha256,
      sourceReceiptSha256: s.source_receipt.sha256, jointDrawsSha256: s.joint_draws.sha256, sourcePlanSha256: s.source_plan.sha256,
      conditionalSharesSha256: s.conditional_shares.sha256, jointMetricArchiveSha256: s.joint_metric_archive.sha256,
      scientificReviewSha256: s.scientific_review.sha256, nativeNumericalReviewSha256: s.native_numerical_review.sha256,
      qualificationSha256: s.qualification.sha256, selectionSha256: s.selection.sha256, builderSha256: s.builder.sha256,
      operatorReviewSha256: reviewPin.sha256 };
    entries.push({ chamber: "house", runId: n.id, resultSha256: n.result.sha256, configuration: structuredClone(n.configuration),
      forecastIssuedAt: native.issued_at, forecastCutoffAt: native.forecast_cutoff, referenceIssuedAt: s.issued_at,
      rows: d.rows, chains: d.chains, winnerRaceCount: 435, pairRaceCount: 412, pairInapplicableRaceCount: 23,
      sourcePins, reference: s });
  }
  const validation = validateHouseLiveCollection(nativeCatalog, results);
  need(validation.valid, validation.issues.join(" "));
  // Native export assigns an actual per-result issue time. The authenticated
  // one-edition catalog/ready packet, not equality of those times, binds edition.
  need(new Set(results.map(r => r.forecast_cutoff)).size === 1, "Mixed forecast cutoff editions");
  const normalized = { schema: "HouseExpectedErrorPublicCatalogV1", schemaVersion: 1, chamber: "house", defaultRunId: nativeCatalog.default_id,
    sourceCatalogSha256: catalogPin.sha256, nativeCatalogSha256: nativePin.sha256, readyPacketSha256: readyPin.sha256,
    evaluationLock: false, observedResultsConsumed: false, entries };
  const publication = { schemaVersion: 1, href: "/data/election-night/house-references.json", sha256: hash(bytesOf(normalized)),
    editions: entries.map(({ chamber, runId, resultSha256, configuration, forecastIssuedAt, forecastCutoffAt, referenceIssuedAt,
      rows, chains, winnerRaceCount, pairRaceCount, pairInapplicableRaceCount, sourcePins }) => ({ chamber, runId, resultSha256,
      qualification: { configuration, forecastIssuedAt, forecastCutoffAt, referenceIssuedAt, rows, chains,
        winnerRaceCount, pairRaceCount, pairInapplicableRaceCount, sourcePins } })) };
  return { catalog: normalized, publication, verification: { authenticatedMetadataArtifacts: rd.authenticated.size,
    delegatedRawArchiveArtifacts: rd.delegated.size, rawDrawRehash: false, observedResultsConsumed: false,
    authority: "Caller-supplied independently trusted operator review pin; exact upstream independent audit for raw arrays." } };
}

/** Writes only a new explicit private directory. Installing public files is a
 * separate reviewed release action. An existing dated export is never replaced.
 */
export async function importHouseElectionNightReferences(options = {}) {
  need(text(options.outputDirectory), "Explicit private output directory required");
  const output = resolve(options.outputDirectory), parent = await realpath(dirname(output));
  need(output === resolve(parent, relative(dirname(output), output)), "Output parent resolves through a symlink");
  const publicRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../public");
  for (const r of [options.sourceRoot, options.reviewRoot, publicRoot]) {
    need(text(r) && !inside(await realpath(r), output), "Output must be outside producer, operator and active public roots");
  }
  const normalized = await normalizeHouseElectionNightReferences(options);
  await mkdir(output); // exclusive, no recursive creation/no overwrite
  try {
    await writeFile(resolve(output, "house-references.json"), bytesOf(normalized.catalog), { flag: "wx" });
    await writeFile(resolve(output, "publication.json"), bytesOf(normalized.publication), { flag: "wx" });
  } catch (error) {
    await rm(output, { recursive: true, force: true });
    throw error;
  }
  return normalized;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const file = process.argv[2];
  need(file && process.argv.length === 3, "Usage: node --experimental-strip-types scripts/import_house_election_night_references.mjs <operator-options.json>");
  const result = await importHouseElectionNightReferences(JSON.parse(await readFile(file, "utf8")));
  console.log(JSON.stringify(result.verification));
}
