// Invented metadata graph ONLY. Placeholder NPZs are deliberately not arrays:
// tests exercise delegated upstream audit trust, never scientific qualification.
import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { makeHouseLiveCatalog, makeHouseLiveResult } from "./house-live-fixtures.mjs";
import { HOUSE_EXPECTED_ERROR_MODAL_TIE_RULE } from "../../scripts/import_house_election_night_references.mjs";

export const digest = b => createHash("sha256").update(b).digest("hex");
const jsonBytes = v => Buffer.from(`${JSON.stringify(v, null, 2)}\n`);
const keys = ["mean_signed_pair_margin_error_pp", "mean_absolute_pair_margin_error_pp", "central90_pair_interval_coverage_count", "modal_group_miss_count"];
const time = "2026-10-26T12:00:00Z";
function metric(mean, max) {
  const m = { mean, sd: 0, quantiles: { p05: mean, p50: mean, p95: mean },
    quantile_convention: max ? "inverse_cdf" : "linear", mean_mcse: 0, rank_rhat: 1,
    units: max ? "count" : "percentage_points" };
  if (max) { m.pmf = Array(max + 1).fill(0); m.pmf[mean] = 1; }
  return m;
}

export async function makeHouseReferenceImportFixture(base, mutate = {}) {
  const sourceRoot = join(base, "producer"), reviewRoot = join(base, "operator");
  const nativeRoot = join(sourceRoot, "native"), collectionRoot = join(sourceRoot, "collection");
  await Promise.all([sourceRoot, reviewRoot, nativeRoot, collectionRoot].map(p => mkdir(p, { recursive: true })));
  const put = async (path, value, root = sourceRoot) => {
    const bytes = Buffer.isBuffer(value) ? value : jsonBytes(value);
    await mkdir(dirname(join(root, path)), { recursive: true }); await writeFile(join(root, path), bytes);
    return { path, sha256: digest(bytes) };
  };
  const local = (p, prefix) => ({ ...p, path: p.path.slice(prefix.length + 1) });
  const driver = await put("code/builder.py", Buffer.from("# synthetic builder\n"));
  const auditor = await put("code/auditor.py", Buffer.from("# synthetic independent auditor\n"));
  const selection = await put("selection.json", { schema: "HouseLiveReleaseSelectionV1", fixture: true });
  const raw = await put("arrays/joint.npz", Buffer.from("Not a real array: upstream-authenticated fixture pin only"));
  await put("native/joint.npz", Buffer.from("Not a real array: upstream-authenticated fixture pin only"));
  const segment = await put("arrays/conditional-shares.npz", Buffer.from("Not a real conditional share archive"));
  const shares = mutate.storageVersion === 2 ? await put("arrays/conditional-shares.index.json", {
    schema: "ConditionalSegmentsManifestV1", source_count: 2, total_rows: 4,
    sources: [{ ...segment, rows: 2, row_offset: 0, row_stop: 2 }, { ...segment, rows: 2, row_offset: 2, row_stop: 4 }],
    fixture: "Metadata only; trusted upstream audit is invented, no actual array claim.",
  }) : segment;
  const storage = mutate.storageVersion === 2 ? { schema: "ConditionalSegmentsManifestV1", manifest: shares, weights: "equal_weight_per_original_joint_row" } : undefined;
  const nativeCatalog = makeHouseLiveCatalog(), natives = [], sourceReceipts = [], sourcePlans = [], sourceAudits = [];
  for (const e of nativeCatalog.entries) {
    const c = e.configuration;
    e.id = c.base === "fundamentals" ? c.uncertainty !== "standard" ? `house-current-${c.uncertainty}` :
      `house-current-${c.polling === "none" ? "fundamentals" : "polling"}` : `house-current-comparison-${c.base}-${c.polling === "none" ? "f" : "fp"}`;
  }
  for (const [i, e] of nativeCatalog.entries.entries()) {
    const native = makeHouseLiveResult(e.configuration);
    native.id = e.id;
    for (let j = 3; j < 23; j++) native.races[j].conditional_shares = null;
    native.joint_draws.sha256 = raw.sha256;
    native.joint_draws.path = "joint.npz";
    native.chamber.joint_draws_sha256 = raw.sha256;
    native.qualification.publication_status = "reviewed"; // invented protocol fixture, never public
    native.issued_at = `2026-10-21T12:00:${String(i).padStart(2, "0")}.000Z`;
    mutate.native?.(native, i);
    natives.push(native);
    const sourcePlan = { schema: storage ? "HouseLiveCombinedPredictionPlanV2" : "HouseLivePredictionPlanV1", configuration: { id: e.id, ...e.configuration },
      pilot: false, settings: { chains: 2, draws: 2 }, source_2026_outcomes_loaded: false };
    if (storage) sourcePlan.conditional_storage = structuredClone(storage);
    mutate.sourcePlan?.(sourcePlan, i);
    const sp = await put(`source/${e.id}/plan.json`, sourcePlan); sourcePlans.push(sp);
    const sourceReceipt = { schema: storage ? "HouseLivePredictionReceiptV2" : "HouseLivePredictionReceiptV1", status: "passed", configuration: { id: e.id, ...e.configuration },
      plan: sp, outputs: [raw, shares] };
    mutate.sourceReceipt?.(sourceReceipt, i);
    sourceReceipts.push(await put(`source/${e.id}/receipt.json`, sourceReceipt));
    sourceAudits.push(await put(`source/${e.id}/audit.json`, { status: "passed", fixture: true, configuration: e.configuration }));
    const n = await put(`native/results/${e.id}.json`, native); e.result = local(n, "native");
  }
  const science = { schema: "HouseLiveScientificCollectionReviewV1", status: "passed", selection_sha256: selection.sha256,
    input_pins: [selection], configurations: natives.map((n, i) => ({ configuration: { id: n.id, ...n.configuration },
      draws: 4, chains: 2, receipt: sourceReceipts[i], independent_audit: sourceAudits[i] })) };
  mutate.science?.(science);
  const sci = await put("science.json", science);
  await put("native/evidence/numerical-review.json", science);
  const q = await put("native/qualification.json", { schema: "HouseLivePublicationQualificationV1", status: "reviewed",
    numerical_review: { path: "evidence/numerical-review.json", sha256: sci.sha256 }, seat_quantile_convention: "inverse_cdf",
    share_quantile_convention: "linear", limitations: ["Synthetic fixture; no scientific admission."] });
  nativeCatalog.qualification = local(q, "native");
  nativeCatalog.source_policy = local(await put("native/source-policy.json", { fixture: true }), "native");
  mutate.nativeCatalog?.(nativeCatalog);
  const nc = await put("native/catalog.json", nativeCatalog);
  const numerical = { schema: "HouseLiveNativeNumericalReviewV1", status: "passed",
    input_pins: [nc, selection, ...sourceAudits, ...nativeCatalog.entries.map(e => ({ ...e.result, path: `native/${e.result.path}` }))],
    results: natives.map(n => ({ id: n.id, status: "reconciled", races: 435, conditional_pairs: 412 })) };
  mutate.numerical?.(numerical);
  const num = await put("native-review.json", numerical);
  const ready = { schema: "HouseLiveReadyPacketV1", status: "scientific_packet_ready_for_website_review", publication_qualified: false,
    catalog: nc, scientific_review: sci, native_review: num };
  mutate.ready?.(ready);
  const readyPin = await put("ready.json", ready);
  const plan = { schema: "HouseExpectedErrorCollectionPlanV1", publication_qualified: false, native_catalog: nc,
    ready_packet: readyPin, selection, native_review: num, configurations: natives.map(n => n.id), drivers: [driver, auditor] };
  mutate.plan?.(plan);
  const pp = await put("collection/plan.json", plan);
  const catalog = { schema: "HouseExpectedErrorCatalogV1", issued_at: time, cycle: 2026, default_id: nativeCatalog.default_id,
    status: "independently_reconciled_model_implied_summaries", native_catalog: nc, ready_packet: readyPin,
    interpretation: "Synthetic model-implied metrics, no results or grades.", update_contract: "Retain dated editions.",
    joint_scope: "Rows within one configuration only.", evaluation_lock: false, website_review_required: true, entries: [] };
  const operatorRows = [], collectionInputs = [nc, readyPin, selection, num, driver, auditor], rawSidecars = [];
  for (const [i, native] of natives.entries()) {
    const nPin = { ...nativeCatalog.entries[i].result, path: `native/${nativeCatalog.entries[i].result.path}` };
    const id = native.id, stem = `collection/results/${id}`;
    const mae = Object.fromEntries(native.races.map(r => [r.race_id, r.conditional_shares ? {
      availability: "available", first_candidate_id: r.conditional_shares.first_candidate_id,
      second_candidate_id: r.conditional_shares.second_candidate_id, denominator: r.conditional_shares.denominator,
      point_margin_pp: 200 * r.conditional_shares.mean - 100, expected_absolute_margin_error_pp: 10,
      qualifications: r.conditional_shares.qualifications,
    } : { availability: "inapplicable", expected_absolute_margin_error_pp: null, reason: "No named pair." }]));
    const modal = native.races.map(r => {
      const p = Math.max(...r.outcomes.map(o => o.p_win)), ties = ["D", "R", "O"].filter(id => r.outcomes.some(o => o.outcome_id === id && o.p_win === p));
      const o = r.outcomes.find(o => o.outcome_id === ties[0]);
      return { race_id: r.race_id, outcome_id: o.outcome_id, party_or_group: o.party, kind: o.kind,
        member_candidate_ids: o.member_candidate_ids, tied_modal_outcomes: ties.length > 1 ? ties : [] };
    });
    const miss = native.races.reduce((n, r) => n + 1 - Math.max(...r.outcomes.map(o => o.p_win)), 0);
    const archive = await put(`${stem}/joint-metrics.npz`, Buffer.from("Synthetic metric bank marker"));
    const s = { schema: "HouseExpectedErrorSidecarV1", status: "derived_pending_independent_review", issued_at: time,
      edition_kind: "dated_forecast_not_election_night_lock", run_id: id, configuration: native.configuration,
      forecast_issued_at: native.issued_at, forecast_cutoff: native.forecast_cutoff, native_result: nPin, catalog: nc, selection,
      scientific_review: sci, native_numerical_review: num, qualification: q, selected_reviews: [sourceAudits[i]], ready_packet: readyPin,
      source_receipt: sourceReceipts[i], source_plan: sourcePlans[i], conditional_shares: shares, joint_draws: raw,
      draw_layout: { chains: 2, draws_per_chain: 2, rows: 4, weights: "equal_weight_per_original_joint_row", source_runs: null,
        ordered_winner_race_ids: native.races.map(r => r.race_id), ordered_pair_race_ids: native.races.filter(r => r.conditional_shares).map(r => r.race_id),
        pair_mask_in_winner_order: native.races.map(r => !!r.conditional_shares) },
      definitions: { modal_tie_rule: HOUSE_EXPECTED_ERROR_MODAL_TIE_RULE, modal_miss: "Exhaustive group winner event.", expected_absolute_error: "Own native pair mean error." },
      race_expected_errors: mae, modal_calls: modal,
      joint_summaries: Object.fromEntries(keys.map((k, i) => [k, metric([0, 10, 370, miss][i], [null, null, 412, 435][i])])),
      joint_metric_archive: archive, precision_scope: "Synthetic joint diagnostics only; no quantile MCSE or separate transformed threshold qualification.",
      qualifications: ["Model-implied, no empirical score or result lock."], native_limitations: native.qualification.limitations, builder: driver };
    if (storage) s.conditional_storage = structuredClone(storage);
    mutate.sidecar?.(s, i);
    rawSidecars.push(structuredClone(s));
    const sp = await put(`${stem}/sidecar.json`, s);
    const sr = { schema: "HouseExpectedErrorSidecarReceiptV1", status: "derived_pending_independent_review",
      input_pins: [nPin, nc, selection, sci, num, q, sourceReceipts[i], sourcePlans[i], shares, raw, readyPin, driver, sourceAudits[i]], outputs: [sp, archive] };
    mutate.receipt?.(sr, i);
    const srp = await put(`${stem}/receipt.json`, sr);
    const a = { schema: "ExpectedErrorSidecarIndependentAuditV1", status: "passed", sidecar: sp, receipt: srp, auditor,
      configuration: native.configuration, rows: 4, chains: 2, draws_per_chain: 2, winner_races: 435, named_pair_races: 412,
      pair_inapplicable_races: 23, max_abs_metric_vector_diff: Object.fromEntries(keys.map(k => [k, 0])), max_abs_per_race_expected_error_diff_pp: 0,
      exact_modal_call_tie_count: modal.filter(c => c.tied_modal_outcomes.length > 1).length };
    mutate.audit?.(a, i);
    const ap = await put(`collection/audits/${id}.json`, a);
    catalog.entries.push({ id, configuration: native.configuration, native_result: nPin, sidecar: local(sp, "collection"),
      receipt: local(srp, "collection"), independent_audit: local(ap, "collection"), source_receipt: sourceReceipts[i], joint_draws: raw,
      rows: 4, chains: 2, winner_races: 435, named_pair_races: 412, pair_inapplicable_races: 23 });
    operatorRows.push({ runId: id, resultSha256: nPin.sha256, sidecarSha256: sp.sha256, sidecarReceiptSha256: srp.sha256, independentAuditSha256: ap.sha256 });
    collectionInputs.push(nPin, sp, srp, ap);
  }
  mutate.catalog?.(catalog);
  const cp = await put("collection/catalog.json", catalog);
  const cr = { schema: "HouseExpectedErrorCollectionReceiptV1", status: "complete_independently_reconciled_collection",
    configurations: 16, publication_qualified: false, input_pins: collectionInputs, outputs: [cp, pp] };
  mutate.collectionReceipt?.(cr);
  const rp = await put("collection/receipt.json", cr);
  const review = { schema: "HouseElectionNightReferenceImportReviewV1", scope: "offline_reference_normalization", status: "approved",
    collectionSha256: cp.sha256, collectionReceiptSha256: rp.sha256, nativeCatalogSha256: nc.sha256, readyPacketSha256: readyPin.sha256,
    auditorSha256: auditor.sha256, results: operatorRows, limitations: ["Invented fixture only."] };
  mutate.review?.(review);
  const op = await put("review.json", review, reviewRoot);
  return { options: { sourceRoot, collectionRoot, nativeRoot, reviewRoot, catalog: local(cp, "collection"),
    receipt: local(rp, "collection"), nativeCatalog: local(nc, "native"), readyPacket: readyPin, review: op }, natives, nativeCatalog, rawSidecars };
}
