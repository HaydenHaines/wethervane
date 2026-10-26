/** Exact-edition projection of an authenticated House expected-error catalog. */
import type { ElectionNightForecast, ElectionNightRace } from "./election-night-forecast.ts";
import type { HouseReferenceQualification } from "./election-night-publication.ts";

interface Pin { path: string; sha256: string }
interface Metric { mean: number; sd: number; quantiles: { p05: number; p50: number; p95: number };
  quantile_convention: string; mean_mcse: number; rank_rhat: number; units: string; pmf?: number[] }
interface Sidecar {
  schema: string; status: string; edition_kind: string; run_id: string;
  configuration: { base: string; polling: string; uncertainty: string };
  issued_at: string; forecast_issued_at: string; forecast_cutoff: string;
  native_result: Pin; catalog: Pin; selection: Pin; scientific_review: Pin; native_numerical_review: Pin;
  qualification: Pin; ready_packet: Pin; source_receipt: Pin; source_plan: Pin; conditional_shares: Pin;
  joint_draws: Pin; joint_metric_archive: Pin; builder: Pin;
  draw_layout: { rows: number; chains: number; draws_per_chain: number; weights: string;
    ordered_winner_race_ids: string[]; ordered_pair_race_ids: string[]; pair_mask_in_winner_order: boolean[] };
  definitions: Record<string, string>;
  race_expected_errors: Record<string, { availability: "available" | "inapplicable";
    first_candidate_id?: string; second_candidate_id?: string; denominator?: string;
    expected_absolute_margin_error_pp: number | null; point_margin_pp?: number; qualifications?: string[]; reason?: string }>;
  modal_calls: { race_id: string; outcome_id: string; party_or_group: string; kind: "candidate" | "group";
    member_candidate_ids: string[]; tied_modal_outcomes: string[] }[];
  joint_summaries: Record<string, Metric>;
  precision_scope: string; qualifications: string[]; native_limitations: string[];
}
interface Entry {
  chamber: "house"; runId: string; resultSha256: string;
  configuration: Sidecar["configuration"]; forecastIssuedAt: string; forecastCutoffAt: string;
  referenceIssuedAt: string; rows: number; chains: number; winnerRaceCount: 435; pairRaceCount: 412;
  pairInapplicableRaceCount: 23; sourcePins: Record<string, string>; reference: Sidecar;
}
interface Catalog {
  schema: "HouseExpectedErrorPublicCatalogV1"; schemaVersion: 1; chamber: "house";
  defaultRunId: string; sourceCatalogSha256: string; nativeCatalogSha256: string; readyPacketSha256: string;
  evaluationLock: false; observedResultsConsumed: false; entries: Entry[];
}
type ModalCall = Sidecar["modal_calls"][number];
export interface HouseExpectedErrorReference {
  chamber: "house"; runId: string; resultSha256: string; referenceIssuedAt: string;
  rows: number; chains: number; winnerRaceCount: 435; pairRaceCount: 412; pairInapplicableRaceCount: 23;
  sourcePins: Record<string, string>;
  perRace: { raceId: string; availability: "available" | "inapplicable";
    expectedAbsoluteMarginErrorPp: number | null; reason: string | null }[];
  summary: {
    modalGroupMissCount: { mean: number; range90: { lower: number; upper: number } };
    meanAbsolutePairMarginError: { mean: number; range90: { lower: number; upper: number } };
    meanSignedPairMarginError: { mean: number; range90: { lower: number; upper: number } };
    central90PairIntervalCoverageCount: { mean: number; range90: { lower: number; upper: number } };
  };
  precisionScope: string; qualifications: string[]; nativeLimitations: string[];
  evaluationLock: false; observedResultsConsumed: false;
}
const hash = (value: unknown): value is string => typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
const text = (value: unknown): value is string => typeof value === "string" && value.trim().length > 0;
const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
const sameRecord = (a: object, b: object) =>
  same(Object.entries(a).sort(([x], [y]) => x.localeCompare(y)),
    Object.entries(b).sort(([x], [y]) => x.localeCompare(y)));
const near = (a: number, b: number, tolerance = 1e-8) => finite(a) && finite(b) && Math.abs(a - b) <= tolerance;
function need(value: unknown): asserts value { if (!value) throw Error("House reference does not match this reviewed forecast."); }
function metric(value: Metric, lower: number, upper: number, rows: number, discrete = false) {
  need(value && finite(value.mean) && value.mean >= lower && value.mean <= upper && finite(value.sd) && value.sd >= 0 &&
    finite(value.mean_mcse) && value.mean_mcse >= 0 && finite(value.rank_rhat) && value.rank_rhat > 0 &&
    value.units === (discrete ? "count" : "percentage_points") && value.quantiles &&
    [value.quantiles.p05, value.quantiles.p50, value.quantiles.p95]
      .every((v, i, a) => finite(v) && v >= lower && v <= upper && (i === 0 || v >= a[i - 1]) &&
        (!discrete || Number.isInteger(v))) &&
    value.quantile_convention === (discrete ? "inverse_cdf" : "linear"));
  if (discrete) {
    need(Array.isArray(value.pmf) && value.pmf.length === upper + 1 &&
      value.pmf.every(p => finite(p) && p >= 0 && p <= 1));
    const counts = value.pmf.map(p => Math.round(p * rows));
    need(counts.every((count, i) => near(value.pmf![i], count / rows, 1e-12)) &&
      counts.reduce((sum, count) => sum + count, 0) === rows);
    const mean = counts.reduce((sum, count, i) => sum + i * count, 0) / rows;
    const sd = Math.sqrt(counts.reduce((sum, count, i) => sum + (i - mean) ** 2 * count, 0) / rows);
    need(near(mean, value.mean, 1e-7) && near(sd, value.sd, 1e-7));
    for (const [p, saved] of [[.05, value.quantiles.p05], [.5, value.quantiles.p50], [.95, value.quantiles.p95]]) {
      let cumulative = 0, found = -1;
      for (let i = 0; i < counts.length; i++) {
        cumulative += counts[i];
        if (cumulative >= Math.ceil(p * rows)) { found = i; break; }
      }
      need(found === saved);
    }
  } else need(value.pmf === undefined);
  return { mean: value.mean, range90: { lower: value.quantiles.p05, upper: value.quantiles.p95 } };
}

/** Exact House modal-call semantics: group identity and D, R, O tie order. */
export function matchesHouseModalCall(race: ElectionNightRace, call: ModalCall): boolean {
  try {
    const maximum = Math.max(...race.outcomes.map(outcome => outcome.winProbability));
    const ties = race.outcomes.filter(outcome => outcome.winProbability === maximum)
      .sort((a, b) => "DRO".indexOf(a.party) - "DRO".indexOf(b.party));
    const chosen = ties[0];
    return !!chosen && call?.race_id === race.id && call.outcome_id === chosen.party &&
      call.party_or_group === chosen.party && call.kind === chosen.kind &&
      same(call.member_candidate_ids, chosen.members.map(member => member.id)) &&
      same(call.tied_modal_outcomes, ties.length > 1 ? ties.map(outcome => outcome.party) : []) &&
      race.modalPick.outcomeId === chosen.id && race.modalPick.kind === chosen.kind &&
      same(race.modalPick.tiedOutcomeIds, ties.map(outcome => outcome.id));
  } catch { return false; }
}

/** The caller authenticates catalog bytes against the House publication SHA. */
export function matchHouseExpectedErrorReference(forecast: ElectionNightForecast, raw: unknown,
  qualification: HouseReferenceQualification): HouseExpectedErrorReference | null {
  try {
    const catalog = raw as Catalog;
    need(forecast.chamber === "house" && forecast.schema === "ElectionNightForecastV1" &&
      forecast.editionStatus === "dated_snapshot" && forecast.summary.tiePolicy === "house_declared_D_R_O_party_order" &&
      forecast.races.length === 435 && forecast.summary.raceCount === 435 && forecast.summary.pairRaceCount === 412 &&
      forecast.houseJointDraws && hash(forecast.run.artifactSha256) &&
      catalog?.schema === "HouseExpectedErrorPublicCatalogV1" && catalog.schemaVersion === 1 &&
      catalog.chamber === "house" && hash(catalog.sourceCatalogSha256) && hash(catalog.nativeCatalogSha256) &&
      hash(catalog.readyPacketSha256) && catalog.evaluationLock === false &&
      catalog.observedResultsConsumed === false && Array.isArray(catalog.entries));
    const matching = catalog.entries.filter(row => row?.chamber === "house" && row.runId === forecast.run.id &&
      row.resultSha256 === forecast.run.artifactSha256);
    need(matching.length === 1);
    const row = matching[0], ref = row.reference, draw = ref.draw_layout;
    need(qualification && sameRecord(row.configuration, qualification.configuration) &&
      sameRecord(row.configuration, forecast.run.configuration) &&
      row.forecastIssuedAt === qualification.forecastIssuedAt && row.forecastIssuedAt === forecast.run.issuedAt &&
      row.forecastCutoffAt === qualification.forecastCutoffAt && row.forecastCutoffAt === forecast.run.cutoffAt &&
      row.referenceIssuedAt === qualification.referenceIssuedAt && row.referenceIssuedAt === ref.issued_at &&
      Number.isFinite(Date.parse(row.referenceIssuedAt)) &&
      Date.parse(row.referenceIssuedAt) >= Date.parse(row.forecastIssuedAt) &&
      row.rows === qualification.rows && row.chains === qualification.chains &&
      row.winnerRaceCount === qualification.winnerRaceCount && row.pairRaceCount === qualification.pairRaceCount &&
      row.pairInapplicableRaceCount === qualification.pairInapplicableRaceCount &&
      row.winnerRaceCount === 435 && row.pairRaceCount === 412 && row.pairInapplicableRaceCount === 23 &&
      sameRecord(row.sourcePins, qualification.sourcePins) && Object.values(row.sourcePins).every(hash) &&
      row.sourcePins.nativeCatalogSha256 === catalog.nativeCatalogSha256 &&
      row.sourcePins.readyPacketSha256 === catalog.readyPacketSha256 &&
      row.sourcePins.collectionSha256 === catalog.sourceCatalogSha256 &&
      row.sourcePins.jointDrawsSha256 === forecast.sourceHashes.jointDrawsSha256 &&
      ref.schema === "HouseExpectedErrorSidecarV1" && ref.status === "derived_pending_independent_review" &&
      ref.edition_kind === "dated_forecast_not_election_night_lock" && ref.run_id === row.runId &&
      sameRecord(ref.configuration, row.configuration) && ref.forecast_issued_at === row.forecastIssuedAt &&
      ref.forecast_cutoff === row.forecastCutoffAt && ref.native_result.sha256 === row.resultSha256 &&
      ref.catalog.sha256 === row.sourcePins.nativeCatalogSha256 &&
      ref.ready_packet.sha256 === row.sourcePins.readyPacketSha256 &&
      ref.selection.sha256 === row.sourcePins.selectionSha256 &&
      ref.scientific_review.sha256 === row.sourcePins.scientificReviewSha256 &&
      ref.native_numerical_review.sha256 === row.sourcePins.nativeNumericalReviewSha256 &&
      ref.qualification.sha256 === row.sourcePins.qualificationSha256 &&
      ref.source_receipt.sha256 === row.sourcePins.sourceReceiptSha256 &&
      ref.source_plan.sha256 === row.sourcePins.sourcePlanSha256 &&
      ref.conditional_shares.sha256 === row.sourcePins.conditionalSharesSha256 &&
      ref.joint_draws.sha256 === row.sourcePins.jointDrawsSha256 &&
      ref.joint_metric_archive.sha256 === row.sourcePins.jointMetricArchiveSha256 &&
      ref.builder.sha256 === row.sourcePins.builderSha256);
    need(draw && Number.isSafeInteger(draw.rows) && draw.rows > 0 && Number.isSafeInteger(draw.chains) && draw.chains > 0 &&
      Number.isSafeInteger(draw.draws_per_chain) && draw.draws_per_chain > 0 &&
      draw.rows === draw.chains * draw.draws_per_chain && draw.rows === row.rows && draw.chains === row.chains &&
      draw.rows === forecast.houseJointDraws.rowCount && draw.chains === forecast.houseJointDraws.chainCount &&
      draw.weights === "equal_weight_per_original_joint_row" &&
      same(draw.ordered_winner_race_ids, forecast.races.map(race => race.id)) &&
      same(draw.ordered_pair_race_ids, forecast.races.filter(race => race.pair).map(race => race.id)) &&
      same(draw.pair_mask_in_winner_order, forecast.races.map(race => race.pair !== null)));
    need(ref.definitions?.modal_tie_rule === "first maximum in declared D, R, O ordering; exact native probability ties recorded" &&
      text(ref.definitions.modal_miss) && text(ref.definitions.expected_absolute_error) &&
      text(ref.precision_scope) && Array.isArray(ref.qualifications) && ref.qualifications.every(text) &&
      Array.isArray(ref.native_limitations) && ref.native_limitations.every(text) &&
      Array.isArray(ref.modal_calls) && ref.modal_calls.length === 435 &&
      ref.race_expected_errors && Object.keys(ref.race_expected_errors).length === 435 &&
      same(Object.keys(ref.race_expected_errors).sort(), forecast.races.map(race => race.id).sort()));
    let mae = 0, supported = 0, absent = 0;
    const perRace = forecast.races.map((race, index) => {
      const expected = ref.race_expected_errors[race.id], call = ref.modal_calls[index];
      need(matchesHouseModalCall(race, call) && expected);
      if (race.pair) {
        need(expected.availability === "available" && expected.first_candidate_id === race.pair.firstCandidateId &&
          expected.second_candidate_id === race.pair.secondCandidateId &&
          expected.denominator === race.pair.denominator && near(expected.point_margin_pp!, race.pair.meanMarginPp) &&
          finite(expected.expected_absolute_margin_error_pp) && expected.expected_absolute_margin_error_pp >= 0 &&
          expected.expected_absolute_margin_error_pp <= 200 && Array.isArray(expected.qualifications) &&
          expected.qualifications.every(text));
        supported++; mae += expected.expected_absolute_margin_error_pp;
        return { raceId: race.id, availability: "available" as const,
          expectedAbsoluteMarginErrorPp: expected.expected_absolute_margin_error_pp, reason: null };
      }
      need(expected.availability === "inapplicable" && expected.expected_absolute_margin_error_pp === null && text(expected.reason));
      absent++;
      return { raceId: race.id, availability: "inapplicable" as const,
        expectedAbsoluteMarginErrorPp: null, reason: expected.reason };
    });
    need(supported === 412 && absent === 23 && Object.keys(ref.joint_summaries).length === 4);
    const summary = {
      meanSignedPairMarginError: metric(ref.joint_summaries.mean_signed_pair_margin_error_pp, -200, 200, row.rows),
      meanAbsolutePairMarginError: metric(ref.joint_summaries.mean_absolute_pair_margin_error_pp, 0, 200, row.rows),
      central90PairIntervalCoverageCount: metric(ref.joint_summaries.central90_pair_interval_coverage_count, 0, 412, row.rows, true),
      modalGroupMissCount: metric(ref.joint_summaries.modal_group_miss_count, 0, 435, row.rows, true),
    };
    need(near(summary.meanAbsolutePairMarginError.mean, mae / 412) &&
      near(summary.modalGroupMissCount.mean, forecast.summary.expectedModalMisses) &&
      near(summary.meanSignedPairMarginError.mean, 0, 1e-7));
    return { chamber: "house", runId: row.runId, resultSha256: row.resultSha256,
      referenceIssuedAt: row.referenceIssuedAt, rows: row.rows, chains: row.chains,
      winnerRaceCount: 435, pairRaceCount: 412, pairInapplicableRaceCount: 23,
      sourcePins: { ...row.sourcePins }, perRace, summary,
      precisionScope: ref.precision_scope, qualifications: [...ref.qualifications],
      nativeLimitations: [...ref.native_limitations], evaluationLock: false, observedResultsConsumed: false };
  } catch { return null; }
}
