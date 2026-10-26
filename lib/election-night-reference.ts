/**
 * Exact-edition projection of an authenticated reference catalog. The caller must
 * verify catalog bytes against the publication manifest. Pin existence alone is
 * neither byte authentication nor scientific admission. This v1 supports only the
 * independently reviewed standard Senate native reference, not observed accuracy.
 */
import type { ElectionNightForecast } from "./election-night-forecast.ts";

export interface ElectionNightReferencePins {
  referenceSha256: string; reviewSha256: string; forecastSha256: string; jointDrawsSha256: string;
  jointRowsSha256: string; receiptSha256: string; numericalGatesSha256: string;
  ownerOutputReviewSha256: string;
  [key: string]: string;
}
export interface NativeReferenceRow {
  race_id: string; state: string; first_candidate_id: string; second_candidate_id: string; target_id: string;
  rule_proxy: string; share_interpretation: string; excluded_options_assumption: string;
  margin_sign: string; denominator: string; point_margin_mean_pp: number;
  interval_90_lower_margin_pp: number; interval_90_upper_margin_pp: number;
  expected_absolute_margin_error_pp: number; first_pair_win_probability: number; modal_pair_winner_id: string;
  forecast_probability_exact_half: boolean; included_in_native_diagnostic_cohort: boolean;
}
export interface NativeReferenceDistribution {
  mean: number; standard_deviation: number;
  quantiles: { p01: number; p05: number; p50: number; p95: number; p99: number };
}
export type ElectionNightReferenceMetric = "mean_absolute_margin_error_pp" | "mean_signed_margin_error_pp" |
  "modal_pair_winner_miss_count" | "interval_90_coverage_count";
export interface ElectionNightReferenceCatalogEntry {
  chamber: "senate"; runId: string; resultSha256: string; targetId: string; lawId: string;
  forecastCutoffAt: string; structuralCutoffAt: string; forecastIssuedAt: string; referenceIssuedAt: string;
  evaluationLock: false; observedResultsConsumed: false; sourcePins: ElectionNightReferencePins;
  reference: {
    drawIdentity: { shape: [number, number]; order: string; weight_per_row: number;
      shared_across_all_races: true; resampling_or_independence_reconstruction: false };
    cohortSha256: string; cohort: NativeReferenceRow[];
    metricDefinitions: Record<ElectionNightReferenceMetric, string>;
    distributions: Record<ElectionNightReferenceMetric, NativeReferenceDistribution>; qualifications: string[];
  };
}
export interface ElectionNightReferenceCatalog { schemaVersion: 1; entries: ElectionNightReferenceCatalogEntry[] }
export interface ElectionNightReferenceDistribution {
  mean: number; standardDeviation: number; range90: { lower: number; upper: number };
  quantiles: NativeReferenceDistribution["quantiles"];
}
export interface ElectionNightReference {
  chamber: "senate"; runId: string; resultSha256: string; targetId: string; lawId: string;
  forecastCutoffAt: string; structuralCutoffAt: string; forecastIssuedAt: string; referenceIssuedAt: string;
  sourcePins: ElectionNightReferencePins; cohortSha256: string;
  drawIdentity: { shape: [4, 10000]; rowCount: 40000; order: string; weightPerRow: number;
    sharedAcrossAllRaces: true; resamplingOrIndependenceReconstruction: false };
  perRace: {
    raceId: string; state: string; firstCandidateId: string; secondCandidateId: string;
    denominator: string; nativeDenominator: string; units: "percentage_points"; orientation: "first_minus_second"; pointRule: "mean";
    meanMarginPp: number; range90: { lower: number; upper: number }; expectedAbsoluteMarginErrorPp: number;
    firstPairWinProbability: number; modalOutcomeId: string; nativeExactHalfTie: boolean;
    ruleProxy: string; shareInterpretation: string; excludedOptionsAssumption: string;
    /** V1 exports no per-race close/decisive probabilities. */
    closeWithin2Pp: null; firstAtLeast10Pp: null; secondAtLeast10Pp: null;
  }[];
  summary: {
    meanAbsoluteMarginError: ElectionNightReferenceDistribution;
    meanSignedMarginError: ElectionNightReferenceDistribution;
    modalWinnerMissCount: ElectionNightReferenceDistribution;
    interval90CoverageCount: ElectionNightReferenceDistribution;
  };
  metricDefinitions: Record<ElectionNightReferenceMetric, string>; qualifications: string[];
  evaluationLock: false; observedResultsConsumed: false; interpretation: string;
}
const TRUSTED = {
  referenceSha256: "48d36fdce404a74569553c859a8b8d66bb9790c0eef614950bfa98ca5da058db",
  reviewSha256: "4ba519f76c03852659c69eec96e84913c0fbcd00c8e8e3d82de6e9e7d845fe05",
  forecastSha256: "a1ffba60d937bfc81d06df51af6ed917a7ddc71fd9860c1d702de7c7c5db67cb",
  jointDrawsSha256: "a7240d654861078c7850e3d0c10c1f287604035de531abf14d82821aa4d5cc31",
  jointRowsSha256: "40e003ab4c7bf6aee9463d78ac3123e0ae702a325412763aec6f28ea3d2693ba",
  receiptSha256: "46dc22f42147ff4ceaf9981055fa9eed22362251af012debde12f9fcd8dfd53c",
  numericalGatesSha256: "77b1f90ef1399fda8cc44c79d01edcd92fb8f422a014ea9c27980e62a1da2044",
  ownerOutputReviewSha256: "f3fcaf56656187f71ab62756307265d05568077186d75479de37f4607e6855f1",
};
const COHORT_SHA = "a97bd621a64170dc8ded1b818adadd4bfaa1ddb27e0a9870d9f69e282010b914";
const NATIVE_DENOMINATOR = "sum_of_named_principal_candidate_votes_under_declared_rule_proxy";
const DISPLAY_DENOMINATOR = "Share of votes for the two named principal contenders";
const ORDER = "unchanged native chain, within-chain draw; C-order flattening";
const METRICS: Record<ElectionNightReferenceMetric, string> = {
  mean_signed_margin_error_pp: "Equal-race mean of joint native margin minus frozen predicted mean margin.",
  mean_absolute_margin_error_pp: "Equal-race mean absolute difference between joint native margin and frozen predicted mean margin.",
  interval_90_coverage_count: "Count of joint native margins inside inclusive saved forecast5th/95th percentile bounds.",
  modal_pair_winner_miss_count: "Count of native principal-pair winners differing from the marginal modal principal; exact0.5 probability selects first candidate explicitly.",
};
const hash = (v: unknown): v is string => typeof v === "string" && /^[a-f0-9]{64}$/.test(v);
const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const close = (a: number, b: number) => finite(a) && finite(b) && Math.abs(a - b) <= 1e-10;
const strings = (v: unknown): v is string[] => Array.isArray(v) && v.length > 0 && v.every(s => typeof s === "string" && s.trim().length > 0);
function requireValue(value: unknown): asserts value { if (!value) throw Error("Reference does not match this exact reviewed forecast."); }
function distribution(value: NativeReferenceDistribution, lower: number, upper: number): ElectionNightReferenceDistribution {
  const q = value.quantiles, values = [q.p01, q.p05, q.p50, q.p95, q.p99];
  requireValue(finite(value.mean) && value.mean >= lower && value.mean <= upper && finite(value.standard_deviation) &&
    value.standard_deviation >= 0 && values.every(x => finite(x) && x >= lower && x <= upper) &&
    values.every((x, i) => i === 0 || x >= values[i - 1]));
  return { mean: value.mean, standardDeviation: value.standard_deviation,
    range90: { lower: q.p05, upper: q.p95 }, quantiles: { ...q } };
}
/** Return null for unavailable, unsupported, stale or malformed editions. Never borrow another model's metrics. */
export function matchElectionNightReference(forecast: ElectionNightForecast, catalog: unknown): ElectionNightReference | null {
  try {
    const value = catalog as ElectionNightReferenceCatalog;
    requireValue(value?.schemaVersion === 1 && Array.isArray(value.entries));
    requireValue(forecast.schema === "ElectionNightForecastV1" && forecast.editionStatus === "dated_snapshot" &&
      forecast.chamber === "senate" && forecast.run.year === 2026 && forecast.run.id === "current-polling" &&
      forecast.run.targetId === "national_senate_aggregate_poll_v1" && forecast.run.lawId === "senate_principal_pair_aggregate_v1" &&
      forecast.run.configuration.base === "fundamentals" && forecast.run.configuration.polling === "standard" &&
      forecast.run.configuration.uncertainty === "standard" && forecast.run.strictAsOf === false && hash(forecast.run.artifactSha256));
    const matches = value.entries.filter(e => e && e.chamber === forecast.chamber && e.runId === forecast.run.id &&
      e.resultSha256 === forecast.run.artifactSha256);
    requireValue(matches.length === 1);
    const entry = matches[0], ref = entry.reference, draw = ref.drawIdentity;
    requireValue(entry.targetId === forecast.run.targetId && entry.lawId === forecast.run.lawId && entry.evaluationLock === false &&
      entry.observedResultsConsumed === false && entry.forecastCutoffAt === forecast.run.cutoffAt && entry.forecastIssuedAt === forecast.run.issuedAt &&
      entry.forecastCutoffAt === "2026-10-26T15:03:57.212405+00:00" && entry.structuralCutoffAt === "2026-10-20T15:21:23Z" &&
      entry.referenceIssuedAt === "2026-10-26T15:12:19.773307+00:00");
    requireValue(forecast.run.sourceCutoffs.length === 2 &&
      forecast.run.sourceCutoffs.find(s => s.label === "Structural inputs")?.at === entry.structuralCutoffAt &&
      forecast.run.sourceCutoffs.find(s => s.label === "Polling inputs")?.at === entry.forecastCutoffAt);
    requireValue(Object.keys(entry.sourcePins).length === Object.keys(TRUSTED).length && Object.values(entry.sourcePins).every(hash) &&
      Object.entries(TRUSTED).every(([key, expected]) => entry.sourcePins[key] === expected));
    for (const key of ["forecastSha256", "jointDrawsSha256", "receiptSha256", "numericalGatesSha256"])
      requireValue(entry.sourcePins[key] === forecast.sourceHashes[key]);
    requireValue(ref.cohortSha256 === COHORT_SHA && draw.shape.length === 2 && draw.shape[0] === 4 && draw.shape[1] === 10000 &&
      draw.weight_per_row === 1 / 40000 && draw.order === ORDER && draw.shared_across_all_races === true &&
      draw.resampling_or_independence_reconstruction === false);
    requireValue(Array.isArray(ref.cohort) && ref.cohort.length === 35 && forecast.races.length === 35 &&
      new Set(ref.cohort.map(row => row.race_id)).size === 35 && new Set(forecast.races.map(r => r.id)).size === 35 &&
      forecast.summary.raceCount === 35 && forecast.summary.pairRaceCount === 35 && forecast.summary.tiePolicy === "lexicographic_outcome_id");
    let expectedMisses = 0, exactTies = 0;
    const perRace: ElectionNightReference["perRace"] = ref.cohort.map((row, index) => {
      const race = forecast.races[index], pair = race.pair;
      requireValue(row.race_id === race.id && row.state === race.state && row.target_id === entry.targetId &&
        row.included_in_native_diagnostic_cohort === true && row.margin_sign === "first_candidate_minus_second_candidate" &&
        row.denominator === NATIVE_DENOMINATOR && pair && pair.denominator === DISPLAY_DENOMINATOR &&
        pair.units === "percentage_points" && pair.orientation === "first_minus_second" && pair.winnerRelationship === "same_event" &&
        row.first_candidate_id === pair.firstCandidateId && row.second_candidate_id === pair.secondCandidateId &&
        row.first_candidate_id !== row.second_candidate_id && race.outcomes.length === 2);
      const [first, second] = race.outcomes;
      requireValue(first.id === row.first_candidate_id && second.id === row.second_candidate_id &&
        first.kind === "candidate" && second.kind === "candidate" && first.members.length === 1 && second.members.length === 1 &&
        first.members[0].id === first.id && second.members[0].id === second.id &&
        first.name === pair.firstCandidateName && second.name === pair.secondCandidateName);
      const p = row.first_pair_win_probability;
      requireValue(finite(p) && p >= 0 && p <= 1 && p === first.winProbability && close(second.winProbability, 1 - p) &&
        pair.firstExceedsHalf === p && row.forecast_probability_exact_half === (p === 0.5));
      const nativeModal = p >= 0.5 ? row.first_candidate_id : row.second_candidate_id;
      const maximum = Math.max(first.winProbability, second.winProbability);
      const ties = race.outcomes.filter(o => o.winProbability === maximum).map(o => o.id).sort();
      // Native v1 chooses the first contender at exactly .5. Guide picks IDs lexically.
      // Accept only when the actual chosen outcome agrees; never reinterpret miss-count rows.
      requireValue(row.modal_pair_winner_id === nativeModal && ties[0] === nativeModal &&
        race.modalPick.outcomeId === nativeModal && race.modalPick.kind === "candidate" && race.modalPick.probability === maximum &&
        JSON.stringify(race.modalPick.tiedOutcomeIds) === JSON.stringify(ties));
      expectedMisses += 1 - maximum; exactTies += Number(p === 0.5);
      requireValue(close(row.point_margin_mean_pp, pair.meanMarginPp) && close(row.interval_90_lower_margin_pp, pair.range90.lower) &&
        close(row.interval_90_upper_margin_pp, pair.range90.upper) && pair.range90.lower <= pair.range90.upper &&
        [row.point_margin_mean_pp, row.interval_90_lower_margin_pp, row.interval_90_upper_margin_pp].every(x => finite(x) && Math.abs(x) <= 100) &&
        finite(row.expected_absolute_margin_error_pp) && row.expected_absolute_margin_error_pp >= 0 && row.expected_absolute_margin_error_pp <= 200);
      requireValue(typeof row.rule_proxy === "string" && row.rule_proxy.length > 0 && typeof row.excluded_options_assumption === "string" &&
        row.excluded_options_assumption.length > 0 && race.notes.includes(row.rule_proxy) && race.notes.includes(row.excluded_options_assumption) &&
        pair.qualifications.includes(row.rule_proxy) && pair.qualifications.includes(row.excluded_options_assumption) &&
        (first.party === "I" ? row.share_interpretation === "opposition_proxy" : first.party === "D" && row.share_interpretation === "democratic_two_party"));
      return { raceId: row.race_id, state: row.state, firstCandidateId: row.first_candidate_id, secondCandidateId: row.second_candidate_id,
        denominator: pair.denominator, nativeDenominator: row.denominator, units: "percentage_points", orientation: "first_minus_second", pointRule: "mean",
        meanMarginPp: row.point_margin_mean_pp, range90: { lower: row.interval_90_lower_margin_pp, upper: row.interval_90_upper_margin_pp },
        expectedAbsoluteMarginErrorPp: row.expected_absolute_margin_error_pp, firstPairWinProbability: p, modalOutcomeId: nativeModal,
        nativeExactHalfTie: row.forecast_probability_exact_half, ruleProxy: row.rule_proxy, shareInterpretation: row.share_interpretation,
        excludedOptionsAssumption: row.excluded_options_assumption, closeWithin2Pp: null, firstAtLeast10Pp: null, secondAtLeast10Pp: null };
    });
    requireValue(close(expectedMisses, forecast.summary.expectedModalMisses) && exactTies === forecast.summary.topTieRaceCount);
    requireValue(Object.keys(ref.distributions).length === 4 && Object.keys(ref.metricDefinitions).length === 4 &&
      Object.entries(METRICS).every(([key, definition]) => ref.metricDefinitions[key as ElectionNightReferenceMetric] === definition));
    const summary = {
      meanAbsoluteMarginError: distribution(ref.distributions.mean_absolute_margin_error_pp, 0, 200),
      meanSignedMarginError: distribution(ref.distributions.mean_signed_margin_error_pp, -200, 200),
      modalWinnerMissCount: distribution(ref.distributions.modal_pair_winner_miss_count, 0, 35),
      interval90CoverageCount: distribution(ref.distributions.interval_90_coverage_count, 0, 35),
    };
    requireValue(close(summary.modalWinnerMissCount.mean, expectedMisses) &&
      close(summary.meanAbsoluteMarginError.mean, perRace.reduce((sum, r) => sum + r.expectedAbsoluteMarginErrorPp, 0) / perRace.length) &&
      close(summary.meanSignedMarginError.mean, 0) && strings(ref.qualifications));
    return { chamber: "senate", runId: entry.runId, resultSha256: entry.resultSha256, targetId: entry.targetId, lawId: entry.lawId,
      forecastCutoffAt: entry.forecastCutoffAt, structuralCutoffAt: entry.structuralCutoffAt, forecastIssuedAt: entry.forecastIssuedAt,
      referenceIssuedAt: entry.referenceIssuedAt, sourcePins: { ...entry.sourcePins }, cohortSha256: ref.cohortSha256,
      drawIdentity: { shape: [4, 10000], rowCount: 40000, order: draw.order, weightPerRow: draw.weight_per_row,
        sharedAcrossAllRaces: true, resamplingOrIndependenceReconstruction: false },
      perRace, summary, metricDefinitions: { ...ref.metricDefinitions }, qualifications: [...ref.qualifications],
      evaluationLock: false, observedResultsConsumed: false,
      interpretation: "Model-implied expectations and 90% ranges from the same 40,000 saved joint rows for all 35 named principal pairs. These are not observed errors, empirical accuracy, calibration validation, Monte Carlo error, grade thresholds or an evaluation lock. A subset requires a separately reviewed reference." };
  } catch { return null; }
}
