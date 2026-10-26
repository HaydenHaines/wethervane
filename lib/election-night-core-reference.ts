/** Pure exact-edition matcher. Caller authenticates catalog bytes; these are own-model fixed32 expectations, not observed accuracy. */
import type { ElectionNightForecast } from "./election-night-forecast.ts";
import type { NativeReferenceDistribution } from "./election-night-reference.ts";
import type { ElectionNightReferenceQualification } from "./election-night-publication.ts";
export interface ElectionNightCoreReferenceDistribution { mean: number; range90: { lower: number; upper: number } }
interface CoreIdentityRow {
  race_id: string; state: string; seat_id: string; event_kind: string; scheduled_general_event_date: string;
  first_candidate: { id: string; name: string; party: string }; second_candidate: { id: string; name: string; party: string };
  denominator: string; rule_proxy: string;
}
interface CoreNativeRow { race_id: string; point_margin_mean_pp: number; interval90_margin_pp: [number, number];
  expected_absolute_margin_error_pp: number; first_pair_win_probability: number; modal_pair_winner_id: string }
interface CoreNativeReference {
  schema: string; status: string; issued_utc: string; arm: "F" | "FP"; native_target_id: string; edition: string; input_cutoff_utc: string;
  cohort_id: string; cohort_size: number; common_target_cohort_sha256: string; cohort: CoreIdentityRow[]; excluded_from_this_reference: string[];
  sources: { path: string; sha256: string }[]; joint_rows: { path: string; sha256: string }; race_axis_binding: string;
  draw_identity: { shape: number[]; weight_per_row: number; order: string; original_race_axis_indices: number[];
    shared_across_selected_races: boolean; resampling: boolean; cross_bank_pairing: boolean };
  distributions: Record<string, NativeReferenceDistribution>; per_race: CoreNativeRow[];
  metric_definitions: Record<string, string>; qualifications: string[];
}
export interface ElectionNightCoreReferenceCatalogEntry {
  chamber: "senate"; runId: string; resultSha256: string; targetId: string; lawId: string;
  forecastIssuedAt: string; forecastCutoffAt: string; evaluationLock: false; observedResultsConsumed: false;
  sourcePins: Record<string, string>; reference: CoreNativeReference;
}
export interface ElectionNightCoreReferenceCatalog { schemaVersion: 1; entries: ElectionNightCoreReferenceCatalogEntry[] }
export interface ElectionNightCoreReference {
  chamber: "senate"; runId: string; resultSha256: string; targetId: string; lawId: string;
  forecastIssuedAt: string; forecastCutoffAt: string; referenceIssuedAt: string;
  cohortId: string; cohortLabel: string; cohortSha256: string; excludedRaceIds: string[]; sourcePins: Record<string, string>;
  drawIdentity: { shape: number[]; rowCount: number; weightPerRow: number; order: string; originalRaceAxisIndices: number[];
    sharedAcrossSelectedRaces: true; resampling: false; crossBankPairing: false; raceAxisBinding: string };
  perRace: { raceId: string; firstCandidateId: string; secondCandidateId: string; meanMarginPp: number;
    range90: { lower: number; upper: number }; expectedAbsoluteMarginErrorPp: number; firstPairWinProbability: number;
    modalOutcomeId: string; denominator: string; nativeDenominator: string; ruleProxy: string }[];
  summary: { meanAbsoluteMarginError: ElectionNightCoreReferenceDistribution; modalWinnerMissCount: ElectionNightCoreReferenceDistribution;
    interval90CoverageCount: ElectionNightCoreReferenceDistribution };
  qualifications: string[]; interpretation: string; evaluationLock: false; observedResultsConsumed: false;
}
const COMMON_PINS = {
  handoffSha256: "e571037c39a1203b79c2027ec72a4e2e7ff3b3340c1648eba39f75b6a7c41fbb",
  reviewSha256: "b82bc61967c3af40274178d79f65ba088e3f5313d32233ec4fbc0c9ac3648c8d",
  reviewFreezeSha256: "a90c5c30a2db502a71ff56b98c5c2fc7966f9b1675539fc30b479d9bca18f3d3",
  policySha256: "efe1bf193e2959284b73ea833af0a136f6c592001a2faf50505e3258e25cf0cd",
  mappingSha256: "9024069ac0f397c2d379663d653d384e32532548ef26772035eda61ce55265e0",
  failedV1EvidenceSha256: "d73ed9b406ba9c1184bca648eaf830a9f8c96ab7f6809e5165d52591554d2550",
};
const EXPECTED = {
  "current-fundamentals": { arm: "F", resultSha256: "37af627c934f519f32b2cf77a326f5658758a24153ee574fdc1a0ba46bfd51c1",
    targetId: "national_senate_aggregate_baseline_v1", issuedAt: "2026-10-20T15:25:53.623020+00:00", cutoffAt: "2026-10-20T15:21:23Z",
    referenceIssuedAt: "2026-10-26T17:25:31.414534+00:00", shape: [100000], rowCount: 100000,
    pins: { referenceSha256: "e8945138896bad678ee4d4dee5ab182e92ca150766deb47595cee6fc4825b120",
      forecastSha256: "c02bdc63a0df27824578c07d6e2b752b831ae4bbd858cabc432ee00fbbf94136",
      jointDrawsSha256: "3244418fd97e377e72dd013e6e3bad3c25948bd80228b3a97953e4599b88a6a8",
      jointRowsSha256: "cc2fd31bba31f5b6393483b859dcce03857b6abf37706caf81336de93a84b001",
      receiptSha256: "b2f083636dbcd8ffe535844bfe7af0af5aa6574b7b8c1358fbafde40135a5ee3" } },
  "current-polling": { arm: "FP", resultSha256: "905ce512c44104d3e48900745cec0f430c9e3954736614468e1a8b728b6da5f0",
    targetId: "national_senate_aggregate_poll_v1", issuedAt: "2026-10-26T15:05:11.698781+00:00", cutoffAt: "2026-10-26T15:03:57.212405+00:00",
    referenceIssuedAt: "2026-10-26T17:25:31.668413+00:00", shape: [4, 10000], rowCount: 40000,
    pins: { referenceSha256: "545ff5c25cc74e681afd10305b7c5a00c0653368cf3a6adda76184b9dd142568",
      forecastSha256: "a1ffba60d937bfc81d06df51af6ed917a7ddc71fd9860c1d702de7c7c5db67cb",
      jointDrawsSha256: "a7240d654861078c7850e3d0c10c1f287604035de531abf14d82821aa4d5cc31",
      jointRowsSha256: "ab6f8c2830ec582f78cad98d14ebf9f48747feccd144215265c1ed74a6ea534a",
      receiptSha256: "46dc22f42147ff4ceaf9981055fa9eed22362251af012debde12f9fcd8dfd53c" } },
} as const;
const EXCLUDED = ["2026:AK:senate:regular", "2026:GA:senate:regular", "2026:ME:senate:regular"];
const COHORT_SHA = "a11a92c18ad01867b5e68a9ac5f56a3398012abddacd980207098f09323e4ed1";
const COHORT_ID = "senate2026_plurality_core32_pre_result_v1";
const ORDER = "unchanged original row order and native chain axes, selected fixed race columns only";
const AXIS_BINDING = "F embeddedrace_ids; FP ordered axis from pinned reviewed source mapping and unchanged forecast builder, since FParchive stores no race_ids array.";
const DEFINITIONS = {
  errors: "Predictive joint margin minus saved mean margin, averaged equally over these32races; absolute error uses abs before averaging.",
  coverage: "Inclusive saved5th/95th percentile interval, count out of32.",
  misses: "Native pair-relative-lead differs from marginal modal principal; exactforecastprobability.5 selectsfirst explicitly.",
};
const QUALIFICATIONS = [
  "Model-implied reference only; no observed election error or grade.",
  "Centered signed error and nominal marginal coverage hold by construction; joint spread retains native dependence.",
  "Each bank is its own empirical measure; no cross-bank row matching or joint difference law.",
  "Preserve true edition dates and earlier-issued F comparison qualification.",
  "This is a new fixed32cohort artifact, not a relabeling of the existing35reference. All35displayrows and conditional-rule limitations remain.",
  "No forecast fit, resampling, actual results, evaluation lock or frontend release.",
];
const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const text = (v: unknown): v is string => typeof v === "string" && v.trim().length > 0;
const near = (a: number, b: number) => finite(a) && finite(b) && Math.abs(a-b) <= 1e-10;
const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
function need(v: unknown): asserts v { if (!v) throw Error("Not the exact qualified fixed32 reference."); }
function expected(f: ElectionNightForecast, qualification?: ElectionNightReferenceQualification) {
  return qualification ? { arm: qualification.arm, resultSha256: f.run.artifactSha256,
    targetId: qualification.targetId, issuedAt: qualification.forecastIssuedAt,
    cutoffAt: qualification.forecastCutoffAt, referenceIssuedAt: qualification.referenceIssuedAt,
    shape: qualification.shape, rowCount: qualification.rowCount, pins: qualification.sourcePins } :
    EXPECTED[f.run.id as keyof typeof EXPECTED];
}
/** Ineligible saved choices have no primary-reference error and never borrow F/FP summaries. */
export function supportsElectionNightCoreReference(f: ElectionNightForecast,
  qualification?: ElectionNightReferenceQualification): boolean {
  try {
    const e = expected(f, qualification);
    return !!e && f.schema === "ElectionNightForecastV1" && f.editionStatus === "dated_snapshot" && f.chamber === "senate" &&
      f.run.year === 2026 && f.run.artifactSha256 === e.resultSha256 && f.run.targetId === e.targetId &&
      f.run.lawId === (qualification?.lawId ?? "senate_principal_pair_aggregate_v1") &&
      (e.arm === "F" || e.arm === "FP") && f.run.issuedAt === e.issuedAt && f.run.cutoffAt === e.cutoffAt &&
      (qualification?.configuration ? f.run.strictAsOf === qualification.strictAsOf &&
        f.run.configuration.base === qualification.configuration.base &&
        f.run.configuration.polling === qualification.configuration.polling &&
        f.run.configuration.uncertainty === qualification.configuration.uncertainty :
        f.run.strictAsOf === (e.arm === "F") && f.run.configuration.base === "fundamentals" &&
        f.run.configuration.polling === (e.arm === "F" ? "none" : "standard") && f.run.configuration.uncertainty === "standard") &&
      f.sourceHashes.forecastSha256 === e.pins.forecastSha256 && f.sourceHashes.jointDrawsSha256 === e.pins.jointDrawsSha256 &&
      f.sourceHashes[e.arm === "F" ? "runReceiptSha256" : "receiptSha256"] === e.pins.receiptSha256 &&
      (e.arm === "F" ? f.run.sourceCutoffs.length === 0 : f.run.sourceCutoffs.length === 2 &&
        f.run.sourceCutoffs.find(s => s.label === "Structural inputs")?.at ===
          (qualification?.structuralCutoffAt ?? "2026-10-20T15:21:23Z") &&
        f.run.sourceCutoffs.find(s => s.label === "Polling inputs")?.at === e.cutoffAt);
  } catch { return false; }
}
function summary(d: NativeReferenceDistribution, lower: number, upper: number, count = false): ElectionNightCoreReferenceDistribution {
  const q = d.quantiles, values = [q.p01,q.p05,q.p50,q.p95,q.p99];
  need(finite(d.mean) && d.mean >= lower && d.mean <= upper && finite(d.standard_deviation) && d.standard_deviation >= 0 &&
    values.every((v,i) => finite(v) && v >= lower && v <= upper && (!count || Number.isInteger(v)) && (i === 0 || v >= values[i-1])));
  return { mean:d.mean, range90:{ lower:q.p05, upper:q.p95 } };
}
/** Null is the only fallback: no nearest edition, cohort substitution or inferred cross-bank law. */
export function matchElectionNightCoreReference(forecast: ElectionNightForecast, catalog: unknown,
  qualification?: ElectionNightReferenceQualification): ElectionNightCoreReference | null {
  try {
    need(supportsElectionNightCoreReference(forecast, qualification));
    const e = expected(forecast, qualification), value = catalog as ElectionNightCoreReferenceCatalog;
    need(value?.schemaVersion === 1 && Array.isArray(value.entries));
    const matches = value.entries.filter(r => r && r.chamber === "senate" && r.runId === forecast.run.id && r.resultSha256 === e.resultSha256);
    need(matches.length === 1); const entry = matches[0], ref = entry.reference, draw = ref.draw_identity;
    need(entry.targetId === e.targetId && entry.lawId === forecast.run.lawId && entry.forecastIssuedAt === e.issuedAt && entry.forecastCutoffAt === e.cutoffAt &&
      entry.evaluationLock === false && entry.observedResultsConsumed === false && ref.schema === "SenateFixedCohortScorecardReferenceV1" &&
      ref.status === "derived_pending_independent_review" && ref.arm === e.arm && ref.native_target_id === e.targetId && ref.edition === e.cutoffAt.slice(0,10) &&
      ref.input_cutoff_utc === e.cutoffAt && ref.issued_utc === e.referenceIssuedAt &&
      ref.cohort_id === (qualification?.cohortId ?? COHORT_ID) && ref.cohort_size === 32 &&
      ref.common_target_cohort_sha256 === (qualification?.cohortSha256 ?? COHORT_SHA) &&
      same(ref.excluded_from_this_reference, EXCLUDED) &&
      (!qualification || same(qualification.excludedRaceIds, EXCLUDED)));
    const trustedPins = qualification ? qualification.sourcePins : { ...COMMON_PINS, ...e.pins };
    need(Object.keys(entry.sourcePins).length === Object.keys(trustedPins).length && Object.entries(trustedPins).every(([k,v]) => entry.sourcePins[k] === v) &&
      ref.joint_rows.sha256 === e.pins.jointRowsSha256 && Array.isArray(ref.sources) &&
      [e.pins.forecastSha256,e.pins.jointDrawsSha256,e.pins.receiptSha256,
        trustedPins.policySha256,trustedPins.mappingSha256].every(pin => ref.sources.some(s => s.sha256 === pin)));
    need(forecast.races.length === 35 && new Set(forecast.races.map(r=>r.id)).size === 35 && forecast.summary.raceCount === 35 && forecast.summary.pairRaceCount === 35 &&
      forecast.summary.tiePolicy === "lexicographic_outcome_id" && EXCLUDED.every(id => forecast.races.some(r=>r.id===id)));
    const indices = forecast.races.flatMap((r,i) => EXCLUDED.includes(r.id) ? [] : [i]);
    need(same(draw.shape,e.shape) && e.rowCount === e.shape.reduce((product,n) => product*n,1) &&
      draw.weight_per_row === 1/e.rowCount && draw.order === ORDER && same(draw.original_race_axis_indices,indices) &&
      draw.shared_across_selected_races === true && draw.resampling === false && draw.cross_bank_pairing === false &&
      ref.race_axis_binding === (qualification?.raceAxisBinding ?? AXIS_BINDING) &&
      Array.isArray(ref.cohort) && ref.cohort.length === 32 && Array.isArray(ref.per_race) && ref.per_race.length === 32);
    let misses = 0;
    const perRace = indices.map((index,i) => {
      const r = forecast.races[index], pair = r.pair, identity = ref.cohort[i], row = ref.per_race[i];
      need(identity.race_id === r.id && row.race_id === r.id && identity.state === r.state && identity.seat_id === r.seatId && identity.scheduled_general_event_date === "2026-11-03" &&
        ["regular","special"].includes(identity.event_kind) && identity.event_kind === r.electionLabel.replace(/ election$/, "").toLowerCase() &&
        pair && pair.denominator === "Share of votes for the two named principal contenders" && pair.units === "percentage_points" && pair.orientation === "first_minus_second" &&
        pair.winnerRelationship === "same_event" && identity.denominator === "sum_of_named_principal_candidate_votes_under_declared_rule_proxy" && text(identity.rule_proxy) && r.outcomes.length === 2);
      const [first,second] = r.outcomes, p = row.first_pair_win_probability;
      need(identity.first_candidate.id === pair.firstCandidateId && identity.second_candidate.id === pair.secondCandidateId && first.id === pair.firstCandidateId && second.id === pair.secondCandidateId &&
        first.id !== second.id && first.kind === "candidate" && second.kind === "candidate" && first.members.length === 1 && second.members.length === 1 &&
        first.members[0].id === first.id && second.members[0].id === second.id && first.party === identity.first_candidate.party && second.party === identity.second_candidate.party &&
        finite(p) && p >= 0 && p <= 1 && p === first.winProbability && near(second.winProbability,1-p) && pair.firstExceedsHalf === p);
      const nativeModal = p >= .5 ? first.id : second.id, maximum = Math.max(p,1-p), ties = r.outcomes.filter(o=>o.winProbability===maximum).map(o=>o.id).sort();
      need(row.modal_pair_winner_id === nativeModal && ties[0] === nativeModal && r.modalPick.outcomeId === nativeModal && r.modalPick.probability === maximum && same(r.modalPick.tiedOutcomeIds,ties));
      misses += 1-maximum;
      need(near(row.point_margin_mean_pp,pair.meanMarginPp) && row.interval90_margin_pp.length === 2 && near(row.interval90_margin_pp[0],pair.range90.lower) &&
        near(row.interval90_margin_pp[1],pair.range90.upper) && row.interval90_margin_pp[0] <= row.interval90_margin_pp[1] &&
        [row.point_margin_mean_pp,...row.interval90_margin_pp].every(v=>finite(v)&&Math.abs(v)<=100) &&
        finite(row.expected_absolute_margin_error_pp) && row.expected_absolute_margin_error_pp >= 0 && row.expected_absolute_margin_error_pp <= 200);
      return { raceId:r.id,firstCandidateId:first.id,secondCandidateId:second.id,meanMarginPp:row.point_margin_mean_pp,
        range90:{lower:row.interval90_margin_pp[0],upper:row.interval90_margin_pp[1]},expectedAbsoluteMarginErrorPp:row.expected_absolute_margin_error_pp,
        firstPairWinProbability:p,modalOutcomeId:nativeModal,denominator:pair.denominator,nativeDenominator:identity.denominator,ruleProxy:identity.rule_proxy };
    });
    need(Object.keys(ref.metric_definitions).length === 3 && Object.entries(DEFINITIONS).every(([k,v])=>ref.metric_definitions[k]===v) &&
      Object.keys(ref.distributions).length === 4 &&
      (qualification ? Array.isArray(ref.qualifications) && ref.qualifications.every(text) : same(ref.qualifications,QUALIFICATIONS)));
    const summaries = { meanAbsoluteMarginError:summary(ref.distributions.mean_absolute_margin_error_pp,0,200),
      modalWinnerMissCount:summary(ref.distributions.modal_pair_winner_miss_count,0,32,true),interval90CoverageCount:summary(ref.distributions.interval_90_coverage_count,0,32,true) };
    need(near(summaries.modalWinnerMissCount.mean,misses) && near(summaries.meanAbsoluteMarginError.mean,perRace.reduce((s,r)=>s+r.expectedAbsoluteMarginErrorPp,0)/32) &&
      near(ref.distributions.mean_signed_margin_error_pp.mean,0));
    return {chamber:"senate",runId:entry.runId,resultSha256:entry.resultSha256,targetId:entry.targetId,lawId:entry.lawId,
      forecastIssuedAt:entry.forecastIssuedAt,forecastCutoffAt:entry.forecastCutoffAt,referenceIssuedAt:ref.issued_utc,
      cohortId:ref.cohort_id, cohortLabel:"Primary grade: 32 plurality candidate pairs",
      cohortSha256:ref.common_target_cohort_sha256,excludedRaceIds:[...EXCLUDED],sourcePins:{...entry.sourcePins},
      drawIdentity:{shape:[...draw.shape],rowCount:e.rowCount,weightPerRow:draw.weight_per_row,order:draw.order,originalRaceAxisIndices:[...indices],
        sharedAcrossSelectedRaces:true,resampling:false,crossBankPairing:false,raceAxisBinding:ref.race_axis_binding},perRace,summary:summaries,
      qualifications:[...ref.qualifications],evaluationLock:false,observedResultsConsumed:false,
      interpretation:"Model-implied expectations and 90% ranges for the fixed 32 plurality candidate pairs from this edition's original joint bank. These are not observed errors, calibration evidence, grade thresholds or an evaluation lock. F and FP have different dates and independent banks; no paired change distribution is defined."};
  } catch { return null; }
}
