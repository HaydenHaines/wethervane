/**
 * Pure editorial rubric for already authenticated comparison contracts and loss
 * records. Callers admit source bytes, the pre-outcome cohort/result policy and
 * scientific event mapping. References bind those decisions; a boolean or a
 * SHA-shaped string alone never establishes approval. No real results are loaded.
 */
export type ElectionNightGrade = "A" | "B" | "C" | "D" | "F";
// Ordered principal identities from the reviewed mapping SHA below; result operators remain separately admitted.
const earlierPrincipalPairs: readonly (readonly string[])[] = Object.freeze([
  Object.freeze(["2026:AK:senate:regular","contender:AK:peltola-mary","contender:AK:sullivan-dan-s","sum_of_named_principal_candidate_votes_under_declared_rule_proxy"]),
  Object.freeze(["2026:AL:senate:regular","contender:AL:everett-wess","contender:AL:barry-moore","sum_of_named_principal_candidate_votes_under_declared_rule_proxy"]),
  Object.freeze(["2026:AR:senate:regular","contender:AR:hallie-shoffner","contender:AR:tom-cotton","sum_of_named_principal_candidate_votes_under_declared_rule_proxy"]),
  Object.freeze(["2026:CO:senate:regular","contender:CO:john-hickenlooper","contender:CO:mark-baisley","sum_of_named_principal_candidate_votes_under_declared_rule_proxy"]),
  Object.freeze(["2026:DE:senate:regular","contender:DE:chris-coons","contender:DE:michael-dr-mike-katz","sum_of_named_principal_candidate_votes_under_declared_rule_proxy"]),
  Object.freeze(["2026:FL:senate:special","contender:FL:angie-nixon","contender:FL:ashley-moody","sum_of_named_principal_candidate_votes_under_declared_rule_proxy"]),
  Object.freeze(["2026:GA:senate:regular","contender:GA:jon-ossoff","contender:GA:mike-collins","sum_of_named_principal_candidate_votes_under_declared_rule_proxy"]),
  Object.freeze(["2026:IA:senate:regular","contender:IA:josh-turek","contender:IA:ashley-hinson","sum_of_named_principal_candidate_votes_under_declared_rule_proxy"]),
  Object.freeze(["2026:ID:senate:regular","contender:ID:todd-achilles","contender:ID:jim-risch","sum_of_named_principal_candidate_votes_under_declared_rule_proxy"]),
  Object.freeze(["2026:IL:senate:regular","contender:IL:juliana-stratton","contender:IL:don-tracy","sum_of_named_principal_candidate_votes_under_declared_rule_proxy"]),
  Object.freeze(["2026:KS:senate:regular","contender:KS:adam-hamilton","contender:KS:roger-marshall","sum_of_named_principal_candidate_votes_under_declared_rule_proxy"]),
  Object.freeze(["2026:KY:senate:regular","contender:KY:charles-booker","contender:KY:andy-barr","sum_of_named_principal_candidate_votes_under_declared_rule_proxy"]),
  Object.freeze(["2026:LA:senate:regular","contender:LA:jamie-davis","contender:LA:julia-letlow","sum_of_named_principal_candidate_votes_under_declared_rule_proxy"]),
  Object.freeze(["2026:MA:senate:regular","contender:MA:edward-j-markey","contender:MA:john-deaton","sum_of_named_principal_candidate_votes_under_declared_rule_proxy"]),
  Object.freeze(["2026:ME:senate:regular","contender:ME:jackson-troy-d","contender:ME:collins-susan-m","sum_of_named_principal_candidate_votes_under_declared_rule_proxy"]),
  Object.freeze(["2026:MI:senate:regular","contender:MI:abdul-el-sayed","contender:MI:mike-rogers","sum_of_named_principal_candidate_votes_under_declared_rule_proxy"]),
  Object.freeze(["2026:MN:senate:regular","contender:MN:peggy-flanagan","contender:MN:michele-tafoya","sum_of_named_principal_candidate_votes_under_declared_rule_proxy"]),
  Object.freeze(["2026:MS:senate:regular","contender:MS:scott-colom","contender:MS:cindy-hyde-smith","sum_of_named_principal_candidate_votes_under_declared_rule_proxy"]),
  Object.freeze(["2026:MT:senate:regular","contender:MT:seth-bodnar","contender:MT:kurt-alme","sum_of_named_principal_candidate_votes_under_declared_rule_proxy"]),
  Object.freeze(["2026:NC:senate:regular","contender:NC:roy-cooper","contender:NC:michael-whatley","sum_of_named_principal_candidate_votes_under_declared_rule_proxy"]),
  Object.freeze(["2026:NE:senate:regular","contender:NE:dan-osborn","contender:NE:pete-ricketts","sum_of_named_principal_candidate_votes_under_declared_rule_proxy"]),
  Object.freeze(["2026:NH:senate:regular","contender:NH:chris-pappas","contender:NH:john-sununu","sum_of_named_principal_candidate_votes_under_declared_rule_proxy"]),
  Object.freeze(["2026:NJ:senate:regular","contender:NJ:cory-booker","contender:NJ:justin-murphy","sum_of_named_principal_candidate_votes_under_declared_rule_proxy"]),
  Object.freeze(["2026:NM:senate:regular","contender:NM:ben-r-lujan","contender:NM:larry-e-marker","sum_of_named_principal_candidate_votes_under_declared_rule_proxy"]),
  Object.freeze(["2026:OH:senate:special","contender:OH:sherrod-brown","contender:OH:jon-husted","sum_of_named_principal_candidate_votes_under_declared_rule_proxy"]),
  Object.freeze(["2026:OK:senate:regular","contender:OK:n-kiyla-jasmine-thomas","contender:OK:kevin-hern","sum_of_named_principal_candidate_votes_under_declared_rule_proxy"]),
  Object.freeze(["2026:OR:senate:regular","contender:OR:jeff-merkley","contender:OR:david-brock-smith","sum_of_named_principal_candidate_votes_under_declared_rule_proxy"]),
  Object.freeze(["2026:RI:senate:regular","contender:RI:john-francis-reed","contender:RI:raymond-t-mckay","sum_of_named_principal_candidate_votes_under_declared_rule_proxy"]),
  Object.freeze(["2026:SC:senate:regular","contender:SC:annie-andrews","contender:SC:darline-graham","sum_of_named_principal_candidate_votes_under_declared_rule_proxy"]),
  Object.freeze(["2026:SD:senate:regular","contender:SD:brian-l-bengs","contender:SD:mike-rounds","sum_of_named_principal_candidate_votes_under_declared_rule_proxy"]),
  Object.freeze(["2026:TN:senate:regular","contender:TN:marquita-bradshaw","contender:TN:bill-hagerty","sum_of_named_principal_candidate_votes_under_declared_rule_proxy"]),
  Object.freeze(["2026:TX:senate:regular","contender:TX:james-talarico","contender:TX:ken-paxton","sum_of_named_principal_candidate_votes_under_declared_rule_proxy"]),
  Object.freeze(["2026:VA:senate:regular","contender:VA:mark-r-warner","contender:VA:bert-mizusawa","sum_of_named_principal_candidate_votes_under_declared_rule_proxy"]),
  Object.freeze(["2026:WV:senate:regular","contender:WV:rachel-fetty-anderson","contender:WV:shelley-moore-capito","sum_of_named_principal_candidate_votes_under_declared_rule_proxy"]),
  Object.freeze(["2026:WY:senate:regular","contender:WY:james-byrd","contender:WY:harriet-hageman","sum_of_named_principal_candidate_votes_under_declared_rule_proxy"]),
]);
const earlierBenchmark = Object.freeze({
  comparisonContractId: "senate2026_conditional_principal_pair_F20261020_FP20261026_comparison_v1",
  reviewSha256: "901b1c869b2ce7c5302ad57cef7805eff860324c78821033052b743c69cd5f9e",
  mappingSha256: "9024069ac0f397c2d379663d653d384e32532548ef26772035eda61ce55265e0",
  freezeManifestSha256: "3abe88f2434279b3cd70849e4fd9158fb50bcbc91c513781843832fade8a706e",
  baseline: Object.freeze({ runId: "current-fundamentals", targetId: "national_senate_aggregate_baseline_v1",
    resultSha256: "37af627c934f519f32b2cf77a326f5658758a24153ee574fdc1a0ba46bfd51c1",
    lawId: "senate_principal_pair_aggregate_v1", issuedAt: "2026-10-20T15:25:53.623020+00:00",
    nativeForecastSha256: "c02bdc63a0df27824578c07d6e2b752b831ae4bbd858cabc432ee00fbbf94136",
    cutoffAt: "2026-10-20T15:21:23Z", horizonCalendarDays: 14 }),
  model: Object.freeze({ runId: "current-polling", targetId: "national_senate_aggregate_poll_v1",
    resultSha256: "905ce512c44104d3e48900745cec0f430c9e3954736614468e1a8b728b6da5f0",
    lawId: "senate_principal_pair_aggregate_v1", issuedAt: "2026-10-26T15:05:11.698781+00:00",
    nativeForecastSha256: "a1ffba60d937bfc81d06df51af6ed917a7ddc71fd9860c1d702de7c7c5db67cb",
    cutoffAt: "2026-10-26T15:03:57.212405+00:00", horizonCalendarDays: 8 }),
  cohortSize: 35,
  publicExplanation: "This compares the October 26 forecast with the October 20 basic electoral-history edition on the same results. The forecasts had different information and lead times; the comparison does not isolate the value of polling.",
  gradeMeaning: "Descriptive performance of these forecasts as issued; no causal layer skill, calibration or challenger-promotion claim.",
});
export const ELECTION_NIGHT_RUBRIC = Object.freeze({
  id: "ElectionNightRubricEditorialDecisionV2",
  decisionSha256: "edb82165b1e6fc76335912108b887b40ded71f0c0de76f3262dcdf39f3e0b79e",
  bands: Object.freeze([
    Object.freeze({ grade: "A", minimumSkill: 0.2 }), Object.freeze({ grade: "B", minimumSkill: 0.1 }),
    Object.freeze({ grade: "C", minimumSkill: 0 }), Object.freeze({ grade: "D", minimumSkill: -0.1 }),
    Object.freeze({ grade: "F", minimumSkill: null }),
  ] as const),
  equalityTolerance: 1e-12,
  winnerMetric: "multicategory Brier sum divided by two",
  intervalMetric: "central 90% interval score in percentage points",
  skillFormula: "1 - model mean loss / declared history mean loss; baseline loss must be positive",
  overallRule: "Lower of the winner and 90% interval letters; no two-component overall letter without both complete components.",
  coinFlipRule: "No better than D when winner skill against fixed coin flip is nonpositive within the declared equality tolerance; preserve a lower F. Senate (.5,.5); House (.5,.5,0) on exhaustive D/R/O. Uniform thirds is a separate House reference.",
  bandsQualification: "Editorial cutoffs for this declared comparison, not universal scientific constants or proof of calibration. Assign from unrounded losses, with absolute skill equality tolerance 1e-12.",
  houseMappingReviewSha256: "70de07db3a0bbb5b637f10daf9b484ccc6d678c6b7dd08a781a8056fcf5f70fa",
  earlierBenchmark,
});
export interface RubricReference { id: string; sha256: string }
export interface RubricForecastBinding {
  chamber: "senate" | "house"; runId: string; resultSha256: string; nativeForecastSha256: string;
  targetId: string; lawId: string; issuedAt: string; cutoffAt: string; horizonCalendarDays: number;
}
interface RubricTargetBase { raceId: string; eventId: string; roundId: string; resultRuleId: string; denominator: string }
export interface RubricWinnerTarget extends RubricTargetBase {
  estimand: "senate_principal_pair_lead" | "house_official_party_winner";
  outcomeIds: string[]; firstCandidateId: string | null; secondCandidateId: string | null;
}
export interface RubricIntervalTarget extends RubricTargetBase {
  estimand: "named_pair_margin"; firstCandidateId: string; secondCandidateId: string;
  units: "percentage_points"; orientation: "first_minus_second"; intervalLevel: 0.9;
}
export type RubricTarget = RubricWinnerTarget | RubricIntervalTarget;
export interface ElectionNightRubricContract {
  schema: "ElectionNightRubricComparisonV1"; id: string; sha256: string; rubricDecisionSha256: string;
  comparisonMode: "matched_horizon" | "explicit_earlier_benchmark"; label: string;
  model: RubricForecastBinding; baseline: RubricForecastBinding;
  comparisonReview: RubricReference; cohortDefinition: RubricReference; resultPolicy: RubricReference;
  earlierBenchmarkPins: { mappingSha256: string; freezeManifestSha256: string } | null;
  houseMappingReviewSha256: string | null;
  winnerCohort: RubricWinnerTarget[]; intervalCohort: RubricIntervalTarget[];
  /** Every winner race outside the predeclared margin cohort is retained here. */
  intervalExclusions: { raceId: string; reason: string }[];
}
export interface RubricResultBinding {
  recordId: string; revision: string; recordSha256: string; sourceSha256: string; resultPolicySha256: string;
  finality: "final" | "certified" | "called" | "provisional" | "unresolved" | "unsupported";
}
export interface RubricLossRecord {
  forecastSha256: string; target: RubricTarget;
  status: "scored" | "unresolved" | "unsupported" | "mismatched";
  metric: "multicategory_brier_sum" | "central90_interval_score_pp";
  result: RubricResultBinding | null;
  /** Winner: full unnormalized Brier sum, not already divided by two. */
  loss: number | null; observedValue: string | number | null; reason: string | null;
}
export interface PairedRubricLossRecord {
  raceId: string; comparisonContractSha256: string; cohortDefinitionSha256: string;
  model: RubricLossRecord; baseline: RubricLossRecord;
}
export interface ElectionNightRubricInput {
  contract: ElectionNightRubricContract; modelForecast: RubricForecastBinding; baselineForecast: RubricForecastBinding;
  winnerRows: PairedRubricLossRecord[]; intervalRows: PairedRubricLossRecord[];
}
export interface RubricPairedAssessment {
  raceId: string; status: "final" | "provisional" | "pending" | "mismatched"; reason: string | null;
  modelLoss: number | null; baselineLoss: number | null; pairedLossDifference: number | null;
  coinFlipLoss: number | null; uniformThreeCategoryLoss: number | null;
}
export interface RubricComponent {
  complete: boolean; grade: ElectionNightGrade | null; historyGrade: ElectionNightGrade | null;
  reason: string | null; skill: number | null; coinFlipSkill: number | null; uniformThreeCategorySkill: number | null;
  coinFlipCapApplied: boolean;
  coverage: { total: number; final: number; provisional: number; pending: number; mismatched: number; unexpected: number };
  raw: { scope: "available_matched_rows"; count: number; finalCount: number; provisionalCount: number;
    modelMeanLoss: number | null; baselineMeanLoss: number | null; meanPairedLossDifference: number | null;
    coinFlipMeanLoss: number | null; uniformThreeCategoryMeanLoss: number | null };
  rows: RubricPairedAssessment[];
}
export interface ElectionNightRubricResult {
  policyId: string; policyDecisionSha256: string; status: "graded" | "not_graded";
  comparison: { id: string; sha256: string; label: string; mode: ElectionNightRubricContract["comparisonMode"];
    matchedHorizon: boolean; explanation: string } | null;
  issues: string[]; winner: RubricComponent; interval: RubricComponent;
  overall: { grade: ElectionNightGrade | null; reason: string | null; coinFlipCapApplied: boolean };
  winnerOnlyGrade: ElectionNightGrade | null;
  intervalExcluded: { raceId: string; reason: string }[];
  unexpectedRows: { component: "winner" | "interval"; raceId: string; reason: string }[];
}
const text = (v: unknown): v is string => typeof v === "string" && v.trim().length > 0;
const hash = (v: unknown): v is string => typeof v === "string" && /^[a-f0-9]{64}$/.test(v);
const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
function requireValue(v: unknown, reason: string): asserts v { if (!v) throw Error(reason); }
function canonical(v: unknown): string { return JSON.stringify(v, (_, x) => x && typeof x === "object" && !Array.isArray(x)
  ? Object.fromEntries(Object.entries(x).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)) : x); }
const same = (a: unknown, b: unknown) => canonical(a) === canonical(b);
const reference = (v: RubricReference) => v && text(v.id) && hash(v.sha256);
function binding(v: RubricForecastBinding): boolean {
  return !!v && ["senate", "house"].includes(v.chamber) && text(v.runId) && hash(v.resultSha256) && hash(v.nativeForecastSha256) &&
    text(v.targetId) && text(v.lawId) && Number.isFinite(Date.parse(v.issuedAt)) && Number.isFinite(Date.parse(v.cutoffAt)) &&
    Date.parse(v.cutoffAt) <= Date.parse(v.issuedAt) && Number.isSafeInteger(v.horizonCalendarDays) && v.horizonCalendarDays >= 0;
}
function baseTarget(t: RubricTarget): boolean { return !!t && [t.raceId, t.eventId, t.roundId, t.resultRuleId, t.denominator].every(text); }
function winnerTarget(t: RubricWinnerTarget, chamber: "senate" | "house"): boolean {
  if (!baseTarget(t) || !Array.isArray(t.outcomeIds) || !t.outcomeIds.every(text) || new Set(t.outcomeIds).size !== t.outcomeIds.length) return false;
  if (chamber === "house") return t.estimand === "house_official_party_winner" && same(t.outcomeIds, ["D", "R", "O"]) &&
    t.firstCandidateId === null && t.secondCandidateId === null;
  return t.estimand === "senate_principal_pair_lead" && text(t.firstCandidateId) && text(t.secondCandidateId) &&
    t.firstCandidateId !== t.secondCandidateId && same(t.outcomeIds, [t.firstCandidateId, t.secondCandidateId]);
}
function intervalTarget(t: RubricIntervalTarget): boolean { return baseTarget(t) && t.estimand === "named_pair_margin" &&
  text(t.firstCandidateId) && text(t.secondCandidateId) && t.firstCandidateId !== t.secondCandidateId &&
  t.units === "percentage_points" && t.orientation === "first_minus_second" && t.intervalLevel === 0.9; }
function validate(input: ElectionNightRubricInput): void {
  const c = input.contract;
  requireValue(c?.schema === "ElectionNightRubricComparisonV1" && text(c.id) && hash(c.sha256) && text(c.label) &&
    c.rubricDecisionSha256 === ELECTION_NIGHT_RUBRIC.decisionSha256 && reference(c.comparisonReview) && reference(c.cohortDefinition) && reference(c.resultPolicy),
    "Missing or mismatched rubric/comparison/cohort/result-policy references.");
  requireValue(binding(c.model) && binding(c.baseline) && same(input.modelForecast, c.model) && same(input.baselineForecast, c.baseline) &&
    c.model.chamber === c.baseline.chamber, "Forecast editions do not match the declared comparison contract.");
  if (c.comparisonMode === "matched_horizon") requireValue(c.model.cutoffAt === c.baseline.cutoffAt &&
    c.model.horizonCalendarDays === c.baseline.horizonCalendarDays && c.earlierBenchmarkPins === null,
    "A matched-horizon comparison requires the same information cutoff and horizon.");
  else {
    const e = ELECTION_NIGHT_RUBRIC.earlierBenchmark;
    requireValue(c.comparisonMode === "explicit_earlier_benchmark" && c.model.chamber === "senate" && c.id === e.comparisonContractId &&
      c.comparisonReview.sha256 === e.reviewSha256 && c.earlierBenchmarkPins?.mappingSha256 === e.mappingSha256 &&
      c.earlierBenchmarkPins?.freezeManifestSha256 === e.freezeManifestSha256 &&
      Object.entries(e.model).every(([k, v]) => c.model[k as keyof RubricForecastBinding] === v) &&
      Object.entries(e.baseline).every(([k, v]) => c.baseline[k as keyof RubricForecastBinding] === v) &&
      c.winnerCohort.length === e.cohortSize && c.intervalCohort.length === e.cohortSize &&
      same(c.winnerCohort.map(t => [t.raceId, t.firstCandidateId, t.secondCandidateId, t.denominator]), earlierPrincipalPairs),
      "Earlier benchmarking is authorized only for the exact reviewed October 20/26 Senate comparison.");
  }
  requireValue(c.model.chamber !== "house" || c.houseMappingReviewSha256 === ELECTION_NIGHT_RUBRIC.houseMappingReviewSha256,
    "House grading requires the explicit reviewed fixed D/R/O party-winner mapping.");
  requireValue(Array.isArray(c.winnerCohort) && c.winnerCohort.length > 0 && c.winnerCohort.every(t => winnerTarget(t, c.model.chamber)) &&
    new Set(c.winnerCohort.map(t => t.raceId)).size === c.winnerCohort.length && Array.isArray(c.intervalCohort) &&
    c.intervalCohort.every(intervalTarget) && new Set(c.intervalCohort.map(t => t.raceId)).size === c.intervalCohort.length,
    "Invalid or duplicate frozen winner/margin targets.");
  if (c.model.chamber === "senate") requireValue(c.intervalCohort.every(t => {
    const w = c.winnerCohort.find(w => w.raceId === t.raceId);
    return w && ["eventId", "roundId", "resultRuleId", "denominator", "firstCandidateId", "secondCandidateId"].every(k =>
      t[k as keyof RubricIntervalTarget] === w[k as keyof RubricWinnerTarget]);
  }), "Senate margin and principal-lead targets must name the same ordered pair, event, round, rule and denominator.");
  const winners = new Set(c.winnerCohort.map(t => t.raceId)), margins = new Set(c.intervalCohort.map(t => t.raceId));
  requireValue(c.intervalCohort.every(t => winners.has(t.raceId)) && Array.isArray(c.intervalExclusions) &&
    c.intervalExclusions.every(t => t && winners.has(t.raceId) && !margins.has(t.raceId) && text(t.reason)) &&
    new Set(c.intervalExclusions.map(t => t.raceId)).size === c.intervalExclusions.length &&
    c.intervalExclusions.length + c.intervalCohort.length === c.winnerCohort.length,
    "Every race outside the fixed margin cohort needs an explicit predeclared exclusion.");
  requireValue(Array.isArray(input.winnerRows) && Array.isArray(input.intervalRows), "Explicit paired loss arrays are required.");
}
function grade(skill: number): ElectionNightGrade {
  return ELECTION_NIGHT_RUBRIC.bands.find(b => b.minimumSkill === null || skill >= b.minimumSkill - ELECTION_NIGHT_RUBRIC.equalityTolerance)!.grade;
}
function worse(a: ElectionNightGrade, b: ElectionNightGrade): ElectionNightGrade {
  const order = ["A", "B", "C", "D", "F"]; return order.indexOf(a) >= order.indexOf(b) ? a : b;
}
function skill(model: number | null, baseline: number | null): number | null {
  if (model === null || baseline === null || baseline <= 0) return null;
  const value = 1 - model / baseline; return finite(value) ? value : null;
}
function assess(c: ElectionNightRubricContract, target: RubricTarget, rows: PairedRubricLossRecord[], winner: boolean): RubricPairedAssessment {
  const result: RubricPairedAssessment = { raceId: target.raceId, status: "pending", reason: null,
    modelLoss: null, baselineLoss: null, pairedLossDifference: null, coinFlipLoss: null, uniformThreeCategoryLoss: null };
  if (rows.length !== 1) return { ...result, status: rows.length ? "mismatched" : "pending",
    reason: rows.length ? "Multiple loss records/revisions; select the admitted revision explicitly." : "No paired result for this locked target." };
  const row = rows[0], m = row.model, b = row.baseline, factor = winner ? 0.5 : 1;
  result.modelLoss = finite(m?.loss) ? m.loss * factor : null; result.baselineLoss = finite(b?.loss) ? b.loss * factor : null;
  const metric = winner ? "multicategory_brier_sum" : "central90_interval_score_pp";
  if (row.comparisonContractSha256 !== c.sha256 || row.cohortDefinitionSha256 !== c.cohortDefinition.sha256 ||
    m?.forecastSha256 !== c.model.resultSha256 || b?.forecastSha256 !== c.baseline.resultSha256 ||
    !same(m?.target, target) || !same(b?.target, target) || m?.metric !== metric || b?.metric !== metric)
    return { ...result, status: "mismatched", reason: "Loss records do not bind the exact comparison, cohort, forecasts, metric and target." };
  if (m.status !== "scored" || b.status !== "scored") return { ...result,
    status: m.status === "mismatched" || b.status === "mismatched" ? "mismatched" : "pending",
    reason: m.reason || b.reason || `Both arms need a separately admitted result score (model ${m.status}; baseline ${b.status}).` };
  if (!m.result || !b.result || !same(m.result, b.result) || !text(m.result.recordId) || !text(m.result.revision) ||
    !hash(m.result.recordSha256) || !hash(m.result.sourceSha256) || m.result.resultPolicySha256 !== c.resultPolicy.sha256 ||
    !same(m.observedValue, b.observedValue))
    return { ...result, status: "mismatched", reason: "Model and baseline must use the same admitted result, revision, source and truth." };
  if (m.result.finality === "unresolved" || m.result.finality === "unsupported") return { ...result,
    modelLoss: null, baselineLoss: null, status: "pending", reason: `Result is ${m.result.finality}; no available observed score.` };
  if (!finite(m.loss) || !finite(b.loss) || m.loss < 0 || b.loss < 0 || (winner ? m.loss > 2 || b.loss > 2 : m.loss > 4000 || b.loss > 4000))
    return { ...result, status: "mismatched", reason: "Invalid raw loss; winner inputs must be full categorical Brier sums in [0,2]." };
  if (winner) {
    const t = target as RubricWinnerTarget;
    if (typeof m.observedValue !== "string" || !t.outcomeIds.includes(m.observedValue))
      return { ...result, status: "mismatched", reason: "Observed category lies outside the declared exhaustive scoring target." };
    const probabilities = c.model.chamber === "house" ? [0.5, 0.5, 0] : [0.5, 0.5];
    result.coinFlipLoss = t.outcomeIds.reduce((sum, id, i) => sum + (probabilities[i] - Number(id === m.observedValue)) ** 2, 0) / 2;
    if (c.model.chamber === "house") result.uniformThreeCategoryLoss = t.outcomeIds.reduce((sum, id) => sum + (1 / 3 - Number(id === m.observedValue)) ** 2, 0) / 2;
  } else if (!finite(m.observedValue) || Math.abs(m.observedValue) > 100)
    return { ...result, status: "mismatched", reason: "Observed named-pair margin must be on the declared percentage-point scale." };
  if (!["final", "certified", "called", "provisional", "unresolved", "unsupported"].includes(m.result.finality))
    return { ...result, status: "mismatched", reason: "Unknown result finality." };
  result.pairedLossDifference = result.modelLoss! - result.baselineLoss!;
  const final = m.result.finality === "final" || m.result.finality === "certified";
  return { ...result, status: final ? "final" : "provisional", reason: final ? null : "Result remains provisional; no complete-cohort letter." };
}
function component(c: ElectionNightRubricContract, targets: RubricTarget[], rows: PairedRubricLossRecord[], winner: boolean,
  issues: string[]): RubricComponent {
  const ids = new Set(targets.map(t => t.raceId)), unexpected = rows.filter(r => !ids.has(r.raceId)).length;
  const assessments = targets.map(t => assess(c, t, rows.filter(r => r.raceId === t.raceId), winner));
  const usable = assessments.filter(r => r.status === "final" || r.status === "provisional");
  const mean = (key: "modelLoss" | "baselineLoss" | "pairedLossDifference" | "coinFlipLoss" | "uniformThreeCategoryLoss") =>
    usable.length && usable.every(r => r[key] !== null) ? usable.reduce((sum, r) => sum + r[key]!, 0) / usable.length : null;
  const coverage = { total: targets.length, final: assessments.filter(r => r.status === "final").length,
    provisional: assessments.filter(r => r.status === "provisional").length, pending: assessments.filter(r => r.status === "pending").length,
    mismatched: assessments.filter(r => r.status === "mismatched").length, unexpected };
  const raw = { scope: "available_matched_rows" as const, count: usable.length, finalCount: coverage.final, provisionalCount: coverage.provisional,
    modelMeanLoss: mean("modelLoss"), baselineMeanLoss: mean("baselineLoss"), meanPairedLossDifference: mean("pairedLossDifference"),
    coinFlipMeanLoss: mean("coinFlipLoss"), uniformThreeCategoryMeanLoss: mean("uniformThreeCategoryLoss") };
  const s = skill(raw.modelMeanLoss, raw.baselineMeanLoss), cs = skill(raw.modelMeanLoss, raw.coinFlipMeanLoss);
  const complete = targets.length > 0 && coverage.final === targets.length && unexpected === 0 && issues.length === 0;
  const historyGrade = complete && s !== null ? grade(s) : null;
  return { complete, grade: historyGrade, historyGrade, skill: s, coinFlipSkill: cs,
    uniformThreeCategorySkill: skill(raw.modelMeanLoss, raw.uniformThreeCategoryMeanLoss),
    coinFlipCapApplied: false, // The editorial cap applies to overall/winner-only letters, not component skill bands.
    reason: issues.length ? issues.join(" ") : !targets.length ? "No predeclared margin cohort; winner-only reporting remains separate." :
      !complete ? "The complete locked cohort requires final/certified matched results with no unexpected rows." :
      raw.baselineMeanLoss === 0 ? "History baseline loss is zero; percentage skill and this component letter are undefined." :
      s === null ? "The skill ratio is not finite/defined; raw paired differences remain primary." : null,
    coverage, raw, rows: assessments };
}
/** Generic scalar paired losses are valid only after their metric/event/result bindings were admitted by the caller. */
export function gradeElectionNightRubric(input: ElectionNightRubricInput): ElectionNightRubricResult {
  const issues: string[] = [];
  try { validate(input); } catch (error) { issues.push(error instanceof Error ? error.message : "Invalid rubric input."); }
  const c = input?.contract;
  const safe = c && binding(c.model) && binding(c.baseline) && reference(c.cohortDefinition) && reference(c.resultPolicy) &&
    Array.isArray(c.winnerCohort) && c.winnerCohort.every(t => winnerTarget(t, c.model.chamber)) && Array.isArray(c.intervalCohort) && c.intervalCohort.every(intervalTarget) &&
    Array.isArray(c.intervalExclusions) && c.intervalExclusions.every(t => t && text(t.raceId) && text(t.reason)) &&
    Array.isArray(input.winnerRows) && input.winnerRows.every(r => r && text(r.raceId)) &&
    Array.isArray(input.intervalRows) && input.intervalRows.every(r => r && text(r.raceId));
  // A malformed outer contract has no admissible scoring cohort to assess.
  if (!safe) {
    const emptyContract = { model: { chamber: "senate" } } as ElectionNightRubricContract;
    const empty = component(emptyContract, [], [], false, issues.length ? issues : ["Invalid rubric input."]);
    return { policyId: ELECTION_NIGHT_RUBRIC.id, policyDecisionSha256: ELECTION_NIGHT_RUBRIC.decisionSha256,
      status: "not_graded", comparison: null, issues, winner: structuredClone(empty), interval: empty,
      overall: { grade: null, reason: "No valid comparison contract.", coinFlipCapApplied: false }, winnerOnlyGrade: null, intervalExcluded: [], unexpectedRows: [] };
  }
  const winner = component(c, c.winnerCohort, input.winnerRows, true, issues), interval = component(c, c.intervalCohort, input.intervalRows, false, issues);
  const unexpectedRows: ElectionNightRubricResult["unexpectedRows"] = [];
  for (const [name, targets, rows] of [["winner", c.winnerCohort, input.winnerRows], ["interval", c.intervalCohort, input.intervalRows]] as const)
    for (const row of rows) if (!targets.some(t => t.raceId === row.raceId)) unexpectedRows.push({ component: name, raceId: row.raceId, reason: "Record is outside the fixed target cohort." });
  const componentFloor = winner.grade && interval.grade ? worse(winner.grade, interval.grade) : null;
  const cap = winner.complete && winner.coinFlipSkill !== null && winner.coinFlipSkill <= ELECTION_NIGHT_RUBRIC.equalityTolerance;
  const overall = componentFloor && cap ? worse(componentFloor, "D") : componentFloor;
  const winnerOnly = c.intervalCohort.length === 0 && winner.grade ? (cap ? worse(winner.grade, "D") : winner.grade) : null;
  return { policyId: ELECTION_NIGHT_RUBRIC.id, policyDecisionSha256: ELECTION_NIGHT_RUBRIC.decisionSha256,
    status: overall ? "graded" : "not_graded", comparison: { id: c.id, sha256: c.sha256, label: c.label, mode: c.comparisonMode,
      matchedHorizon: c.comparisonMode === "matched_horizon", explanation: c.comparisonMode === "explicit_earlier_benchmark" ? earlierBenchmark.publicExplanation : "Same information cutoff and horizon; scores concern the exact declared targets and results." },
    issues, winner, interval, overall: { grade: overall, reason: overall ? null : "Both complete, defined component letters are required for an overall letter.",
      coinFlipCapApplied: overall !== null && overall !== componentFloor },
    winnerOnlyGrade: winnerOnly, intervalExcluded: c.intervalExclusions.map(t => ({ ...t })), unexpectedRows };
}
