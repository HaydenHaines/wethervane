/** Private presentation study, not an owner export schema, artifact parser or API.
 * Every accepted draft is invented. These checks reconcile declarations; they do
 * not authenticate a real source, posterior, endpoint crosswalk or forecast.
 */
export type PairComparisonArm = "count" | "F" | "FP";
export const pairComparisonArms: readonly PairComparisonArm[] = ["count", "F", "FP"];
export const pairComparisonArmLabels: Record<PairComparisonArm, string> = {
  count: "Experimental vote-count model", F: "Earlier model without race polls (F)", FP: "Earlier model with race polls (FP)",
};
export interface PairEndpoint {
  personId: string;
  name: string;
  sourceCandidateId: string;
  modelCandidateId: string;
  identitySha256: string;
}
export interface NamedPairIdentity {
  id: string;
  raceId: string;
  contestId: string;
  eventId: string;
  electionDate: string;
  first: PairEndpoint;
  second: PairEndpoint;
}
export interface PairArmBinding {
  comparisonContextSha256: string;
  pairSpecificationSha256: string;
  sourceId: string;
  artifactSha256: string;
  pairId: string;
  raceId: string;
  contestId: string;
  eventId: string;
  electionDate: string;
  first: Omit<PairEndpoint, "name">;
  second: Omit<PairEndpoint, "name">;
  estimand: "candidate_marks";
  denominator: { kind: "named_pair_counts"; firstPersonId: string; secondPersonId: string };
}
export interface PairDistributionSummary {
  mean: number;
  median: number;
  intervals: { level: number; lower: number; upper: number }[];
}
export interface PairArmView {
  binding: PairArmBinding;
  distribution:
    | { status: "available"; undefinedMass: 0; summary: PairDistributionSummary }
    | { status: "undefined_share"; undefinedMass: number; summary: null; reason: string }
    | { status: "unavailable"; undefinedMass: null; summary: null; reason: string };
  evaluation:
    | { status: "scored"; score: { crps: number; weightedIntervalScore: number; absoluteMeanError: number } }
    | { status: "unavailable"; score: null; reason: string };
  precision: { status: "unavailable"; reason: string };
}
export interface CandidatePairComparisonDraft {
  kind: "candidate_pair_comparison_preparation";
  provenance: { kind: "invented_fixture"; label: string };
  label: string;
  binding: {
    requestSha256: string;
    comparisonContextSha256: string;
    pairSpecificationSha256: string;
    cycle: number;
    informationCutoff: string;
    horizonDays: 14;
    sources: Record<PairComparisonArm, { id: string; artifactSha256: string }>;
  };
  qualification: {
    experimental: true;
    posterior: "not_established";
    adoption: "not_qualified";
    informationSets: "different";
    explanation: string;
  };
  /** Fixed expected roster, including pairs with no supplied results. */
  requiredPairs: NamedPairIdentity[];
  rows: {
    pairId: string;
    observedFirstPairShare: number | null;
    arms: Record<PairComparisonArm, PairArmView>;
  }[];
}
export type CandidatePairComparisonReadiness =
  | { productionActivated: false; valid: false; display: false; problems: string[] }
  | {
      productionActivated: false; valid: true; display: true; problems: [];
      view: CandidatePairComparisonDraft;
      coverage: {
        expectedPairs: number; suppliedPairs: number; fullyScoredPairs: number;
        scoredByArm: Record<PairComparisonArm, number>; missingPairIds: string[];
      };
    };

const record = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);
const text = (v: unknown): v is string => typeof v === "string" && v.trim().length > 0;
const hash = (v: unknown): v is string => typeof v === "string" && /^[a-f0-9]{64}$/.test(v);
const probability = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v) && v >= 0 && v <= 1;
const nonnegative = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v) && v >= 0;
const day = (v: unknown): v is string => typeof v === "string" && /^\d{4}-\d{2}-\d{2}$/.test(v)
  && Number.isFinite(Date.parse(`${v}T00:00:00Z`)) && new Date(`${v}T00:00:00Z`).toISOString().slice(0, 10) === v;
const endpoint = (v: unknown): v is PairEndpoint => record(v)
  && [v.personId, v.name, v.sourceCandidateId, v.modelCandidateId].every(text) && hash(v.identitySha256);
const endpointMatches = (v: unknown, expected: PairEndpoint) => record(v)
  && ["personId", "sourceCandidateId", "modelCandidateId", "identitySha256"].every(key => v[key] === expected[key as keyof PairEndpoint]);
const exactArms = (v: unknown): v is Record<PairComparisonArm, unknown> => record(v)
  && Object.keys(v).length === 3 && pairComparisonArms.every(arm => Object.hasOwn(v, arm));

/** Validate invented supplied summaries. No scoring, pooling or missing-mass renormalization. */
export function candidatePairComparisonReadiness(input: unknown): CandidatePairComparisonReadiness {
  const fail = (...messages: string[]): CandidatePairComparisonReadiness => ({
    productionActivated: false, valid: false, display: false, problems: messages,
  });
  if (!record(input) || input.kind !== "candidate_pair_comparison_preparation" || !record(input.provenance)
      || input.provenance.kind !== "invented_fixture" || !text(input.provenance.label) || !text(input.label))
    return fail("Explicit invented-fixture provenance is required; this is not a real result adapter.");
  const b = input.binding, q = input.qualification;
  if (!record(b) || ![b.requestSha256, b.comparisonContextSha256, b.pairSpecificationSha256].every(hash)
      || !Number.isInteger(b.cycle) || Number(b.cycle) < 1900 || Number(b.cycle) > 2200
      || !text(b.informationCutoff) || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$/.test(b.informationCutoff)
      || !Number.isFinite(Date.parse(b.informationCutoff)) || b.horizonDays !== 14
      || !exactArms(b.sources)) return fail("The source records or the date 14 days before the example election are incomplete.");
  for (const arm of pairComparisonArms) {
    const source = b.sources[arm];
    if (!record(source) || !text(source.id) || !hash(source.artifactSha256))
      return fail("Each model needs a named source and an exact saved-result record.");
  }
  if (!record(q) || q.experimental !== true || q.posterior !== "not_established" || q.adoption !== "not_qualified"
      || q.informationSets !== "different" || !text(q.explanation))
    return fail("The example must remain experimental, unapproved for public use and clear about different input data.");
  if (!Array.isArray(input.requiredPairs) || input.requiredPairs.length === 0 || !Array.isArray(input.rows))
    return fail("A nonempty fixed pair roster and explicit result rows are required.");
  const pairs = new Map<string, NamedPairIdentity>();
  const races = new Set<string>(), contests = new Set<string>();
  for (const pair of input.requiredPairs) {
    if (!record(pair) || ![pair.id, pair.raceId, pair.contestId, pair.eventId].every(text) || !day(pair.electionDate)
        || !endpoint(pair.first) || !endpoint(pair.second)) return fail("Each expected pair needs an election date, event and two named candidates.");
    if (pair.first.personId === pair.second.personId || pair.first.sourceCandidateId === pair.second.sourceCandidateId
        || pair.first.modelCandidateId === pair.second.modelCandidateId)
      return fail("A pair must contain two distinct named endpoints.");
    if (pairs.has(String(pair.id)) || races.has(String(pair.raceId)) || contests.has(String(pair.contestId)))
      return fail("The fixed roster cannot repeat a pair, race or contest.");
    if (Number(pair.electionDate.slice(0, 4)) !== b.cycle || Date.parse(`${pair.electionDate}T00:00:00Z`) - Date.parse(String(b.informationCutoff)) !== 14 * 86400000)
      return fail("Each example election date must be exactly 14 days after the declared input cutoff.");
    pairs.set(String(pair.id), pair as unknown as NamedPairIdentity);
    races.add(String(pair.raceId)); contests.add(String(pair.contestId));
  }
  const supplied = new Set<string>();
  const scoredByArm = { count: 0, F: 0, FP: 0 };
  let fullyScoredPairs = 0;
  for (const row of input.rows) {
    if (!record(row) || !text(row.pairId) || !pairs.has(row.pairId) || supplied.has(row.pairId) || !exactArms(row.arms))
      return fail("Result rows must identify each expected pair at most once and retain all three model slots.");
    if (row.observedFirstPairShare !== null && !probability(row.observedFirstPairShare))
      return fail("An observed pair share must be in [0,1] or explicitly unavailable.");
    supplied.add(row.pairId);
    const pair = pairs.get(row.pairId)!;
    let allScored = true;
    for (const arm of pairComparisonArms) {
      const view = row.arms[arm], source = b.sources[arm] as Record<string, unknown>;
      if (!record(view) || !record(view.binding)) return fail("Every model slot needs a bound result or an explicit unavailable state.");
      const binding = view.binding;
      if (binding.comparisonContextSha256 !== b.comparisonContextSha256 || binding.pairSpecificationSha256 !== b.pairSpecificationSha256
          || binding.sourceId !== source.id || binding.artifactSha256 !== source.artifactSha256
          || binding.pairId !== pair.id || binding.raceId !== pair.raceId || binding.contestId !== pair.contestId
          || binding.eventId !== pair.eventId || binding.electionDate !== pair.electionDate
          || !endpointMatches(binding.first, pair.first) || !endpointMatches(binding.second, pair.second))
        return fail("The model summaries do not refer to the same ordered candidates, source records, race, event and date.");
      if (binding.estimand !== "candidate_marks" || !record(binding.denominator)
          || binding.denominator.kind !== "named_pair_counts" || binding.denominator.firstPersonId !== pair.first.personId
          || binding.denominator.secondPersonId !== pair.second.personId)
        return fail("Shares must use the same two named candidates in the same order; shares among all candidates cannot substitute.");
      const distribution = view.distribution, evaluation = view.evaluation;
      if (!record(distribution) || !record(evaluation) || !record(view.precision)
          || view.precision.status !== "unavailable" || !text(view.precision.reason))
        return fail("Predictions, errors and missing calculation-precision information must stay separate.");
      if (distribution.status === "available") {
        const summary = distribution.summary;
        if (distribution.undefinedMass !== 0 || !record(summary) || !probability(summary.mean) || !probability(summary.median)
            || !Array.isArray(summary.intervals) || summary.intervals.length === 0)
          return fail("A displayed share needs valid summary values and no outcomes in which that share cannot be calculated.");
        const levels = new Set<number>();
        for (const interval of summary.intervals) {
          if (!record(interval) || !probability(interval.level) || interval.level === 0 || interval.level === 1
              || levels.has(interval.level) || !probability(interval.lower) || !probability(interval.upper) || interval.lower > interval.upper)
            return fail("Predictive intervals need distinct stated levels and ordered bounded shares.");
          levels.add(interval.level);
        }
      } else if (distribution.status === "undefined_share") {
        if (!probability(distribution.undefinedMass) || distribution.undefinedMass === 0 || distribution.summary !== null || !text(distribution.reason))
          return fail("The chance of an incalculable share must remain visible; no replacement summary can be supplied.");
      } else if (distribution.status === "unavailable") {
        if (distribution.undefinedMass !== null || distribution.summary !== null || !text(distribution.reason))
          return fail("A missing result must remain unknown, not a zero share or a claim that every share could be calculated.");
      } else return fail("Unsupported predictive state.");
      if (evaluation.status === "scored") {
        if (distribution.status !== "available" || row.observedFirstPairShare === null || !record(evaluation.score)
            || ![evaluation.score.crps, evaluation.score.weightedIntervalScore, evaluation.score.absoluteMeanError].every(nonnegative))
          return fail("Scores require a defined prediction, observed share and finite nonnegative supplied losses.");
        const summary = distribution.summary as Record<string, unknown>;
        if (Math.abs(Number(evaluation.score.absoluteMeanError) - Math.abs(Number(summary.mean) - row.observedFirstPairShare)) > 1e-12)
          return fail("The supplied absolute error must agree with the displayed mean and observed pair share.");
        scoredByArm[arm] += 1;
      } else if (evaluation.status !== "unavailable" || evaluation.score !== null || !text(evaluation.reason))
        return fail("Missing scores require null values and a reason; missing is not zero loss.");
      else allScored = false;
    }
    if (allScored) fullyScoredPairs += 1;
  }
  return { productionActivated: false, valid: true, display: true, problems: [],
    view: input as unknown as CandidatePairComparisonDraft,
    coverage: { expectedPairs: pairs.size, suppliedPairs: supplied.size, fullyScoredPairs, scoredByArm,
      missingPairIds: [...pairs.keys()].filter(id => !supplied.has(id)) } };
}
