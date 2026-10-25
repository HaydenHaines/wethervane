/** Private, invented data contract for frontend preparation only. This is not an owner DTO.
 * Values are supplied; this module neither fits forecasts nor calculates scoring rules.
 */
export type Availability = "scored" | "outcome_missing" | "undefined_mass" | "unavailable" | "not_requested";
export type MetricId = "pair_crps" | "pair_wis" | "leading_brier" | "leading_log" | "vector_energy" | "report_crps" | "report_wis" | "report_log";
export type MetricUnit = "share_fraction" | "dimensionless" | "report_count";
export type Precision = { status: "adequate" | "limited" | "not_reported"; note: string; rule?: string };
export type MetricBinding = {
  target: string; orderedRosterIds: string[]; cutoff: string; outcomeView: string;
  metricDefinition: string; weighting: string; unit: MetricUnit; reportProfile?: string;
};
export type SuppliedMetric = {
  state: "complete" | "partial" | "unavailable" | "not_requested";
  value: number | null; infinite: boolean; scoredIds: string[]; binding: MetricBinding;
  reason?: string; precision: Precision;
};
export type PairIdentity = {
  id: string; raceId: string; eventId: string; contestId: string; electionDate: string;
  first: { id: string; name: string }; second: { id: string; name: string };
};
export type PairRow = {
  state: Availability; undefinedMass: number | null; observedShare: number | null;
  crps: number | null; wis: number | null;
  intervals: { level: number; lower: number; upper: number; width: number; covered: boolean | null }[];
  leading: null | { probability: number; tieProbability: number; observed: boolean | null;
    brier: number | null; logLoss: number | null; logLossInfinite: boolean };
  shape?: { near50: boolean; probMarginWithin2pp: number; probMarginAtLeast10pp: number;
    expectedSignedMarginPp: number; medianSignedMarginPp: number };
  reason?: string;
};
export type ReportIdentity = {
  id: string; eventId: string; date: string; name: string; releaseId: string;
  profile: string; definition: string; sourceQualityFlags: string[]; sourceVariance: number | null;
  effectProvenance: { release: string; state: string };
};
export type ReportRow = {
  state: Availability | "inactive_event"; eventPresent: boolean; undefinedMass: number | null;
  observedCount: number | null; predictedMean: number | null; crps: number | null; wis: number | null;
  intervals: { level: number; lower: number; upper: number; width: number; covered: boolean | null }[];
  logLoss: number | null; logLossInfinite: boolean; reason?: string;
};
export type EnergyRow = {
  state: "scored" | "withheld" | "not_requested"; value: number | null;
  undefinedMass: number | null; missingObservedIds: string[]; positiveWeightMissingIds: string[];
  orderedPairIds: string[]; rows: number; weightSum: number | null; reason?: string;
};
export type ModelCycle = {
  pairs: Record<string, PairRow>; reports: Record<string, ReportRow>; energy: EnergyRow;
  metrics: Partial<Record<MetricId, SuppliedMetric>>;
};
export type EvaluationCycle = {
  id: string; label: string; cutoff: string; outcomeView: string;
  pairRoster: PairIdentity[]; reportRoster: ReportIdentity[];
  models: Record<string, ModelCycle>;
};
export type EvaluationScoreboardDraftV1 = {
  schema: "PrivateEvaluationScoreboardDraftV1";
  provenance: { kind: "invented_fixture"; label: string; declaredSourceHashes: string[] };
  label: string;
  status: { run: "preparing" | "running" | "complete" | "not_reported";
    evidence: "experimental" | "not_assessed";
    adoption: "not_qualified" | "not_assessed"; note: string };
  models: { id: string; name: string; ingredients: string[]; informationKey: string }[];
  cycles: EvaluationCycle[];
};

export const metricMeta: Record<MetricId, { label: string; unit: MetricUnit; explanation: string }> = {
  pair_crps: { label: "Pair-share full-distribution error (CRPS)", unit: "share_fraction", explanation: "Compares the whole predicted named-pair share distribution with the recorded share. Lower is better." },
  pair_wis: { label: "Pair-share range error (WIS)", unit: "share_fraction", explanation: "Balances narrow prediction ranges against ranges that miss the recorded share. Lower is better." },
  leading_brier: { label: "Pair-leading probability error (Brier)", unit: "dimensionless", explanation: "Tests the chance that the first named candidate gets more of these two candidates’ votes. Ties are separate; this is not legal winner probability. Lower is better." },
  leading_log: { label: "Pair-leading natural-log loss", unit: "dimensionless", explanation: "Penalizes probability assigned away from the recorded strictly-leading event. Lower is better." },
  vector_energy: { label: "Whole-vector energy score", unit: "share_fraction", explanation: "Scores all ordered named-pair shares together using Euclidean distance divided by the square root of the number of pairs. Lower is better; it cannot identify an individual pair’s contribution." },
  report_crps: { label: "Administrative report-total error (CRPS)", unit: "report_count", explanation: "Compares the predicted count distribution with recorded administrative report totals. Lower is better." },
  report_wis: { label: "Administrative report range error (WIS)", unit: "report_count", explanation: "Balances report-count range width with misses. Lower is better." },
  report_log: { label: "Administrative report natural-log loss", unit: "dimensionless", explanation: "Scores the recorded administrative total under the predicted distribution. Lower is better." },
};

const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const probability = (v: unknown) => finite(v) && v >= 0 && v <= 1;
const nonnegative = (v: unknown) => finite(v) && v >= 0;
const unique = (xs: string[]) => new Set(xs).size === xs.length;
const sameSet = (a: string[], b: string[]) => a.length === b.length && a.every(x => b.includes(x));
const sameOrder = (a: string[], b: string[]) => a.length === b.length && a.every((x, i) => x === b[i]);
const text = (x: unknown): x is string => typeof x === "string" && x.trim().length > 0;
const knownMetricIds = Object.keys(metricMeta) as MetricId[];
const expectedIds = (cycle: EvaluationCycle, metric: MetricId) => metric.startsWith("report_")
  ? cycle.reportRoster.map(x => x.id) : metric === "vector_energy" ? ["whole_vector"] : cycle.pairRoster.map(x => x.id);
const scoredRowIds = (cycle: EvaluationCycle, result: ModelCycle, metric: MetricId) => metric === "vector_energy"
  ? result.energy.state === "scored" ? ["whole_vector"] : []
  : expectedIds(cycle, metric).filter(id => {
    if (metric.startsWith("report_")) {
      const row = result.reports[id];
      return row?.state === "scored" && (metric === "report_crps" ? nonnegative(row.crps)
        : metric === "report_wis" ? nonnegative(row.wis) : nonnegative(row.logLoss) || row.logLossInfinite);
    }
    const row = result.pairs[id];
    return row?.state === "scored" && (metric === "pair_crps" ? nonnegative(row.crps)
      : metric === "pair_wis" ? nonnegative(row.wis)
        : metric === "leading_brier" ? nonnegative(row.leading?.brier)
          : nonnegative(row.leading?.logLoss) || row.leading?.logLossInfinite === true);
  });

export type ScoreboardReadiness = { valid: false; problems: string[] } | { valid: true; view: EvaluationScoreboardDraftV1 };
/** Validate display consistency only. Declared hashes do not authenticate underlying files. */
export function evaluationScoreboardReadiness(input: unknown): ScoreboardReadiness {
  const problems: string[] = [];
  const fail = (message: string) => { problems.push(message); };
  if (!input || typeof input !== "object") return { valid: false, problems: ["No invented evaluation draft was supplied."] };
  const d = input as EvaluationScoreboardDraftV1;
  if (d.schema !== "PrivateEvaluationScoreboardDraftV1" || d.provenance?.kind !== "invented_fixture"
    || !text(d.provenance.label) || !Array.isArray(d.provenance.declaredSourceHashes) || !text(d.label))
    return { valid: false, problems: ["Explicit invented-fixture provenance is required."] };
  if (!d.status || !Array.isArray(d.models) || !Array.isArray(d.cycles)
    || d.models.some(m => !m || typeof m !== "object") || d.cycles.some(c => !c || typeof c !== "object")
    || !unique(d.models.map(m => m.id))
    || !unique(d.cycles.map(c => c.id))) return { valid: false, problems: ["The model and cycle roster is invalid."] };
  if (!["preparing", "running", "complete", "not_reported"].includes(d.status.run)
    || !["experimental", "not_assessed"].includes(d.status.evidence)
    || !["not_qualified", "not_assessed"].includes(d.status.adoption)
    || !text(d.status.note)) fail("Status dimensions are incomplete.");
  for (const model of d.models) if (!text(model.id) || !text(model.name) || !text(model.informationKey)
    || !Array.isArray(model.ingredients) || model.ingredients.some(x => !text(x))) fail("A model stack is incomplete.");
  for (const cycle of d.cycles) {
    if (!text(cycle.id) || !text(cycle.label) || !text(cycle.cutoff) || !text(cycle.outcomeView)
      || !Array.isArray(cycle.pairRoster) || !Array.isArray(cycle.reportRoster) || !cycle.models
      || cycle.pairRoster.some(p => !p || typeof p !== "object") || cycle.reportRoster.some(r => !r || typeof r !== "object")
      || !unique(cycle.pairRoster.map(p => p.id)) || !unique(cycle.reportRoster.map(r => r.id))) { fail("A cycle roster is incomplete."); continue; }
    for (const report of cycle.reportRoster) if (!text(report.profile) || !text(report.definition)
      || !text(report.releaseId) || !Array.isArray(report.sourceQualityFlags)
      || !(report.sourceVariance === null || nonnegative(report.sourceVariance))
      || !text(report.effectProvenance?.release) || !text(report.effectProvenance?.state)) fail(`${cycle.id}: report source metadata is incomplete.`);
    for (const model of d.models) {
      const r = cycle.models[model.id];
      if (!r) continue;
      if (!r.pairs || !r.reports || !r.energy || !r.metrics) { fail(`${cycle.id}/${model.id}: model result is incomplete.`); continue; }
      for (const [id, p] of Object.entries(r.pairs)) {
        if (!cycle.pairRoster.some(x => x.id === id)) { fail(`${cycle.id}: pair outside expected roster.`); continue; }
        if (!p || !["scored", "outcome_missing", "undefined_mass", "unavailable", "not_requested"].includes(p.state)) { fail(`${cycle.id}/${id}: invalid pair state.`); continue; }
        if (!Array.isArray(p.intervals)) { fail(`${cycle.id}/${id}: pair ranges missing.`); continue; }
        if (p.state === "undefined_mass" && !(p.undefinedMass !== null && probability(p.undefinedMass) && p.undefinedMass > 0)) fail(`${cycle.id}/${id}: positive undefined mass required.`);
        if (p.state === "scored" && (!(probability(p.observedShare) && p.undefinedMass === 0 && nonnegative(p.crps) && nonnegative(p.wis))
          || !p.leading || !probability(p.leading.probability) || !probability(p.leading.tieProbability)
          || p.leading.probability + p.leading.tieProbability > 1 + 1e-12 || typeof p.leading.observed !== "boolean"
          || !nonnegative(p.leading.brier) || !(nonnegative(p.leading.logLoss) && !p.leading.logLossInfinite
            || p.leading.logLoss === null && p.leading.logLossInfinite))) fail(`${cycle.id}/${id}: invalid scored pair.`);
        if (p.state === "scored" && p.leading && p.observedShare !== null && probability(p.observedShare)
          && p.leading.observed !== (p.observedShare > .5)) fail(`${cycle.id}/${id}: strictly-leading observation disagrees with pair share.`);
        if (p.state !== "scored" && (p.crps !== null || p.wis !== null)) fail(`${cycle.id}/${id}: unavailable pair has a score.`);
        if (p.state === "outcome_missing" && (p.observedShare !== null || p.undefinedMass !== 0
          || p.leading && (p.leading.observed !== null || p.leading.brier !== null || p.leading.logLoss !== null || p.leading.logLossInfinite)))
          fail(`${cycle.id}/${id}: missing outcome has an observed score.`);
        if (p.state === "outcome_missing" && (p.intervals.some(v => v?.covered !== null)
          || p.leading && (!probability(p.leading.probability) || !probability(p.leading.tieProbability)
            || p.leading.probability + p.leading.tieProbability > 1 + 1e-12)))
          fail(`${cycle.id}/${id}: missing outcome forecast is inconsistent.`);
        if (p.state === "undefined_mass" && (p.leading !== null || p.shape || p.intervals.length))
          fail(`${cycle.id}/${id}: undefined predictive mass must withhold summary and shape.`);
        if ((p.state === "unavailable" || p.state === "not_requested") && (p.leading !== null || p.shape || p.intervals.length))
          fail(`${cycle.id}/${id}: unavailable pair has predictive values.`);
        if (!Array.isArray(p.intervals) || p.intervals.some(v => !v || !probability(v.level) || !probability(v.lower)
          || !probability(v.upper) || v.lower > v.upper || !nonnegative(v.width)
          || Math.abs(v.width - (v.upper - v.lower)) > 1e-9 || !(v.covered === null || typeof v.covered === "boolean"))) fail(`${cycle.id}/${id}: invalid pair interval.`);
        if (p.state === "scored" && probability(p.observedShare)
          && p.intervals.some(v => v && v.covered !== (v.lower <= p.observedShare! && p.observedShare! <= v.upper)))
          fail(`${cycle.id}/${id}: pair range coverage disagrees with recorded share.`);
        if (p.shape && (!probability(p.shape.probMarginWithin2pp) || !probability(p.shape.probMarginAtLeast10pp)
          || !finite(p.shape.expectedSignedMarginPp) || !finite(p.shape.medianSignedMarginPp))) fail(`${cycle.id}/${id}: invalid margin shape.`);
      }
      for (const [id, q] of Object.entries(r.reports)) {
        if (!cycle.reportRoster.some(x => x.id === id)) { fail(`${cycle.id}: report outside expected roster.`); continue; }
        if (!q || !["scored", "outcome_missing", "undefined_mass", "unavailable", "not_requested", "inactive_event"].includes(q.state)) { fail(`${cycle.id}/${id}: invalid report state.`); continue; }
        if (q.state === "inactive_event" && (q.eventPresent || q.observedCount !== null || q.predictedMean !== null)) fail(`${cycle.id}/${id}: inactive event has a count.`);
        if (!Array.isArray(q.intervals)) { fail(`${cycle.id}/${id}: report ranges missing.`); continue; }
        if (q.state === "undefined_mass" && !(q.undefinedMass !== null && probability(q.undefinedMass) && q.undefinedMass > 0)) fail(`${cycle.id}/${id}: positive report undefined mass required.`);
        if (q.state === "scored" && (!q.eventPresent || q.undefinedMass !== 0 || !nonnegative(q.observedCount) || !nonnegative(q.predictedMean)
          || !nonnegative(q.crps) || !nonnegative(q.wis) || !(nonnegative(q.logLoss) && !q.logLossInfinite
            || q.logLoss === null && q.logLossInfinite))) fail(`${cycle.id}/${id}: invalid scored report.`);
        if (q.state !== "scored" && (q.crps !== null || q.wis !== null)) fail(`${cycle.id}/${id}: unavailable report has a score.`);
        if (q.state === "outcome_missing" && (!q.eventPresent || q.observedCount !== null || q.undefinedMass !== 0
          || !nonnegative(q.predictedMean) || q.logLoss !== null || q.logLossInfinite
          || q.intervals.some(v => v?.covered !== null)))
          fail(`${cycle.id}/${id}: missing report outcome has a score.`);
        if (q.state === "undefined_mass" && (q.predictedMean !== null || q.intervals.length || q.logLoss !== null || q.logLossInfinite))
          fail(`${cycle.id}/${id}: undefined report mass must withhold summary and score.`);
        if ((q.state === "inactive_event" || q.state === "unavailable" || q.state === "not_requested")
          && (q.predictedMean !== null || q.intervals.length || q.logLoss !== null || q.logLossInfinite))
          fail(`${cycle.id}/${id}: unavailable report has predictive values.`);
        if (!Array.isArray(q.intervals) || q.intervals.some(v => !v || !probability(v.level) || !nonnegative(v.lower)
          || !nonnegative(v.upper) || v.lower > v.upper || !nonnegative(v.width)
          || Math.abs(v.width - (v.upper - v.lower)) > 1e-6 || !(v.covered === null || typeof v.covered === "boolean"))) fail(`${cycle.id}/${id}: invalid report interval.`);
        if (q.state === "scored" && nonnegative(q.observedCount)
          && q.intervals.some(v => v && v.covered !== (v.lower <= q.observedCount! && q.observedCount! <= v.upper)))
          fail(`${cycle.id}/${id}: report range coverage disagrees with recorded total.`);
      }
      const e = r.energy;
      if (!Array.isArray(e.orderedPairIds) || !Array.isArray(e.missingObservedIds) || !Array.isArray(e.positiveWeightMissingIds)) {
        fail(`${cycle.id}: vector roster is incomplete.`); continue;
      }
      if (!["scored", "withheld", "not_requested"].includes(e.state)
        || !sameOrder(e.orderedPairIds, cycle.pairRoster.map(x => x.id))
        || !Number.isInteger(e.rows) || e.rows < 0) fail(`${cycle.id}: vector roster is incomplete.`);
      if (e.state === "scored" && (!nonnegative(e.value) || e.undefinedMass !== 0 || e.missingObservedIds.length
        || e.positiveWeightMissingIds.length || !nonnegative(e.weightSum) || e.weightSum === null || e.weightSum <= 0
        || cycle.pairRoster.some(pair => r.pairs[pair.id]?.state !== "scored"))) fail(`${cycle.id}: incomplete vector cannot be scored.`);
      if (e.state !== "scored" && e.value !== null) fail(`${cycle.id}: withheld vector has a score.`);
      if (e.undefinedMass !== null && !probability(e.undefinedMass)) fail(`${cycle.id}: invalid vector undefined mass.`);
      for (const [key, metric] of Object.entries(r.metrics) as [MetricId, SuppliedMetric][]) {
        if (!knownMetricIds.includes(key) || !metric?.binding || !metric.precision) { fail(`${cycle.id}: unknown metric.`); continue; }
        if (!["complete", "partial", "unavailable", "not_requested"].includes(metric.state)
          || !Array.isArray(metric.scoredIds) || !Array.isArray(metric.binding.orderedRosterIds)) {
          fail(`${cycle.id}/${key}: metric state or roster is incomplete.`); continue;
        }
        const expected = expectedIds(cycle, key), eligible = scoredRowIds(cycle, r, key), b = metric.binding;
        if (b.unit !== metricMeta[key].unit || b.cutoff !== cycle.cutoff || b.outcomeView !== cycle.outcomeView
          || !Array.isArray(b.orderedRosterIds) || !sameOrder(b.orderedRosterIds, expected) || !text(b.target) || !text(b.metricDefinition) || !text(b.weighting)
          || key.startsWith("report_") && (!text(b.reportProfile) || cycle.reportRoster.some(x => x.profile !== b.reportProfile)))
          fail(`${cycle.id}/${key}: units or scope binding mismatch.`);
        if (!["adequate", "limited", "not_reported"].includes(metric.precision.status) || !text(metric.precision.note))
          fail(`${cycle.id}/${key}: precision status is incomplete.`);
        if (!unique(metric.scoredIds) || metric.scoredIds.some(id => !eligible.includes(id))) fail(`${cycle.id}/${key}: scored IDs exceed usable rows.`);
        if (metric.state === "complete" && (expected.length === 0 || !sameSet(metric.scoredIds, expected))) fail(`${cycle.id}/${key}: incomplete roster cannot claim a complete score.`);
        if (metric.state === "partial" && (sameSet(metric.scoredIds, expected) || metric.scoredIds.length === 0)) fail(`${cycle.id}/${key}: invalid partial coverage.`);
        if (metric.state === "complete" || metric.state === "partial") {
          if (!(nonnegative(metric.value) && !metric.infinite || metric.value === null && metric.infinite)) fail(`${cycle.id}/${key}: finite or infinite loss required.`);
          if (metric.infinite && key !== "leading_log" && key !== "report_log") fail(`${cycle.id}/${key}: only log loss can be infinite.`);
          if (key === "vector_energy" && metric.value !== e.value) fail(`${cycle.id}/${key}: vector summary disagrees with supplied whole-vector score.`);
        } else if (metric.value !== null || metric.infinite || metric.scoredIds.length) fail(`${cycle.id}/${key}: unavailable metric has a value.`);
      }
    }
  }
  return problems.length ? { valid: false, problems } : { valid: true, view: d };
}

export function coverageFor(cycle: EvaluationCycle, modelId: string) {
  const r = cycle.models[modelId];
  return {
    pairs: { expected: cycle.pairRoster.length, supplied: r ? Object.keys(r.pairs).length : 0,
      scored: r ? cycle.pairRoster.filter(x => r.pairs[x.id]?.state === "scored").length : 0 },
    reports: { expected: cycle.reportRoster.length, supplied: r ? Object.keys(r.reports).length : 0,
      observed: r ? cycle.reportRoster.filter(x => r.reports[x.id]?.observedCount !== null && r.reports[x.id]?.observedCount !== undefined).length : 0,
      scored: r ? cycle.reportRoster.filter(x => r.reports[x.id]?.state === "scored").length : 0 },
  };
}

const bindingMatch = (a: MetricBinding, b: MetricBinding) => a.target === b.target
  && a.cutoff === b.cutoff && a.outcomeView === b.outcomeView && a.metricDefinition === b.metricDefinition
  && a.weighting === b.weighting && a.unit === b.unit && a.reportProfile === b.reportProfile
  && a.orderedRosterIds.length === b.orderedRosterIds.length && a.orderedRosterIds.every((id, i) => id === b.orderedRosterIds[i]);
const sameObservedOutcomes = (cycle: EvaluationCycle, firstId: string, secondId: string, id: MetricId, scoredIds: string[]) => {
  const a = cycle.models[firstId], b = cycle.models[secondId];
  if (!a || !b) return false;
  const ids = id === "vector_energy" ? cycle.pairRoster.map(p => p.id) : scoredIds;
  return ids.every(key => id.startsWith("report_")
    ? a.reports[key]?.observedCount === b.reports[key]?.observedCount
    : a.pairs[key]?.observedShare === b.pairs[key]?.observedShare);
};
export type MetricComparison = { scopeMatch: "matched" | "unmatched" | "not_assessed";
  informationMatch: "same" | "different" | "not_assessed"; delta: number | null; reason: string };
/** A supplied-score subtraction, never a scorer or a component-effect estimate. */
export function compareSuppliedMetric(d: EvaluationScoreboardDraftV1, cycle: EvaluationCycle,
  firstId: string, secondId: string, id: MetricId): MetricComparison {
  const a = cycle.models[firstId]?.metrics[id], b = cycle.models[secondId]?.metrics[id];
  const first = d.models.find(m => m.id === firstId), second = d.models.find(m => m.id === secondId);
  const informationMatch = first && second ? first.informationKey === second.informationKey ? "same" : "different" : "not_assessed";
  if (!a || !b) return { scopeMatch: "not_assessed", informationMatch, delta: null, reason: "One or both metric results were not supplied." };
  if (!bindingMatch(a.binding, b.binding) || !sameSet(a.scoredIds, b.scoredIds)
    || !sameObservedOutcomes(cycle, firstId, secondId, id, a.scoredIds)) return {
    scopeMatch: "unmatched", informationMatch, delta: null,
    reason: "Scored target, ordered roster, cutoff, outcomes, definition, units, weights, profile, or scored IDs differ." };
  if (a.state !== "complete" || b.state !== "complete" || a.infinite || b.infinite
    || a.value === null || b.value === null) return { scopeMatch: "matched", informationMatch, delta: null,
      reason: "A complete finite score for both models is required for a direct difference." };
  return { scopeMatch: "matched", informationMatch, delta: a.value - b.value,
    reason: informationMatch === "different" ? "Descriptive difference only: model input information differs; no component or causal effect is isolated."
      : "Descriptive difference for the same scored scope. This does not establish future accuracy." };
}
