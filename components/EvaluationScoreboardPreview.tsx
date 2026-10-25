import { useId } from "react";
import {
  compareSuppliedMetric, coverageFor, evaluationScoreboardReadiness, metricMeta,
  type EvaluationScoreboardDraftV1, type MetricId, type ModelCycle, type PairRow,
  type ReportRow, type SuppliedMetric,
} from "../lib/evaluation-scoreboard-view.ts";

const grid = { display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(min(100%,245px),1fr))", gap: 14 } as const;
const num = (n: number, digits = 2) => n !== 0 && Math.abs(n) < 10 ** -digits
  ? n.toExponential(2) : new Intl.NumberFormat("en-US", { maximumFractionDigits: digits }).format(n);
const points = (n: number) => `${num(n * 100)} percentage points`;
const mass = (n: number | null) => n === null ? "unknown" : n === 0 ? "0%" : n === 1 ? "100%"
  : n < .001 ? "less than 0.1%" : n > .999 ? "more than 99.9%" : `${num(n * 100, 1)}%`;
const scoreParts = (metric: SuppliedMetric) => ({
  value: metric.infinite ? "∞" : metric.value === null ? "Unavailable"
    : metric.binding.unit === "share_fraction" ? num(metric.value * 100) : num(metric.value),
  unit: metric.binding.unit === "share_fraction" ? "percentage points"
    : metric.binding.unit === "report_count" ? "report counts" : "dimensionless",
});
const stateLabel = (state: string) => ({ scored: "Scored", outcome_missing: "Recorded outcome missing",
  undefined_mass: "Undefined predictive mass", unavailable: "Unavailable", not_requested: "Not requested",
  inactive_event: "Inactive event" } as Record<string, string>)[state] || state;

function IntervalReadout({ results, expected, kind }: { results: (PairRow | ReportRow | undefined)[]; expected: number; kind: "pair" | "report" }) {
  const levels = [...new Set(results.flatMap(row => row?.intervals.map(x => x.level) || []))].sort((a, b) => a - b);
  if (!levels.length) return <p>No range coverage or width was supplied.</p>;
  return <div>{levels.map(level => {
    const present = results.flatMap(row => row?.state === "scored" ? row.intervals.filter(x => x.level === level && x.covered !== null) : []);
    const covered = present.filter(x => x.covered).length;
    const widths = results.flatMap(row => row?.state === "scored" || row?.state === "outcome_missing"
      ? row.intervals.filter(x => x.level === level).map(x => x.width) : []);
    return <p key={level}><strong>{mass(level)} nominal range:</strong> {covered} of {expected} expected recorded results covered
      {present.length < expected ? ` (partial: ${present.length} evaluable)` : ""}; width {widths.length
        ? `${kind === "pair" ? points(Math.min(...widths)) : `${num(Math.min(...widths))} report counts`}${widths.length > 1 ? ` to ${kind === "pair" ? points(Math.max(...widths)) : `${num(Math.max(...widths))} report counts`}` : ""} across supplied ranges`
        : "unavailable"}. Coverage and width describe different aspects of range quality.</p>;
  })}</div>;
}

function MetricCard({ id, metric, comparison }: { id: MetricId; metric?: SuppliedMetric;
  comparison?: ReturnType<typeof compareSuppliedMetric> }) {
  const meta = metricMeta[id];
  return <article className="count-quantity" aria-label={meta.label} style={{ minWidth: 0 }}>
    <h4>{meta.label}</h4>
    {metric ? <>
      <p><strong>{metric.state === "complete" ? "Complete supplied score" : metric.state === "partial" ? "Supplied partial score" : metric.state === "not_requested" ? "Not requested" : "Unavailable"}</strong></p>
      {metric.state === "complete" || metric.state === "partial" ? <>
        <p className="count-value" style={{ fontVariantNumeric: "tabular-nums", fontSize: "clamp(21px, 5vw, 30px)" }}>
          <span style={{ whiteSpace: "nowrap", font: "inherit", color: "inherit" }}>{scoreParts(metric).value} <small style={{ font: "14px Arial, sans-serif" }}>{scoreParts(metric).unit}</small></span>
        </p>
        {metric.infinite && <p>Infinite empirical loss: the supplied probability for an observed event was zero. This is different from a missing or zero loss.</p>}
      </>
        : <p>{metric.reason || "No value was supplied."} Missing is not zero.</p>}
      <p>{metric.scoredIds.length} of {metric.binding.orderedRosterIds.length} expected {id === "vector_energy" ? "whole vectors" : id.startsWith("report_") ? "reports" : "named pairs"} scored.
        {metric.state === "partial" ? " This value covers only the named scored IDs; no full-roster mean is claimed." : ""}</p>
      <p><strong>Numerical precision:</strong> {metric.precision.status.replaceAll("_", " ")}. {metric.precision.note}</p>
    </> : <p><strong>Unavailable.</strong> No result supplied; missing is not zero.</p>}
    <p>{meta.explanation}</p>
    {comparison && <p><strong>Scope match:</strong> {comparison.scopeMatch.replaceAll("_", " ")}. <strong>Information match:</strong> {comparison.informationMatch === "different" ? "Different inputs" : comparison.informationMatch === "same" ? "Same information" : "Not assessed"}.
      {comparison.delta !== null ? <> <strong>Descriptive difference:</strong> {metric?.binding.unit === "share_fraction" ? points(comparison.delta) : metric?.binding.unit === "report_count" ? `${num(comparison.delta)} report counts` : `${num(comparison.delta)} dimensionless`} (selected minus comparison).</> : null} {comparison.reason}</p>}
  </article>;
}

function PairDetail({ pair, row }: { pair: EvaluationScoreboardDraftV1["cycles"][number]["pairRoster"][number]; row?: PairRow }) {
  return <article className="count-quantity" style={{ minWidth: 0 }}>
    <h4>{pair.first.name} and {pair.second.name}</h4>
    <p><strong>{row ? stateLabel(row.state) : "Missing forecast row"}.</strong> {row?.reason || ""}</p>
    <p>First named candidate’s votes divided by these two candidates’ combined votes. Other candidates are outside this denominator.</p>
    {row?.state === "undefined_mass" && <p>Share undefined in {mass(row.undefinedMass)} of predicted outcomes; the remaining mass is not reweighted.</p>}
    {row?.observedShare !== null && row?.observedShare !== undefined && <p>Recorded pair share: {mass(row.observedShare)}.</p>}
    {row?.state === "scored" && <>
      <p>Full-distribution error (CRPS): {points(row.crps!)}. Range error (WIS): {points(row.wis!)}. Lower is better.</p>
      {row.leading && <p>First candidate strictly leads this named pair with probability {mass(row.leading.probability)}; tie probability {mass(row.leading.tieProbability)}.
        Recorded strict lead: {row.leading.observed ? "yes" : "no"}; Brier error {num(row.leading.brier!)} dimensionless;
        natural-log loss {row.leading.logLossInfinite ? "∞ (infinite loss)" : row.leading.logLoss === null ? "unavailable" : `${num(row.leading.logLoss)} dimensionless`}.
        This is not legal winner or chamber probability.</p>}
      {row.shape?.near50 && <p>A near-even chance of leading can still describe a wide or split forecast: chance of a margin within 2 percentage points {mass(row.shape.probMarginWithin2pp)};
        chance of at least 10 percentage points {mass(row.shape.probMarginAtLeast10pp)}. Mean signed margin {num(row.shape.expectedSignedMarginPp)} percentage points;
        median signed margin {num(row.shape.medianSignedMarginPp)} percentage points. These margin fields are already in percentage points.</p>}
    </>}
    {(row?.state === "scored" || row?.state === "outcome_missing") && <>
      {row.intervals.map(i => <p key={i.level}>{mass(i.level)} nominal range: {mass(i.lower)}–{mass(i.upper)}; width {points(i.width)}; recorded share {i.covered === null ? "not assessed" : i.covered ? "covered" : "outside range"}.</p>)}
      {row.state === "outcome_missing" && row.leading && <p>Forecast chance that the first candidate strictly leads this named pair: {mass(row.leading.probability)}; tie chance {mass(row.leading.tieProbability)}. No recorded outcome or leading error was supplied.</p>}
    </>}
    <details className="count-checks"><summary>Pair identity and event</summary><p>Race {pair.raceId}; contest {pair.contestId}; event {pair.eventId}; date {pair.electionDate}.</p>
      <p>Ordered endpoints: {pair.first.name} ({pair.first.id}) then {pair.second.name} ({pair.second.id}).</p></details>
  </article>;
}

function ReportDetail({ report, row }: { report: EvaluationScoreboardDraftV1["cycles"][number]["reportRoster"][number]; row?: ReportRow }) {
  return <article className="count-quantity" style={{ minWidth: 0 }}>
    <h4>{report.name}</h4><p><strong>{row ? stateLabel(row.state) : "Missing forecast row"}.</strong> {row?.reason || ""}</p>
    <p>Profile: {report.profile}. Definition: {report.definition}. These are reported administrative totals, not exact physical participants.</p>
    {row?.state === "undefined_mass" && <p>Undefined predictive mass {mass(row.undefinedMass)}; no reweighting or score.</p>}
    {row?.state === "inactive_event" && <p>The event was inactive; count and mean are unavailable. This is different from an active report of zero.</p>}
    {row?.eventPresent && row.observedCount === 0 && <p>Active event with a recorded total of 0 report counts.</p>}
    {row?.state === "scored" && <>
      <p>Recorded {num(row.observedCount!)} report counts; predicted mean {num(row.predictedMean!)} report counts.</p>
      <p>Full-distribution error (CRPS) {num(row.crps!)} report counts; range error (WIS) {num(row.wis!)} report counts.
        Natural-log loss {row.logLossInfinite ? "∞ (infinite loss)" : row.logLoss === null ? "unavailable" : `${num(row.logLoss)} dimensionless`}.</p>
    </>}
    {row?.state === "outcome_missing" && row.predictedMean !== null && <p>Predicted mean {num(row.predictedMean)} report counts; recorded total unavailable, so no error is shown.</p>}
    {(row?.state === "scored" || row?.state === "outcome_missing") && row.intervals.map(i => <p key={i.level}>{mass(i.level)} nominal range: {num(i.lower)}–{num(i.upper)} report counts; width {num(i.width)} report counts; recorded total {i.covered === null ? "not assessed" : i.covered ? "covered" : "outside range"}.</p>)}
    <details className="count-checks"><summary>Report source and effect provenance</summary><p>Event {report.eventId}; date {report.date}; release {report.releaseId}.</p>
      <p>Source quality: {report.sourceQualityFlags.length ? report.sourceQualityFlags.join("; ") : "No flags supplied"}. Source variance {report.sourceVariance === null ? "unknown" : `${num(report.sourceVariance)} report counts squared`}.</p>
      <p>Release effect: {report.effectProvenance.release}. State effect: {report.effectProvenance.state}.</p></details>
  </article>;
}

/** Private static renderer. Parent supplies selected year/model context; there is no route or session state. */
export default function EvaluationScoreboardPreview({ view, selectedCycleId, selectedModelId, compareModelId }: {
  view: EvaluationScoreboardDraftV1; selectedCycleId: string; selectedModelId: string; compareModelId?: string;
}) {
  const baseId = useId().replaceAll(":", "");
  const readiness = evaluationScoreboardReadiness(view);
  const banner = <div className="count-fixture-banner" role="note"><strong>Illustrative data only · Private forecast-quality study</strong>
    <p>All people, models, values and source records in this preview are invented. Declared hashes do not authenticate real data. This is not an actual qualified forecast or a public recommendation.</p></div>;
  if (!readiness.valid) return <section className="count-preview" aria-label="Private evaluation scoreboard">{banner}
    <div className="count-warning"><strong>Evaluation unavailable.</strong><ul>{readiness.problems.map((p, i) => <li key={i}>{p}</li>)}</ul></div></section>;
  const draft = readiness.view, cycle = draft.cycles.find(c => c.id === selectedCycleId);
  const model = draft.models.find(m => m.id === selectedModelId);
  const comparisonModel = draft.models.find(m => m.id === compareModelId);
  const result = cycle?.models[selectedModelId];
  const coverage = cycle ? coverageFor(cycle, selectedModelId) : null;
  const ids: MetricId[] = ["pair_crps", "pair_wis", "leading_brier", "leading_log", "vector_energy", "report_crps", "report_wis", "report_log"];
  return <section className="count-preview" aria-labelledby={`${baseId}-title`} style={{ maxWidth: 1100, width: "100%", minWidth: 0, overflowWrap: "anywhere" }}>
    {banner}<header><div className="eyebrow">Private frontend preparation</div><h2 id={`${baseId}-title`}>{draft.label}</h2>
      <p>{draft.provenance.label}</p></header>
    {!cycle || !model ? <div className="count-warning"><strong>Selection unavailable.</strong> The parent-selected cycle or model is not in this supplied study.</div> : <>
      <section className="count-section" aria-labelledby={`${baseId}-setup`}><h3 id={`${baseId}-setup`}>Forecast setup · {cycle.label}</h3>
        <p><strong>{model.name}</strong> · information cutoff {cycle.cutoff} · recorded outcome view {cycle.outcomeView}.</p>
        <ul>{model.ingredients.map((x, i) => <li key={i}>{x}</li>)}</ul>
        {comparisonModel && <p>Compared with {comparisonModel.name}. Input information: {model.informationKey === comparisonModel.informationKey ? "same" : "different; any matched score difference is descriptive and does not isolate a component effect"}.</p>}
        <dl className="count-outcomes"><div><dt>Run</dt><dd>{draft.status.run.replaceAll("_", " ")}</dd></div>
          <div><dt>Evidence</dt><dd>{draft.status.evidence.replaceAll("_", " ")}</dd></div>
          <div><dt>Public adoption</dt><dd>{draft.status.adoption.replaceAll("_", " ")}</dd></div>
          <div><dt>Pairs scored</dt><dd>{coverage?.pairs.scored} of {coverage?.pairs.expected} expected</dd></div>
          <div><dt>Reports observed / scored</dt><dd>{coverage?.reports.observed} / {coverage?.reports.scored} of {coverage?.reports.expected} expected</dd></div></dl>
        <p>{draft.status.note} These status fields describe different questions. Historical cycles are cases; pair and report rows show coverage within them, not independent forecasts.</p>
      </section>
      <section className="count-section" aria-labelledby={`${baseId}-scores`}><h3 id={`${baseId}-scores`}>Forecast-quality scoreboard</h3>
        <p>Scores below are supplied by the invented fixture. No score is calculated from the displayed rows. A complete claim requires every expected item, including missing ones.</p>
        <div style={grid}>{ids.map(id => <MetricCard key={id} id={id} metric={result?.metrics[id]}
          comparison={compareModelId && cycle.models[compareModelId] ? compareSuppliedMetric(draft, cycle, selectedModelId, compareModelId, id) : undefined} />)}</div>
        <article className="count-quantity" style={{ marginTop: 14 }}><h4>Prediction range coverage and width</h4>
          <p>Coverage counts recorded results inside each nominal range; width says how broad those ranges are. Read both together. A predictive range describes possible results, not numerical calculation precision.</p>
          <h5>Named-pair shares</h5><IntervalReadout kind="pair" expected={cycle.pairRoster.length} results={cycle.pairRoster.map(x => result?.pairs[x.id])} />
          <h5>Administrative report totals</h5><IntervalReadout kind="report" expected={cycle.reportRoster.length} results={cycle.reportRoster.map(x => result?.reports[x.id])} />
        </article>
        <p>Administrative totals can have different definitions and source quality across cycles. The 2018 participating-voter profile and later counted-ballot profiles must stay separate when supplied. No cross-definition or cross-cycle mean is formed here.</p>
      </section>
      <details className="count-checks"><summary>Cycle and scope breakdown</summary>
        <div role="region" aria-label="Cycle coverage table" tabIndex={0} style={{ maxWidth: "100%", overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse", minWidth: 520 }}><thead><tr><th scope="col">Cycle</th><th scope="col">Cutoff</th><th scope="col">Named pairs</th><th scope="col">Reports observed / scored</th><th scope="col">Result</th></tr></thead>
          <tbody>{draft.cycles.map(c => { const cv = coverageFor(c, selectedModelId); return <tr key={c.id}><th scope="row">{c.label}</th><td>{c.cutoff}</td>
            <td>{cv.pairs.scored} scored / {cv.pairs.expected} expected</td><td>{cv.reports.observed} observed / {cv.reports.scored} scored / {cv.reports.expected} expected</td>
            <td>{c.models[selectedModelId] ? "Supplied" : "Unavailable"}</td></tr>; })}</tbody></table></div>
        <p>{draft.cycles.length} historical cycles supplied. Rows within a cycle are coverage, not independent replications. No equal-cycle score is inferred.</p>
        <h4>Expected named pairs · {cycle.label}</h4><div style={grid}>{cycle.pairRoster.map(p => <PairDetail key={p.id} pair={p} row={result?.pairs[p.id]} />)}</div>
        <h4>Expected administrative reports · {cycle.label}</h4><div style={grid}>{cycle.reportRoster.map(r => <ReportDetail key={r.id} report={r} row={result?.reports[r.id]} />)}</div>
        <h4>Whole ordered vector</h4><p>{result?.energy.state === "scored" ? `Supplied energy ${points(result.energy.value!)} for ${result.energy.orderedPairIds.length} ordered pair shares; ${result.energy.rows} joint rows, original weight sum ${num(result.energy.weightSum!)}.`
          : `Withheld or not requested: ${result?.energy.reason || "No vector was supplied."} Undefined mass ${mass(result?.energy.undefinedMass ?? null)}; missing observed coordinates ${result?.energy.missingObservedIds.join(", ") || "none"}; positive-weight missing coordinates ${result?.energy.positiveWeightMissingIds.join(", ") || "none"}. No coordinate is dropped or reweighted.`}</p>
      </details>
      <details className="count-checks"><summary>Technical definitions and declared evidence</summary>
        <p>This private V1 view model is not an owner-approved browser export. Source hashes are declarations only and do not authenticate files or qualify a posterior.</p>
        <p>Declared source hashes: {draft.provenance.declaredSourceHashes.length ? draft.provenance.declaredSourceHashes.join(", ") : "none"}.</p>
        <p>Energy uses the whole ordered named-pair share vector and Euclidean distance divided by √(number of pairs). It is not a sum of per-pair effects. Losses in native share fractions are displayed after multiplying by 100 to percentage points; signed margin fields are already in points. Brier and natural-log losses are dimensionless; administrative report errors are counts. A null loss with an infinite flag is infinite, not a missing or zero loss.</p>
        <p>Reported administrative totals can differ from exact participation. Source variance may be unknown. Release and state effects are identified within individual report records. Neither a displayed difference nor a range proves future accuracy.</p>
        {result && ids.map(id => { const m = result.metrics[id]; return m ? <p key={id}><strong>{metricMeta[id].label} binding:</strong> target {m.binding.target}; cutoff {m.binding.cutoff}; outcome view {m.binding.outcomeView}; definition {m.binding.metricDefinition}; weighting {m.binding.weighting}; unit {m.binding.unit}; report profile {m.binding.reportProfile || "not applicable"}; ordered roster {m.binding.orderedRosterIds.join(", ")}; scored IDs {m.scoredIds.join(", ") || "none"}. {m.precision.rule || "No precision rule supplied."}</p> : null; })}
      </details>
    </>}
  </section>;
}
