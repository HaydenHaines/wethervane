import type { HouseComparisonState } from "./useHouseComparison";

export default function HouseComparisonControls({ comparison, busy }: { comparison: HouseComparisonState; busy: boolean }) {
  if (!comparison.choices.length && !comparison.requestedId) return null;
  const known = comparison.choices.some(run => run.id === comparison.requestedId);
  return <div className="house-comparison-controls stack-field">
    <label htmlFor="house-comparison">Compare with</label>
    <select id="house-comparison" value={comparison.requestedId || ""} disabled={busy}
      aria-describedby="house-comparison-help" onChange={event => comparison.choose(event.target.value)}>
      <option value="">No comparison</option>
      {comparison.requestedId && !known && <option value={comparison.requestedId} disabled>Unavailable comparison</option>}
      {comparison.choices.map(run => <option key={run.id} value={run.id} disabled={!run.available}>
        {run.label}{run.available ? "" : " · Unavailable"}
      </option>)}
    </select>
    <p id="house-comparison-help">Compare another saved model for the same election year. The selected model stays in the charts. Changing the year clears the comparison.</p>
  </div>;
}

export function HouseComparisonStatus({ comparison, label, year, configurationId }: { comparison: HouseComparisonState; label: string; year: number; configurationId: string }) {
  if (comparison.busy) return <div className="compare-banner" role="status"><strong>Loading comparison: {comparison.requested?.label || "saved model"}</strong><span>The page still shows {label}, {year}. Differences will appear after the comparison is verified.</span></div>;
  if (comparison.error) return <div className="compare-banner" role="alert"><strong>Comparison unavailable</strong><span>{comparison.error} The selected model and its scores remain visible.</span><button onClick={comparison.retry}>Retry comparison</button><button onClick={() => comparison.choose("")}>Remove comparison</button></div>;
  if (!comparison.ready) return null;
  const reference = comparison.ready.reference.result;
  const context2018 = year === 2018 && [reference.run.configurationId, configurationId].includes("house_relative_context_v1");
  return <section className="house-comparison-note" aria-label="House model comparison">
    <strong>{label} compared with {reference.run.label} · {year}</strong>
    <p>Compare the models’ scores and winning chances for the same districts. Changes show the selected model minus the compared model. They do not tell us how certain a difference is. Different scores can favor different models.</p>
    {context2018 && <p>The 2018 election-context model applies a White House party effect under a Republican president even though its training elections covered only Democratic presidents. This extends beyond the situations observed in training.</p>}
  </section>;
}
