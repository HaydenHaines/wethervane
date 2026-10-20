import type {
  ExplorerBundle,
  ExplorerRegistry,
  ExplorerRun,
} from "../lib/explorer-types";
import { availableRuns } from "../lib/explorer-state";

type Props = {
  registry: ExplorerRegistry;
  bundle: ExplorerBundle;
  busy: boolean;
  compare: boolean;
  notice: string | null;
  onYear: (year: number) => void;
  onDate: (id: string) => void;
  onComponents: (patch: Partial<ExplorerRun["components"]>) => void;
  onCompare: (compare: boolean) => void;
  onShare: () => void;
  onReset: () => void;
};
export default function StackControls({
  registry,
  bundle,
  busy,
  compare,
  notice,
  onYear,
  onDate,
  onComponents,
  onCompare,
  onShare,
  onReset,
}: Props) {
  const run = bundle.run;
  const choices = availableRuns(registry, run.year, run.dateId);
  const years = [...new Set(registry.runs.map((entry) => entry.year))]
    .sort((a, b) => b - a);
  const dates = [
    ...new Map(
      availableRuns(registry, run.year).map((r) => [r.dateId, r.dateLabel]),
    ).entries(),
  ];
  const baseOptions = [...new Set(choices.map((r) => r.components.base))];
  const pollOptions = [...new Set(choices.map((r) => r.components.polling))];
  const uncertainties = [
    ...new Map(
      choices
        .filter(
          (r) =>
            r.components.base === run.components.base &&
            r.components.polling === run.components.polling,
        )
        .map((r) => [r.components.uncertainty, r.uncertaintyLabel]),
    ).entries(),
  ];
  return (
    <section
      className="stack-builder"
      aria-labelledby="stack-title"
      aria-busy={busy}
    >
      <div className="stack-builder-top">
        <div>
          <h2 className="section-title" id="stack-title">
            Build your perspective
          </h2>
          <div className="small">
            Choose the components. Every view updates together.
          </div>
        </div>
        <div className="stack-actions">
          <label className="compare-toggle">
            <input
              type="checkbox"
              checked={compare}
              disabled={busy}
              onChange={(e) => onCompare(e.target.checked)}
            />
            Compare default
          </label>
          <button onClick={onReset} disabled={busy}>
            Reset stack
          </button>
          <button onClick={onShare}>Share view ↗</button>
        </div>
      </div>
      <div className="stack-period">
        <div className="stack-field">
          <label>Election year</label>
          <div className="segmented" role="group" aria-label="Election year">
            {years.map((year) => (
              <button
                key={year}
                aria-pressed={run.year === year}
                disabled={busy || !registry.runs.some((r) => r.year === year)}
                onClick={() => onYear(year)}
              >
                {year}
              </button>
            ))}
          </div>
        </div>
        <div className="stack-field">
          <label htmlFor="forecast-date">Forecast date</label>
          <select
            id="forecast-date"
            value={run.dateId}
            disabled={busy || dates.length === 1}
            onChange={(e) => onDate(e.target.value)}
          >
            {dates.map(([id, label]) => (
              <option key={id} value={id}>
                {label}
              </option>
            ))}
          </select>
          <p>
            {dates.length === 1
              ? "Only this saved forecast date is available."
              : "Only saved forecast dates are listed."}
            {run.horizonDays !== null
              ? ` ${run.horizonDays} days before the election.`
              : ""}
          </p>
        </div>
      </div>
      <div className="stack-components">
        <div className="stack-field">
          <label htmlFor="base-model">Base model</label>
          <select
            id="base-model"
            value={run.components.base}
            disabled={busy || baseOptions.length === 1}
            onChange={(e) =>
              onComponents({
                base: e.target.value as ExplorerRun["components"]["base"],
              })
            }
          >
            {baseOptions.map((value) => (
              <option key={value} value={value}>
                {value === "winner"
                  ? "Historical winner model"
                  : "Fundamentals"}
              </option>
            ))}
          </select>
          <p>
            {run.historical
              ? "A historical reconstruction, with group probabilities where candidates share an outcome."
              : "Aggregate state fundamentals on the two-principal-contender scale."}
          </p>
        </div>
        <div className="stack-field">
          <label htmlFor="polling-component">Polling component</label>
          <select
            id="polling-component"
            value={run.components.polling}
            disabled={busy || pollOptions.length === 1}
            onChange={(e) =>
              onComponents({
                polling: e.target.value as ExplorerRun["components"]["polling"],
              })
            }
          >
            {pollOptions.map((value) => (
              <option key={value} value={value}>
                {
                  {
                    none: "No polling",
                    standard: "Standard polling",
                    persistent: "Persistent polling error",
                  }[value]
                }
              </option>
            ))}
          </select>
          <p>Each choice loads an existing compatible run.</p>
        </div>
        <div className="stack-field">
          <label htmlFor="uncertainty">Uncertainty</label>
          <select
            id="uncertainty"
            value={run.components.uncertainty}
            disabled={busy || uncertainties.length === 1}
            onChange={(e) => onComponents({ uncertainty: e.target.value })}
          >
            {uncertainties.map(([id, label]) => (
              <option key={id} value={id}>
                {label}
              </option>
            ))}
          </select>
          <p>{run.uncertaintyDescription}</p>
        </div>
      </div>
      <p className="stack-current">
        <strong>{run.label}</strong> ·{" "}
        {run.historical
          ? "Historical research reconstruction"
          : "Current research preview"}
        {run.sensitivity ? " · Sensitivity setting" : ""}
        {!run.strictAsOf
          ? " · Retrospective proxies; not a strict as-of forecast"
          : ""}
      </p>
      <div className="stack-status" role="status">
        {busy
          ? "Loading and checking the selected saved stack…"
          : notice || "All views follow your selected components."}
      </div>
    </section>
  );
}
