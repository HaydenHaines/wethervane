import { useState } from "react";
import type { ExplorerBundle, ExplorerRegistry } from "../lib/explorer-types";
import type {
  CompositionCatalog,
  CompositionRequest,
  CompositionResolution,
  SavedConfiguration,
  SavedView,
  ConfigurationEvidence,
} from "../lib/composition-types";
import { availableRuns } from "../lib/explorer-state";

type Props = {
  registry: ExplorerRegistry;
  bundle: ExplorerBundle;
  busy: boolean;
  compare: boolean;
  notice: string | null;
  catalog: CompositionCatalog | null;
  request: CompositionRequest | null;
  resolution: CompositionResolution | null;
  displayedConfiguration: SavedConfiguration | null;
  onYear: (year: number) => void;
  onDate: (id: string) => void;
  onComponents: (patch: Record<string, string>) => void;
  onPreset: (id: string) => void;
  onCompare: (compare: boolean) => void;
  onShare: () => void;
  onReset: () => void;
  savedViews: SavedView[];
  onSave: (name: string) => void;
  onRestore: (id: string) => void;
  onDelete: (id: string) => void;
  baselineRunId: string | null;
  onBaseline: (id: string | null) => void;
};

const fieldId = (slot: string) =>
  ({
    base: "base-model",
    polling: "polling-component",
    uncertainty: "uncertainty",
  })[slot] || `component-${slot}`;
const sameChoices = (a: Record<string, string>, b: Record<string, string>) =>
  Object.keys(a).length === Object.keys(b).length &&
  Object.entries(a).every(([key, value]) => b[key] === value);
const displayTime = (at: string) =>
  new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    timeZone: "UTC",
    timeZoneName: "short",
  }).format(new Date(at));

export default function StackControls({
  registry,
  bundle,
  busy,
  compare,
  notice,
  catalog,
  request,
  resolution,
  displayedConfiguration,
  onYear,
  onDate,
  onComponents,
  onPreset,
  onCompare,
  onShare,
  onReset,
  savedViews,
  onSave,
  onRestore,
  onDelete,
  baselineRunId,
  onBaseline,
}: Props) {
  const [viewName, setViewName] = useState("");
  const run = bundle.run,
    year = request?.year ?? run.year,
    dateId = request?.dateId ?? run.dateId;
  const years = [...new Set(registry.runs.map((entry) => entry.year))].sort(
    (a, b) => b - a,
  );
  const dates = [
    ...new Map(
      availableRuns(registry, year).map((entry) => [
        entry.dateId,
        entry.dateLabel,
      ]),
    ).entries(),
  ];
  const matchesDisplayed =
    !!request &&
    !!displayedConfiguration &&
    (!resolution ||
      (resolution.state === "ready" &&
        resolution.configurationId === displayedConfiguration.id)) &&
    displayedConfiguration.runId === run.id &&
    request.year === displayedConfiguration.year &&
    request.dateId === displayedConfiguration.dateId &&
    sameChoices(request.components, displayedConfiguration.components);
  const canSave = !busy && matchesDisplayed;
  const comparable = registry.runs.filter(
    (entry) =>
      entry.year === run.year &&
      entry.dateId === run.dateId &&
      entry.lawId === run.lawId &&
      entry.targetId === run.targetId &&
      entry.id !== registry.defaults[String(run.year)],
  );
  const unknownBaseline =
    baselineRunId &&
    !comparable.some((entry) => entry.id === baselineRunId) &&
    baselineRunId !== registry.defaults[String(run.year)];
  const currentEvidence = displayedConfiguration?.evidence;
  const requestedEvidence = resolution?.evidence;
  const stateLabel = busy
    ? "Loading requested forecast"
    : resolution?.state === "requires_computation"
      ? "Requires computation"
      : resolution?.state === "incompatible"
        ? "Unsupported combination"
        : matchesDisplayed
          ? "Ready"
          : resolution?.state === "ready"
            ? "Requested result is not displayed"
            : "Saved forecast displayed";

  return (
    <details
      className="stack-builder collapsible-stack"
      aria-labelledby="stack-title"
      aria-busy={busy}
    >
      <summary className="stack-builder-summary">
        <span className="stack-summary-main">
          <span id="stack-title">Build your perspective</span>
          <span className="stack-summary-displayed">{run.label}</span>
        </span>
        <span className="stack-summary-actions">
          {(!matchesDisplayed || busy) && (
            <span className="stack-summary-status">{stateLabel}</span>
          )}
          <span className="stack-summary-closed">Expand to customize</span>
          <span className="stack-summary-open">Collapse controls</span>
          <span className="stack-summary-arrow" aria-hidden="true">
            ⌄
          </span>
        </span>
      </summary>
      <div className="stack-builder-body">
        <div className="stack-builder-top">
          <div>
            <div className="section-title">Customize this view</div>
            <div className="small">
              Choose saved components or inspect experimental alternatives.
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
              {baselineRunId === null
                ? "Compare default"
                : "Compare configuration"}
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
              {years.map((value) => (
                <button
                  key={value}
                  aria-pressed={year === value}
                  disabled={busy}
                  onClick={() => onYear(value)}
                >
                  {value}
                </button>
              ))}
            </div>
          </div>
          <div className="stack-field">
            <label htmlFor="forecast-date">Forecast date</label>
            <select
              id="forecast-date"
              value={dateId}
              disabled={busy || dates.length <= 1}
              onChange={(e) => onDate(e.target.value)}
            >
              {!dates.some(([id]) => id === dateId) && (
                <option value={dateId}>{dateId} · Not available</option>
              )}
              {dates.map(([id, label]) => (
                <option key={id} value={id}>
                  {label}
                </option>
              ))}
            </select>
            <p>
              {dates.length === 1
                ? "One saved forecast date is available."
                : "Choose an available forecast date."}
              {year === run.year && run.horizonDays !== null
                ? ` ${run.horizonDays} days before the election.`
                : ""}
            </p>
          </div>
        </div>

        {catalog && (
          <div className="composition-presets">
            <span>Preserved presets</span>
            <div role="group" aria-label="Preserved forecast presets">
              {catalog.presets.map((preset) => {
                const entry = registry.runs.find((r) => r.id === preset.runId);
                return (
                  <button
                    key={preset.id}
                    onClick={() => onPreset(preset.id)}
                    disabled={busy || !entry}
                    aria-pressed={matchesDisplayed && run.id === preset.runId}
                    title={preset.description}
                  >
                    {entry && <span>{entry.year}</span>}
                    {preset.label}
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {catalog && request ? (
          <div className="stack-components">
            {catalog.slots.map((slot) => {
              const options = slot.options.filter((option) =>
                option.years.includes(year),
              );
              const selected = slot.options.find(
                (option) => option.id === request.components[slot.id],
              );
              return (
                <div className="stack-field" key={slot.id}>
                  <label htmlFor={fieldId(slot.id)}>{slot.label}</label>
                  <select
                    id={fieldId(slot.id)}
                    value={request.components[slot.id] || ""}
                    disabled={
                      busy ||
                      options.length === 0 ||
                      (options.length === 1 &&
                        options[0].availability === "saved")
                    }
                    onChange={(e) =>
                      onComponents({ [slot.id]: e.target.value })
                    }
                  >
                    {!options.some(
                      (option) => option.id === request.components[slot.id],
                    ) && (
                      <option
                        value={request.components[slot.id] || ""}
                        disabled
                      >
                        {request.components[slot.id] || "Choose a component"} ·
                        Unsupported for {year}
                      </option>
                    )}
                    {options.map((option) => (
                      <option
                        key={option.id}
                        value={option.id}
                        disabled={option.availability === "unavailable"}
                      >
                        {option.label}
                        {option.availability === "requires_computation"
                          ? " · Needs computation"
                          : option.availability === "unavailable"
                            ? " · Unavailable"
                            : ""}
                      </option>
                    ))}
                  </select>
                  <p>
                    {selected?.experimental && (
                      <span className="component-flag">Experimental</span>
                    )}
                    {selected?.description || slot.description}
                  </p>
                </div>
              );
            })}
          </div>
        ) : (
          <p className="composition-catalog-note">
            Component choices are unavailable. The saved forecast below remains
            available.
          </p>
        )}

        <div
          className={`composition-state${!matchesDisplayed ? " unresolved" : ""}`}
          role="status"
        >
          <strong>{stateLabel}</strong>
          {resolution && !matchesDisplayed && <p>{resolution.reason}</p>}
          {resolution?.state === "requires_computation" && (
            <p>
              {catalog?.execution.reason ||
                "This configuration needs a supported computation service."}
            </p>
          )}
          <p>
            <span>{matchesDisplayed ? "Displaying" : "Still displaying"}:</span>{" "}
            <strong>{run.label}</strong>
            {run.historical
              ? " · Historical reconstruction"
              : " · Research preview"}
            .
          </p>
          {notice && <p>{notice}</p>}
        </div>

        <div className="composition-secondary">
          <div className="stack-field comparison-field">
            <label htmlFor="comparison-baseline">Comparison baseline</label>
            <select
              id="comparison-baseline"
              value={
                baselineRunId === registry.defaults[String(run.year)]
                  ? ""
                  : baselineRunId || ""
              }
              disabled={busy}
              onChange={(e) => onBaseline(e.target.value || null)}
            >
              <option value="">{run.year} default</option>
              {unknownBaseline && (
                <option value={baselineRunId!} disabled>
                  Unavailable comparison
                </option>
              )}
              {comparable.map((entry) => (
                <option key={entry.id} value={entry.id}>
                  {entry.label}
                </option>
              ))}
            </select>
            <p>
              Same year, date, target and outcome definition. Further data
              checks apply. The historical scorecard always uses this reference; the checkbox adds comparison details below.
            </p>
          </div>
          <details className="composition-saved">
            <summary>
              Saved views <span>({savedViews.length}) · This browser</span>
            </summary>
            <div className="save-view-form">
              <label htmlFor="saved-view-name">Name this view</label>
              <div>
                <input
                  id="saved-view-name"
                  value={viewName}
                  onChange={(e) => setViewName(e.target.value)}
                  placeholder="e.g. My historical comparison"
                  maxLength={80}
                />
                <button
                  disabled={!canSave || !viewName.trim()}
                  onClick={() => onSave(viewName.trim())}
                >
                  Save view
                </button>
              </div>
              {!canSave && (
                <p>
                  Save becomes available when the requested configuration is the
                  displayed result.
                </p>
              )}
            </div>
            {savedViews.length === 0 ? (
              <p className="small">Named views are stored in this browser.</p>
            ) : (
              <ul className="saved-view-list">
                {savedViews.map((view) => (
                  <li key={view.id}>
                    <button
                      className="saved-view-restore"
                      aria-label={`Restore ${view.name}`}
                      disabled={busy}
                      onClick={() => onRestore(view.id)}
                    >
                      {view.name}
                      <span>
                        {registry.runs.find((entry) => entry.id === view.runId)
                          ?.year || "Saved"}
                      </span>
                    </button>
                    <button
                      className="saved-view-delete"
                      aria-label={`Delete ${view.name}`}
                      onClick={() => onDelete(view.id)}
                    >
                      Delete
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </details>
        </div>

        <details className="composition-details">
          <summary>Configuration, evidence & data dates</summary>
          <div className="configuration-display">
            <h3>Displayed forecast</h3>
            <p>{run.label}</p>
            <dl>
              <dt>Configuration identity</dt>
              <dd>
                <code>
                  {displayedConfiguration?.id ||
                    "Configuration metadata unavailable"}
                </code>
              </dd>
              <dt>Result identity</dt>
              <dd>
                <code>
                  {displayedConfiguration?.runId || run.id}
                  {displayedConfiguration
                    ? ` · SHA256 ${displayedConfiguration.resultSha256}`
                    : ""}
                </code>
              </dd>
              <dt>Target / outcome definition</dt>
              <dd>
                <code>
                  {run.targetId} / {run.lawId}
                </code>
              </dd>
              {displayedConfiguration?.cutoffs.map((cutoff) => (
                <div
                  className="configuration-cutoff"
                  key={cutoff.label + cutoff.at}
                >
                  <dt>{cutoff.label}</dt>
                  <dd>
                    <time dateTime={cutoff.at} title={cutoff.at}>
                      {displayTime(cutoff.at)}
                    </time>
                  </dd>
                </div>
              ))}
            </dl>
            {currentEvidence && <Evidence evidence={currentEvidence} />}
          </div>
          {!matchesDisplayed && resolution && (
            <div className="configuration-request">
              <h3>Requested configuration</h3>
              <p>{resolution.reason}</p>
              <dl>
                {Object.entries(request?.components || {}).map(
                  ([slot, value]) => (
                    <div className="configuration-cutoff" key={slot}>
                      <dt>
                        {catalog?.slots.find((s) => s.id === slot)?.label ||
                          slot}
                      </dt>
                      <dd>
                        {catalog?.slots
                          .find((s) => s.id === slot)
                          ?.options.find((option) => option.id === value)
                          ?.label || value}
                      </dd>
                    </div>
                  ),
                )}
              </dl>
              {requestedEvidence && <Evidence evidence={requestedEvidence} />}{" "}
              {resolution.requiredSteps.length > 0 && (
                <>
                  <h4>Required before a result can be displayed</h4>
                  <ul>
                    {resolution.requiredSteps.map((step) => (
                      <li key={step}>{step}</li>
                    ))}
                  </ul>
                </>
              )}
            </div>
          )}
          {catalog?.unavailable.length ? (
            <div className="configuration-unavailable">
              <h3>Capabilities outside the active editor</h3>
              {catalog.unavailable.map((capability) => (
                <p key={capability.id}>
                  <strong>{capability.label}:</strong> {capability.reason}
                </p>
              ))}
            </div>
          ) : null}
        </details>
      </div>
    </details>
  );
}

function Evidence({ evidence }: { evidence: ConfigurationEvidence }) {
  return (
    <div className="configuration-evidence">
      <h4>Evidence · {evidence.status.replaceAll("_", " ")}</h4>
      <dl>
        {[
          { label: "Implementation", value: evidence.implementation },
          { label: "Numerical checks", value: evidence.numerical },
          { label: "Predictive evidence", value: evidence.predictive },
          { label: "Scope", value: evidence.scope },
        ].map((item) => (
          <div className="configuration-cutoff" key={item.label}>
            <dt>{item.label}</dt>
            <dd>{item.value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
