import { forecastLabel } from "../lib/forecast-copy";
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

// Explanations only: catalog IDs, availability and scientific records stay unchanged.
const choiceExplanation: Record<string, string> = {
  fundamentals:
    "Starts from electoral history, combining information into a race-level estimate before adding any selected polls. This is the saved fundamentals model.",
  winner:
    "A reference model for past elections that assigns chances to winner outcomes rather than vote shares. It uses the historical winner rule preserved in these saved comparisons.",
  recent_calendar_history:
    "An experimental reference using recent electoral history. Saved 2026 results are available with and without polls. Completing the calculation does not establish that it predicts better than the default.",
  senate_history_refreshed_v1:
    "An experimental starting estimate using refreshed electoral history. Saved 2026 results are available with and without polls; other combinations may need a new calculation or joint model fit.",
  senate_presidential_slope_v1:
    "An experimental starting estimate that learns how earlier presidential voting relates to Senate results. A slope describes the strength of that relationship. Saved 2026 results are available with and without polls.",
  senate_relative_context_v1:
    "An experimental starting estimate using a state's political lean relative to the nation, the president's party, and whether the election is a midterm. Saved 2026 results are available with and without polls.",
  senate_relative_lean_v1:
    "An experimental starting estimate using state voting relative to the national result and how national patterns carry into the next election. Saved 2026 results are available with and without polls.",
  none: "Does not add polling evidence to the starting model. Only saved combinations can be displayed.",
  standard: "Uses the saved standard setting for this part of the model.",
  persistent:
    "Tests an adjustment learned from recurring polling errors for the same pollster and state in earlier elections. It uses the same selected polls. Standard polling already allows errors to be shared within a race; this adds a separate learned adjustment. Available only for past elections.",
  "error-low":
    "Tests smaller assumed polling errors. Error scales describe how far polls may differ from the underlying race; this changes the uncertainty assumptions, not which candidate a poll reports ahead.",
  "error-high":
    "Tests larger assumed polling errors. Error scales describe how far polls may differ from the underlying race; this changes the uncertainty assumptions, not which candidate a poll reports ahead.",
  "cycle-low":
    "Tests a smaller election-wide polling error shared across races. SD means standard deviation: the scale of that error. Shared errors allow many races to shift together.",
  "cycle-high":
    "Tests a larger election-wide polling error shared across races. SD means standard deviation: the scale of that error. Shared errors allow many races to shift together.",
};
function explainChoice(slot: string, id: string, fallback: string) {
  if (id === "standard")
    return slot === "polling"
      ? "Adds polls using a fixed bell-shaped (Gaussian) model for measurement error: polls are evidence with uncertainty, not exact election results. Only saved combinations can be displayed."
      : "Uses the saved default assumptions about how much results can vary and how races can move together.";
  if (id.startsWith("loading-"))
    return "National loading controls how strongly a shared national shift moves races together. The multiplier changes that shared effect; ×0 removes this effect, not all uncertainty. These are saved tests for past elections, not current forecasts.";
  return choiceExplanation[id] || fallback;
}

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
  const defaultEntry = registry.runs.find((entry) => entry.id === registry.defaults[String(run.year)]);
  const defaultComparable = defaultEntry?.dateId === run.dateId && defaultEntry?.targetId === run.targetId && defaultEntry?.lawId === run.lawId;
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
          <span className="stack-summary-displayed">{forecastLabel(run.label)}</span>
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
              A stack is your combination of model choices. Select saved results
              or inspect experimental alternatives; experimental choices may
              need a new calculation before they can be shown.
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
            <span>Preserved presets · ready-made combinations</span>
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
                        {forecastLabel(option.label)}
                        {option.availability === "requires_computation"
                          ? " · Needs a new calculation"
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
                    {explainChoice(
                      slot.id,
                      selected?.id || "",
                      selected?.description || slot.description,
                    )}
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
              This combination needs a new model calculation. This site can
              display saved results but cannot run that calculation.
            </p>
          )}
          <p>
            <span>{matchesDisplayed ? "Displaying" : "Still displaying"}:</span>{" "}
            <strong>{forecastLabel(run.label)}</strong>
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
              <option value="" disabled={!defaultComparable}>{run.year} default{defaultComparable ? "" : " · Different forecast definition"}</option>
              {unknownBaseline && (
                <option value={baselineRunId!} disabled>
                  Unavailable comparison
                </option>
              )}
              {comparable.map((entry) => (
                <option key={entry.id} value={entry.id}>
                  {forecastLabel(entry.label)}
                </option>
              ))}
            </select>
            <p>
              A baseline is the reference result you compare against. Choices
              must predict the same election and type of outcome on the same
              date. The historical scorecard uses this reference; the checkbox
              adds comparison details below.
            </p>
            {!defaultComparable && <p>The default uses a different forecast definition. Select one of the matching saved comparisons here to see changes between models.</p>}
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
          <p>
            A configuration records the model choices; a result is the saved
            output of those choices. Cutoffs are the latest dates allowed for
            the stated source inputs. A historical reconstruction revisits a
            past election and may use information assembled later; it is not
            automatically a test of what was knowable at the time.
          </p>
          <div className="configuration-display">
            <h3>Displayed forecast</h3>
            <p>{forecastLabel(run.label)}</p>
            <dl>
              <dt>Configuration identity · exact model-choice record</dt>
              <dd>
                <code>
                  {displayedConfiguration?.id ||
                    "Configuration metadata unavailable"}
                </code>
              </dd>
              <dt>Result identity · exact saved output</dt>
              <dd>
                <code>
                  {displayedConfiguration?.runId || run.id}
                  {displayedConfiguration
                    ? ` · SHA256 ${displayedConfiguration.resultSha256}`
                    : ""}
                </code>
              </dd>
              <dt>Target / outcome definition · what is predicted</dt>
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
            <p className="small">
              Technical IDs identify the exact records. SHA256 is a file
              fingerprint used to check that the saved result has not changed.
            </p>
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
              <h3>Choices this editor cannot run</h3>
              <p>
                Changing individual model weights needs a new fit (estimating
                model settings from data). Blending forecasts means combining
                their predictions while preserving how races move together; this
                editor does not provide that service.
              </p>
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
      <p>
        Implementation says what has been built. Numerical checks test whether
        the calculations behave as intended. Predictive evidence tests
        predictions against election outcomes; passing calculation checks alone
        does not establish forecast accuracy.
      </p>
      <details>
        <summary>Read the technical evidence record</summary>
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
        <p className="small">
          In this record, aggregate means race-level rather than voter-by-voter
          modeling; a prior is a starting estimate before adding evidence; a fit
          estimates model settings; a posterior is the estimate after evidence
          is added. Joint output describes races simulated together.
          Strict-as-of validation allows only information available at the
          stated forecast date. A sealed prediction is a saved record, not by
          itself proof of accuracy. An adapter connects one result format to
          another. Support means the set of possible outcomes; categorical means one winner category rather than vote share. Bindings link a model to its verified inputs. A frozen resolver maps choices to existing saved results; it does not calculate a new forecast.
        </p>
      </details>
    </div>
  );
}
