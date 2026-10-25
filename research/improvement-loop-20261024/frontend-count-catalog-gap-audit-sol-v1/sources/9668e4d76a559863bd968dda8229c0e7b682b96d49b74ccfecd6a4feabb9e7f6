"use client";

import { useEffect, useState, type ReactNode } from "react";
import type { HouseLiveCatalogView, HouseLiveView } from "../lib/house-live-view";
import type { HouseLiveConfiguration } from "../lib/house-live-result";
import { houseLiveModelSlices } from "../lib/house-live-model-slices";
import { resolveHouseLiveSelection, sameHouseLiveConfiguration } from "../lib/house-live-selection";
import { houseLiveStatusLabel, houseLiveCutoffRole } from "../lib/house-live-copy";
import ModelSliceControls from "./ModelSliceControls";

const controlSlots = { "house-live-base": "base", "house-live-polling": "polling",
  "house-live-uncertainty": "uncertainty" } as const;

function utcTime(value: string): string {
  return new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", year: "numeric",
    hour: "numeric", minute: "2-digit", timeZone: "UTC", timeZoneName: "short" }).format(new Date(value));
}

export default function HouseLiveControls({ view, catalog, expanded, onExpandedChange, busy, error, onSelect, onRetry, children }: {
  view: HouseLiveView;
  catalog: HouseLiveCatalogView;
  expanded: boolean;
  onExpandedChange: (expanded: boolean) => void;
  busy: boolean;
  error: string;
  onSelect: (runId: string) => void;
  onRetry: () => void;
  children?: ReactNode;
}) {
  const [selected, setSelected] = useState<HouseLiveConfiguration>({ ...view.run.configuration });
  useEffect(() => setSelected({ ...view.run.configuration }), [view.run.id]);
  const resolution = resolveHouseLiveSelection(catalog, selected);
  const displayed = resolution.state === "ready" && resolution.entry.id === view.run.id &&
    sameHouseLiveConfiguration(selected, view.run.configuration);
  const update = (id: string, value: string) => {
    const slot = controlSlots[id as keyof typeof controlSlots];
    if (!slot || busy) return;
    const next = { ...selected, [slot]: value } as HouseLiveConfiguration;
    setSelected(next);
    const nextResolution = resolveHouseLiveSelection(catalog, next);
    if (nextResolution.state === "ready" && nextResolution.entry.id !== view.run.id)
      onSelect(nextResolution.entry.id);
  };
  const reset = () => {
    if (busy) return;
    const entry = catalog.entries.find(item => item.id === catalog.defaultId);
    if (!entry) return;
    setSelected({ ...entry.configuration });
    // Also clear a failed pending choice in the parent when the default is already displayed.
    onSelect(entry.id);
  };
  const defaultChoice = catalog.entries.find(item => item.id === catalog.defaultId);
  const draftIsDefault = !!defaultChoice && sameHouseLiveConfiguration(selected, defaultChoice.configuration);

  return <>
    <details className="control-card house-configuration" open={expanded}
      onToggle={event => onExpandedChange(event.currentTarget.open)} aria-busy={busy}>
      <summary>Build your perspective <span className="small">{view.run.label}{busy ? " · Loading selection…" : ""}</span></summary>
      <p>Choose an electoral-history model, current polling, and uncertainty setting. An exact saved choice loads immediately. Each result keeps its own list of candidates, assumptions about how districts move together, and evidence.</p>
      <ModelSliceControls slices={houseLiveModelSlices(catalog, selected)} busy={busy}
        explain={(_slot, _value, fallback) => fallback} onChange={update} />
      <div className="stack-actions">
        <button type="button" disabled={busy || (view.run.id === catalog.defaultId && draftIsDefault)}
          onClick={reset}>Reset to published default</button>
      </div>
      <p className="small" role="status" aria-live="polite">
        {displayed ? "These choices match the result now displayed." : resolution.state === "ready"
          ? `Saved choice: ${resolution.entry.label}. The displayed forecast changes only after verification.`
          : resolution.reason}
      </p>
      <dl>
        <dt>Displayed model</dt><dd>{view.run.label}</dd>
        <dt>Forecast cutoff</dt><dd>{utcTime(view.run.cutoffAt)}</dd>
        <dt>Issued</dt><dd>{utcTime(view.run.issuedAt)}</dd>
        <dt>Input review</dt><dd>{houseLiveStatusLabel(view.qualification.inputStatus, "input")}</dd>
        <dt>Calculation checks</dt><dd>{houseLiveStatusLabel(view.qualification.numericalStatus, "numerical")}</dd>
        <dt>Publication review</dt><dd>{houseLiveStatusLabel(view.qualification.publicationStatus, "publication")}</dd>
        <dt>Forecast skill</dt><dd>{houseLiveStatusLabel(view.qualification.forecastSkill, "skill")}</dd>
      </dl>
      <details className="assumptions"><summary>Source cutoffs</summary>
        <ul>{view.run.sourceCutoffs.map((source, index) => <li key={`${source.role}-${source.at}-${index}`}>
          {houseLiveCutoffRole(source.role)}: {utcTime(source.at)}</li>)}</ul>
      </details>
      <p className="small">A saved result can be available even when its future accuracy is unknown. Calculation checks assess whether the saved computation ran as intended. These forecasts predict winner groups for 435 districts; they do not predict turnout or total vote counts.</p>
      {children}
    </details>
    {busy && <div className="compare-banner" role="status"><strong>Loading the selected House result…</strong><span>Still displaying {view.run.label} until the new saved result is verified.</span></div>}
    {error && <div className="compare-banner" role="alert"><strong>Unable to change House models</strong><span>{error} The displayed result remains available.</span><button type="button" onClick={onRetry}>Try again</button></div>}
  </>;
}
