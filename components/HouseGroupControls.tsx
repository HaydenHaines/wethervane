"use client";

import { useEffect, useState, type ReactNode } from "react";
import type { HouseGroupPublication } from "../lib/house-group-catalog.ts";
import { groupFundamentalsId, houseFundamentalsLabel } from "../lib/house-group-presentation.ts";
import { houseGroupModelSlices } from "../lib/house-group-model-slices.ts";
import ModelSliceControls from "./ModelSliceControls";

export default function HouseGroupControls({ selectionKey, expanded, onExpandedChange, publication, busy, error, onChoose, onRetry, onLegacy, children }: {
  selectionKey: string;
  expanded: boolean;
  onExpandedChange: (expanded: boolean) => void;
  publication: HouseGroupPublication;
  busy: boolean;
  error: string;
  onChoose: (runId: string) => void;
  onRetry: () => void;
  onLegacy: () => void;
  children?: ReactNode;
}) {
  const [shareUrl, setShareUrl] = useState("");
  const [notice, setNotice] = useState("");
  const { result, entry, catalog } = publication;
  useEffect(() => { setShareUrl(""); setNotice(""); }, [result.id, selectionKey]);
  const years = [...new Set(catalog.runs.map(run => run.cycle))].sort((a, b) => b - a);
  const fundamentals = groupFundamentalsId(result.configuration);
  const currentLabel = houseFundamentalsLabel(fundamentals);
  const requestedLabel = catalog.runs.find(run => run.id === entry.id)?.label || currentLabel;

  function chooseYear(year: number) {
    const matching = catalog.runs.find(run => run.cycle === year && run.fundamentalsId === fundamentals);
    const fallback = catalog.runs.find(run => run.cycle === year);
    const next = matching || fallback;
    if (next) onChoose(next.id);
  }
  async function share() {
    const url = location.href;
    setShareUrl(url);
    try { await navigator.clipboard.writeText(url); setNotice("Link copied. It restores this model, election year, and selected view."); }
    catch { setNotice("Copy the link below to share this view."); }
  }

  return <>
    <details className="control-card house-configuration" open={expanded} onToggle={event => onExpandedChange(event.currentTarget.open)} aria-busy={busy}>
      <summary>Build your perspective <span className="small">{currentLabel} · Full 435 D/R/O group forecasts{busy ? " · Loading selection…" : ""}</span></summary>
      <p>Choose an election year and saved House model. This research result estimates which Democratic, Republican, or Other candidate group wins each district. It does not assign each candidate a separate win chance or predict House control.</p>
      <div className="house-selection-fields">
        <div className="stack-field">
          <span className="control-field-label" id="house-group-election-year-label">Election year</span>
          <div className="segmented year-control" role="group" aria-labelledby="house-group-election-year-label">
            {years.map(year => <button key={year} disabled={busy || year === result.cycle} aria-pressed={year === result.cycle} onClick={() => chooseYear(year)}>{year}</button>)}
          </div>
        </div>
      </div>
      <p className="small">Changing year keeps {currentLabel} when that model is available for the year.</p>
      <div className="stack-actions">
        <button disabled={busy || entry.id === catalog.initialRunId} onClick={() => onChoose(catalog.initialRunId)}>Reset group view</button>
        <button disabled={busy} onClick={share}>Share House view ↗</button>
      </div>
      <ModelSliceControls slices={houseGroupModelSlices(publication)} busy={busy}
        explain={(slot, value, fallback) => slot === "house-winner-target" && value === "legacy" ? "The original saved D/R candidate forecasts cover a smaller reviewed subset. They remain a separate result." : fallback}
        onChange={(id, value) => {
          if (id === "house-group-run") onChoose(value);
          if (id === "house-winner-target" && value === "legacy") onLegacy();
        }} />
      <dl>
        <dt>Research status</dt><dd>Experimental research</dd>
        <dt>Author review</dt><dd>Research review complete</dd>
        <dt>Forecast coverage</dt><dd>{result.races.length} of 435 districts</dd>
        <dt>Source support</dt><dd>Candidate lists assembled after the election</dd>
        <dt>Displayed model</dt><dd>{requestedLabel}</dd>
      </dl>
      <p className="small">The author’s checks are complete; independent review has not yet been received. Some source information uses final candidate lists assembled after each election, so these are historical reconstructions rather than forecasts published before election day.</p>
      {children}
      {notice && <p role="status" className="small">{notice}</p>}
      {shareUrl && <div className="share-link"><label htmlFor="house-group-share-url">Shareable House view</label><input id="house-group-share-url" readOnly value={shareUrl} onFocus={event => event.target.select()} /></div>}
    </details>
    {busy && <div className="compare-banner" role="status"><strong>Loading selected House forecast…</strong><span>Still displaying {currentLabel} for {result.cycle} until the selected result is verified.</span></div>}
    {error && <div className="compare-banner" role="alert"><strong>Unable to change House group forecasts</strong><span>{error} The previous result remains visible.</span><button onClick={onRetry}>Try selection again</button></div>}
  </>;
}
