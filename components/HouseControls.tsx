import { useState, type ReactNode } from "react";
import type { HouseCatalog } from "../lib/house-catalog";
import type { HouseResult } from "../lib/house-result";
import { houseRunForYear } from "../lib/house-selection";
import { houseModelSlices } from "../lib/house-model-slices";
import ModelSliceControls from "./ModelSliceControls";
import { houseEvidenceLabel, houseSelectionNote } from "../lib/house-copy";

export default function HouseControls({ result, catalog, busy, requestedId, error, onSelect, onRetry, children }: {
  result: HouseResult;
  catalog?: HouseCatalog;
  busy: boolean;
  requestedId: string | null;
  error: string;
  onSelect: (id: string) => void;
  onRetry: () => void;
  children?: ReactNode;
}) {
  const [shareUrl, setShareUrl] = useState("");
  const [notice, setNotice] = useState("");
  const years = catalog ? [...new Set(catalog.runs.map((run) => run.year))].sort((a, b) => b - a) : [result.run.year];
  const choices = catalog?.runs.filter((run) => run.year === result.run.year) ?? [];
  const requested = catalog?.runs.find((run) => run.id === requestedId);
  async function share() {
    setShareUrl(location.href);
    try { await navigator.clipboard.writeText(location.href); setNotice("Link copied. It restores this model, election year and selected race."); }
    catch { setNotice("Copy the link below to share this view."); }
  }
  return (
    <>
      <details className="control-card house-configuration" aria-busy={busy}>
        <summary>Build your perspective <span className="small">{result.run.label}{busy ? " · Loading selection…" : ""}</span></summary>
        <p>Choose an election year and a saved model. Each choice loads its own reviewed results, race coverage and testing evidence. Changing the year does not assume that district boundaries stayed the same.</p>
        {catalog ? (
          <>
            <div className="house-selection-fields">
              <div className="stack-field">
                <span className="control-field-label" id="house-election-year-label">Election year</span>
                <div className="segmented year-control" role="group" aria-labelledby="house-election-year-label">
                  {years.map((year) => <button key={year} disabled={busy || !houseRunForYear(catalog, result.run.id, year)} aria-pressed={result.run.year === year} onClick={() => {
                    const id = houseRunForYear(catalog, result.run.id, year);
                    if (id) onSelect(id);
                  }}>{year}</button>)}
                </div>
              </div>
            </div>
            {choices.some((run) => !run.available) && <details className="assumptions"><summary>Why some choices are unavailable</summary>{choices.filter((run) => !run.available).map((run) => <p key={run.id}><strong>{run.label}:</strong> {run.unavailableReason}</p>)}</details>}
            <p className="small">{houseSelectionNote(catalog.selectionNote)} The initial selection is a reference, not a model chosen for having the best historical score.</p>
            {catalog.capabilities?.polling === false && <p className="small">These published models use electoral history without polls. Polling integration is not available in this collection.</p>}
            <div className="stack-actions">
              <button disabled={busy || result.run.id === catalog.defaultRunId} onClick={() => onSelect(catalog.defaultRunId)}>Reset House view</button>
              <button disabled={busy} onClick={share}>Share House view ↗</button>
            </div>
          </>
        ) : <p>This saved combination is selected. Additional model choices will appear when reviewed results are published.</p>}
        <ModelSliceControls slices={houseModelSlices(result, catalog)} busy={busy}
          explain={(_slot, _value, fallback) => fallback} onChange={(id, value) => { if (id === "house-model") onSelect(value); }} />
        <p className="small">Results for {result.races.length} of 435 districts. Shared uncertainty and candidate context remain bundled into the selected House model.</p>
        <dl>
          <dt>Displayed model</dt><dd>{result.run.label}</dd>
          <dt>Research status</dt><dd>{result.run.experimental ? "Experimental" : "Standard"}</dd>
          <dt>Testing evidence</dt><dd>{houseEvidenceLabel(result.run.evidenceLabel)}</dd>
        </dl>
        <p className="small">“Experimental” describes the stage of research. A completed calculation does not by itself establish forecasting accuracy.</p>
        {children}
        {notice && <p role="status" className="small">{notice}</p>}
        {shareUrl && <div className="share-link"><label htmlFor="house-share-url">Shareable House view</label><input id="house-share-url" readOnly value={shareUrl} onFocus={(event) => event.target.select()} /></div>}
      </details>
      {busy && <div className="compare-banner" role="status"><strong>Loading {requested?.label || "the selected House model"}…</strong><span>Still displaying {result.run.label} ({result.run.year}) until the new result is verified.</span></div>}
      {error && <div className="compare-banner" role="alert"><strong>Unable to change House models</strong><span>{error} The previous forecast remains visible.</span><button onClick={onRetry}>Try selection again</button></div>}
    </>
  );
}
