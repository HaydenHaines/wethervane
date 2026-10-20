"use client";

import { useEffect, useId, useMemo, useState } from "react";
import type { HousePublicationWithCatalog } from "../lib/house-catalog";
import {
  HOUSE_SAVED_VIEWS_STORAGE_KEY,
  MAX_HOUSE_SAVED_VIEWS,
  createHouseSavedView,
  houseSavedViewAvailability,
  parseHouseSavedViews,
  serializeHouseSavedViews,
  verifyHouseSavedView,
  type HouseSavedView,
  type HouseSavedResult,
} from "../lib/house-saved-views";

export default function HouseSavedViews({ publication, raceId, mapView, busy, selectionError, comparison, onRestore }: {
  publication: Extract<HousePublicationWithCatalog, { status: "available" }>;
  raceId: string;
  mapView: "forecast" | "actual";
  busy: boolean;
  selectionError: string;
  comparison?: HouseSavedResult;
  onRestore: (view: HouseSavedView) => void;
}) {
  const inputId = useId();
  const [views, setViews] = useState<HouseSavedView[]>([]);
  const [name, setName] = useState("");
  const [notice, setNotice] = useState("");
  const [storageReady, setStorageReady] = useState(false);
  const [storageChecked, setStorageChecked] = useState(false);
  const { result, resultSha256, catalog } = publication;

  useEffect(() => {
    function readStored() {
      try {
        setViews(parseHouseSavedViews(localStorage.getItem(HOUSE_SAVED_VIEWS_STORAGE_KEY)));
        setStorageReady(true);
      } catch {
        setStorageReady(false);
        setNotice("Saved House views cannot be read in this browser. You can still copy the page address to keep a link to the displayed result.");
      } finally {
        setStorageChecked(true);
      }
    }
    readStored();
    const onStorage = (event: StorageEvent) => {
      if (event.key === HOUSE_SAVED_VIEWS_STORAGE_KEY || event.key === null) readStored();
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  function persist(next: HouseSavedView[], success: string): boolean {
    try {
      localStorage.setItem(HOUSE_SAVED_VIEWS_STORAGE_KEY, serializeHouseSavedViews(next));
      setViews(next);
      setNotice(success);
      return true;
    } catch {
      setNotice("This change could not be saved in your browser. Your saved list has not changed. You can still copy the page address to keep a link to the displayed result.");
      return false;
    }
  }
  const selectedRace = result.races.find(race => race.id === raceId);
  const actualAvailable = result.run.historical;
  const selectionValid = !!selectedRace && (mapView !== "actual" || actualAvailable);
  const saveReason = busy ? "Wait for the requested House result to finish loading before saving."
    : selectionError ? "The requested result could not be loaded. Retry or return to a completed result before saving."
      : !storageReady ? "Browser storage must be available before a view can be saved."
        : !selectionValid ? mapView === "actual" && selectedRace ? "Actual winners is unavailable for this election. Choose Forecast before saving." : "Select a district in the displayed result before saving."
          : views.length >= MAX_HOUSE_SAVED_VIEWS ? `You have ${MAX_HOUSE_SAVED_VIEWS} saved House views. Delete one to save another.`
            : "";
  function save() {
    if (saveReason || !name.trim()) return;
    try {
      const view = createHouseSavedView(name, { result, resultSha256 }, { raceId, mapView, comparison });
      if (persist([...views, view], `Saved ${view.label} in this browser.`)) setName("");
    } catch {
      setNotice("This view could not be saved. Select a district in the displayed result and try again.");
    }
  }
  const entries = useMemo(() => views.map(view => {
    const sameResult = view.runId === result.run.id && view.resultSha256 === resultSha256;
    const canVerify = sameResult && (!view.comparison || (comparison?.result.run.id === view.comparison.runId && comparison.resultSha256 === view.comparison.resultSha256));
    const exact = canVerify ? verifyHouseSavedView(view, { result, resultSha256 }, comparison) : null;
    const preflight = houseSavedViewAvailability(view, catalog);
    return { view,
      available: canVerify ? exact?.status === "verified" : preflight.status === "catalog_compatible",
      unavailableReason: canVerify ? exact?.reason : preflight.reason,
    };
  }), [views, result, resultSha256, catalog, comparison]);

  return <details className="composition-saved house-saved-views">
    <summary>Saved House views <span aria-hidden="true">({views.length})</span></summary>
    <p className="small">Saved in this browser on this device. These views are not synced.</p>
    <div className="save-view-form">
      <label htmlFor={inputId}>Name this House view</label>
      <div>
        <input id={inputId} value={name} maxLength={80} onChange={event => setName(event.target.value)}
          onKeyDown={event => { if (event.key === "Enter") { event.preventDefault(); save(); } }}
          aria-describedby={saveReason ? `${inputId}-save-reason` : undefined} />
        <button type="button" disabled={!!saveReason || !name.trim()} onClick={save}>Save House view</button>
      </div>
    </div>
    {saveReason && <p id={`${inputId}-save-reason`} className="small">{saveReason}</p>}
    {notice && <p className="small house-saved-notice" role="status">{notice}</p>}
    {views.length ? <ul className="saved-view-list">
      {entries.map(({ view, available, unavailableReason }) => {
        const district = view.raceId.split(":")[1] || "Saved district";
        return <li key={view.id} className={available ? "" : "house-saved-unavailable"}>
          <div className="house-saved-description">
            <button type="button" className="saved-view-restore" aria-label={`Restore ${view.label}`}
              disabled={busy || !available} onClick={() => onRestore(Object.freeze({ ...view, ...(view.comparison ? { comparison: Object.freeze({ ...view.comparison }) } : {}) }))}>
              {view.label}
            </button>
            <p className="small">{view.year} · {view.runLabel || "Saved House model"} · {district} · {view.mapView === "actual" ? "Actual winners" : "Forecast"}</p>
            {view.comparison && <p className="small">Compared with {view.comparison.runLabel || "the exact saved comparison model"}</p>}
            {!available && <p className="small house-saved-reason"><strong>Unavailable:</strong> {unavailableReason || "This exact saved result cannot currently be restored."}</p>}
          </div>
          <button type="button" className="saved-view-delete" aria-label={`Delete ${view.label}`}
            onClick={() => persist(views.filter(entry => entry.id !== view.id), `Deleted ${view.label} from this browser.`)}>Delete</button>
        </li>;
      })}
    </ul> : <p className="small">{!storageChecked ? "Loading saved House views…" : storageReady ? "No House views saved in this browser yet." : "Saved views are unavailable in this browser."}</p>}
  </details>;
}
