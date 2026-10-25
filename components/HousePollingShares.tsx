"use client";

import { useEffect, useState } from "react";
import type { HouseReviewedCatalogEntry } from "../lib/house-reviewed-catalog";
import type { HousePollingShareResult } from "../lib/house-polling-shares";
import { loadHousePollingShares } from "../lib/house-reviewed-client";
import { probability } from "../lib/presentation";

/** Vote share is a separate saved quantity, never inferred from group-win chances. */
export default function HousePollingShares({ entry, raceId, eligible }: {
  entry: HouseReviewedCatalogEntry; raceId: string; eligible: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [result, setResult] = useState<HousePollingShareResult | null>(null);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    if (!open || !eligible || result) return;
    const controller = new AbortController();
    setError("");
    loadHousePollingShares(entry, controller.signal).then(value => {
      if (!controller.signal.aborted) setResult(value);
    }).catch(reason => {
      if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "The saved vote shares could not be loaded.");
    });
    return () => controller.abort();
  }, [entry, open, eligible, result, attempt]);
  const race = result?.races.find(candidate => candidate.id === raceId);
  return <details className="house-polling-shares" open={open} onToggle={event => setOpen(event.currentTarget.open)}>
    <summary>Vote shares with polling</summary>
    <p>These shares count Democratic and Republican votes only, excluding Other candidates and write-ins. They describe vote totals within that pair, not the chance of winning the district.</p>
    {!eligible ? <p>This district is outside the two-party vote-share study. Its group winning chances are still shown above.</p>
      : error ? <div role="alert"><p>{error}</p><button onClick={() => setAttempt(value => value + 1)}>Retry vote shares</button></div>
      : !result ? <p role="status">Loading this model’s saved vote shares…</p>
      : !race ? <p role="alert">This district’s vote-share result is unavailable in the saved result.</p>
      : <>
        <p className="small">Same model and election year as the map · Polling included</p>
        <dl>{race.outcomes.map(outcome => <div key={outcome.id}>
          <dt>{outcome.name} ({outcome.party})</dt>
          <dd>{outcome.share ? <><strong>{probability(outcome.share.mean)}</strong> average share<br />Middle 90% of modeled shares: {probability(outcome.share.quantiles.p05)}–{probability(outcome.share.quantiles.p95)}</> : "Vote shares unavailable"}</dd>
        </div>)}</dl>
      </>}
  </details>;
}
