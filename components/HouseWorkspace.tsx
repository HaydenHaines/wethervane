"use client";

import { useEffect, useRef, useState } from "react";
import { loadHouseLiveAvailability } from "../lib/house-live-client";
import { resolveHouseEdition, type HouseAvailabilityInput } from "../lib/house-live-edition";
import HouseExplorer from "./HouseExplorer";
import HouseLiveWorkspace from "./HouseLiveWorkspace";

/** Keep edition intent in the URL; current publication errors never imply historical data. */
export default function HouseWorkspace({ onReturn }: { onReturn: () => void }) {
  const [search, setSearch] = useState<string | null>(null);
  const [availability, setAvailability] = useState<HouseAvailabilityInput>({ state: "pending" });
  const [attempt, setAttempt] = useState(0);
  const generation = useRef(0);

  useEffect(() => {
    const restore = () => setSearch(location.search);
    restore();
    window.addEventListener("popstate", restore);
    return () => window.removeEventListener("popstate", restore);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const current = ++generation.current;
    setAvailability({ state: "pending" });
    loadHouseLiveAvailability(controller.signal).then(available => {
      if (controller.signal.aborted || current !== generation.current) return;
      // A child may have normalized its URL while this independent check ran.
      setSearch(location.search);
      setAvailability({ state: "response", body: { schema: "HouseLiveAvailabilityV1", available } });
    }).catch(() => {
      if (controller.signal.aborted || current !== generation.current) return;
      setSearch(location.search);
      setAvailability({ state: "error" });
    });
    return () => controller.abort();
  }, [attempt]);

  const resolution = search === null ? null : resolveHouseEdition(search, availability);
  const edition = resolution?.status === "selected" ? resolution.edition : null;
  const currentAvailable = availability.state === "response" &&
    (availability.body as { available: boolean }).available;
  const unpublished = availability.state === "response" && !currentAvailable;
  function select(next: "current" | "historical") {
    if (next === edition) return;
    const url = `/?chamber=house&edition=${next}`;
    history.pushState(null, "", url);
    setSearch(location.search);
  }
  function retry() {
    ++generation.current;
    setAvailability({ state: "pending" });
    setAttempt(value => value + 1);
  }

  return <>
    <div className="house-edition-navigation">
      <div className="chamber-tabs" role="group" aria-label="House election edition">
        <button type="button" aria-pressed={edition === "current"}
          disabled={unpublished && edition !== "current"} onClick={() => select("current")}>Current · 2026</button>
        <button type="button" aria-pressed={edition === "historical"}
          onClick={() => select("historical")}>Historical elections</button>
      </div>
      {unpublished && edition !== "current" ? <p className="small">The 2026 forecast is awaiting publication review. Historical results are available below.</p> : null}
      {availability.state === "error" && edition ? <p className="small" role="status">
        Current forecast availability could not be checked. <button type="button" onClick={retry}>Check again</button>
      </p> : null}
    </div>
    {edition === "current" ? <HouseLiveWorkspace onHistorical={() => select("historical")} onReturn={onReturn} />
      : edition === "historical" ? <HouseExplorer onReturn={onReturn} />
      : resolution?.status === "invalid" || resolution?.status === "error" || resolution?.status === "not_house"
        ? <main className="load-page"><div className="error-box">
          <h1>{resolution.status === "error" ? "House forecast availability could not be checked" : "This House link could not be opened"}</h1>
          <p role="alert">{resolution.reason}</p>
          {resolution.status === "error" ? <button type="button" onClick={retry}>Try again</button> : null}
          <button type="button" onClick={() => select("current")}>Open current House forecast</button>
          <button type="button" onClick={() => select("historical")}>Explore historical House research</button>
          <button type="button" onClick={onReturn}>Explore the Senate</button>
        </div></main>
        : <main className="load-page" role="status">Checking the House forecast…</main>}
  </>;
}
