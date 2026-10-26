"use client";

import { useEffect, useRef, useState } from "react";
import type { ChamberId } from "../lib/chamber";
import SenateWorkspace from "./SenateWorkspace";
import HouseWorkspace from "./HouseWorkspace";
import ElectionNightGuide from "./ElectionNightGuide";

const fromUrl = (): ChamberId =>
  new URLSearchParams(location.search).get("chamber") === "house"
    ? "house"
    : "senate";

export default function ChamberExplorer() {
  const [chamber, setChamber] = useState<ChamberId | null>(null);
  const urls = useRef({ senate: "/", house: "/?chamber=house" });
  const tabs = useRef<Partial<Record<ChamberId, HTMLButtonElement | null>>>({});
  const [scorecardOpen, setScorecardOpen] = useState(false);
  const scorecardButton = useRef<HTMLButtonElement>(null);
  const previousScroll = useRef(0);
  const previousChamber = useRef<ChamberId | null>(null);
  function closeScorecard() {
    setScorecardOpen(false);
    requestAnimationFrame(() => {
      scorecardButton.current?.focus({ preventScroll: true });
      window.scrollTo({ top: previousScroll.current, behavior: "instant" });
    });
  }
  useEffect(() => {
    const restore = () => {
      const next = fromUrl();
      if (previousChamber.current !== next) previousScroll.current = 0;
      previousChamber.current = next;
      urls.current[next] = location.pathname + location.search + location.hash;
      setChamber(next);
    };
    restore();
    window.addEventListener("popstate", restore);
    return () => window.removeEventListener("popstate", restore);
  }, []);

  function select(next: ChamberId) {
    if (next === chamber) return;
    if (scorecardOpen) previousScroll.current = 0;
    previousChamber.current = next;
    if (chamber)
      urls.current[chamber] =
        location.pathname + location.search + location.hash;
    history.pushState(null, "", urls.current[next]);
    setChamber(next);
  }

  return (
    <div className="shell">
      <a className="skip-link" href="#races">
        Skip to the races
      </a>
      <header className="masthead">
        <a className="wordmark" href="/" aria-label="Ballotmancy home">
          <span className="brandmark" aria-hidden="true">
            b.
          </span>
          ballotmancy
        </a>
        <nav className="mastnav" aria-label="Main navigation">
          <a href="#races">The races</a>
          <a href="#methodology">How to read this</a>
          <button type="button" className="night-guide-trigger" ref={scorecardButton}
            aria-expanded={scorecardOpen} aria-controls="election-night-scorecard" disabled={!chamber}
            onClick={() => {
              if (scorecardOpen) closeScorecard();
              else { previousScroll.current = window.scrollY; setScorecardOpen(true); }
            }}>Election night scorecard</button>
          <span className="preview">
            <i />
            Research preview
          </span>
        </nav>
      </header>
      <div className="chamber-navigation">
        <div role="tablist" aria-label="Chamber" className="chamber-tabs">
          {(["senate", "house"] as const).map((id) => (
            <button
              key={id}
              ref={(node) => {
                tabs.current[id] = node;
              }}
              type="button"
              role="tab"
              id={`${id}-tab`}
              aria-controls={`${id}-panel`}
              aria-selected={chamber === id}
              tabIndex={(chamber || "senate") === id ? 0 : -1}
              onClick={() => select(id)}
              onKeyDown={(event) => {
                let next: ChamberId;
                if (event.key === "Home") next = "senate";
                else if (event.key === "End") next = "house";
                else if (["ArrowLeft", "ArrowRight"].includes(event.key))
                  next = id === "senate" ? "house" : "senate";
                else return;
                event.preventDefault();
                select(next);
                tabs.current[next]?.focus();
              }}
            >
              {id === "senate" ? "Senate" : "House"}
            </button>
          ))}
        </div>
        <span className="small">Two chambers. Your perspective.</span>
      </div>
      {scorecardOpen && chamber && <ElectionNightGuide key={chamber} chamber={chamber} onClose={closeScorecard} />}
      {chamber ? (
        <div
          role="tabpanel"
          id={`${chamber}-panel`}
          aria-labelledby={`${chamber}-tab`}
          tabIndex={0}
        >
          {chamber === "senate" ? (
            <SenateWorkspace />
          ) : (
            <HouseWorkspace onReturn={() => select("senate")} />
          )}
        </div>
      ) : (
        <div className="load-page" role="status">
          Preparing your perspective…
        </div>
      )}
    </div>
  );
}
