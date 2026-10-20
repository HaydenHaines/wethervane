"use client";

import { useEffect, useMemo, useState } from "react";
import { geoAlbersUsa, geoPath } from "d3-geo";
import type { FeatureCollection, Feature, Geometry } from "geojson";
import {
  STATE_NAMES,
  COLORS,
  partyName,
  partyColor,
  percent,
  probability,
  dateLabel,
  candidateShortName,
  raceRating,
  raceColor,
  raceTextColor,
} from "../lib/presentation";

import type {
  ForecastSnapshot as Snapshot,
  RaceForecast as Race,
} from "../lib/site-types";
import TippingRibbon from "./TippingRibbon";
import type { TippingSnapshot } from "../lib/tipping-types";

type ModelKey = "polling" | "fundamentals";
type Geo = FeatureCollection<
  Geometry,
  { state_abbr: string; state_name: string }
>;
const SCENARIOS = [
  { id: "campaign_pledge_unaligned", label: "Opposition unaligned" },
  {
    id: "independent_opposition_caucuses_D",
    label: "Opposition independents → D",
  },
  {
    id: "independent_opposition_caucuses_R",
    label: "Opposition independents → R",
  },
];
const closest = (a: Race, b: Race) =>
  Math.abs(a.firstWinProbability - 0.5) - Math.abs(b.firstWinProbability - 0.5);
function raceLabel(race: Race) {
  return /special/i.test(race.id + race.seatId)
    ? "Special election"
    : "Regular election";
}

function tippingMatchesForecast(
  tipping: TippingSnapshot,
  forecast: Snapshot,
): boolean {
  try {
    if (
      tipping.schemaVersion !== 1 ||
      tipping.method !== "caucus_rank_threshold_v1" ||
      tipping.sources.pollingForecastSha256 !==
        forecast.sources.pollingSha256 ||
      tipping.sources.fundamentalsForecastSha256 !==
        forecast.sources.fundamentalsSha256
    )
      return false;
    return (["polling", "fundamentals"] as const).every((key) => {
      const summary = tipping.models[key];
      const expected = [...forecast.models[key].races].sort(
        (a, b) =>
          b.firstShareMean - a.firstShareMean || a.id.localeCompare(b.id),
      );
      if (
        summary.raceOrder.length !== expected.length ||
        !summary.raceOrder.every((id, i) => id === expected[i].id)
      )
        return false;
      return SCENARIOS.every(({ id }) => {
        const scenario =
          summary.scenarios[id as keyof typeof summary.scenarios];
        return [scenario.D, scenario.R].every(
          (caucus) =>
            expected.every(
              (race) =>
                Number.isFinite(caucus.frequencyByRace[race.id]) &&
                caucus.frequencyByRace[race.id] >= 0 &&
                caucus.frequencyByRace[race.id] <= 1,
            ) &&
            (caucus.projectedTippingRaceId === null ||
              expected.some(
                (race) => race.id === caucus.projectedTippingRaceId,
              )),
        );
      });
    });
  } catch {
    return false;
  }
}

export default function SenateExplorer() {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [tipping, setTipping] = useState<TippingSnapshot | null>(null);
  const [tippingFailed, setTippingFailed] = useState(false);
  const [geo, setGeo] = useState<Geo | null>(null);
  const [error, setError] = useState("");
  const [modelKey, setModelKey] = useState<ModelKey>("polling");
  const [scenario, setScenario] = useState(SCENARIOS[0].id);
  const [selectedId, setSelectedId] = useState("");
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState("competitive");
  useEffect(() => {
    let active = true;
    fetch("/data/forecast.json")
      .then((r) => {
        if (!r.ok)
          throw Error("The saved forecast snapshot could not be loaded.");
        return r.json();
      })
      .then((data: Snapshot) => {
        if (active) {
          setSnapshot(data);
          setSelectedId(
            [...data.models.polling.races].sort(closest)[0]?.id || "",
          );
        }
      })
      .catch((e) => {
        if (active) setError(e.message);
      });
    fetch("/data/tipping-points.json")
      .then((r) => {
        if (!r.ok) throw Error("Tipping data unavailable");
        return r.json();
      })
      .then((data: TippingSnapshot) => {
        if (active) setTipping(data);
      })
      .catch(() => {
        if (active) setTippingFailed(true);
      });
    fetch("/states-us.geojson")
      .then((r) => {
        if (!r.ok) throw Error("Map geography unavailable");
        return r.json();
      })
      .then((data) => {
        if (active) setGeo(data);
      })
      .catch(() => {});
    return () => {
      active = false;
    };
  }, []);
  const model = snapshot?.models[modelKey];
  const races = model?.races;
  const selected = races?.find((r) => r.id === selectedId) || races?.[0];
  const control = model?.controls[scenario];
  const tippingValid =
    tipping && snapshot ? tippingMatchesForecast(tipping, snapshot) : false;
  const shownRaces = useMemo(() => {
    const text = query.toLowerCase().trim();
    return [...(races || [])]
      .filter(
        (r) =>
          !text ||
          `${STATE_NAMES[r.state]} ${r.state} ${r.candidates.map((c) => c.name).join(" ")} ${raceLabel(r)}`
            .toLowerCase()
            .includes(text),
      )
      .sort(
        sort === "state"
          ? (a, b) =>
              (STATE_NAMES[a.state] || a.state).localeCompare(
                STATE_NAMES[b.state] || b.state,
              ) || a.id.localeCompare(b.id)
          : closest,
      );
  }, [races, query, sort]);
  const selectRace = (id: string) => {
    setSelectedId(id);
    if (window.matchMedia("(max-width: 800px)").matches) {
      requestAnimationFrame(() => {
        document
          .getElementById("race-detail-title")
          ?.focus({ preventScroll: true });
        document.getElementById("race-detail")?.scrollIntoView({
          behavior: window.matchMedia("(prefers-reduced-motion: reduce)")
            .matches
            ? "instant"
            : "smooth",
          block: "start",
        });
      });
    }
  };
  const selectState = (state: string) => {
    const matches = races?.filter((r) => r.state === state).sort(closest);
    if (matches?.length) selectRace(matches[0].id);
  };
  const stateRaces = selected
    ? races?.filter((r) => r.state === selected.state) || []
    : [];
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
          <span className="preview">
            <i />
            Research preview
          </span>
        </nav>
      </header>
      {error ? (
        <main className="load-page">
          <div className="error-box">
            <h1>Snapshot unavailable</h1>
            <p>{error}</p>
            <button onClick={() => location.reload()}>Try again</button>
          </div>
        </main>
      ) : !snapshot || !model || !selected || !control ? (
        <main className="load-page" aria-busy="true">
          <p>Preparing the Senate perspective…</p>
        </main>
      ) : (
        <main>
          <section className="hero">
            <div>
              <div className="eyebrow">
                2026 United States Senate · {races!.length} elections
              </div>
              <h1>
                The Senate,
                <br />
                in perspective.
              </h1>
              <p>
                A closer look at the races that could shape the chamber.
                <br className="desktop-break" /> Explore two research models,
                their uncertainty, and the assumptions behind them.
              </p>
            </div>
            <div className="edition">
              THE RESEARCH EDITION
              <strong>{dateLabel(snapshot.issuedAt)}</strong>Evidence through{" "}
              {dateLabel(snapshot.cutoffAt)}.
            </div>
          </section>
          <div className="ribbon-overview">
            {" "}
            <div className="toolbar">
              <div
                className="segmented"
                role="group"
                aria-label="Research model"
              >
                <button
                  aria-pressed={modelKey === "polling"}
                  onClick={() => setModelKey("polling")}
                >
                  Polling model
                </button>
                <button
                  aria-pressed={modelKey === "fundamentals"}
                  onClick={() => setModelKey("fundamentals")}
                >
                  Fundamentals model
                </button>
              </div>
              <div className="scenario">
                <label htmlFor="scenario" className="small">
                  Independent alignment
                </label>
                <select
                  id="scenario"
                  value={scenario}
                  onChange={(e) => setScenario(e.target.value)}
                >
                  {SCENARIOS.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            {tipping && tippingValid ? (
              <TippingRibbon
                summary={tipping.models[modelKey]}
                scenario={scenario}
                races={races!}
                selectedId={selected.id}
                onSelect={selectRace}
              />
            ) : (
              <section className="tipping-card">
                <p className="small">
                  {tippingFailed || (tipping && !tippingValid)
                    ? "The tipping-point snapshot is unavailable. Map and race summaries remain available below."
                    : "Loading the path to a majority…"}
                </p>
              </section>
            )}
          </div>
          <section className="workspace" aria-label="Senate race explorer">
            <div className="workspace-main">
              <div className="map-card">
                <div className="map-head">
                  <div>
                    <h2>A nation of distinct races</h2>
                    <div className="small">
                      Select a state to explore its election.
                    </div>
                  </div>
                  <span className="small">2026 Senate</span>
                </div>
                {geo ? (
                  <SenateMap
                    geo={geo}
                    races={races!}
                    selected={selected}
                    onSelect={selectState}
                  />
                ) : (
                  <div className="map-loading">
                    Map unavailable. Explore every race in the list below.
                  </div>
                )}
                <div className="map-legend">
                  {[
                    { label: "Safe D", color: COLORS.D },
                    { label: "Likely D", color: COLORS.likelyD },
                    { label: "Safe R", color: COLORS.R },
                    { label: "Likely R", color: COLORS.likelyR },
                    { label: "Safe I", color: COLORS.I },
                    { label: "Likely I", color: COLORS.likelyI },
                    { label: "Tossup", color: COLORS.competitive },
                  ].map((x) => (
                    <span key={x.label}>
                      <i className="dot" style={{ color: x.color }} />
                      {x.label}
                    </span>
                  ))}
                </div>
                <div className="map-foot">
                  <span>
                    Safe ≥98% · Likely 65–&lt;98% · Tossup &lt;65% leader win
                    chance.
                  </span>
                  <span>Gray: no Senate election · Area ≠ votes</span>
                </div>
              </div>
              <section className="control-card" aria-labelledby="control-title">
                <div className="section-top">
                  <div>
                    <h2 id="control-title" className="section-title">
                      Who could control the chamber?
                    </h2>
                    <div className="small">
                      {modelKey === "polling" ? "Polling" : "Fundamentals"}{" "}
                      model · chamber scenario
                    </div>
                  </div>
                </div>
                <div className="control-probs" aria-live="polite">
                  {[
                    {
                      label: "Democratic control",
                      p: control.democraticMajorityProbability,
                      color: COLORS.D,
                    },
                    {
                      label: "Republican control",
                      p: control.republicanMajorityProbability,
                      color: COLORS.R,
                    },
                    {
                      label: "Unresolved alignment",
                      p: control.unresolvedOrganizingProbability,
                      color: COLORS.unresolved,
                    },
                  ].map((x) => (
                    <div key={x.label} style={{ color: x.color }}>
                      <span className="prob-label">
                        <span className="dot" />
                        {x.label}
                      </span>
                      <span className="prob-number">{probability(x.p)}</span>
                    </div>
                  ))}
                </div>
                <div className="prob-track" aria-hidden="true">
                  <span
                    style={{
                      width: `${control.democraticMajorityProbability * 100}%`,
                      background: COLORS.D,
                    }}
                  />
                  <span
                    style={{
                      width: `${control.republicanMajorityProbability * 100}%`,
                      background: COLORS.R,
                    }}
                  />
                  <span
                    style={{
                      width: `${control.unresolvedOrganizingProbability * 100}%`,
                      background: COLORS.unresolved,
                    }}
                  />
                </div>
                <p className="control-note">
                  {control.description} Unresolved means control depends on
                  organizing arrangements; it does not mean an undecided
                  election.
                </p>
              </section>
              <section
                className="race-section"
                id="races"
                aria-labelledby="races-title"
              >
                <div className="section-top">
                  <h2 className="section-title" id="races-title">
                    The races
                  </h2>
                  <span className="small">
                    {shownRaces.length} of {races!.length} elections
                  </span>
                </div>
                <div className="race-tools">
                  <div className="search-wrap">
                    <span className="search-icon" aria-hidden="true">
                      ⌕
                    </span>
                    <input
                      aria-label="Search states or candidates"
                      placeholder="Find a state or candidate"
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                    />
                  </div>
                  <select
                    className="sort-select"
                    aria-label="Sort races"
                    value={sort}
                    onChange={(e) => setSort(e.target.value)}
                  >
                    <option value="competitive">Most competitive</option>
                    <option value="state">State A–Z</option>
                  </select>
                </div>
                <table className="race-table">
                  <thead>
                    <tr>
                      <th scope="col">State / election</th>
                      <th scope="col">Leading contender</th>
                      <th scope="col">Win probability</th>
                      <th scope="col">Outlook</th>
                    </tr>
                  </thead>
                  <tbody>
                    {shownRaces.map((r) => {
                      const first =
                        r.firstWinProbability >= r.secondWinProbability;
                      const leader = r.candidates[first ? 0 : 1];
                      const p = first
                        ? r.firstWinProbability
                        : r.secondWinProbability;
                      return (
                        <tr
                          key={r.id}
                          className={selected.id === r.id ? "selected" : ""}
                        >
                          <td>
                            <button
                              className="race-name"
                              aria-pressed={selected.id === r.id}
                              onClick={() => selectRace(r.id)}
                            >
                              {STATE_NAMES[r.state] || r.state}
                              <span className="race-sub">{raceLabel(r)}</span>
                            </button>
                          </td>
                          <td>
                            <span style={{ color: partyColor(leader.party) }}>
                              {leader.name}
                            </span>
                            <span className="race-sub">
                              {partyName(leader.party)}
                            </span>
                          </td>
                          <td className="chance-cell">
                            <span className="win-label">{probability(p)}</span>
                            <div className="mini-prob" aria-hidden="true">
                              <span
                                style={{
                                  width: `${r.firstWinProbability * 100}%`,
                                  background: partyColor(r.candidates[0].party),
                                }}
                              />
                              <span
                                style={{
                                  width: `${r.secondWinProbability * 100}%`,
                                  background: partyColor(r.candidates[1].party),
                                }}
                              />
                            </div>
                          </td>
                          <td>
                            <span
                              className="small"
                              style={{ color: raceTextColor(r) }}
                            >
                              {raceRating(r)}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
                {!shownRaces.length && (
                  <p className="empty">
                    No matching races. Try another state or candidate.
                  </p>
                )}
              </section>
            </div>
            <aside
              id="race-detail"
              className="detail"
              aria-label="Selected race detail"
              aria-live="polite"
            >
              <div className="detail-kicker">
                <span className="eyebrow" style={{ margin: 0 }}>
                  Race in focus
                </span>
                <span
                  className="race-tag"
                  style={{
                    color: raceTextColor(selected),
                    background: `${raceColor(selected)}22`,
                  }}
                >
                  {raceRating(selected)}
                </span>
              </div>
              <h2 id="race-detail-title" tabIndex={-1}>
                {STATE_NAMES[selected.state] || selected.state}
              </h2>
              <div className="small">U.S. Senate · {raceLabel(selected)}</div>
              {stateRaces.length > 1 && (
                <>
                  <label className="sr-only" htmlFor="seat-select">
                    Choose election in {STATE_NAMES[selected.state]}
                  </label>
                  <select
                    className="race-selector"
                    id="seat-select"
                    value={selected.id}
                    onChange={(e) => selectRace(e.target.value)}
                  >
                    {stateRaces.map((r) => (
                      <option key={r.id} value={r.id}>
                        {raceLabel(r)} ·{" "}
                        {r.candidates
                          .map((c) => candidateShortName(c.name))
                          .join(" / ")}
                      </option>
                    ))}
                  </select>
                </>
              )}
              <hr className="detail-divider" />
              <h3 className="detail-label">Chance of winning</h3>
              <div className="candidate-grid">
                {selected.candidates.map((c, i) => {
                  const p =
                    i === 0
                      ? selected.firstWinProbability
                      : selected.secondWinProbability;
                  const color = partyColor(c.party);
                  return (
                    <div className="candidate" key={c.id}>
                      <div className="candidate-top">
                        <div>
                          <div className="candidate-name">{c.name}</div>
                          <div className="party-label">
                            {partyName(c.party)}
                            {c.incumbent ? " · Incumbent" : ""}
                          </div>
                        </div>
                        <strong className="candidate-prob" style={{ color }}>
                          {probability(p)}
                        </strong>
                      </div>
                      <div className="candidate-track">
                        <span
                          style={{ width: `${p * 100}%`, background: color }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
              <hr className="detail-divider" />
              <h3 className="detail-label">Principal contender vote share</h3>
              <div className="intervals-grid">
                {selected.candidates.map((c, i) => {
                  const mean =
                    i === 0
                      ? selected.firstShareMean
                      : 1 - selected.firstShareMean;
                  const lo =
                    i === 0
                      ? selected.shareQuantiles.p05
                      : 1 - selected.shareQuantiles.p95;
                  const hi =
                    i === 0
                      ? selected.shareQuantiles.p95
                      : 1 - selected.shareQuantiles.p05;
                  const color = partyColor(c.party);
                  return (
                    <div className="interval-row" key={c.id}>
                      <div className="interval-label">
                        <span>{candidateShortName(c.name)}</span>
                        <strong>{percent(mean)}</strong>
                      </div>
                      <div
                        className="interval-line"
                        aria-label={`${c.name}: mean ${percent(mean)}, 90% interval ${percent(lo)} to ${percent(hi)}`}
                      >
                        <span
                          className="interval-range"
                          style={{
                            left: `${lo * 100}%`,
                            width: `${(hi - lo) * 100}%`,
                            background: color,
                          }}
                        />
                        <span
                          className="interval-mean"
                          style={{ left: `${mean * 100}%`, background: color }}
                        />
                      </div>
                      <p className="detail-note" style={{ margin: "7px 0 0" }}>
                        90% interval: {percent(lo)}–{percent(hi)}
                      </p>
                    </div>
                  );
                })}
              </div>
              <p className="detail-note">
                Win probabilities of 0% or 100% reflect this saved simulation,
                not certainty. Share is conditional on votes for these two
                contenders. The interval describes modeled vote share, not the
                chance of winning.
              </p>
              <hr className="detail-divider" />
              <div className="metrics">
                <div className="metric">
                  <span className="small">Evidence model</span>
                  <strong style={{ fontSize: 19 }}>
                    {modelKey === "polling" ? "Polling" : "Fundamentals"}
                  </strong>
                </div>
                {selected.pollCount !== undefined && (
                  <div className="metric">
                    <span className="small">Admitted observations</span>
                    <strong>{selected.pollCount}</strong>
                  </div>
                )}
              </div>
              {selected.notes.length > 0 && (
                <details className="assumptions">
                  <summary>Race assumptions</summary>
                  {selected.notes.map((note, i) => (
                    <p className="detail-note" key={i}>
                      {note}
                    </p>
                  ))}
                </details>
              )}
              <hr className="detail-divider" />
              <h3 className="detail-label">Official candidate sources</h3>
              <div className="official-links">
                {selected.candidates.map((c) => (
                  <span key={c.id}>
                    {c.sourceUrls.length ? (
                      <a
                        href={c.sourceUrls[0]}
                        target="_blank"
                        rel="noreferrer"
                      >
                        {c.name} ↗
                      </a>
                    ) : (
                      <span>{c.name}: source unavailable</span>
                    )}
                  </span>
                ))}
              </div>
              <p className="source-date">
                Evidence cutoff: {dateLabel(model.cutoffAt)}
                <br />
                Snapshot issued: {dateLabel(model.issuedAt)}. Research output;
                not a live forecast.
              </p>
            </aside>
          </section>
          <section id="methodology" className="methodology">
            <div>
              <div className="eyebrow">A little context goes a long way</div>
              <h2>Read the uncertainty.</h2>
              <p className="small">
                A transparent preview of saved model outputs.
              </p>
              <a
                className="download"
                href="/data/forecast.json"
                download="ballotmancy-senate-snapshot.json"
              >
                Download this snapshot ↓
              </a>
            </div>
            <div className="method-content">
              <div>
                <h3>Two ways of looking at a race</h3>
                <p>
                  Switch between a polling model and a fundamentals model to
                  compare their saved summaries. The controls do not run new
                  inference. These aggregate state models are research previews;
                  their probabilities have not been established as calibrated
                  forecasts. Polling coverage is a limited audited subset, not
                  an exhaustive census.
                </p>
              </div>
              <div>
                <h3>A simpler contest than the real ballot</h3>
                <p>
                  Each race represents two named principal contenders. Share
                  excludes other candidates, and the model assigns no
                  probability to a third-party win. Election rules are
                  approximate, including states with ranked-choice or runoff
                  systems. There are no county forecasts here.
                </p>
              </div>
              <div>
                <h3>Party and control are different</h3>
                <p>
                  Independent contenders remain independent throughout. Chamber
                  control also depends on caucus arrangements and organizing
                  rules. Choose a scenario to see those assumptions. The default
                  leaves opposition independents unaligned and preserves
                  unresolved organizing outcomes.
                </p>
              </div>
              <div>
                <h3>A snapshot with a cutoff</h3>
                <p>
                  Issued {dateLabel(snapshot.issuedAt)}, using evidence
                  available by {dateLabel(snapshot.cutoffAt)}. This page does
                  not update live. Win probability, expected vote share, and
                  share intervals answer different questions; a narrow race can
                  remain deeply uncertain.
                </p>
              </div>
            </div>
          </section>
          <footer className="footer">
            <span className="footer-brand">ballotmancy</span>
            <span>
              Independent research. Visible assumptions.
              <br />
              2026 Senate research preview · {dateLabel(snapshot.issuedAt)}
            </span>
          </footer>
        </main>
      )}
    </div>
  );
}

function SenateMap({
  geo,
  races,
  selected,
  onSelect,
}: {
  geo: Geo;
  races: Race[];
  selected: Race;
  onSelect: (state: string) => void;
}) {
  const projection = useMemo(
    () =>
      geoAlbersUsa().fitExtent(
        [
          [25, 20],
          [935, 570],
        ],
        geo,
      ),
    [geo],
  );
  const path = geoPath(projection);
  const byState = new Map<string, Race[]>();
  for (const race of races)
    byState.set(race.state, [...(byState.get(race.state) || []), race]);
  return (
    <svg
      className="map-svg"
      viewBox="0 0 960 600"
      role="group"
      aria-label="United States Senate races. Use Tab to select a state with an election."
    >
      <title>2026 Senate race map</title>
      {geo.features.map((feature) => {
        const state = feature.properties.state_abbr;
        const matches = byState.get(state);
        const race = matches ? [...matches].sort(closest)[0] : undefined;
        const label = `${feature.properties.state_name}${race ? `, ${matches!.length} election${matches!.length > 1 ? "s" : ""}, ${raceRating(race)}` : ", no Senate election"}`;
        return (
          <path
            key={state}
            d={path(feature as Feature) || undefined}
            fill={race ? raceColor(race) : COLORS.inactive}
            className={`map-state${selected.state === state ? " selected" : ""}`}
            role={race ? "button" : undefined}
            tabIndex={race ? 0 : undefined}
            aria-label={label}
            aria-pressed={race ? selected.state === state : undefined}
            onClick={race ? () => onSelect(state) : undefined}
            onKeyDown={
              race
                ? (e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      onSelect(state);
                    }
                  }
                : undefined
            }
          >
            <title>{label}</title>
          </path>
        );
      })}
      {geo.features.map((feature) => {
        const state = feature.properties.state_abbr;
        const centroid = path.centroid(feature);
        if (
          !Number.isFinite(centroid[0]) ||
          ["RI", "DE", "DC", "MD", "CT", "NJ", "MA"].includes(state)
        )
          return null;
        return (
          <text
            key={state}
            className={`map-label${!byState.has(state) ? " inactive" : raceRating([...byState.get(state)!].sort(closest)[0]) === "Likely" ? " likely" : ""}`}
            x={centroid[0]}
            y={centroid[1]}
            textAnchor="middle"
            dominantBaseline="middle"
          >
            {state}
          </text>
        );
      })}
      <text x="65" y="580" fill="#707975" fontSize="9">
        Alaska and Hawaii shown in insets; scale varies.
      </text>
    </svg>
  );
}
