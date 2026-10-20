"use client";

import { useEffect, useRef, useState } from "react";
import {
  houseReadinessInput,
  type HouseResult,
} from "../lib/house-result";
import { parseHousePublicationWithCatalog, type HousePublicationWithCatalog } from "../lib/house-catalog";
import { houseSelectionForResult, houseSelectionUrl } from "../lib/house-selection";
import HouseControls from "./HouseControls";
import { validateChamberReadiness } from "../lib/chamber";
import {
  COLORS,
  chamberOutlook,
  competitiveOrder,
  outcomeColor,
  probability,
  raceRating,
  STATE_NAMES,
} from "../lib/presentation";
import DistrictMap from "./DistrictMap";
import HouseRibbon from "./HouseRibbon";
import ChamberControl from "./ChamberControl";
import RaceListToggle from "./RaceListToggle";

function HouseFooter() {
  return (
    <footer className="footer">
      <span className="footer-brand">ballotmancy</span>
      <span>
        Independent research. Visible assumptions.
        <br />
        House research preview
      </span>
    </footer>
  );
}

export default function HouseExplorer({ onReturn }: { onReturn: () => void }) {
  const [publication, setPublication] = useState<HousePublicationWithCatalog | null>(null);
  const previous = useRef<HousePublicationWithCatalog | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(true);
  const [request, setRequest] = useState<{ id: string | null; navigation: "initial" | "history" | "push"; attempt: number } | null>(null);
  useEffect(() => {
    const restore = (navigation: "initial" | "history") => {
      const params = new URLSearchParams(location.search);
      if (params.get("chamber") !== "house") return;
      const id = params.get("run");
      if (navigation === "history" && previous.current?.status === "available" && id === previous.current.result.run.id) {
        // Also cancel any pending selection if Back returns to the displayed run.
        setRequest(null);
        setError("");
        setBusy(false);
        return;
      }
      setRequest({ id, navigation, attempt: Date.now() });
    };
    restore("initial");
    const onHistory = () => restore("history");
    window.addEventListener("popstate", onHistory);
    return () => window.removeEventListener("popstate", onHistory);
  }, []);
  useEffect(() => {
    if (!request) return;
    const controller = new AbortController();
    setError("");
    setBusy(true);
    const query = request.id ? `?${new URLSearchParams({ run: request.id })}` : "";
    fetch(`/api/chambers/house${query}`, { signal: controller.signal, cache: "no-store" })
      .then(async (response) => {
        if (!response.ok) throw Error(response.status === 400 ? "This model selection is not available in the published collection." : "The House view is temporarily unavailable.");
        const value = parseHousePublicationWithCatalog(await response.json());
        if (!value) throw Error("The House publication could not be verified.");
        if (value.status === "available") {
          if (request.id && value.result.run.id !== request.id) throw Error("The returned result does not match the requested model.");
          if (!validateChamberReadiness(houseReadinessInput(value.result, value.geography)).canShowRaces) throw Error("The House results could not be verified.");
        }
        if (controller.signal.aborted) return;
        if (value.status === "available") {
          const prior = request.navigation === "push" && previous.current?.status === "available" ? previous.current.result : null;
          const url = houseSelectionForResult(value.result, prior, location.search);
          if (request.navigation === "push") history.pushState(null, "", url);
          else history.replaceState(null, "", url);
        }
        previous.current = value;
        setPublication(value);
        setBusy(false);
      })
      .catch((reason) => {
        if (!controller.signal.aborted) { setError(reason.message); setBusy(false); }
      });
    return () => controller.abort();
  }, [request]);
  const retry = () => setRequest((current) => ({ id: current?.id ?? null, navigation: current?.navigation ?? "initial", attempt: (current?.attempt ?? 0) + 1 }));
  const choose = (id: string) => setRequest({ id, navigation: "push", attempt: Date.now() });
  if (!publication)
    return (
      <main className="load-page" aria-busy={busy}>
        {error ? (
          <div className="error-box">
            <h1>House view unavailable</h1>
            <p>{error}</p>
            <button onClick={retry}>Try again</button>
            {request?.id && <button onClick={() => setRequest({ id: null, navigation: "initial", attempt: Date.now() })}>Load published default</button>}
            <button onClick={onReturn}>Explore the Senate</button>
          </div>
        ) : <p role="status">Preparing the House perspective…</p>}
      </main>
    );
  if (publication.status === "available")
    return <HouseResults publication={publication} busy={busy} requestedId={request?.id ?? null} error={error} onChoose={choose} onRetry={retry} />;
  return (
    <main>
      <section className="hero house-preparation-hero">
        <div>
          <div className="eyebrow">United States House · In preparation</div>
          <h1>
            The House,
            <br />
            in preparation.
          </h1>
          <p>
            A district-by-district perspective is taking shape. The first
            release will compare predictions for past elections with the actual
            winners. A current forecast will follow when it is ready.
          </p>
        </div>
        <div className="edition">
          THE NEXT PERSPECTIVE<strong>435 seats · 218 for a majority</strong>
          Results have not been published yet.
        </div>
      </section>
      <section
        className="house-preparation"
        id="races"
        aria-labelledby="house-preparation-title"
      >
        <div className="house-preparation-intro">
          <div className="eyebrow">Same page. A wider view.</div>
          <h2 id="house-preparation-title">Every district has a place.</h2>
          <p>
            Switch between the House and Senate here. Each view shows the races,
            the model assumptions, and how much the results could vary.
          </p>
          <button className="house-return" onClick={onReturn}>
            Explore the Senate <span aria-hidden="true">→</span>
          </button>
        </div>
        <div className="house-preparation-features">
          <article>
            <span aria-hidden="true">01</span>
            <h3>A path to a majority</h3>
            <p>
              A ribbon of districts ordered from strongest Democratic to
              strongest Republican chances, above the map.
            </p>
          </article>
          <article>
            <span aria-hidden="true">02</span>
            <h3>Zoom in on your district</h3>
            <p>
              An election-specific district map, with state focus and race
              details that follow your selection.
            </p>
          </article>
          <article>
            <span aria-hidden="true">03</span>
            <h3>See how the model did</h3>
            <p>
              Historical results with an at-a-glance scorecard and clear
              coverage of which races were tested.
            </p>
          </article>
        </div>
      </section>
      <section className="methodology" id="methodology">
        <div>
          <div className="eyebrow">Before the first release</div>
          <h2>Evidence comes first.</h2>
        </div>
        <div className="method-content">
          <div>
            <h3>Historical results first</h3>
            <p>
              The initial research covers races in past elections that meet the
              study’s inclusion rules. Included and excluded districts will be
              listed. These historical tests will be labeled separately from a
              current House forecast.
            </p>
          </div>
          <div>
            <h3>Coverage stays visible</h3>
            <p>
              Results for some districts can show how the model did in those
              races. Estimating who wins the House needs a complete forecast
              that allows districts to shift together—for example, in a national
              political swing. A tipping point identifies a race at the seat
              threshold for a majority.
            </p>
          </div>
        </div>
      </section>
      <HouseFooter />
    </main>
  );
}

function HouseResults({ publication, busy, requestedId, error, onChoose, onRetry }: {
  publication: Extract<HousePublicationWithCatalog, { status: "available" }>;
  busy: boolean;
  requestedId: string | null;
  error: string;
  onChoose: (id: string) => void;
  onRetry: () => void;
}) {
  const { result, geography } = publication;
  const readiness = validateChamberReadiness(
    houseReadinessInput(result, geography),
  );
  const first = [...result.races].sort(competitiveOrder)[0];
  const [selectedId, setSelectedId] = useState(first.id);
  const [query, setQuery] = useState("");
  const [expanded, setExpanded] = useState(false);
  const detail = useRef<HTMLElement>(null);
  const [mapView, setMapView] = useState<"forecast" | "actual">("forecast");
  const selected = result.races.find((race) => race.id === selectedId) || first;
  const scenario = readiness.canShowControl
    ? result.chamberProof?.scenarios[0]
    : undefined;
  const outlook = chamberOutlook(scenario?.control);
  useEffect(() => {
    detail.current?.scrollTo({ top: 0 });
  }, [selected.id]);
  useEffect(() => {
    const restore = () => {
      const params = new URLSearchParams(location.search);
      if (params.get("chamber") !== "house") return;
      setSelectedId(
        result.races.find((race) => race.id === params.get("race"))?.id ||
          first.id,
      );
      setMapView(
        result.run.historical && params.get("view") === "actual"
          ? "actual"
          : "forecast",
      );
    };
    restore();
    window.addEventListener("popstate", restore);
    return () => window.removeEventListener("popstate", restore);
  }, [result, first.id]);
  function select(id: string, view = mapView, focusDetails = true) {
    history.pushState(null, "", houseSelectionUrl(result, id, view));
    setSelectedId(id);
    setMapView(view);
    if (focusDetails && window.matchMedia("(max-width: 800px)").matches)
      requestAnimationFrame(() => {
        document
          .getElementById("house-detail-title")
          ?.focus({ preventScroll: true });
        document.getElementById("house-race-detail")?.scrollIntoView({
          behavior: window.matchMedia("(prefers-reduced-motion: reduce)")
            .matches
            ? "instant"
            : "smooth",
          block: "start",
        });
      });
  }
  const shown = [...result.races]
    .filter((race) =>
      `${race.district.districtKey} ${STATE_NAMES[race.state]} ${race.outcomes.map((o) => o.name).join(" ")}`
        .toLowerCase()
        .includes(query.toLowerCase().trim()),
    )
    .sort(competitiveOrder);
  useEffect(() => { setExpanded(false); setQuery(""); }, [result.run.id]);
  const visible = expanded ? shown : shown.slice(0, 10);
  const tipping = readiness.canShowTipping
    ? result.tippingProof?.scenarios.find((s) => s.id === scenario?.id)
    : undefined;
  return (
    <main>
      <section className="hero">
        <div>
          <div className="eyebrow">
            {result.run.year} United States House · {result.races.length}{" "}
            modeled districts
          </div>
          <h1>
            The{" "}
            <span
              className="chamber-name"
              data-chamber-outlook={outlook.party}
              style={{ color: outlook.color }}
              title={outlook.label}
            >
              House
            </span>
            ,
            <br />
            in perspective.
          </h1>
          <p>
            {result.run.label}.{" "}
            {result.run.historical
              ? "Historical model performance, with the tested races and source limitations in view."
              : "Explore the districts and the chamber together."}
          </p>
        </div>
        <div className="edition">
          {result.run.historical
            ? "THE HISTORICAL EDITION"
            : "THE RESEARCH EDITION"}
          <strong>
            {result.metadata.full_chamber
              ? "Full chamber forecast"
              : "Included districts only"}
          </strong>
          {result.races.length} of 435 districts
        </div>
      </section>
      <HouseControls result={result} catalog={publication.catalog} busy={busy} requestedId={requestedId} error={error} onSelect={onChoose} onRetry={onRetry} />
      {result.run.historical && <HouseScoreboard result={result} />}
      <HouseRibbon
        result={result}
        selectedId={selected.id}
        onSelect={select}
        showMajorityPosition={readiness.canShowControl}
      />
      {tipping && (
        <div className="house-tipping-summary">
          {(["D", "R"] as const).map((party) => {
            const [id, frequency] = Object.entries(
              tipping.frequencies[party],
            ).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))[0];
            return (
              <p key={party}>
                {party === "D" ? "Democratic" : "Republican"} majority
                threshold:{" "}
                <button onClick={() => select(id)}>
                  {result.races.find((r) => r.id === id)!.district.districtKey}
                </button>{" "}
                appears most often ({probability(frequency)}) in the saved
                simulations, which model districts together rather than
                independently.
              </p>
            );
          })}
          <p className="small">
            This percentage counts simulations where the district occupies the
            majority threshold in the ordered races, including simulations where
            its party falls short. It is not the chance that changing this one
            result would change who controls the House.
          </p>
        </div>
      )}
      <div className="house-results-layout">
        <div>
          <section className="map-card" aria-label="House map">
            <div className="map-head">
              <div>
                <h2>District by district</h2>
                <p className="small">
                  {result.races.length} modeled districts. Unmodeled districts
                  remain gray.
                </p>
              </div>
              {result.run.historical && (
                <div className="segmented" role="group" aria-label="Map view">
                  {(["forecast", "actual"] as const).map((view) => (
                    <button
                      key={view}
                      aria-pressed={view === mapView}
                      onClick={() => select(selected.id, view, false)}
                    >
                      {view === "forecast" ? "Forecast" : "Actual winners"}
                    </button>
                  ))}
                </div>
              )}
            </div>
            {geography ? (
              <DistrictMap
                races={result.races.map((race) => ({
                  race,
                  districtKey: race.district.districtKey,
                }))}
                identity={{
                  cycle: result.run.year,
                  mapVintage: result.metadata.mapVintage,
                }}
                geography={geography}
                selectedRaceId={selected.id}
                onSelect={select}
                mapView={mapView}
              />
            ) : (
              <div className="district-map-unavailable">
                <h3>District map unavailable</h3>
                <p>District boundaries must be checked
                against this election’s map before results can be placed on it.
                Explore the races below.</p>
              </div>
            )}
            <p className="small">
              D means Democratic; R means Republican. Safe means the leading
              candidate has more than a 90% chance to win; Likely means more
              than 60% up to 90%; Tossup means neither candidate exceeds 60%.
              These are model estimates, not guarantees.
            </p>
            <div className="map-legend">
              {(mapView === "actual"
                ? [
                    { label: "Democratic winner", color: COLORS.D },
                    { label: "Republican winner", color: COLORS.R },
                  ]
                : [
                    { label: "Safe D", color: COLORS.D },
                    { label: "Likely D", color: COLORS.likelyD },
                    { label: "Tossup", color: COLORS.competitive },
                    { label: "Likely R", color: COLORS.likelyR },
                    { label: "Safe R", color: COLORS.R },
                  ]
              ).map((item) => (
                <span key={item.label}>
                  <i className="dot" style={{ color: item.color }} />
                  {item.label}
                </span>
              ))}
            </div>
          </section>
          {scenario ? (
            <ChamberControl scenario={scenario} chamber="house" />
          ) : (
            <section
              className="control-card house-control-unavailable"
              aria-label="House control availability"
            >
              <h2 className="section-title">Who could control the chamber?</h2>
              <p>
                Not available for this result. Estimating who wins the House
                needs all 435 districts and simulations that allow them to shift
                together, rather than treating every race as independent.
              </p>
              <span className="small">435 seats · 218 for a majority</span>
            </section>
          )}
          <section
            id="races"
            className="race-section"
            aria-labelledby="house-races-title"
          >
            <div className="section-top">
              <h2 id="house-races-title" className="section-title">
                The districts
              </h2>
              <span className="small">
                {visible.length} of {shown.length}{" "}
                {query.trim() ? "matching races" : "modeled races"}
              </span>
            </div>
            <div className="race-tools">
              <input
                aria-label="Search districts or candidates"
                placeholder="Find a district, state, or candidate"
                value={query}
                onChange={(event) => {
                  setQuery(event.target.value);
                  setExpanded(false);
                }}
              />
            </div>
            <div className="house-race-list" id="house-race-list">
              {visible.map((race) => (
                <button
                  key={race.id}
                  aria-pressed={race.id === selected.id}
                  onClick={() => select(race.id)}
                >
                  <strong>{race.district.districtKey}</strong>
                  <span>{race.outcomes.map((o) => o.name).join(" / ")}</span>
                  <b>{raceRating(race)}</b>
                </button>
              ))}
            </div>
            <RaceListToggle
              expanded={expanded}
              total={shown.length}
              controls="house-race-list"
              onToggle={() => setExpanded(!expanded)}
            />
            {!shown.length && (
              <p>
                No matching districts. Try another district, state, or
                candidate.
              </p>
            )}
          </section>
        </div>
        <aside
          ref={detail}
          tabIndex={0}
          id="house-race-detail"
          className="detail"
          aria-label="Selected House race detail"
          aria-live="polite"
        >
          <div className="eyebrow">
            District in focus · {raceRating(selected)}
          </div>
          <h2 id="house-detail-title" tabIndex={-1}>
            {selected.district.districtKey}
          </h2>
          <p className="small">
            {STATE_NAMES[selected.state]} · {selected.electionLabel}
          </p>
          <h3 className="detail-label">Chance of winning</h3>
          {selected.outcomes.map((outcome) => (
            <div className="candidate" key={outcome.id}>
              <div className="candidate-top">
                <div>
                  <div className="candidate-name">{outcome.name}</div>
                  <span className="party-label">
                    {outcome.party === "D" ? "Democratic" : "Republican"}
                  </span>
                </div>
                <strong
                  className="candidate-prob"
                  style={{ color: outcomeColor(outcome) }}
                >
                  {probability(outcome.winProbability)}
                </strong>
              </div>
              <div className="candidate-track">
                <span
                  style={{
                    width: `${outcome.winProbability * 100}%`,
                    background: outcomeColor(outcome),
                  }}
                />
              </div>
              {outcome.share && (
                <p className="detail-note">
                  Average share of Democratic and Republican votes{" "}
                  {probability(outcome.share.mean)} · Middle 90% of modeled
                  shares {probability(outcome.share.quantiles.p05)}–
                  {probability(outcome.share.quantiles.p95)}
                </p>
              )}
            </div>
          ))}
          {selected.observed && (
            <p className="observed-result">
              Actual winner: <strong>{selected.observed.winnerName}</strong>
            </p>
          )}
          <p className="detail-note">
            Winning chances describe how often a candidate wins in the model;
            they are not vote percentages. Any vote shares shown count
            Democratic and Republican votes only, excluding other candidates.
            Missing districts have no forecast.
          </p>
          {selected.notes.map((note, index) => (
            <p className="detail-note" key={index}>
              {note}
            </p>
          ))}
        </aside>
      </div>
      <section className="methodology" id="methodology">
        <div>
          <div className="eyebrow">Read the evidence</div>
          <h2>Coverage and context.</h2>
        </div>
        <div className="method-content">
          <div>
            <h3>How to read this result</h3>
            <p>
              {result.run.informationMode === "strict_asof"
                ? "This result uses evidence reconstructed for its stated forecast date, with later information excluded under its documented rules."
                : "Some historical inputs were assembled later as substitutes for information about the past (retrospective proxies). This comparison does not establish how accurately the model could have forecast using only information available on that date."}
            </p>
            {result.limitations.map((note, index) => (
              <p key={index}>{note}</p>
            ))}
          </div>
          <div>
            <h3>Which races are included?</h3>
            <p>
              {result.races.length} of 435 districts have results. Each excluded
              district has an explicit reason.
            </p>
            <details>
              <summary>
                View excluded districts (
                {result.coverage.filter((row) => !row.eligible).length})
              </summary>
              <ul className="house-exclusions">
                {result.coverage
                  .filter((row) => !row.eligible)
                  .map((row) => (
                    <li key={row.raceId}>
                      <b>{row.districtKey}</b>: {row.exclusionReason}
                    </li>
                  ))}
              </ul>
            </details>
            <details>
              <summary>Publication fingerprints</summary>
              <p>
                These IDs identify the model choices and the exact saved result.
                SHA256 fingerprints are long codes used to check that a file has
                not changed; they do not measure forecast quality.
              </p>
              <p className="house-digest">
                Configuration: {result.run.configurationId}
                <br />
                {result.run.configurationSha256}
                <br />
                Result: {publication.resultSha256}
              </p>
            </details>
          </div>
        </div>
      </section>
      <HouseFooter />
    </main>
  );
}

function HouseScoreboard({ result }: { result: HouseResult }) {
  const observed = result.races.filter((race) => race.observed);
  const decisive = observed.filter(
    (race) =>
      Math.abs(
        race.outcomes[0].winProbability - race.outcomes[1].winProbability,
      ) > 1e-12,
  );
  const correct = decisive.filter(
    (race) =>
      [...race.outcomes].sort((a, b) => b.winProbability - a.winProbability)[0]
        .id === race.observed!.outcomeId,
  ).length;
  const brier = observed.length
    ? observed.reduce(
        (sum, race) =>
          sum +
          race.outcomes.reduce(
            (score, outcome) =>
              score +
              (outcome.winProbability -
                Number(outcome.id === race.observed!.outcomeId)) **
                2,
            0,
          ),
        0,
      ) / observed.length
    : null;
  return (
    <section
      className="historical-scoreboard"
      aria-labelledby="house-scoreboard-title"
    >
      <div className="scoreboard-heading">
        <div>
          <div className="eyebrow">
            {result.run.year} historical reconstruction
          </div>
          <h2 className="section-title" id="house-scoreboard-title">
            How this model did
          </h2>
          <p>
            {result.run.label} ·{" "}
            {result.metadata.full_chamber
              ? "Full chamber"
              : "Eligible races only"}
          </p>
        </div>
      </div>
      <div className="scoreboard-cards house-scoreboard-cards">
        <article className="scoreboard-card">
          <h3>Favorites that won</h3>
          <div className="scoreboard-value">
            {decisive.length ? correct : "—"}
            <span> / {decisive.length}</span>
          </div>
          <p>
            A favorite is the candidate with the higher modeled winning chance.
            Counts only races with a known winner and one clear favorite.{" "}
            {observed.length - decisive.length} exact ties excluded.
          </p>
        </article>
        <article className="scoreboard-card">
          <h3>Probability error</h3>
          <div className="scoreboard-value">
            {brier === null ? "—" : brier.toFixed(3)}
          </div>
          <p>
            Brier score, 0–2 scale. Lower is better; 0 is perfect. It compares
            the assigned probabilities with what happened, penalizing confident
            wrong predictions more. Uses {observed.length} races with actual results.
          </p>
        </article>
        <article className="scoreboard-card">
          <h3>Tested coverage</h3>
          <div className="scoreboard-value">
            {observed.length}
            <span> / 435</span>
          </div>
          <p>
            Districts with forecasts and known actual winners. These scores
            describe the tested races; they do not establish accuracy for
            districts left out of the study.
          </p>
        </article>
      </div>
    </section>
  );
}
