"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";
import { geoAlbersUsa, geoPath } from "d3-geo";
import type { FeatureCollection, MultiPolygon, Polygon } from "geojson";
import greatLakesDisplay from "../data/house-live/geography/great-lakes-display.json";
import type {
  DistrictFeatureCollection,
  DistrictGeography,
  DistrictGeographyIdentity,
  DistrictRace,
} from "../lib/district-geography";
import {
  joinDistrictGeography,
  parseDistrictKey,
} from "../lib/district-geography";
import { districtForProjection } from "../lib/district-projection";
import {
  COLORS,
  STATE_NAMES,
  leadingOutcome,
  outcomeColor,
  probability,
  raceColor,
  raceRating,
} from "../lib/presentation";

const GREAT_LAKES_STATES = new Set(["IL", "IN", "MI", "MN", "NY", "OH", "PA", "WI"]);
const GREAT_LAKES = greatLakesDisplay as unknown as FeatureCollection<Polygon | MultiPolygon, { name: string }>;

export interface DistrictMapProps {
  races: readonly DistrictRace[];
  identity: DistrictGeographyIdentity;
  geography: DistrictGeography;
  selectedRaceId: string | null;
  onSelect: (raceId: string) => void;
  initialState?: string;
  mapView?: "forecast" | "actual";
}

function districtLabel(key: string): string {
  const parsed = parseDistrictKey(key)!;
  return `${STATE_NAMES[parsed.state] || parsed.state} ${parsed.district === "00" ? "at-large district" : `district ${parsed.district}`}`;
}

export default function DistrictMap({
  races,
  identity,
  geography,
  selectedRaceId,
  onSelect,
  initialState = "",
  mapView = "forecast",
}: DistrictMapProps) {
  const uniqueId = useId();
  const [focusState, setFocusState] = useState(initialState);
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const pathElements = useRef(new Map<string, SVGPathElement>());
  const priorSelection = useRef(selectedRaceId);
  const joined = useMemo(
    () => joinDistrictGeography(races, identity, geography),
    [races, identity, geography],
  );
  const stateOptions = useMemo(
    () => [...new Set([
      ...joined.matches.map((match) => match.race.state),
      ...joined.unmatchedFeatures.map((feature) => parseDistrictKey(feature.properties.districtKey)!.state),
    ])].sort(),
    [joined],
  );
  const visible = useMemo(
    () =>
      joined.matches.filter(
        (match) => !focusState || match.race.state === focusState,
      ),
    [joined, focusState],
  );
  const unmatched = useMemo(
    () =>
      joined.unmatchedFeatures.filter(
        (feature) =>
          !focusState ||
          parseDistrictKey(feature.properties.districtKey)?.state ===
            focusState,
      ),
    [joined, focusState],
  );
  const displayed = useMemo<DistrictFeatureCollection>(
    () => ({
      type: "FeatureCollection",
      features: [...visible.map((match) => match.feature), ...unmatched].map(
        districtForProjection,
      ),
    }),
    [visible, unmatched],
  );
  const projection = useMemo(
    () =>
      displayed.features.length
        ? geoAlbersUsa().fitExtent(
            [
              [22, 20],
              [938, 558],
            ],
            displayed,
          )
        : geoAlbersUsa(),
    [displayed],
  );
  const path = geoPath(projection);
  const showGreatLakes = !focusState || GREAT_LAKES_STATES.has(focusState);
  const projected = new Map(
    displayed.features.map((feature) => [
      feature.properties.districtKey,
      feature,
    ]),
  );
  useEffect(() => {
    if (focusState && !stateOptions.includes(focusState)) setFocusState("");
  }, [focusState, stateOptions]);
  useEffect(() => {
    if (priorSelection.current !== selectedRaceId && focusState) {
      const selected = joined.matches.find(
        (match) => match.race.id === selectedRaceId,
      );
      if (selected && selected.race.state !== focusState)
        setFocusState(selected.race.state);
    }
    priorSelection.current = selectedRaceId;
  }, [selectedRaceId, focusState, joined]);
  useEffect(() => {
    setActiveKey(
      (previous) =>
        visible.find((match) => match.race.id === selectedRaceId)?.identity
          .districtKey ||
        visible.find((match) => match.identity.districtKey === previous)
          ?.identity.districtKey ||
        visible[0]?.identity.districtKey ||
        null,
    );
  }, [visible, selectedRaceId]);

  function moveFocus(key: string, event: React.KeyboardEvent<SVGPathElement>) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      const match = visible.find((entry) => entry.identity.districtKey === key);
      if (match) onSelect(match.race.id);
      return;
    }
    const index = visible.findIndex(
      (match) => match.identity.districtKey === key,
    );
    const nextIndex =
      event.key === "Home"
        ? 0
        : event.key === "End"
          ? visible.length - 1
          : ["ArrowRight", "ArrowDown"].includes(event.key)
            ? (index + 1) % visible.length
            : ["ArrowLeft", "ArrowUp"].includes(event.key)
              ? (index - 1 + visible.length) % visible.length
              : null;
    if (nextIndex === null) return;
    event.preventDefault();
    const next = visible[nextIndex]?.identity.districtKey;
    if (next) {
      setActiveKey(next);
      pathElements.current.get(next)?.focus();
    }
  }

  if (joined.status === "invalid" || joined.status === "unavailable")
    return (
      <section
        className="district-map-unavailable"
        aria-label="District map unavailable"
      >
        <h3>District map unavailable</h3>
        <p>
          Race cycle {identity.cycle} · {identity.mapVintage}
        </p>
        <ul>
          {joined.diagnostics.map((diagnostic, index) => (
            <li key={diagnostic.code + index}>{diagnostic.message}</li>
          ))}
        </ul>
      </section>
    );
  const active = visible.find(
    (match) => match.identity.districtKey === activeKey,
  );
  return (
    <section className="district-map" aria-labelledby={`${uniqueId}-title`}>
      <div className="district-map-toolbar">
        <div>
          <h3 id={`${uniqueId}-title`}>{identity.cycle} House districts</h3>
          <p>District boundaries verified for this election.</p>
        </div>
        <div>
          <label htmlFor={`${uniqueId}-state`}>Focus on a state</label>
          <select
            id={`${uniqueId}-state`}
            value={focusState}
            onChange={(event) => setFocusState(event.target.value)}
          >
            <option value="">All states</option>
            {stateOptions.map((state) => (
              <option key={state} value={state}>
                {STATE_NAMES[state] || state}
              </option>
            ))}
          </select>
        </div>
      </div>
      <p className="district-map-instructions" id={`${uniqueId}-instructions`}>
        Tab to the map, use arrow keys to move between districts, and press
        Enter to select. Focus on a state for more detail. Small districts are
        easier to find in the searchable race list below. Great Lakes water is
        shown as nonselectable water; selectable map shapes represent districts.
      </p>
      {displayed.features.length > 0 ? (
        <svg
          className="map-svg district-map-svg"
          viewBox="0 0 960 590"
          role="group"
          aria-label={`${identity.cycle} House district map${focusState ? `, ${STATE_NAMES[focusState] || focusState}` : ""}`}
          aria-describedby={`${uniqueId}-instructions`}
        >
          <title>
            {identity.cycle} House districts · {identity.mapVintage}
          </title>
          {unmatched.map((feature) => (
            <path
              key={feature.properties.districtKey}
              data-district-key={feature.properties.districtKey}
              data-forecast="unavailable"
              d={
                path(projected.get(feature.properties.districtKey)!) ||
                undefined
              }
              fill={COLORS.inactive}
              stroke="var(--paper, #f7f5ef)"
              strokeWidth="1"
            >
              <title>
                {districtLabel(feature.properties.districtKey)}: not included in this model
              </title>
            </path>
          ))}
          {visible.map((match) => {
            const key = match.identity.districtKey,
              race = match.race;
            const leader = leadingOutcome(race);
            const actual = race.observed
              ? race.outcomes.find(
                  (outcome) => outcome.id === race.observed!.outcomeId,
                )
              : undefined;
            const label = `${districtLabel(key)}, ${race.electionLabel}, ${mapView === "actual" ? (race.observed ? `observed winner ${race.observed.winnerName}` : "observed winner unavailable") : `${raceRating(race)}, ${leader.name}${leader.kind === "group" ? " group outcome" : ""}, ${probability(leader.winProbability)} chance of winning`}`;
            return (
              <path
                key={match.identityKey}
                data-district-key={key}
                data-forecast="available"
                ref={(node) => {
                  if (node) pathElements.current.set(key, node);
                  else pathElements.current.delete(key);
                }}
                className={`map-state district-map-shape${selectedRaceId === race.id ? " selected" : ""}`}
                d={path(projected.get(key)!) || undefined}
                fill={
                  mapView === "actual"
                    ? actual
                      ? outcomeColor(actual)
                      : COLORS.inactive
                    : raceColor(race)
                }
                stroke="var(--paper, #f7f5ef)"
                strokeWidth="1"
                role="button"
                tabIndex={activeKey === key ? 0 : -1}
                aria-label={label}
                aria-pressed={selectedRaceId === race.id}
                onFocus={() => setActiveKey(key)}
                onClick={() => {
                  setActiveKey(key);
                  onSelect(race.id);
                }}
                onKeyDown={(event) => moveFocus(key, event)}
              >
                <title>{label}</title>
              </path>
            );
          })}
          {showGreatLakes && GREAT_LAKES.features.map((lake) => (
            <path
              key={lake.properties.name}
              data-great-lake={lake.properties.name}
              d={path(districtForProjection(lake)) || undefined}
              fill="var(--map-water, #dce9ed)"
              stroke="var(--paper, #f7f5ef)"
              strokeWidth="1"
              pointerEvents="visiblePainted"
              aria-hidden="true"
              focusable="false"
            />
          ))}
          {!focusState && (
            <text x="24" y="580" fill="var(--muted, #707975)" fontSize="10">
              Alaska and Hawaii use insets; geographic scale varies.
            </text>
          )}
        </svg>
      ) : (
        <p className="district-map-empty">
          No matched district geometry is available for this view.
        </p>
      )}
      <div className="district-map-active" aria-live="polite">
        {active ? (
          <>
            <strong>{districtLabel(active.identity.districtKey)}</strong>
            <span>
              {" "}
              · {active.race.electionLabel}
              {mapView === "forecast"
                ? ` · ${raceRating(active.race)}`
                : active.race.observed
                  ? ` · Winner: ${active.race.observed.winnerName}`
                  : " · Observed result unavailable"}
            </span>
          </>
        ) : (
          "No district selected."
        )}
      </div>
      <p className="district-map-coverage">
        {joined.matches.length} of {races.length} forecast districts match this
        election’s district map.{" "}
        {mapView === "actual" && races.some(({ race }) => !race.observed)
          ? "Gray districts have no recorded winner here or are not included in this model."
          : joined.unmatchedFeatures.length > 0 ? "Gray districts are not included in this model." : "Every forecast district has matching boundaries."}
      </p>
      {joined.diagnostics.length > 0 && (
        <details className="district-map-diagnostics">
          <summary>Geography coverage notes</summary>
          <ul>
            {joined.diagnostics.map((diagnostic, index) => (
              <li key={diagnostic.code + index}>
                {diagnostic.message}
                {diagnostic.keys?.length
                  ? ` ${diagnostic.keys.join(", ")}`
                  : ""}
              </li>
            ))}
          </ul>
        </details>
      )}
      <details className="district-map-provenance">
        <summary>District boundary source</summary>
        {geography.metadata.sourceName && <p>
          {geography.metadata.sourceUrl?.startsWith("https://")
            ? <a href={geography.metadata.sourceUrl} target="_blank" rel="noreferrer">{geography.metadata.sourceName}</a>
            : geography.metadata.sourceName}
          {geography.metadata.sourceNote && <> · {geography.metadata.sourceNote}</>}
        </p>}
        <p>The election year and boundary version identify which district map is used. SHA256 is a file fingerprint that checks the map data has not changed.</p>
        <p>
          Election year: {geography.metadata.cycle} · Boundary version:{" "}
          {geography.metadata.mapVintage}
        </p>
        <p>
          Verified map file fingerprint (SHA256): <code>{geography.metadata.sourceSha256}</code>
        </p>
      </details>
    </section>
  );
}
