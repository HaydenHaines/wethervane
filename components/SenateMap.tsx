"use client";

import { useMemo } from "react";
import { geoAlbersUsa, geoPath } from "d3-geo";
import type { FeatureCollection, Feature, Geometry } from "geojson";
import type { ExplorerRace as Race } from "../lib/explorer-types";
import {
  COLORS,
  raceColor,
  raceRating,
  competitiveOrder,
  partyColor,
  outcomeColor,
} from "../lib/presentation";

export type SenateGeography = FeatureCollection<
  Geometry,
  { state_abbr: string; state_name: string }
>;

type Geo = SenateGeography;

export default function SenateMap({
  geo,
  races,
  selected,
  onSelect,
  mapView = "forecast",
  absentLabel = "no Senate election",
}: {
  geo: Geo;
  races: Race[];
  selected: Race;
  onSelect: (state: string) => void;
  mapView?: "forecast" | "actual";
  absentLabel?: string;
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
      <title>{selected.id.slice(0, 4)} Senate race map</title>
      {geo.features.map((feature) => {
        const state = feature.properties.state_abbr;
        const matches = byState.get(state);
        const race =
          matches?.find((r) => r.id === selected.id) ||
          (matches ? [...matches].sort(competitiveOrder)[0] : undefined);
        const label = `${feature.properties.state_name}${race ? `, ${matches!.length} election${matches!.length > 1 ? "s" : ""}, ${mapView === "actual" ? `observed winner ${race.observed?.winnerName || "unavailable"}` : raceRating(race)}` : `, ${absentLabel}`}`;
        return (
          <path
            key={state}
            d={path(feature as Feature) || undefined}
            fill={
              race
                ? mapView === "actual"
                  ? race.observed
                    ? outcomeColor(
                        race.outcomes.find(
                          (o) => o.id === race.observed!.outcomeId,
                        )!,
                      )
                    : COLORS.inactive
                  : raceColor(race)
                : COLORS.inactive
            }
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
            className={`map-label${!byState.has(state) ? " inactive" : mapView === "forecast" && raceRating(byState.get(state)!.find((r) => r.id === selected.id) || [...byState.get(state)!].sort(competitiveOrder)[0]) === "Likely" ? " likely" : ""}`}
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
