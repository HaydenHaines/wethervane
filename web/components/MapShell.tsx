"use client";
import { useState, useEffect, useCallback } from "react";
import DeckGL from "@deck.gl/react";
import { GeoJsonLayer } from "@deck.gl/layers";
import { fetchCounties, fetchSuperTypes, type CountyRow } from "@/lib/api";
import { useMapContext } from "@/components/MapContext";
import { CommunityPanel } from "@/components/CommunityPanel";
import { TypePanel } from "@/components/TypePanel";

// Colorblind-accessible super-type palette (stained glass colors)
export const SUPER_TYPE_COLORS: [number, number, number][] = [
  [31, 119, 180],    // blue
  [255, 127, 14],    // orange
  [44, 160, 44],     // green
  [214, 39, 40],     // red
  [148, 103, 189],   // purple
  [140, 86, 75],     // brown
  [227, 119, 194],   // pink
  [127, 127, 127],   // gray
];

// Super-type display names (fetched from API, with fallback defaults)
export const SUPER_TYPE_NAMES: Record<number, string> = {
  0: "Rural White Conservative",
  1: "Small-Town Mixed",
  2: "Suburban Professional",
  3: "Black Belt & Diverse",
  4: "Hispanic South Florida",
  5: "North Georgia Exurban",
  6: "Deep Rural Georgia",
  7: "Metro Atlanta Professional",
};

// Legacy community colors (fallback when type data not present)
const COMMUNITY_COLORS: [number, number, number][] = [
  [78, 121, 167],
  [89, 161, 79],
  [176, 122, 161],
  [255, 157, 167],
  [156, 117, 95],
  [242, 142, 43],
  [186, 176, 172],
  [255, 210, 0],
  [148, 103, 189],
  [140, 162, 82],
];

const INITIAL_VIEW = {
  longitude: -84.5,
  latitude: 31.5,
  zoom: 5.8,
  pitch: 0,
  bearing: 0,
};

export default function MapShell() {
  const { selectedCommunityId, setSelectedCommunityId, selectedTypeId, setSelectedTypeId } = useMapContext();
  const [geojson, setGeojson] = useState<any>(null);
  const [tractGeojson, setTractGeojson] = useState<any>(null);
  const [countyMap, setCountyMap] = useState<Record<string, CountyRow>>({});
  const [hasTypeData, setHasTypeData] = useState(false);
  const [showTracts, setShowTracts] = useState(false);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);

  useEffect(() => {
    Promise.all([
      fetch("/counties-fl-ga-al.geojson").then((r) => r.json()),
      fetchCounties(),
      fetchSuperTypes().catch(() => []),
      fetch("/tract-communities.geojson").then((r) => r.json()).catch(() => null),
    ]).then(([geo, counties, superTypes, tractGeo]) => {
      if (tractGeo) setTractGeojson(tractGeo);
      const map: Record<string, CountyRow> = {};
      let typeDataPresent = false;
      counties.forEach((c) => {
        map[c.county_fips] = c;
        if (c.super_type !== null) typeDataPresent = true;
      });
      // Populate super-type names from API
      superTypes.forEach((st: any) => {
        SUPER_TYPE_NAMES[st.super_type_id] = st.display_name;
      });
      setCountyMap(map);
      setHasTypeData(typeDataPresent);
      const enriched = {
        ...geo,
        features: geo.features.map((f: any) => ({
          ...f,
          properties: {
            ...f.properties,
            community_id: map[f.properties.county_fips]?.community_id ?? -1,
            dominant_type: map[f.properties.county_fips]?.dominant_type ?? -1,
            super_type: map[f.properties.county_fips]?.super_type ?? -1,
          },
        })),
      };
      setGeojson(enriched);
    });
  }, []);

  const getColor = useCallback(
    (f: any): [number, number, number, number] => {
      // Tract communities and county types both use super_type for coloring
      const st: number = f.properties?.super_type ?? -1;
      const dt: number = f.properties?.dominant_type ?? f.properties?.type_id ?? -1;

      if (st >= 0 || hasTypeData) {
        const isSelected = selectedTypeId !== null && dt === selectedTypeId;
        const base = st >= 0 && st < SUPER_TYPE_COLORS.length
          ? SUPER_TYPE_COLORS[st]
          : [180, 180, 180] as [number, number, number];
        return isSelected
          ? [...base, 255] as [number, number, number, number]
          : [...base, 180] as [number, number, number, number];
      }
      // Fallback: community colors
      const cid: number = f.properties?.community_id ?? -1;
      const isSelected = selectedCommunityId !== null && cid === selectedCommunityId;
      const base = cid >= 0 && cid < COMMUNITY_COLORS.length ? COMMUNITY_COLORS[cid] : [180, 180, 180];
      return isSelected ? [...base, 255] as [number, number, number, number] : [...base, 180] as [number, number, number, number];
    },
    [selectedCommunityId, selectedTypeId, hasTypeData, showTracts]
  );

  const getLineWidth = useCallback(
    (f: any): number => {
      if (hasTypeData) {
        const dt: number = f.properties?.dominant_type ?? -1;
        return selectedTypeId !== null && dt === selectedTypeId ? 800 : 200;
      }
      const cid: number = f.properties?.community_id ?? -1;
      return selectedCommunityId !== null && cid === selectedCommunityId ? 800 : 200;
    },
    [selectedCommunityId, selectedTypeId, hasTypeData]
  );

  const activeData = showTracts && tractGeojson ? tractGeojson : geojson;
  const layerId = showTracts && tractGeojson ? "tract-communities" : "counties";

  const layers = activeData
    ? [
        new GeoJsonLayer({
          id: layerId,
          data: activeData,
          pickable: true,
          stroked: true,
          filled: true,
          getFillColor: getColor as any,
          getLineColor: [80, 80, 80, 120],
          getLineWidth,
          lineWidthUnits: "meters",
          updateTriggers: {
            getFillColor: [selectedCommunityId, selectedTypeId, hasTypeData, showTracts],
            getLineWidth: [selectedCommunityId, selectedTypeId, hasTypeData, showTracts],
          },
          onHover: ({ object, x, y }: any) => {
            if (object) {
              if (showTracts && tractGeojson) {
                const st = object.properties?.super_type;
                const tid = object.properties?.type_id;
                const n = object.properties?.n_tracts;
                const area = object.properties?.area_sqkm;
                const stName = st >= 0 ? (SUPER_TYPE_NAMES[st] || `Super-type ${st}`) : "?";
                setTooltip({ x, y, text: `${stName}\nType ${tid} · ${n} tracts · ${Math.round(area)} km²` });
              } else {
                const name = object.properties?.county_name || object.properties?.county_fips;
                if (hasTypeData) {
                  const st = object.properties?.super_type;
                  const dt = object.properties?.dominant_type;
                  const stName = st >= 0 ? (SUPER_TYPE_NAMES[st] || `Super-type ${st}`) : "?";
                  setTooltip({ x, y, text: `${name}\n${stName} (Type ${dt})` });
                } else {
                  const cid = object.properties?.community_id;
                  setTooltip({ x, y, text: `${name}\nCommunity ${cid}` });
                }
              }
            } else {
              setTooltip(null);
            }
          },
          onClick: ({ object }: any) => {
            if (object) {
              if (hasTypeData) {
                const dt = object.properties?.dominant_type;
                if (dt !== undefined && dt >= 0) {
                  setSelectedTypeId(dt === selectedTypeId ? null : dt);
                  setSelectedCommunityId(null);
                }
              } else {
                const cid = object.properties?.community_id;
                if (cid !== undefined && cid >= 0) {
                  setSelectedCommunityId(cid === selectedCommunityId ? null : cid);
                  setSelectedTypeId(null);
                }
              }
            }
          },
        }),
      ]
    : [];

  // Build legend entries based on available data
  const legendEntries = hasTypeData
    ? SUPER_TYPE_COLORS.map((color, i) => ({
        color,
        label: SUPER_TYPE_NAMES[i] || `Super-type ${i}`,
      }))
    : COMMUNITY_COLORS.map((color, i) => ({
        color,
        label: `Community ${i}`,
      }));

  // Determine which super-types are actually present in data
  const activeSuperTypes = new Set<number>();
  if (hasTypeData) {
    Object.values(countyMap).forEach((c) => {
      if (c.super_type !== null) activeSuperTypes.add(c.super_type);
    });
  }

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      <DeckGL
        initialViewState={INITIAL_VIEW}
        controller={true}
        layers={layers}
        style={{ background: "#e8ecf0" }}
      />

      {tooltip && (
        <div style={{
          position: "absolute",
          left: tooltip.x + 12,
          top: tooltip.y + 12,
          background: "white",
          border: "1px solid var(--color-border)",
          borderRadius: "4px",
          padding: "6px 10px",
          fontSize: "12px",
          fontFamily: "var(--font-sans)",
          pointerEvents: "none",
          whiteSpace: "pre-line",
          boxShadow: "0 2px 6px rgba(0,0,0,0.1)",
        }}>
          {tooltip.text}
        </div>
      )}

      {/* County/Tract toggle */}
      {tractGeojson && (
        <button
          onClick={() => setShowTracts(!showTracts)}
          style={{
            position: "absolute",
            top: 12,
            left: 16,
            background: showTracts ? "#2166ac" : "white",
            color: showTracts ? "white" : "#333",
            border: "1px solid var(--color-border)",
            borderRadius: "4px",
            padding: "6px 14px",
            fontSize: "12px",
            fontFamily: "var(--font-sans)",
            cursor: "pointer",
            fontWeight: 600,
            boxShadow: "0 1px 3px rgba(0,0,0,0.15)",
          }}
        >
          {showTracts ? "Tract Communities" : "County Types"} ▾
        </button>
      )}

      {/* Legend */}
      <div style={{
        position: "absolute",
        bottom: 24,
        left: 16,
        background: "white",
        border: "1px solid var(--color-border)",
        borderRadius: "4px",
        padding: "8px 12px",
        fontSize: "11px",
        fontFamily: "var(--font-sans)",
      }}>
        {hasTypeData
          ? legendEntries
              .filter((_, i) => activeSuperTypes.has(i))
              .map((entry, idx) => (
                <div key={idx} style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "2px" }}>
                  <div style={{
                    width: 12, height: 12, borderRadius: 2,
                    background: `rgb(${entry.color.join(",")})`,
                  }} />
                  <span style={{ color: "var(--color-text-muted)" }}>{entry.label}</span>
                </div>
              ))
          : legendEntries.map((entry, i) => (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "2px" }}>
                <div style={{
                  width: 12, height: 12, borderRadius: 2,
                  background: `rgb(${entry.color.join(",")})`,
                }} />
                <span style={{ color: "var(--color-text-muted)" }}>{entry.label}</span>
              </div>
            ))
        }
      </div>

      {/* Side panels */}
      {selectedCommunityId !== null && !hasTypeData && (
        <CommunityPanel
          communityId={selectedCommunityId}
          onClose={() => setSelectedCommunityId(null)}
        />
      )}

      {selectedTypeId !== null && hasTypeData && (
        <TypePanel
          typeId={selectedTypeId}
          onClose={() => setSelectedTypeId(null)}
        />
      )}
    </div>
  );
}
