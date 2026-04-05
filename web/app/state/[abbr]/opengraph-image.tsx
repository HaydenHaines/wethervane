import { ImageResponse } from "next/og";

export const runtime = "edge";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";
export const alt = "WetherVane state election forecast";

const API_BASE = process.env.API_URL || "http://localhost:8002";

interface StateRaceItem {
  slug: string;
  race_type: string;
  state: string;
  year: number;
  has_predictions: boolean;
  n_polls: number;
}

interface StateCounty {
  county_fips: string;
  state_abbr: string;
  pred_dem_share: number | null;
}

// Inline state names — edge runtime cannot import from lib/config
const STATE_NAMES: Record<string, string> = {
  AL: "Alabama", AK: "Alaska", AZ: "Arizona", AR: "Arkansas", CA: "California",
  CO: "Colorado", CT: "Connecticut", DE: "Delaware", DC: "District of Columbia",
  FL: "Florida", GA: "Georgia", HI: "Hawaii", ID: "Idaho", IL: "Illinois",
  IN: "Indiana", IA: "Iowa", KS: "Kansas", KY: "Kentucky", LA: "Louisiana",
  ME: "Maine", MD: "Maryland", MA: "Massachusetts", MI: "Michigan",
  MN: "Minnesota", MS: "Mississippi", MO: "Missouri", MT: "Montana",
  NE: "Nebraska", NV: "Nevada", NH: "New Hampshire", NJ: "New Jersey",
  NM: "New Mexico", NY: "New York", NC: "North Carolina", ND: "North Dakota",
  OH: "Ohio", OK: "Oklahoma", OR: "Oregon", PA: "Pennsylvania",
  RI: "Rhode Island", SC: "South Carolina", SD: "South Dakota", TN: "Tennessee",
  TX: "Texas", UT: "Utah", VT: "Vermont", VA: "Virginia", WA: "Washington",
  WV: "West Virginia", WI: "Wisconsin", WY: "Wyoming",
};

export default async function Image({
  params,
}: {
  params: Promise<{ abbr: string }>;
}) {
  const { abbr } = await params;
  const upperAbbr = abbr.toUpperCase();
  const stateName = STATE_NAMES[upperAbbr];

  // Fetch county count and race metadata in parallel
  let countyCount = 0;
  let races: StateRaceItem[] = [];

  try {
    const [countiesRes, racesRes] = await Promise.all([
      fetch(`${API_BASE}/api/v1/counties`, { next: { revalidate: 86400 } }),
      fetch(`${API_BASE}/api/v1/forecast/race-metadata`, { next: { revalidate: 3600 } }),
    ]);

    if (countiesRes.ok) {
      const allCounties: StateCounty[] = await countiesRes.json();
      countyCount = allCounties.filter((c) => c.state_abbr === upperAbbr).length;
    }

    if (racesRes.ok) {
      const allRaces: StateRaceItem[] = await racesRes.json();
      // Filter to this state's races with predictions
      races = allRaces.filter((r) => r.state === upperAbbr && r.has_predictions);
    }
  } catch {
    /* fallback to generic card */
  }

  const displayName = stateName ?? upperAbbr;
  const raceCount = races.length;

  // Generic card for unknown state
  if (!stateName) {
    return new ImageResponse(
      (
        <div
          style={{
            width: "100%",
            height: "100%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "#f7f8fa",
            fontFamily: "Georgia, serif",
            fontSize: 48,
            color: "#222",
          }}
        >
          WetherVane — State Not Found
        </div>
      ),
      { ...size },
    );
  }

  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          background: "#f7f8fa",
          padding: "48px 56px",
          fontFamily: "system-ui, sans-serif",
        }}
      >
        {/* Top accent bar — neutral WetherVane brand color */}
        <div
          style={{
            display: "flex",
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            height: 6,
            background: "#4a7fb5",
          }}
        />

        {/* State abbreviation badge */}
        <div
          style={{
            display: "flex",
            width: 72,
            height: 72,
            borderRadius: 12,
            background: "white",
            border: "2px solid #e0e0e0",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 28,
            fontWeight: 800,
            color: "#222",
            marginBottom: 24,
            letterSpacing: "0.05em",
          }}
        >
          {upperAbbr}
        </div>

        {/* State name + subtitle */}
        <div
          style={{
            fontSize: 60,
            fontWeight: 700,
            color: "#222",
            fontFamily: "Georgia, serif",
            lineHeight: 1.1,
          }}
        >
          {displayName}
        </div>
        <div
          style={{
            fontSize: 30,
            color: "#666",
            marginTop: 8,
            fontFamily: "Georgia, serif",
          }}
        >
          2026 Election Forecast
        </div>

        {/* Stats row */}
        <div style={{ display: "flex", gap: 16, marginTop: "auto", marginBottom: 16 }}>
          {countyCount > 0 && (
            <div
              style={{
                display: "flex",
                padding: "12px 24px",
                borderRadius: 8,
                fontSize: 20,
                background: "white",
                border: "1px solid #e0e0e0",
                color: "#444",
              }}
            >
              {countyCount} {countyCount === 1 ? "county" : "counties"} in model
            </div>
          )}
          {raceCount > 0 && (
            <div
              style={{
                display: "flex",
                padding: "12px 24px",
                borderRadius: 8,
                fontSize: 20,
                background: "white",
                border: "1px solid #e0e0e0",
                color: "#444",
              }}
            >
              {raceCount} tracked {raceCount === 1 ? "race" : "races"}
            </div>
          )}
          {countyCount === 0 && raceCount === 0 && (
            <div
              style={{
                display: "flex",
                padding: "12px 24px",
                borderRadius: 8,
                fontSize: 20,
                background: "white",
                border: "1px solid #e0e0e0",
                color: "#666",
              }}
            >
              County-level analysis
            </div>
          )}
        </div>

        {/* Footer branding */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginTop: 8,
          }}
        >
          <div
            style={{
              fontSize: 22,
              fontWeight: 700,
              color: "#222",
              fontFamily: "Georgia, serif",
              letterSpacing: "0.02em",
            }}
          >
            WetherVane
          </div>
          <div style={{ fontSize: 16, color: "#999" }}>
            wethervane.hhaines.duckdns.org
          </div>
        </div>
      </div>
    ),
    { ...size },
  );
}
