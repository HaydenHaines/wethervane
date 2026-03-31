import Link from "next/link";
import { stripStateSuffix } from "@/lib/config/states";

// ── Types ─────────────────────────────────────────────────────────────────────

export interface TypeCounty {
  county_fips: string;
  county_name: string | null;
  state_abbr: string;
}

export interface TypeCountyListProps {
  counties: TypeCounty[];
  typeName: string;
  countyWord: string;
  nCounties: number;
}

// ── Component ─────────────────────────────────────────────────────────────────

export function TypeCountyList({
  counties,
  typeName,
  countyWord,
  nCounties,
}: TypeCountyListProps) {
  // Group counties by state for display, then sort states alphabetically.
  // Grouping here (not in the page) keeps the page component free of display logic.
  const countiesByState: Record<string, TypeCounty[]> = {};
  for (const county of counties) {
    if (!countiesByState[county.state_abbr]) {
      countiesByState[county.state_abbr] = [];
    }
    countiesByState[county.state_abbr].push(county);
  }
  const sortedStates = Object.keys(countiesByState).sort();

  return (
    <section style={{ marginBottom: 40 }}>
      <h2
        style={{
          fontFamily: "var(--font-serif)",
          fontSize: 22,
          marginBottom: 8,
        }}
      >
        Member Counties
      </h2>
      <p
        style={{
          fontSize: 14,
          color: "var(--color-text-muted)",
          marginBottom: 16,
        }}
      >
        {nCounties} {countyWord} classified as <strong>{typeName}</strong>
      </p>

      {sortedStates.map((stateAbbr) => (
        <div key={stateAbbr} style={{ marginBottom: 20 }}>
          <h3
            style={{
              fontFamily: "var(--font-serif)",
              fontSize: 13,
              fontWeight: 600,
              color: "var(--color-text-muted)",
              textTransform: "uppercase",
              letterSpacing: "0.07em",
              marginBottom: 8,
            }}
          >
            {stateAbbr}
          </h3>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: "4px 16px",
            }}
          >
            {countiesByState[stateAbbr].map((county) => (
              <Link
                key={county.county_fips}
                href={`/county/${county.county_fips}`}
                style={{
                  padding: "6px 0",
                  fontSize: 14,
                  color: "var(--color-dem)",
                  textDecoration: "none",
                  borderBottom: "1px solid var(--color-bg)",
                }}
              >
                {stripStateSuffix(county.county_name)}
              </Link>
            ))}
          </div>
        </div>
      ))}
    </section>
  );
}
