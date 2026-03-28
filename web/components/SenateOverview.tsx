"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { fetchSenateOverview, type SenateOverviewData, type SenateRaceData } from "@/lib/api";
import { DUSTY_INK, type Rating } from "@/lib/colors";
import { SenateControlBar, type ControlBarRace } from "@/components/SenateControlBar";
import { RaceCard } from "@/components/RaceCard";

const KEY_RATINGS: Set<string> = new Set(["tossup", "lean_d", "lean_r"]);

function toControlBarRace(r: SenateRaceData): ControlBarRace {
  return {
    state: r.state,
    rating: r.rating as Rating,
    margin: r.margin,
    slug: r.slug,
    race: r.race,
  };
}

export function SenateOverview() {
  const router = useRouter();
  const [data, setData] = useState<SenateOverviewData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchSenateOverview()
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);

  function handleRaceClick(slug: string) {
    router.push(`/forecast/${slug}`);
  }

  if (error) {
    return (
      <div style={{ padding: "40px 20px", color: DUSTY_INK.textMuted, fontFamily: "var(--font-sans)", fontSize: "14px" }}>
        Failed to load Senate overview: {error}
      </div>
    );
  }

  if (!data) {
    return (
      <div style={{ padding: "40px 20px", color: DUSTY_INK.textMuted, fontFamily: "var(--font-sans)", fontSize: "13px" }}>
        Loading...
      </div>
    );
  }

  const keyRaces = data.races.filter(r => KEY_RATINGS.has(r.rating));
  const otherRaces = data.races.filter(r => !KEY_RATINGS.has(r.rating));

  return (
    <div style={{ padding: "20px 16px", maxWidth: "720px" }}>
      {/* Label */}
      <div style={{
        fontFamily: "var(--font-sans)",
        fontSize: "11px",
        fontVariant: "small-caps",
        letterSpacing: "1.2px",
        color: DUSTY_INK.textSubtle,
        marginBottom: "6px",
      }}>
        2026 United States Senate
      </div>

      {/* Headline */}
      <h1 style={{
        fontFamily: "var(--font-serif)",
        fontSize: "28px",
        fontWeight: 700,
        color: DUSTY_INK.text,
        margin: "0 0 4px",
        lineHeight: 1.2,
      }}>
        {data.headline}
      </h1>

      {/* Subtitle */}
      <p style={{
        fontFamily: "var(--font-sans)",
        fontSize: "14px",
        color: DUSTY_INK.textMuted,
        margin: "0 0 20px",
        lineHeight: 1.4,
      }}>
        {data.subtitle}
      </p>

      {/* Control bar */}
      <SenateControlBar
        races={data.races.map(toControlBarRace)}
        demSeats={data.dem_seats_safe}
        gopSeats={data.gop_seats_safe}
        onRaceClick={handleRaceClick}
      />

      {/* Key Races */}
      {keyRaces.length > 0 && (
        <section style={{ marginBottom: "24px" }}>
          <h2 style={{
            fontFamily: "var(--font-sans)",
            fontSize: "12px",
            fontWeight: 600,
            textTransform: "uppercase",
            letterSpacing: "0.8px",
            color: DUSTY_INK.textMuted,
            margin: "0 0 10px",
          }}>
            Key Races
          </h2>
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
            gap: "10px",
          }}>
            {keyRaces.map((r) => (
              <RaceCard
                key={r.slug}
                state={r.state}
                race={r.race}
                slug={r.slug}
                rating={r.rating as Rating}
                margin={r.margin}
                nPolls={r.n_polls}
                onClick={() => handleRaceClick(r.slug)}
              />
            ))}
          </div>
        </section>
      )}

      {/* Other Races */}
      {otherRaces.length > 0 && (
        <section>
          <h2 style={{
            fontFamily: "var(--font-sans)",
            fontSize: "12px",
            fontWeight: 600,
            textTransform: "uppercase",
            letterSpacing: "0.8px",
            color: DUSTY_INK.textMuted,
            margin: "0 0 10px",
          }}>
            Other Races
          </h2>
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
            gap: "10px",
          }}>
            {otherRaces.map((r) => (
              <RaceCard
                key={r.slug}
                state={r.state}
                race={r.race}
                slug={r.slug}
                rating={r.rating as Rating}
                margin={r.margin}
                nPolls={r.n_polls}
                onClick={() => handleRaceClick(r.slug)}
              />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
