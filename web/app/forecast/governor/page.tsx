"use client";

import { useState, useEffect } from "react";
import { RaceCardGrid } from "@/components/forecast/RaceCardGrid";
import { ErrorAlert } from "@/components/shared/ErrorAlert";
import { Skeleton } from "@/components/ui/skeleton";
import type { SenateRaceData } from "@/lib/api";

/**
 * Governor overview page.
 * Uses /forecast/races to get governor race slugs.
 * TODO: Switch to dedicated /governor/overview API when available.
 */
export default function GovernorPage() {
  const [races, setRaces] = useState<SenateRaceData[] | null>(null);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const res = await fetch("/api/v1/forecast/races");
        if (!res.ok) throw new Error("Failed to load races");
        const allRaces: string[] = await res.json();
        const govRaces = allRaces
          .filter((r) => r.includes("Governor"))
          .map((race) => {
            const parts = race.split(" ");
            const state = parts[1];
            return {
              state,
              race,
              slug: race.toLowerCase().replace(/\s+/g, "-"),
              rating: "tossup",
              margin: 0,
              n_polls: 0,
            };
          });
        setRaces(govRaces);
      } catch (e) {
        setError(e as Error);
      }
    }
    load();
  }, []);

  if (error) return <ErrorAlert title="Failed to load Governor races" />;

  if (!races) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-8 w-64" />
        <div className="grid grid-cols-3 gap-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-28 rounded-lg" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div>
      <h1 className="font-serif text-2xl font-bold mb-4">2026 Governor Races</h1>
      <p className="text-sm mb-6" style={{ color: "var(--color-text-muted)" }}>
        {races.length} governor races tracked. Detailed predictions available as polling data arrives.
      </p>
      <RaceCardGrid races={races} title="All Governor Races" />
    </div>
  );
}
