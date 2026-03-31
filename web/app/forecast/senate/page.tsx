"use client";

import { useSenateOverview } from "@/lib/hooks/use-senate-overview";
import { BalanceBar } from "@/components/forecast/BalanceBar";
import { RaceCardGrid } from "@/components/forecast/RaceCardGrid";
import { ErrorAlert } from "@/components/shared/ErrorAlert";
import { Skeleton } from "@/components/ui/skeleton";
import { KEY_RATING_SET } from "@/lib/config/ratings";
import { ELECTION_YEAR } from "@/lib/config/election";

export default function SenatePage() {
  const { data, error, isLoading, mutate } = useSenateOverview();

  if (error) {
    return <ErrorAlert title="Failed to load Senate forecast" retry={() => mutate()} />;
  }

  if (isLoading || !data) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-8 w-full" />
        <div className="grid grid-cols-3 gap-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-28 rounded-lg" />
          ))}
        </div>
      </div>
    );
  }

  const keyRaces = data.races.filter((r) => KEY_RATING_SET.has(r.rating));
  const otherRaces = data.races.filter((r) => !KEY_RATING_SET.has(r.rating));

  return (
    <div>
      <h1 className="font-serif text-2xl font-bold mb-4">{ELECTION_YEAR} United States Senate</h1>

      <BalanceBar
        races={data.races}
        demSeats={data.dem_seats_safe}
        gopSeats={data.gop_seats_safe}
      />

      {keyRaces.length > 0 && (
        <RaceCardGrid races={keyRaces} title="Key Races" />
      )}

      {otherRaces.length > 0 && (
        <RaceCardGrid races={otherRaces} title="Other Races" />
      )}
    </div>
  );
}
