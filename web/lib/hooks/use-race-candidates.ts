import useSWR from "swr";
import {
  fetchRaceCandidates,
  type RaceCandidatesResponse,
} from "@/lib/api";

/**
 * Fetch candidate badge data for a race.
 *
 * Sabermetrics data changes only when the pipeline is re-run, so a long
 * deduping interval is appropriate.  Returns null gracefully when no
 * badge data exists for a race's candidates.
 */
export function useRaceCandidates(raceKey: string | null) {
  return useSWR<RaceCandidatesResponse>(
    raceKey ? ["race-candidates", raceKey] : null,
    () => fetchRaceCandidates(raceKey!),
    {
      revalidateOnFocus: false,
      dedupingInterval: 3_600_000, // 1 hour
    },
  );
}
