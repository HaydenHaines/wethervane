import useSWR from "swr";
import {
  fetchCandidateProfile,
  fetchCandidatePredecessor,
  type CandidateBadgesResponse,
  type PredecessorInfo,
} from "@/lib/api";

/**
 * Fetch the full badge + race history profile for a single candidate.
 *
 * Returns the complete CandidateBadgesResponse including races[] field.
 * Sabermetrics data is static between pipeline runs — 1 hour dedup interval.
 */
export function useCandidateProfile(bioguideId: string | null) {
  return useSWR<CandidateBadgesResponse>(
    bioguideId ? ["candidate-profile", bioguideId] : null,
    () => fetchCandidateProfile(bioguideId!),
    {
      revalidateOnFocus: false,
      dedupingInterval: 3_600_000, // 1 hour
    },
  );
}

/**
 * Fetch the predecessor candidate for a single-race candidate.
 *
 * Returns PredecessorInfo or null (when no predecessor exists or the candidate
 * has multiple races).  Only fetches when the profile indicates n_races === 1.
 */
export function useCandidatePredecessor(
  bioguideId: string | null,
  nRaces: number | undefined,
) {
  // Only look up predecessor for single-race candidates
  const shouldFetch = bioguideId !== null && nRaces === 1;

  return useSWR<PredecessorInfo | null>(
    shouldFetch ? ["candidate-predecessor", bioguideId] : null,
    () => fetchCandidatePredecessor(bioguideId!),
    {
      revalidateOnFocus: false,
      dedupingInterval: 3_600_000, // 1 hour
    },
  );
}
