import useSWR from "swr";
import {
  fetchCandidateList,
  type CandidateListParams,
  type CandidateListResponse,
} from "@/lib/api";

/**
 * Fetch the candidate directory list with optional filtering.
 *
 * Sabermetrics data changes only when the pipeline is re-run, so a long
 * deduping interval is appropriate.  The full list is loaded once and
 * client-side filtering is applied on top — this avoids round-trips as the
 * user types into the search box.
 *
 * Pass ``null`` as params to suspend the fetch (e.g. when a required filter
 * hasn't been selected yet).
 */
export function useCandidatesList(params: CandidateListParams | null = {}) {
  // Build a stable cache key from the params object so SWR deduplicates correctly.
  const key = params !== null
    ? ["candidates-list", JSON.stringify(params)]
    : null;

  return useSWR<CandidateListResponse>(
    key,
    () => fetchCandidateList(params ?? {}),
    {
      revalidateOnFocus: false,
      dedupingInterval: 3_600_000, // 1 hour
    },
  );
}
