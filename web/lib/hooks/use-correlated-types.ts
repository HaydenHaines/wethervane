import useSWR from "swr";
import { fetchCorrelatedTypes, type CorrelatedTypeData } from "@/lib/api";

/**
 * Returns the N types most electorally correlated with the given type,
 * using the Ledoit-Wolf regularized observed covariance matrix.
 *
 * Pass null to skip fetching (conditional fetching pattern).
 * Refreshes every 30 minutes — stable between model retrains.
 */
export function useCorrelatedTypes(typeId: number | null, n = 4) {
  return useSWR<CorrelatedTypeData[]>(
    typeId != null ? `correlated-types-${typeId}-n${n}` : null,
    () => fetchCorrelatedTypes(typeId!, n),
    {
      revalidateOnFocus: false,
      dedupingInterval: 1_800_000, // 30 min
    },
  );
}
