import useSWR from "swr";
import { fetchFundamentals, type FundamentalsData } from "@/lib/api";

/**
 * Fundamentals model data — refreshes every hour.
 *
 * The fundamentals data changes infrequently (monthly economic releases),
 * so a longer deduping interval is appropriate here.
 */
export function useFundamentals() {
  return useSWR<FundamentalsData>("fundamentals", fetchFundamentals, {
    revalidateOnFocus: false,
    dedupingInterval: 3_600_000, // 1 hour
  });
}
