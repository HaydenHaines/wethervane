import useSWR from "swr";
import { fetchSeatHistory, type SeatHistoryEntry } from "@/lib/api";

/**
 * Fetch the Senate seat balance time series.
 *
 * Returns all snapshot dates with projected D/R seat counts.
 * Refreshes every 15 minutes — snapshots are added twice-weekly by cron.
 */
export function useSeatHistory() {
  return useSWR<SeatHistoryEntry[]>("seat-history", fetchSeatHistory, {
    revalidateOnFocus: false,
    dedupingInterval: 900_000, // 15 min
  });
}
