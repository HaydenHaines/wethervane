import useSWR from "swr";
import { fetchSenateScrollyContext, type SenateScrollyContextData } from "@/lib/api";

/** Scrollytelling narrative context — refreshes every 10 minutes (changes only when predictions regenerate). */
export function useSenateScrollyContext() {
  return useSWR<SenateScrollyContextData>("senate-scrolly-context", fetchSenateScrollyContext, {
    revalidateOnFocus: false,
    dedupingInterval: 600_000, // 10 min
  });
}
