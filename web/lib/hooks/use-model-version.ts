import useSWR from "swr";
import { COMMUNITY_COUNT_FALLBACK } from "@/lib/config/election";

interface ModelVersionData {
  version_id: string;
  k: number | null;
  /** Number of electoral community types (J in the KMeans model). */
  j: number | null;
  holdout_r: string | null;
  shift_type: string | null;
  created_at: string | null;
}

async function fetchModelVersion(): Promise<ModelVersionData> {
  const API_BASE = process.env.NEXT_PUBLIC_API_URL
    ? `${process.env.NEXT_PUBLIC_API_URL}/api/v1`
    : "/api/v1";
  const res = await fetch(`${API_BASE}/model/version`);
  if (!res.ok) throw new Error(`/model/version failed: ${res.status}`);
  return res.json();
}

/**
 * Returns the current model version metadata.
 *
 * The `communityCount` field resolves to the model's `j` value (number of
 * KMeans types) when the API is reachable, falling back to
 * COMMUNITY_COUNT_FALLBACK when it is not.
 */
export function useModelVersion() {
  const result = useSWR<ModelVersionData>("model-version", fetchModelVersion, {
    // Model version changes only on retrain — no need to revalidate frequently.
    revalidateOnFocus: false,
    dedupingInterval: 3_600_000, // 1 hour
  });

  const communityCount = result.data?.j ?? COMMUNITY_COUNT_FALLBACK;

  return { ...result, communityCount };
}
