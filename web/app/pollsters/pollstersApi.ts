export interface PollsterEntry {
  pollster: string;
  rank: number;
  n_polls: number;
  n_races: number;
  rmse_pp: number;
  mean_error_pp: number;
}

export interface PollsterAccuracyResponse {
  description: string;
  n_pollsters: number;
  pollsters: PollsterEntry[];
}

export interface PollCoverageMetadata {
  total_polls: number;
  polls_with_xt_data: number;
  polls_analyzed: number;
  active_xt_columns: string[];
  mappable_xt_columns: string[];
  oversample_threshold: number;
  undersample_threshold: number;
}

export interface PollCoverageAffectedType {
  type_id: number;
  display_name: string;
  group_share: number;
  state_weight: number;
  exposure: number;
}

export interface PollCoverageGap {
  demographic_group: string;
  label: string;
  poll_share: number;
  population_share: number;
  ratio: number;
  status: string;
  affected_types: PollCoverageAffectedType[];
}

export interface PollCoveragePollResult {
  race: string;
  state: string;
  pollster: string;
  date: string;
  n_sample: number | null;
  n_groups_analyzed: number;
  n_undersampled: number;
  n_oversampled: number;
  gaps: PollCoverageGap[];
}

export interface PollCoverageTopAffectedType {
  type_label: string;
  n_races_affected: number;
}

export interface PollCoverageGroupSummary {
  label: string;
  n_undersampled: number;
  n_oversampled: number;
  n_representative: number;
  n_total_polls: number;
  top_affected_types: PollCoverageTopAffectedType[];
}

export interface PollCoverageUndersampledRankingEntry {
  group: string;
  label: string;
  n_polls_undersampled: number;
}

export interface PollCoverageSummary {
  by_group: Record<string, PollCoverageGroupSummary>;
  undersampled_ranking: PollCoverageUndersampledRankingEntry[];
}

export interface PollCoverageReportResponse {
  metadata: PollCoverageMetadata;
  summary: PollCoverageSummary;
  per_poll_results: PollCoveragePollResult[];
}

export type PollCoverageFetchResult =
  | { status: "ok"; data: PollCoverageReportResponse }
  | { status: "unavailable"; message: string }
  | { status: "error"; message: string };

const SERVER_API_BASE = process.env.API_URL || "http://localhost:8002";
export const POLLSTER_ACCURACY_ENDPOINT = "/api/v1/pollsters/accuracy";
export const POLLSTER_COVERAGE_ENDPOINT = "/api/v1/pollsters/coverage";

async function readErrorMessage(res: Response): Promise<string> {
  try {
    const payload = (await res.json()) as { detail?: string };
    if (typeof payload.detail === "string" && payload.detail.trim()) {
      return payload.detail;
    }
  } catch {
    // Fall through to generic message below.
  }

  return `Request failed with status ${res.status}.`;
}

async function readCoverageResponse(input: string, init?: RequestInit): Promise<PollCoverageFetchResult> {
  try {
    const res = await fetch(input, init);

    if (res.ok) {
      return {
        status: "ok",
        data: (await res.json()) as PollCoverageReportResponse,
      };
    }

    const message = await readErrorMessage(res);
    if (res.status === 503) {
      return { status: "unavailable", message };
    }

    return { status: "error", message };
  } catch {
    return {
      status: "error",
      message: "Could not load poll coverage diagnostics.",
    };
  }
}

export async function fetchPollsterAccuracy(): Promise<PollsterAccuracyResponse | null> {
  try {
    const res = await fetch(`${SERVER_API_BASE}${POLLSTER_ACCURACY_ENDPOINT}`, {
      next: { revalidate: 86400 },
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export async function fetchPollsterCoverage(): Promise<PollCoverageFetchResult> {
  return readCoverageResponse(`${SERVER_API_BASE}${POLLSTER_COVERAGE_ENDPOINT}`, {
    headers: { Accept: "application/json" },
    next: { revalidate: 86400 },
  });
}

export async function fetchPollsterCoverageClient(
  input = POLLSTER_COVERAGE_ENDPOINT,
): Promise<PollCoverageFetchResult> {
  return readCoverageResponse(input, {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
}
