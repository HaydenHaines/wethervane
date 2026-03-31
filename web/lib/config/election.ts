/**
 * Election cycle configuration — update this file for each new election cycle.
 *
 * COMMUNITY_COUNT is a fallback only. The authoritative value comes from
 * /api/v1/model/version (field: j). Components that can fetch from the API
 * should use that value instead of this constant.
 */

/** Four-digit election year. Overridable via NEXT_PUBLIC_ELECTION_YEAR env var. */
export const ELECTION_YEAR = process.env.NEXT_PUBLIC_ELECTION_YEAR ?? "2026";

/** Human-readable cycle label for display strings. */
export const ELECTION_CYCLE = `${ELECTION_YEAR} Midterms`;

/** Number of governor races on the ballot this cycle. */
export const GOVERNOR_RACES_COUNT = 36;

/**
 * Fallback community count when the API is unavailable.
 *
 * The authoritative count is the `j` field from /api/v1/model/version.
 * This constant exists only so the UI degrades gracefully rather than
 * showing nothing when the API is down.
 */
export const COMMUNITY_COUNT_FALLBACK = 100;
