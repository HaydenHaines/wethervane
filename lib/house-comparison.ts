import type { HouseResult } from "./house-result";
import { HOUSE_SHARE_DENOMINATOR, parseHouseResult } from "./house-result.ts";

export const HOUSE_COMPARISON_SOURCE_PINS = {
  sourceCatalog:
    "c189fe1100acec043f6ace325c81c1e2da8765c5f20a454b31e3825821d9cb7f",
  reviewReceipt:
    "70f371b56d63909c847d0aa539efb28a5c4a538c3ed2e33052ea4f0a28eea9ee",
  sourceCapabilityPolicy:
    "618e5870cc224d06d949e93bef4ba80e1d044445ff6a91877657837eb81f2897",
  sourceHandoff:
    "910692edc82c183704f37bb9737985ddc9b94cbd44b6f95a52a3754e237f9118",
  sourcePanel:
    "b07c227432fa4768822e0db083107df6675d7d4168b7fc9a493180c16b1bfe80",
  interpretationReceipt:
    "fd73a4a2d36d775d7c3ff0d2e1c512a237b3b4e4ade38179af60405d841a0e00",
  publishedCatalog:
    "82c63f9ba70bcd8a9fbbb2037d5bb48d3e9ee8190462f1583caa9a3c8ddb092b",
} as const;
export interface HouseComparisonRun {
  runId: string;
  configurationId: string;
  configurationSha256: string;
  resultSha256: string;
  canonicalResultSha256: string;
  planSha256: string;
  fitReceiptSha256: string;
  jointArtifactId: string;
  jointSourceSha256: string;
}
export interface HouseComparisonGroup {
  year: number;
  mapVintage: string;
  raceCount: number;
  raceIds: string[];
  supportSha256: string;
  identitySha256: string;
  runs: HouseComparisonRun[];
}
export interface HouseComparisonManifest {
  schemaVersion: 1;
  id: "reviewed-house-same-year-20261020-v1";
  targetId: "house_aggregate_fundamentals_retrospective_v1";
  denominator: string;
  nominalHorizonDays: 14;
  informationMode: "retrospective_proxy_strictly_earlier_fitted_labels";
  scope: "eligible_race_subset";
  sourcePins: Record<keyof typeof HOUSE_COMPARISON_SOURCE_PINS, string>;
  groups: HouseComparisonGroup[];
  limitations: string[];
}
export interface HouseComparisonResult {
  result: HouseResult;
  resultSha256: string;
}
export type HouseComparisonAvailability =
  | { status: "eligible"; reason: null; group: HouseComparisonGroup }
  | { status: "unavailable"; reason: string };
export type VerifiedHouseComparison = {
  status: "comparable";
  reason: null;
  primary: HouseComparisonResult;
  reference: HouseComparisonResult;
  group: HouseComparisonGroup;
};
export type HouseComparisonVerification =
  VerifiedHouseComparison | { status: "unavailable"; reason: string };
export interface HouseResultScore {
  observed: number;
  decisive: number;
  correct: number;
  ties: number;
  brier: number | null;
}
export interface HouseComparisonSummary {
  primary: HouseResultScore;
  reference: HouseResultScore;
  brierDifference: number | null;
  races: {
    raceId: string;
    candidates: {
      id: string;
      primaryProbability: number;
      referenceProbability: number;
      difference: number;
    }[];
  }[];
}

const object = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === "object" && !Array.isArray(value);
const text = (value: unknown): value is string =>
  typeof value === "string" &&
  value.length > 0 &&
  value.length <= 1000 &&
  !/[\u0000-\u001f\u007f]/.test(value);
const hash = (value: unknown): value is string =>
  typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
const unique = (ids: string[]) => new Set(ids).size === ids.length;
const equalIds = (a: string[], b: string[]) =>
  a.length === b.length &&
  unique(a) &&
  unique(b) &&
  a.every((id) => b.includes(id));

/** Trusted static manifest, checked against this source-bound release's identities. */
export function parseHouseComparisonManifest(
  value: unknown,
): HouseComparisonManifest | null {
  try {
    if (
      !object(value) ||
      value.schemaVersion !== 1 ||
      value.id !== "reviewed-house-same-year-20261020-v1" ||
      value.targetId !== "house_aggregate_fundamentals_retrospective_v1" ||
      value.denominator !== HOUSE_SHARE_DENOMINATOR ||
      value.nominalHorizonDays !== 14 ||
      value.informationMode !==
        "retrospective_proxy_strictly_earlier_fitted_labels" ||
      value.scope !== "eligible_race_subset" ||
      !object(value.sourcePins) ||
      !Array.isArray(value.groups) ||
      !value.groups.length ||
      value.groups.length > 128 ||
      !Array.isArray(value.limitations) ||
      !value.limitations.every(text)
    )
      return null;
    if (
      Object.keys(value.sourcePins).length !==
        Object.keys(HOUSE_COMPARISON_SOURCE_PINS).length ||
      Object.entries(HOUSE_COMPARISON_SOURCE_PINS).some(
        ([key, pin]) =>
          (value.sourcePins as Record<string, unknown>)[key] !== pin,
      )
    )
      return null;
    const years = new Set<number>(),
      runIds = new Set<string>();
    for (const group of value.groups) {
      if (
        !object(group) ||
        !Number.isSafeInteger(group.year) ||
        years.has(group.year as number) ||
        !text(group.mapVintage) ||
        !Number.isInteger(group.raceCount) ||
        (group.raceCount as number) < 1 ||
        (group.raceCount as number) > 435 ||
        !hash(group.supportSha256) ||
        !hash(group.identitySha256) ||
        !Array.isArray(group.raceIds) ||
        group.raceIds.length !== group.raceCount ||
        !group.raceIds.every(
          (id) =>
            typeof id === "string" &&
            new RegExp(`^${group.year}:[A-Z]{2}-[0-9]{2}:house:regular$`).test(
              id,
            ),
        ) ||
        !unique(group.raceIds) ||
        !Array.isArray(group.runs) ||
        group.runs.length < 2 ||
        group.runs.length > 128
      )
        return null;
      years.add(group.year as number);
      const configurations = new Set<string>();
      for (const run of group.runs) {
        if (
          !object(run) ||
          !text(run.runId) ||
          runIds.has(run.runId) ||
          !text(run.configurationId) ||
          configurations.has(run.configurationId) ||
          !text(run.jointArtifactId) ||
          ![
            run.configurationSha256,
            run.resultSha256,
            run.canonicalResultSha256,
            run.planSha256,
            run.fitReceiptSha256,
            run.jointSourceSha256,
          ].every(hash) ||
          run.planSha256 !== run.configurationSha256
        )
          return null;
        runIds.add(run.runId);
        configurations.add(run.configurationId);
      }
    }
    return structuredClone(value) as unknown as HouseComparisonManifest;
  } catch {
    return null;
  }
}

/** Canonical object digest is separate from the immutable file-byte digest. */
export function houseComparisonCanonicalJson(value: unknown): string {
  const normalize = (node: unknown): unknown => {
    if (Array.isArray(node)) return node.map(normalize);
    if (object(node))
      return Object.fromEntries(
        Object.keys(node)
          .sort()
          .filter((key) => node[key] !== undefined)
          .map((key) => [key, normalize(node[key])]),
      );
    if (typeof node === "number" && !Number.isFinite(node))
      throw Error("Nonfinite comparison content.");
    return node;
  };
  return JSON.stringify(normalize(value));
}
export async function houseComparisonDigest(value: unknown): Promise<string> {
  if (!globalThis.crypto?.subtle)
    throw Error("Secure result verification is unavailable in this browser.");
  const bytes = new TextEncoder().encode(houseComparisonCanonicalJson(value));
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
}

/** Candidate identities and observed truth, never prediction values or draw pairings. */
export function houseComparisonIdentity(result: HouseResult): unknown {
  return result.races
    .map((race) => ({
      id: race.id,
      state: race.state,
      seatId: race.seatId,
      district: race.district,
      electionLabel: race.electionLabel,
      outcomes: [...race.outcomes]
        .sort((a, b) => a.id.localeCompare(b.id))
        .map((outcome) => ({
          id: outcome.id,
          name: outcome.name,
          party: outcome.party,
          kind: outcome.kind,
          members: outcome.members,
          denominator: outcome.share?.denominator,
        })),
      observed: race.observed,
      pollCount: race.pollCount,
      notes: race.notes,
    }))
    .sort((a, b) => a.id.localeCompare(b.id));
}
export function houseComparisonSupport(result: HouseResult): unknown {
  return {
    chamber: result.metadata.chamber,
    chamberSize: result.metadata.chamber_size,
    majorityThreshold: result.metadata.majority_threshold,
    scope: result.metadata.scope,
    fullChamber: result.metadata.full_chamber,
    cycle: result.metadata.cycle,
    mapVintage: result.metadata.mapVintage,
    coverage: [...result.coverage].sort((a, b) =>
      a.raceId.localeCompare(b.raceId),
    ),
  };
}

/** IDs alone are preflight, not result or scientific compatibility verification. */
export function houseComparisonAvailability(
  manifest: HouseComparisonManifest,
  primaryRunId: string,
  referenceRunId: string,
): HouseComparisonAvailability {
  if (primaryRunId === referenceRunId)
    return {
      status: "unavailable",
      reason: "Choose a different model to compare.",
    };
  const parsed = parseHouseComparisonManifest(manifest);
  if (!parsed)
    return {
      status: "unavailable",
      reason: "The reviewed House comparison record is unavailable.",
    };
  const group = parsed.groups.find((group) =>
    group.runs.some((run) => run.runId === primaryRunId),
  );
  if (!group || !group.runs.some((run) => run.runId === referenceRunId))
    return {
      status: "unavailable",
      reason:
        "Compare published House models from the same election year and reviewed race collection.",
    };
  return { status: "eligible", reason: null, group };
}

/** Independently verify both complete result snapshots; never match simulation rows. */
export async function verifyHouseComparison(
  manifest: HouseComparisonManifest,
  primary: HouseComparisonResult,
  reference: HouseComparisonResult,
): Promise<HouseComparisonVerification> {
  try {
    const first = structuredClone(primary),
      second = structuredClone(reference);
    const availability = houseComparisonAvailability(
      manifest,
      first.result.run.id,
      second.result.run.id,
    );
    if (availability.status !== "eligible") return availability;
    const group = availability.group;
    for (const publication of [first, second]) {
      const result = parseHouseResult(publication.result),
        binding = group.runs.find((run) => run.runId === result?.run.id);
      if (
        !result ||
        !binding ||
        publication.resultSha256 !== binding.resultSha256 ||
        result.run.configurationId !== binding.configurationId ||
        result.run.configurationSha256 !== binding.configurationSha256 ||
        result.run.year !== group.year ||
        !result.run.historical ||
        result.run.informationMode !== manifest.informationMode ||
        result.metadata.mapVintage !== group.mapVintage ||
        result.metadata.scope !== manifest.scope ||
        result.metadata.full_chamber ||
        !equalIds(
          result.races.map((race) => race.id),
          group.raceIds,
        ) ||
        result.races.length !== group.raceCount ||
        result.races.some(
          (race) =>
            !race.observed ||
            race.outcomes.some(
              (outcome) =>
                !outcome.share ||
                outcome.share.denominator !== manifest.denominator,
            ),
        ) ||
        result.jointDraws?.id !== binding.jointArtifactId ||
        result.jointDraws?.sourceSha256 !== binding.jointSourceSha256
      )
        return {
          status: "unavailable",
          reason:
            "A loaded House result does not match its exact reviewed model, source, election and district coverage.",
        };
      if (
        (await houseComparisonDigest(result)) !==
          binding.canonicalResultSha256 ||
        (await houseComparisonDigest(houseComparisonSupport(result))) !==
          group.supportSha256 ||
        (await houseComparisonDigest(houseComparisonIdentity(result))) !==
          group.identitySha256
      )
        return {
          status: "unavailable",
          reason:
            "The loaded result, eligible districts, candidates or actual results differ from the reviewed comparison.",
        };
    }
    return {
      status: "comparable",
      reason: null,
      primary: first,
      reference: second,
      group,
    };
  } catch {
    return {
      status: "unavailable",
      reason:
        "The exact House comparison could not be verified. The displayed model can still be explored.",
    };
  }
}

/** Existing categorical win score, not a share score or an independent-race test. */
export function summarizeHouseResult(result: HouseResult): HouseResultScore {
  const parsed = parseHouseResult(result);
  if (!parsed)
    throw new TypeError("A valid House result is required for scores.");
  const observed = parsed.races.filter((race) => race.observed);
  const decisive = observed.filter(
    (race) =>
      Math.abs(
        race.outcomes[0].winProbability - race.outcomes[1].winProbability,
      ) > 1e-12,
  );
  return {
    observed: observed.length,
    decisive: decisive.length,
    correct: decisive.filter(
      (race) =>
        [...race.outcomes].sort(
          (a, b) => b.winProbability - a.winProbability,
        )[0].id === race.observed!.outcomeId,
    ).length,
    ties: observed.length - decisive.length,
    brier: observed.length
      ? observed.reduce(
          (sum, race) =>
            sum +
            race.outcomes.reduce(
              (score, outcome) =>
                score +
                (outcome.winProbability -
                  Number(outcome.id === race.observed!.outcomeId)) **
                  2,
              0,
            ),
          0,
        ) / observed.length
      : null,
  };
}
/** Differences are primary minus reference; probability differences are fractions. */
export function summarizeHouseComparison(
  comparison: VerifiedHouseComparison,
): HouseComparisonSummary {
  if (comparison.status !== "comparable")
    throw new TypeError(
      "Verify the same-year House comparison before summarizing it.",
    );
  const primary = summarizeHouseResult(comparison.primary.result),
    reference = summarizeHouseResult(comparison.reference.result);
  const byId = new Map(
    comparison.reference.result.races.map((race) => [race.id, race]),
  );
  return {
    primary,
    reference,
    brierDifference:
      primary.brier !== null && reference.brier !== null
        ? primary.brier - reference.brier
        : null,
    races: comparison.primary.result.races.map((race) => ({
      raceId: race.id,
      candidates: race.outcomes.map((outcome) => {
        const base = byId
          .get(race.id)!
          .outcomes.find((candidate) => candidate.id === outcome.id)!;
        return {
          id: outcome.id,
          primaryProbability: outcome.winProbability,
          referenceProbability: base.winProbability,
          difference: outcome.winProbability - base.winProbability,
        };
      }),
    })),
  };
}
