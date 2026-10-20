import type { HouseComparisonResult } from "./house-comparison";
import {
  HOUSE_COMPARISON_SOURCE_PINS,
  houseComparisonDigest,
} from "./house-comparison.ts";
import { parseHouseResult } from "./house-result.ts";

export const HOUSE_SCORE_EVIDENCE_SOURCE_PINS = {
  ...HOUSE_COMPARISON_SOURCE_PINS,
  scoreReceipt:
    "c9e77fbf7d6d2263ecc1479ed324416e4284b995ea5ecca6509ba4271c0b7fbf",
  perRaceScores:
    "be2fb76fdd18822a0d4dfedefb1e1f18509c5822b575e8e9a0b5f047145cfc81",
  cycleScores:
    "86b822702b74b82f9f0c1b04afefb11c5aca426f8969dfe1f9dcda431d94ab6c",
  observedLabels:
    "45eb03bc69205763cfbef5014ce9e505dc0aa1cfe8a18ee8d7ca9b7a2d5f9c37",
  rosterFeatures:
    "3bbacb9487d45280fc131fa2f7754767b7939c32f5c282af4fc48af99d81a9c4",
  precisionReceipt:
    "0c9fb0eaf1d96ad9cefe8b3ed896765b63c3197f08f79590e6b283438fb31eca",
  reportReceipt:
    "09e755eda4140f41d449b61f114f1d24364a3488f67a79de514f89b0dd9d9739",
} as const;
// Canonical entries digest pins saved metrics as well as exact result contents.
export const HOUSE_SCORE_EVIDENCE_ENTRIES_SHA256 =
  "4f0857ae006132265194ffa29d9d0033e9fbb5c495c12b65f93f6d99f5978495";
export interface HouseScoreEvidenceEntry {
  runId: string;
  configurationId: string;
  configurationSha256: string;
  resultSha256: string;
  canonicalResultSha256: string;
  fitReceiptSha256: string;
  year: number;
  mapVintage: string;
  raceIds: string[];
  raceCount: number;
  units: "share_fraction";
  intervalLevel: 0.9;
  scope: "eligible_race_subset";
  total: number;
  covered: number;
  meanAbsoluteError: number;
  meanIntervalWidth: number;
  crps: number;
  intervalScore: number;
}
export interface HouseScoreEvidenceManifest {
  schemaVersion: 1;
  id: "reviewed-house-share-tests-20261020-v1";
  sourcePins: Record<keyof typeof HOUSE_SCORE_EVIDENCE_SOURCE_PINS, string>;
  qualifications: string[];
  entries: HouseScoreEvidenceEntry[];
}
export type HouseScoreEvidenceVerification =
  | { status: "verified"; reason: null; summary: HouseScoreEvidenceEntry }
  | { status: "unavailable"; reason: string };
const record = (v: unknown): v is Record<string, unknown> =>
  !!v && typeof v === "object" && !Array.isArray(v);
const hash = (v: unknown) => typeof v === "string" && /^[a-f0-9]{64}$/.test(v);
const text = (v: unknown) =>
  typeof v === "string" &&
  v.length > 0 &&
  v.length < 2000 &&
  !/[\u0000-\u001f\u007f]/.test(v);
/** Structural check only; async verification also checks the immutable summary digest. */
export function parseHouseScoreEvidenceManifest(
  value: unknown,
): HouseScoreEvidenceManifest | null {
  try {
    if (
      !record(value) ||
      value.schemaVersion !== 1 ||
      value.id !== "reviewed-house-share-tests-20261020-v1" ||
      !record(value.sourcePins) ||
      !Array.isArray(value.entries) ||
      value.entries.length !== 16 ||
      !Array.isArray(value.qualifications) ||
      !value.qualifications.length ||
      !value.qualifications.every(text)
    )
      return null;
    if (
      Object.keys(value.sourcePins).length !==
        Object.keys(HOUSE_SCORE_EVIDENCE_SOURCE_PINS).length ||
      Object.entries(HOUSE_SCORE_EVIDENCE_SOURCE_PINS).some(
        ([k, v]) => (value.sourcePins as Record<string, unknown>)[k] !== v,
      )
    )
      return null;
    const ids = new Set();
    for (const e of value.entries) {
      if (
        !record(e) ||
        !text(e.runId) ||
        ids.has(e.runId) ||
        !text(e.configurationId) ||
        !text(e.mapVintage) ||
        !Number.isInteger(e.year) ||
        ![
          e.configurationSha256,
          e.resultSha256,
          e.canonicalResultSha256,
          e.fitReceiptSha256,
        ].every(hash) ||
        e.units !== "share_fraction" ||
        e.intervalLevel !== 0.9 ||
        e.scope !== "eligible_race_subset" ||
        !Number.isInteger(e.total) ||
        (e.total as number) < 1 ||
        (e.total as number) > 435 ||
        e.raceCount !== e.total ||
        !Number.isInteger(e.covered) ||
        (e.covered as number) < 0 ||
        (e.covered as number) > (e.total as number) ||
        !Array.isArray(e.raceIds) ||
        e.raceIds.length !== e.total ||
        !e.raceIds.every(text) ||
        new Set(e.raceIds).size !== e.total
      )
        return null;
      if (
        ![e.meanAbsoluteError, e.meanIntervalWidth, e.crps].every(
          (n) =>
            typeof n === "number" && Number.isFinite(n) && n >= 0 && n <= 1,
        ) ||
        typeof e.intervalScore !== "number" ||
        !Number.isFinite(e.intervalScore) ||
        e.intervalScore < 0 ||
        e.intervalScore > 21
      )
        return null;
      ids.add(e.runId);
    }
    return structuredClone(value) as unknown as HouseScoreEvidenceManifest;
  } catch {
    return null;
  }
}
/** Exact selected result only. No array reconstruction, inferred scores or interval claims. */
export async function verifyHouseScoreEvidence(
  manifest: unknown,
  publication: HouseComparisonResult,
): Promise<HouseScoreEvidenceVerification> {
  const unavailable = {
    status: "unavailable" as const,
    reason:
      "Vote-share testing evidence could not be verified for this exact saved result.",
  };
  try {
    const parsed = parseHouseScoreEvidenceManifest(manifest),
      snapshot = structuredClone(publication);
    if (
      !parsed ||
      (await houseComparisonDigest(parsed.entries)) !==
        HOUSE_SCORE_EVIDENCE_ENTRIES_SHA256
    )
      return unavailable;
    const result = parseHouseResult(snapshot.result),
      summary = parsed.entries.find((e) => e.runId === result?.run.id);
    if (
      !result ||
      !summary ||
      snapshot.resultSha256 !== summary.resultSha256 ||
      result.run.configurationId !== summary.configurationId ||
      result.run.configurationSha256 !== summary.configurationSha256 ||
      result.run.year !== summary.year ||
      result.metadata.mapVintage !== summary.mapVintage ||
      result.metadata.scope !== summary.scope ||
      result.metadata.full_chamber ||
      !result.run.historical ||
      result.races.length !== summary.total ||
      result.races.some((r) => !summary.raceIds.includes(r.id)) ||
      (await houseComparisonDigest(result)) !== summary.canonicalResultSha256
    )
      return unavailable;
    return { status: "verified", reason: null, summary };
  } catch {
    return unavailable;
  }
}
