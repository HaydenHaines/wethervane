/** Runtime site publication decision for optional, exact-edition reference data. */
export interface ElectionNightReferenceQualification {
  targetId: string; lawId: string; forecastIssuedAt: string; forecastCutoffAt: string;
  referenceIssuedAt: string; cohortSha256: string; sourcePins: Record<string, string>;
  shape: number[]; rowCount: number;
  configuration?: { base: string; polling: string; uncertainty: string }; strictAsOf?: boolean;
  structuralCutoffAt?: string;
  arm?: "F" | "FP"; cohortId?: string; excludedRaceIds?: string[]; raceAxisBinding?: string;
}
export interface ElectionNightReferencePublication {
  schemaVersion: number; href: string; sha256: string;
  editions: { chamber: string; runId: string; resultSha256: string;
    qualification?: ElectionNightReferenceQualification }[];
}
export interface ElectionNightReferencePublicationIndex {
  schemaVersion: 1; full: ElectionNightReferencePublication; primary: ElectionNightReferencePublication;
}
const hash = (value: unknown): value is string => typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
const text = (value: unknown): value is string => typeof value === "string" && value.trim().length > 0;
function publication(value: unknown, href: string): value is ElectionNightReferencePublication {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const p = value as ElectionNightReferencePublication;
  return p.schemaVersion === 1 && p.href === href && hash(p.sha256) && Array.isArray(p.editions) &&
    p.editions.every(e => e && ["senate", "house"].includes(e.chamber) &&
      typeof e.runId === "string" && /^[A-Za-z0-9_-]+$/.test(e.runId) && hash(e.resultSha256) &&
      e.qualification && text(e.qualification.targetId) && text(e.qualification.lawId) &&
      text(e.qualification.forecastIssuedAt) && text(e.qualification.forecastCutoffAt) &&
      text(e.qualification.referenceIssuedAt) && hash(e.qualification.cohortSha256) &&
      e.qualification.sourcePins && typeof e.qualification.sourcePins === "object" &&
      !Array.isArray(e.qualification.sourcePins) &&
      Object.values(e.qualification.sourcePins).every(hash) && hash(e.qualification.sourcePins.reviewSha256) &&
      hash(e.qualification.sourcePins.referenceSha256) && hash(e.qualification.sourcePins.forecastSha256) &&
      hash(e.qualification.sourcePins.jointDrawsSha256) &&
      Array.isArray(e.qualification.shape) && e.qualification.shape.length > 0 &&
      e.qualification.shape.every(n => Number.isSafeInteger(n) && n > 0) &&
      Number.isSafeInteger(e.qualification.rowCount) && e.qualification.rowCount > 0 &&
      e.qualification.shape.reduce((product, n) => product * n, 1) === e.qualification.rowCount) &&
    p.editions.every(e => e.qualification?.configuration === undefined ||
      (e.qualification.configuration && typeof e.qualification.configuration === "object" &&
        text(e.qualification.configuration.base) && text(e.qualification.configuration.polling) &&
        text(e.qualification.configuration.uncertainty) && typeof e.qualification.strictAsOf === "boolean")) &&
    p.editions.every(e => href.endsWith("/references.json") ?
      text(e.qualification?.structuralCutoffAt) && hash(e.qualification.sourcePins.receiptSha256) &&
        hash(e.qualification.sourcePins.numericalGatesSha256) :
      e.chamber === "senate" && ["F", "FP"].includes(e.qualification?.arm ?? "") &&
        text(e.qualification?.cohortId) && text(e.qualification?.raceAxisBinding) &&
        text(e.qualification?.structuralCutoffAt) &&
        Array.isArray(e.qualification?.excludedRaceIds) &&
        e.qualification.excludedRaceIds.every(text) &&
        hash(e.qualification?.sourcePins?.policySha256) && hash(e.qualification?.sourcePins?.mappingSha256)) &&
    new Set(p.editions.map(e => `${e.chamber}:${e.runId}:${e.resultSha256}`)).size === p.editions.length;
}
export function parseElectionNightPublicationIndex(value: unknown): ElectionNightReferencePublicationIndex | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const p = value as ElectionNightReferencePublicationIndex;
  return p.schemaVersion === 1 && publication(p.full, "/data/election-night/references.json") &&
    publication(p.primary, "/data/election-night/core-references.json") ? p : null;
}
