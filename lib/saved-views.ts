import type {
  CompositionCatalog,
  SavedConfiguration,
  SavedView,
} from "./composition-types";

const MAX_VIEWS = 30;
const MAX_NAME_LENGTH = 80;
const MAX_ID_LENGTH = 512;
const MAX_STORED_LENGTH = 128_000;
const CONTROL_CHARACTERS = /[\u0000-\u001f\u007f]/;

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function identifier(value: unknown): value is string {
  return typeof value === "string" && value.length > 0 && value.length <= MAX_ID_LENGTH &&
    value === value.trim() && !CONTROL_CHARACTERS.test(value);
}

function optionalIdentifier(value: unknown): value is string | null {
  return value === null || identifier(value);
}

function fingerprint(value: unknown): value is string {
  return typeof value === "string" && /^[a-fA-F0-9]{64}$/.test(value);
}

/** Parse browser-owned data without changing or rebinding its scientific identity. */
export function parseSavedViews(value: string | null, catalog: CompositionCatalog): SavedView[] {
  if (typeof value !== "string" || value.length > MAX_STORED_LENGTH) return [];
  let parsed: unknown;
  try {
    parsed = JSON.parse(value);
  } catch {
    return [];
  }
  if (!Array.isArray(parsed) || !Array.isArray(catalog?.saved)) return [];
  const accepted: SavedView[] = [];
  const ids = new Set<string>();
  const availableRuns = new Set(catalog.saved.map((item) => item?.runId));
  // Bound both parsing work and the stored collection; entries beyond the cap
  // are ignored rather than allowing an unbounded malformed collection.
  for (const item of parsed.slice(0, MAX_VIEWS)) {
    if (!record(item) || !identifier(item.id) || ids.has(item.id) ||
      typeof item.name !== "string" || !item.name.trim() ||
      item.name.length > MAX_NAME_LENGTH || CONTROL_CHARACTERS.test(item.name) ||
      !identifier(item.configurationId) || !identifier(item.runId) ||
      !fingerprint(item.resultSha256) || !optionalIdentifier(item.scenarioId) ||
      !optionalIdentifier(item.raceId) || typeof item.compare !== "boolean" ||
      !["forecast", "actual"].includes(item.mapView as string)) continue;
    const baseline = item.baselineRunId === undefined ? null : item.baselineRunId;
    if (!optionalIdentifier(baseline) || (baseline !== null && !availableRuns.has(baseline))) continue;
    const configuration = catalog.saved.find((candidate) =>
      candidate?.id === item.configurationId && candidate.runId === item.runId &&
      candidate.resultSha256 === item.resultSha256,
    );
    if (!configuration) continue;
    ids.add(item.id);
    accepted.push({
      id: item.id,
      name: item.name.trim(),
      configurationId: item.configurationId,
      runId: item.runId,
      resultSha256: item.resultSha256,
      scenarioId: item.scenarioId,
      raceId: item.raceId,
      compare: item.compare,
      baselineRunId: baseline,
      mapView: item.mapView as SavedView["mapView"],
    });
  }
  return accepted;
}

/** Create a fresh local view; persistence remains the caller's responsibility. */
export function createSavedView(
  name: string,
  configuration: SavedConfiguration,
  selection: {
    runId: string;
    scenarioId: string | null;
    raceId: string | null;
    compare: boolean;
    mapView: "forecast" | "actual";
    baselineRunId?: string | null;
  },
): SavedView {
  if (!identifier(configuration.id) || !identifier(configuration.runId) || !fingerprint(configuration.resultSha256))
    throw new TypeError("The saved configuration has no valid bound result identity.");
  if (selection.runId !== configuration.runId)
    throw new TypeError("The displayed run does not match the saved configuration.");
  if (!optionalIdentifier(selection.scenarioId) || !optionalIdentifier(selection.raceId) ||
    !optionalIdentifier(selection.baselineRunId ?? null) || typeof selection.compare !== "boolean" ||
    !["forecast", "actual"].includes(selection.mapView))
    throw new TypeError("The selected view contains invalid fields.");
  if (typeof name !== "string") throw new TypeError("A saved view name must be text.");
  let boundedName = name.trim().replace(/[\u0000-\u001f\u007f]/g, " ").slice(0, MAX_NAME_LENGTH);
  // Avoid cutting a Unicode surrogate pair at the UTF-16 input length limit.
  if (/[\uD800-\uDBFF]$/.test(boundedName)) boundedName = boundedName.slice(0, -1);
  boundedName = boundedName.trim() || "Saved view";
  return {
    id: globalThis.crypto.randomUUID(),
    name: boundedName,
    configurationId: configuration.id,
    runId: configuration.runId,
    resultSha256: configuration.resultSha256,
    scenarioId: selection.scenarioId,
    raceId: selection.raceId,
    compare: selection.compare,
    baselineRunId: selection.baselineRunId ?? null,
    mapView: selection.mapView,
  };
}
