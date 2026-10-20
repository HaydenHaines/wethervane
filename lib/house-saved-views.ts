import type { HouseCatalog } from "./house-catalog";
import type { HouseResult } from "./house-result";
import { parseHouseResult } from "./house-result.ts";

export const HOUSE_SAVED_VIEWS_STORAGE_KEY = "ballotmancy.house.saved-views.v1";
export const MAX_HOUSE_SAVED_VIEWS = 30;
const MAX_STORED_LENGTH = 128_000;
const MAX_LABEL_LENGTH = 80;
const CONTROL = /[\u0000-\u001f\u007f]/;

/** Browser-owned view identity. Labels never substitute for scientific bindings. */
export interface HouseSavedView {
  id: string;
  label: string;
  runLabel?: string;
  savedAt: string;
  runId: string;
  resultSha256: string;
  configurationSha256: string;
  year: number;
  mapVintage: string;
  raceId: string;
  mapView: "forecast" | "actual";
}
export interface HouseSavedResult {
  /** A publication already admitted by the caller's loader. */
  result: HouseResult;
  resultSha256: string;
}
export interface HouseSavedViewAvailability {
  status: "catalog_compatible" | "unavailable";
  reason: string | null;
  /** This stage cannot verify selected district, map vintage or actual result. */
  verification: "catalog_only";
}
export type HouseSavedViewVerification =
  | {
      status: "verified";
      reason: null;
      verification: "exact_result";
      selection: { raceId: string; mapView: "forecast" | "actual" };
    }
  | {
      status: "unavailable";
      reason: string;
      verification: "exact_result";
      selection?: never;
    };

const object = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === "object" && !Array.isArray(value);
const text = (value: unknown, maximum = 512): value is string =>
  typeof value === "string" &&
  value.length > 0 &&
  value.length <= maximum &&
  value === value.trim() &&
  !CONTROL.test(value);
const hash = (value: unknown): value is string =>
  typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
const runId = (value: unknown): value is string =>
  typeof value === "string" && /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,159}$/.test(value);
function timestamp(value: unknown): value is string {
  if (typeof value !== "string" || value.length > 40) return false;
  const date = new Date(value);
  return Number.isFinite(date.getTime()) && date.toISOString() === value;
}
function canonicalView(value: unknown): HouseSavedView | null {
  if (
    !object(value) ||
    !text(value.id, 160) ||
    !text(value.label, MAX_LABEL_LENGTH) ||
    (value.runLabel !== undefined && !text(value.runLabel, 1000)) ||
    !timestamp(value.savedAt) ||
    !runId(value.runId) ||
    !hash(value.resultSha256) ||
    !hash(value.configurationSha256) ||
    typeof value.year !== "number" ||
    !Number.isSafeInteger(value.year) ||
    value.year < 1 ||
    value.year > 9999 ||
    !text(value.mapVintage) ||
    !text(value.raceId) ||
    !new RegExp(`^${value.year}:[A-Z]{2}-[0-9]{2}:house:regular$`).test(
      value.raceId,
    ) ||
    !["forecast", "actual"].includes(value.mapView as string)
  )
    return null;
  return {
    id: value.id,
    label: value.label,
    ...(value.runLabel === undefined ? {} : { runLabel: value.runLabel }),
    savedAt: value.savedAt,
    runId: value.runId,
    resultSha256: value.resultSha256,
    configurationSha256: value.configurationSha256,
    year: value.year,
    mapVintage: value.mapVintage,
    raceId: value.raceId,
    mapView: value.mapView as HouseSavedView["mapView"],
  };
}

/** Retain valid stale/missing-catalog entries so the user can inspect or delete them. */
export function parseHouseSavedViews(raw: string | null): HouseSavedView[] {
  if (typeof raw !== "string" || raw.length > MAX_STORED_LENGTH) return [];
  try {
    const stored: unknown = JSON.parse(raw);
    if (
      !object(stored) ||
      stored.schemaVersion !== 1 ||
      !Array.isArray(stored.views)
    )
      return [];
    const ids = new Set<string>(),
      accepted: HouseSavedView[] = [];
    for (const value of stored.views.slice(0, MAX_HOUSE_SAVED_VIEWS)) {
      const view = canonicalView(value);
      if (!view || ids.has(view.id)) continue;
      ids.add(view.id);
      accepted.push(view);
    }
    return accepted;
  } catch {
    return [];
  }
}

/** Versioned, bounded serialization; unknown fields never enter browser storage. */
export function serializeHouseSavedViews(views: HouseSavedView[]): string {
  if (!Array.isArray(views))
    throw new TypeError("House saved views must be an array.");
  const accepted: HouseSavedView[] = [],
    ids = new Set<string>();
  for (const value of views.slice(0, MAX_HOUSE_SAVED_VIEWS)) {
    const view = canonicalView(value);
    if (!view || ids.has(view.id)) continue;
    ids.add(view.id);
    accepted.push(view);
  }
  const raw = JSON.stringify({ schemaVersion: 1, views: accepted });
  if (raw.length > MAX_STORED_LENGTH)
    throw new TypeError("House saved views exceed the browser storage bound.");
  return raw;
}

/** Catalog preflight only. A matching entry still needs the exact result loaded. */
export function houseSavedViewAvailability(
  view: HouseSavedView,
  catalog?: HouseCatalog | null,
): HouseSavedViewAvailability {
  const unavailable = (reason: string): HouseSavedViewAvailability => ({
    status: "unavailable",
    reason,
    verification: "catalog_only",
  });
  if (!canonicalView(view))
    return unavailable("This saved view has invalid identity fields.");
  if (!catalog || !Array.isArray(catalog.runs))
    return unavailable(
      "The House catalog is not available yet; this saved view has been kept.",
    );
  const matches = catalog.runs.filter((run) => run.id === view.runId);
  if (matches.length !== 1)
    return unavailable(
      "The exact saved House model is no longer in this catalog. This view has not been replaced with another result.",
    );
  const run = matches[0];
  if (!run.available)
    return unavailable(
      run.unavailableReason ||
        "The saved House model is currently unavailable.",
    );
  if (run.resultSha256 !== view.resultSha256)
    return unavailable(
      "The result has changed since this view was saved. The original result is needed to restore it.",
    );
  if (
    run.configurationSha256 !== view.configurationSha256 ||
    run.year !== view.year
  )
    return unavailable(
      "The saved configuration or election year differs from the catalog.",
    );
  if (view.mapView === "actual" && !run.historical)
    return unavailable(
      "Actual-winner view is unavailable for this saved model.",
    );
  return {
    status: "catalog_compatible",
    reason: null,
    verification: "catalog_only",
  };
}

/**
 * Verify exact dated district selection after loading an admitted publication.
 * This checks identity bindings, not the file's cryptographic bytes or geometry
 * admission. A missing map can still restore this exact race in the list/detail.
 */
export function verifyHouseSavedView(
  view: HouseSavedView,
  publication: HouseSavedResult,
): HouseSavedViewVerification {
  const unavailable = (reason: string): HouseSavedViewVerification => ({
    status: "unavailable",
    reason,
    verification: "exact_result",
  });
  try {
    if (!canonicalView(view) || !publication || !hash(publication.resultSha256))
      return unavailable(
        "The saved view or loaded result identity is invalid.",
      );
    const result = parseHouseResult(publication.result);
    if (!result)
      return unavailable("The loaded House result did not pass validation.");
    if (
      view.runId !== result.run.id ||
      view.resultSha256 !== publication.resultSha256 ||
      view.configurationSha256 !== result.run.configurationSha256 ||
      view.year !== result.run.year
    )
      return unavailable(
        "The loaded result does not match the exact saved model, configuration, result and year.",
      );
    if (view.mapVintage !== result.metadata.mapVintage)
      return unavailable(
        "The saved district map edition differs from the loaded result. This selection has not been replaced with another district.",
      );
    const matches = result.races.filter((race) => race.id === view.raceId);
    if (matches.length !== 1)
      return unavailable(
        "The exact saved race is not included in this result. No substitute district has been selected.",
      );
    const race = matches[0];
    if (
      race.district.cycle !== view.year ||
      race.district.mapVintage !== view.mapVintage ||
      !result.coverage.some(
        (row) =>
          row.eligible &&
          row.raceId === race.id &&
          row.districtKey === race.district.districtKey,
      )
    )
      return unavailable(
        "The saved race does not match the districts included for this election and map.",
      );
    if (view.mapView === "actual" && !result.run.historical)
      return unavailable(
        "Actual-winner view is unavailable for a model of an election that has not happened yet.",
      );
    return {
      status: "verified",
      reason: null,
      verification: "exact_result",
      selection: { raceId: view.raceId, mapView: view.mapView },
    };
  } catch {
    return unavailable("The loaded result could not verify this saved view.");
  }
}

/** Create a local view from the displayed, verified result, never from a draft. */
export function createHouseSavedView(
  label: string,
  publication: HouseSavedResult,
  selection: { raceId: string; mapView: "forecast" | "actual" },
  identity?: { id?: string; savedAt?: string },
): HouseSavedView {
  if (typeof label !== "string")
    throw new TypeError("A saved view label must be text.");
  let bounded = label
    .trim()
    .replace(/[\u0000-\u001f\u007f]/g, " ")
    .slice(0, MAX_LABEL_LENGTH);
  if (/[\uD800-\uDBFF]$/.test(bounded)) bounded = bounded.slice(0, -1);
  bounded = bounded.trim() || "Saved House view";
  if (!publication?.result?.run || !selection)
    throw new TypeError(
      "A displayed House result and race selection are required.",
    );
  const { result } = publication;
  const value: HouseSavedView = {
    id: identity?.id ?? globalThis.crypto.randomUUID(),
    label: bounded,
    runLabel: result.run.label,
    savedAt: identity?.savedAt ?? new Date().toISOString(),
    runId: result.run.id,
    resultSha256: publication.resultSha256,
    configurationSha256: result.run.configurationSha256,
    year: result.run.year,
    mapVintage: result.metadata.mapVintage,
    raceId: selection.raceId,
    mapView: selection.mapView,
  };
  const canonical = canonicalView(value);
  if (!canonical)
    throw new TypeError(
      "This House view has invalid identity or selection fields.",
    );
  const verified = verifyHouseSavedView(canonical, publication);
  if (verified.status !== "verified") throw new TypeError(verified.reason);
  return canonical;
}
