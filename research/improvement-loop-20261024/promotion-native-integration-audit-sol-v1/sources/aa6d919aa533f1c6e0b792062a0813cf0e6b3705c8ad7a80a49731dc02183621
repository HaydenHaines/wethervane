import type { HouseLiveCatalogView, HouseLiveRunChoice } from "./house-live-view.ts";
import { houseLiveConfigurationKey, parseHouseLiveConfiguration,
  type HouseLiveConfiguration } from "./house-live-result.ts";

export type HouseLiveSelectionResolution =
  | { state: "ready"; entry: HouseLiveRunChoice; reason: string }
  | { state: "requires_computation" | "unsupported"; entry: null; reason: string };

/** Resolve only an exact, reviewed saved choice. This never constructs a forecast. */
export function resolveHouseLiveSelection(
  catalog: HouseLiveCatalogView,
  selection: HouseLiveConfiguration,
): HouseLiveSelectionResolution {
  if (!catalog || !Array.isArray(catalog.entries) || !selection ||
    !["fundamentals", "recent_calendar_history", "house_history_refreshed_v1", "house_presidential_slope_v1",
      "house_relative_lean_v1", "house_relative_context_v1"].includes(selection.base) ||
    !["none", "standard"].includes(selection.polling) ||
    !["standard", "cycle-low", "cycle-high", "error-low", "error-high"].includes(selection.uncertainty))
    return { state: "unsupported", entry: null, reason: "This component choice is not supported by the House model contract." };
  const key = houseLiveConfigurationKey(selection);
  const matches = catalog.entries.filter(entry =>
    parseHouseLiveConfiguration(entry.configuration) && houseLiveConfigurationKey(entry.configuration) === key);
  if (matches.length === 1 && /^[0-9a-f]{64}$/.test(matches[0].resultSha256))
    return { state: "ready", entry: matches[0], reason: "An exact saved, reviewed result is available for these choices." };
  if (matches.length > 1)
    return { state: "unsupported", entry: null, reason: "The saved House catalog has an ambiguous component choice." };
  if (selection.uncertainty !== "standard" && (selection.base !== "fundamentals" || selection.polling !== "standard"))
    return { state: "requires_computation", entry: null,
      reason: "This sensitivity has no saved result with the selected history and polling choices. It needs a new joint calculation and review; this site cannot run one." };
  return { state: "requires_computation", entry: null,
    reason: "No exact saved House result matches these choices. A new calculation and review would be required; this site cannot run one." };
}

export function sameHouseLiveConfiguration(a: HouseLiveConfiguration, b: HouseLiveConfiguration): boolean {
  return houseLiveConfigurationKey(a) === houseLiveConfigurationKey(b);
}
