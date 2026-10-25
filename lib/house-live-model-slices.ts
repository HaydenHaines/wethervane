import type { ComponentOption } from "./composition-types.ts";
import type { HouseLiveCatalogView } from "./house-live-view.ts";
import type { HouseLiveConfiguration } from "./house-live-result.ts";
import { MODEL_SLICES, type ModelSliceId, type ModelSliceView } from "./model-slices.ts";
import { resolveHouseLiveSelection } from "./house-live-selection.ts";

const BASES = [
  { id: "fundamentals", label: "Electoral history", description: "Uses earlier election results as the starting estimate for current House contests." },
  { id: "recent_calendar_history", label: "Recent calendar history", description: "Emphasizes election cycles from the recent calendar." },
  { id: "house_history_refreshed_v1", label: "Refreshed history", description: "Uses refreshed earlier House and presidential results." },
  { id: "house_presidential_slope_v1", label: "Learned presidential slope", description: "Learns how earlier district presidential voting relates to House results." },
  { id: "house_relative_lean_v1", label: "Relative district and national lean", description: "Separates district presidential lean from the national result." },
  { id: "house_relative_context_v1", label: "Relative lean with presidential context", description: "Adds the president's party and midterm context to relative lean." },
] as const;
const UNCERTAINTIES = [
  { id: "standard", label: "Standard uncertainty", description: "Uses this saved model's ordinary assumptions about polling errors and districts moving together." },
  { id: "cycle-low", label: "Smaller shared polling error (× 0.5)", description: "Multiplies only the election-wide polling-error standard deviation by 0.5. Standard deviation describes the typical size of an error." },
  { id: "cycle-high", label: "Larger shared polling error (× 2)", description: "Multiplies only the election-wide polling-error standard deviation by 2. Standard deviation describes the typical size of an error." },
  { id: "error-low", label: "Smaller overall polling error (× 2/3)", description: "Multiplies all modeled polling-error standard deviations by 2/3, including shared and race-level errors. Standard deviation describes the typical size of an error." },
  { id: "error-high", label: "Larger overall polling error (× 1.5)", description: "Multiplies all modeled polling-error standard deviations by 1.5, including shared and race-level errors. Standard deviation describes the typical size of an error." },
] as const;

const unavailable: Partial<Record<ModelSliceId, string>> = {
  population: "This House forecast has no population model. The district map locates races; it does not estimate residents or voter counts.",
  community: "This House forecast does not estimate community composition or overlapping groups.",
  participation: "This House forecast does not model turnout or predict how many people vote.",
};

/** Show all eight roles while offering only exact saved three-slot compositions. */
export function houseLiveModelSlices(catalog: HouseLiveCatalogView, selected: HouseLiveConfiguration): ModelSliceView[] {
  const option = (slot: keyof HouseLiveConfiguration, id: string, label: string, description: string): ComponentOption => {
    const candidate = { ...selected, [slot]: id } as HouseLiveConfiguration;
    const resolution = resolveHouseLiveSelection(catalog, candidate);
    return { id, label, description, years: [2026],
      availability: resolution.state === "ready" ? "saved" : resolution.state === "requires_computation" ? "requires_computation" : "unavailable",
      experimental: false };
  };
  return MODEL_SLICES.map(slice => {
    if (slice.id === "fundamentals") return {
      id: slice.id, label: slice.label, purpose: "Choose one saved electoral-history base for the current House contests.",
      unavailableReason: null,
      controls: [{ id: "house-live-base", label: "Electoral-history model", value: selected.base, required: true,
        description: "Each choice belongs to a complete saved forecast.",
        options: BASES.map(base => option("base", base.id, base.label, base.description)) }],
    };
    if (slice.id === "environment") return {
      id: slice.id, label: slice.label, purpose: "Uncertainty includes errors that can shift many districts together.",
      unavailableReason: null,
      controls: [{ id: "house-live-uncertainty", label: "Uncertainty scenario", value: selected.uncertainty, required: true,
        description: "These settings change assumptions about the typical size of polling errors. Each saved setting is a separate complete forecast.",
        options: UNCERTAINTIES.map(row => option("uncertainty", row.id, row.label, row.description)) }],
    };
    if (slice.id === "measurement") return {
      id: slice.id, label: slice.label, purpose: "Choose a separate saved forecast with or without selected current polls.",
      unavailableReason: null,
      controls: [{ id: "house-live-polling", label: "Polling evidence", value: selected.polling, required: false,
        description: "A polling choice loads a complete saved forecast; it does not recompute the displayed run.",
        toggle: { label: "Include current polling evidence", offValue: "none", onValue: "standard" },
        options: [
          option("polling", "none", "Without polls", "Uses the saved electoral-history model without poll measurements."),
          option("polling", "standard", "With polls", "Uses the saved current polling measurements and their documented uncertainty."),
        ] }],
    };
    if (slice.id === "choice") return {
      id: slice.id, label: slice.label, purpose: "Outcomes are Democratic, Republican, or Other candidate groups for each district.",
      controls: [], unavailableReason: null, builtIn: true,
      builtInReason: "The list of current candidates and the possible winner groups are fixed in each saved result. A group's chance is not divided among its candidates.",
    };
    if (slice.id === "counts_rules") return {
      id: slice.id, label: slice.label, purpose: "Each saved result declares its treatment of district election rules.",
      controls: [], unavailableReason: null, builtIn: true,
      builtInReason: "Each result includes its own treatment of district election rules and, when supported, House seat totals. It does not predict total vote counts, and these rules cannot be switched separately.",
    };
    return { id: slice.id, label: slice.label, purpose: slice.purpose, controls: [], unavailableReason: unavailable[slice.id] ?? slice.unavailable };
  });
}
