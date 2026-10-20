import type { HouseCatalog } from "./house-catalog";
import type { HouseResult } from "./house-result";
import { MODEL_SLICES, type ModelSliceId, type ModelSliceView } from "./model-slices.ts";

/**
 * House source-capability-policy-v1: historical aggregate fundamentals only.
 * This adapter does not inherit Senate providers or turn capability flags into
 * component admission. Future House components need their own reviewed adapter.
 */
const HOUSE_REASONS: Record<ModelSliceId, string> = {
  population: "A House population model is not connected. These results use dated district rosters and maps, rather than modeling the people represented in each district.",
  community: "House community composition is not estimated in these district-level forecasts. No separate community component is available.",
  fundamentals: "No House fundamentals choices are available for this result.",
  environment: "Shared election-wide uncertainty is bundled into each saved House model. There is no separate national-environment or shared-error selector in this collection.",
  measurement: "Polling is not connected to this collection. There is no House polling evidence to switch on or off here.",
  participation: "A House turnout and participation model is not connected. These results predict conditional two-party shares, not voter counts.",
  choice: "Candidate context and the Democratic–Republican candidate pair are fixed within each saved House model. A separate candidate-choice component is not available.",
  counts_rules: "The races covered by this study do not make up the full 435-seat House. Full-chamber coverage is missing, and a separate vote-count or election-rule selector is unavailable.",
};

/** Exact saved configurations remain the unit of choice; no component refitting. */
export function houseModelSlices(result: HouseResult, catalog?: HouseCatalog): ModelSliceView[] {
  const choices = catalog?.runs.filter(run => run.year === result.run.year) ?? [];
  const boundChoices = choices.some(run => run.id === result.run.id && run.available)
    ? choices
    : [{ ...result.run, available: true, unavailableReason: null }];
  return MODEL_SLICES.map(slice => ({
    id: slice.id,
    label: slice.label,
    purpose: slice.id === "fundamentals"
      ? "Choose a saved House model built from earlier election results."
      : slice.id === "environment"
        ? "Represents influences that can move House districts together."
        : slice.id === "counts_rules"
          ? "Defines vote-count and election-rule accounting for the House."
          : slice.purpose,
    unavailableReason: slice.id === "fundamentals" ? null
      : slice.id === "measurement" && catalog?.capabilities?.polling
        ? "No separately selectable House polling component is provided. A capability label alone does not supply a control for changing the saved model."
        : slice.id === "counts_rules" && result.metadata.full_chamber
          ? "Fixed chamber calculations are included in this result. A separate House vote-count or election-rule selector is not available."
          : HOUSE_REASONS[slice.id],
    controls: slice.id === "fundamentals" ? [{
      id: "house-model",
      label: "House model",
      description: "Results for the House races covered by this study. Each choice loads its saved model, including the assumptions and uncertainty built into it. These are historical reconstructions using some source information assembled later.",
      value: result.run.id,
      required: true,
      options: boundChoices.map(run => ({
        id: run.id,
        label: run.label,
        description: run.available
          ? "A saved House fundamentals result fitted on earlier elections. Its component effects cannot be independently removed from this output."
          : run.unavailableReason || "This House result is not available for publication.",
        years: [run.year],
        availability: run.available ? "saved" as const : "unavailable" as const,
        experimental: run.experimental,
      })),
    }] : [],
  }));
}
