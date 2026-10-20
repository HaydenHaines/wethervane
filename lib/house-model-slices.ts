import type { HouseCatalog } from "./house-catalog";
import type { HouseResult } from "./house-result";
import { MODEL_SLICES, type ModelSliceId, type ModelSliceView } from "./model-slices.ts";
import { houseModelDescription } from "./house-copy.ts";

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
  counts_rules: "Supported two-candidate race handling is fixed in this saved result. This study does not model vote counts or cover all 435 districts, so it cannot produce a House seat total or chamber-control forecast.",
};

const HOUSE_BUILT_IN_REASONS: Partial<Record<ModelSliceId, string>> = {
  environment: "Shared election-wide uncertainty is bundled into each saved House model; there is no separate control to change it.",
  choice: "Candidate context and the Democratic–Republican candidate pair are fixed in each saved House model; there is no separate candidate-choice control.",
};

/** Exact saved configurations remain the unit of choice; no component refitting. */
export function houseModelSlices(result: HouseResult, catalog?: HouseCatalog): ModelSliceView[] {
  const choices = catalog?.runs.filter(run => run.year === result.run.year) ?? [];
  const boundChoices = choices.some(run => run.id === result.run.id && run.available)
    ? choices
    : [{ ...result.run, available: true, unavailableReason: null }];
  return MODEL_SLICES.map(slice => {
    const builtIn = slice.id === "environment" || slice.id === "choice" || slice.id === "counts_rules";
    const builtInReason = slice.id === "counts_rules" && result.metadata.full_chamber
      ? "Chamber calculations and supported race rules are fixed in this saved result. There is no separate House vote-count or election-rule control."
      : HOUSE_BUILT_IN_REASONS[slice.id] ?? (slice.id === "counts_rules" ? HOUSE_REASONS.counts_rules : null);
    return {
    id: slice.id,
    label: slice.label,
    purpose: slice.id === "fundamentals"
      ? "Choose a saved House model built from earlier election results."
      : slice.id === "environment"
        ? "Represents influences that can move House districts together."
        : slice.id === "counts_rules"
          ? "Race rules and any chamber calculations follow this saved result; they are not separate controls."
          : slice.purpose,
    unavailableReason: slice.id === "fundamentals" ? null
      : builtIn ? null
        : slice.id === "measurement" && catalog?.capabilities?.polling
          ? "No separately selectable House polling component is provided. A capability label alone does not supply a control for changing the saved model."
          : HOUSE_REASONS[slice.id],
    ...(builtIn ? { builtIn: true, builtInReason } : {}),
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
          ? houseModelDescription(run.configurationId, run.year)
          : run.unavailableReason || "This House result is not available for publication.",
        years: [run.year],
        availability: run.available ? "saved" as const : "unavailable" as const,
        experimental: run.experimental,
      })),
    }] : [],
  };
  });
}
