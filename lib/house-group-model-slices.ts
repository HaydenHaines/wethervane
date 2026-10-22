import type { ComponentOption } from "./composition-types.ts";
import type { HouseGroupPublication } from "./house-group-catalog.ts";
import { MODEL_SLICES, type ModelSliceId, type ModelSliceView } from "./model-slices.ts";
import { groupFundamentalsId, houseFundamentalsLabel } from "./house-group-presentation.ts";

const reasons: Record<ModelSliceId, string> = {
  population: "A population model is not connected to these forecasts. District maps are available to locate the races, but the map does not add population or demographic estimates to the model.",
  community: "Community composition is not estimated in these forecasts; no separate component is available.",
  fundamentals: "No saved group forecasts are available for this election year.",
  environment: "Shared election-wide uncertainty is bundled into each saved model and cannot be selected separately.",
  measurement: "Polling is not connected to this collection; no polling component can be switched on or off.",
  participation: "A turnout model is not connected. These results do not predict voter counts.",
  choice: "Candidate groups and the group-win target are fixed within each saved result; candidates within a group do not receive separate probabilities.",
  counts_rules: "The result uses an aggregate approximation for the listed election rule. It does not simulate vote counts, ranked-choice rounds, or runoffs, and it does not estimate House control.",
};

/** Maps the reviewed full-roster publication to the shared eight-role builder. */
export function houseGroupModelSlices(publication: HouseGroupPublication): ModelSliceView[] {
  const { result, entry, catalog } = publication;
  const yearRuns = catalog.runs.filter(run => run.cycle === result.cycle);
  const boundRuns = yearRuns.some(run => run.id === entry.id) ? yearRuns : [entry];
  return MODEL_SLICES.map(slice => {
    const builtIn = slice.id === "environment" || slice.id === "counts_rules";
    let controls: ModelSliceView["controls"] = [];
    if (slice.id === "fundamentals") {
      const options: ComponentOption[] = boundRuns.map(run => ({
        id: run.id,
        label: houseFundamentalsLabel(run.fundamentalsId),
        description: `Saved ${run.cycle} House group result using ${houseFundamentalsLabel(run.fundamentalsId)}. Each result keeps its documented assumptions and uncertainty bundled together.`,
        years: [run.cycle], availability: "saved", experimental: true,
      }));
      controls = [{ id: "house-group-run", label: "House model", description: "Choose a saved full-roster result for this election year.", value: entry.id, options, required: true }];
    } else if (slice.id === "choice") {
      controls = [{
        id: "house-winner-target", label: "Winner outcome set", description: "Choose whether to view the original candidate forecasts or full-roster party/group-win forecasts.",
        value: "full435", required: true,
        options: [
          { id: "legacy", label: "Original D/R candidate forecasts", description: "Returns to the separately published original forecast for its supported races.", years: [result.cycle], availability: "saved", experimental: true },
          { id: "full435", label: "Full 435 D/R/O group forecasts", description: "Forecasts which Democratic, Republican, or Other candidate group wins each district; it does not assign probabilities to every candidate.", years: [result.cycle], availability: "saved", experimental: true },
        ],
      }];
    }
    return {
      id: slice.id, label: slice.label,
      purpose: slice.id === "fundamentals" ? "Choose a saved House model for the selected election year."
        : slice.id === "choice" ? "Choose whether the result is about individual candidates or winning party/candidate groups."
          : slice.id === "counts_rules" ? "Election-rule handling is part of the saved group forecast, not a separate simulator."
            : slice.purpose,
      controls,
      unavailableReason: controls.length ? null : builtIn ? null : reasons[slice.id],
      ...(builtIn ? { builtIn: true, builtInReason: reasons[slice.id] } : {}),
    };
  });
}
