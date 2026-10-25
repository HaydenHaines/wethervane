import type { ComponentOption, CompositionCatalog, CompositionRequest } from "./composition-types";

/** Scientific responsibilities are stable; each contract supplies its own controls. */
export const MODEL_SLICES = [
  { id: "population", label: "Population & geography", purpose: "Defines who is represented, where, and for which dates.", unavailable: "A population model is not connected yet. These forecasts use a fixed list of contests." },
  { id: "community", label: "Community composition", purpose: "Models measured groups and uncertainty about how they overlap.", unavailable: "Community composition is not estimated in these forecasts." },
  { id: "fundamentals", label: "Electoral history", purpose: "Sets a starting estimate from earlier elections and candidate context.", unavailable: "No starting-estimate choices are available for this result." },
  { id: "environment", label: "National environment", purpose: "Sets uncertainty and how strongly races can move together.", unavailable: "No environment choices are available for this result." },
  { id: "measurement", label: "Polls & surveys", purpose: "Adds polling evidence while allowing for survey error.", unavailable: "No polling choices are available for this result." },
  { id: "participation", label: "Turnout & participation", purpose: "Models who votes and how turnout varies.", unavailable: "A turnout model is not connected yet. These forecasts do not predict voter counts." },
  { id: "choice", label: "Candidate choice", purpose: "Models choices among candidates and ballot options.", unavailable: "A separate choice model is not connected yet. The saved forecasts define their own possible winners." },
  { id: "counts_rules", label: "Vote counts & election rules", purpose: "Turns voting choices into counts, winners and chamber outcomes.", unavailable: "A separate vote-count model is not connected yet." },
] as const;

export type ModelSliceId = typeof MODEL_SLICES[number]["id"];
export interface SliceControl {
  id: string;
  label: string;
  description: string;
  value: string;
  options: ComponentOption[];
  /** Only a scientifically defined off/identity choice makes a control optional. */
  toggle?: { label: string; offValue: string; onValue: string };
  required: boolean;
}
export interface ModelSliceView {
  id: ModelSliceId;
  label: string;
  purpose: string;
  controls: SliceControl[];
  unavailableReason: string | null;
  /** This role is present in the saved model, but has no separate control. */
  builtIn?: boolean;
  builtInReason?: string | null;
}

/** Adapt the preserved three-slot contract without claiming finer independence. */
export function savedForecastSlices(catalog: CompositionCatalog, request: CompositionRequest, rememberedPolling: string): ModelSliceView[] {
  const slots: Partial<Record<ModelSliceId, string>> = { fundamentals: "base", environment: "uncertainty", measurement: "polling" };
  return MODEL_SLICES.map((slice) => {
    const slot = catalog.slots.find((entry) => entry.id === slots[slice.id]);
    const options = slot?.options.filter((option) => option.years.includes(request.year)) ?? [];
    const value = slot ? request.components[slot.id] ?? "" : "";
    const enabledPolls = options.filter((option) => option.id !== "none" && option.availability !== "unavailable");
    const onValue = enabledPolls.find((option) => option.id === rememberedPolling)?.id
      ?? enabledPolls.find((option) => option.id === "standard")?.id ?? enabledPolls[0]?.id;
    const toggle = slot?.id === "polling" && options.some((option) => option.id === "none") && onValue
      ? { label: "Include polling evidence", offValue: "none", onValue }
      : undefined;
    const builtIn = slice.id === "choice" || slice.id === "counts_rules";
    return {
      id: slice.id, label: slice.label, purpose: slice.purpose,
      controls: slot && options.length ? [{ id: slot.id, label: slot.label, description: slot.description, value, options, required: !toggle, ...(toggle ? { toggle } : {}) }] : [],
      unavailableReason: slot && options.length || builtIn ? null : slice.unavailable,
      ...(slice.id === "choice" ? {
        builtIn: true,
        builtInReason: "The candidate options are fixed in each saved forecast. There is no separate candidate-choice control.",
      } : slice.id === "counts_rules" ? {
        builtIn: true,
        builtInReason: "Chamber calculations and tie rules are fixed in each saved Senate result. Total vote counts are not predicted, and there is no separate control to change these rules.",
      } : {}),
    };
  });
}
