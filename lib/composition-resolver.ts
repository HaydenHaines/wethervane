import type { ExplorerRegistry, ExplorerRunEntry } from "./explorer-types";
import type {
  CompositionCatalog,
  CompositionRequest,
  CompositionResolution,
  SavedConfiguration,
} from "./composition-types";

function incompatible(request: CompositionRequest, reason: string): CompositionResolution {
  return {
    state: "incompatible",
    request,
    configurationId: null,
    resultId: null,
    configuration: null,
    run: null,
    reason,
    requiredSteps: [],
    evidence: null,
    outputs: [],
    executionAvailable: false,
  };
}

function slotIds(catalog: CompositionCatalog): string[] | null {
  const ids = catalog.slots.map((slot) => slot.id);
  return ids.length > 0 && new Set(ids).size === ids.length ? ids : null;
}

function sameComponents(
  left: Record<string, string>,
  right: Record<string, string>,
  requiredSlots: string[],
): boolean {
  return requiredSlots.every((slot) => left[slot] === right[slot]) &&
    Object.keys(left).length === requiredSlots.length && Object.keys(right).length === requiredSlots.length;
}

function validSha256(value: string): boolean {
  return /^[0-9a-f]{64}$/.test(value);
}

function registryMatches(
  configuration: SavedConfiguration,
  entry: ExplorerRunEntry,
  requiredSlots: string[],
): boolean {
  return /^cfg-[0-9a-f]{64}$/.test(configuration.id) &&
    validSha256(configuration.resultSha256) &&
    validSha256(entry.sha256) &&
    configuration.runId === entry.id &&
    configuration.resultSha256 === entry.sha256 &&
    configuration.year === entry.year &&
    configuration.dateId === entry.dateId &&
    configuration.targetId === entry.targetId &&
    configuration.lawId === entry.lawId &&
    sameComponents(configuration.components, entry.components, requiredSlots);
}

function ready(
  request: CompositionRequest,
  configuration: SavedConfiguration,
  run: ExplorerRunEntry,
): CompositionResolution {
  return {
    state: "ready",
    request,
    configurationId: configuration.id,
    resultId: configuration.resultSha256,
    configuration,
    run,
    reason: "This exact component selection has a saved, hash-identified forecast.",
    requiredSteps: [],
    evidence: configuration.evidence,
    outputs: configuration.outputs,
    executionAvailable: false,
  };
}

/**
 * Resolve only exact frozen outputs or explicitly declared metadata-only
 * computation families. This function never fits, queues, or coerces a choice.
 */
export function resolveComposition(
  catalog: CompositionCatalog,
  registry: ExplorerRegistry,
  request: CompositionRequest,
): CompositionResolution {
  if (!request || typeof request !== "object") {
    return incompatible(request, "A composition request is required.");
  }
  if (request.catalogId !== catalog.id) {
    return incompatible(request, "The requested catalog version is unavailable.");
  }
  const requiredSlots = slotIds(catalog);
  if (!requiredSlots) return incompatible(request, "The deployment catalog has invalid component slots.");
  if (!Number.isInteger(request.year) || typeof request.dateId !== "string" || !request.dateId) {
    return incompatible(request, "Choose a valid election year and date.");
  }
  const components = request.components;
  if (!components || typeof components !== "object" || Array.isArray(components)) {
    return incompatible(request, "Choose one value for each component slot.");
  }
  const suppliedSlots = Object.keys(components);
  if (suppliedSlots.some((id) => !requiredSlots.includes(id))) {
    return incompatible(request, "This component selection contains an unknown slot.");
  }
  if (requiredSlots.some((id) => typeof components[id] !== "string" || !components[id])) {
    return incompatible(request, "Choose one value for each component slot.");
  }
  if (suppliedSlots.length !== requiredSlots.length) {
    return incompatible(request, "This request is missing a required component slot.");
  }

  const options = new Map(catalog.slots.map((slot) => [slot.id, new Map(slot.options.map((option) => [option.id, option]))]));
  for (const slotId of requiredSlots) {
    const option = options.get(slotId)?.get(components[slotId]);
    if (!option || !option.years.includes(request.year) || option.availability === "unavailable") {
      return incompatible(request, `The selected ${slotId} option is unavailable for ${request.year}.`);
    }
  }

  const availableDates = new Set(registry.runs.filter((run) => run.year === request.year).map((run) => run.dateId));
  if (!availableDates.has(request.dateId)) {
    return incompatible(request, `No saved or reviewed composition is available for ${request.year} at that date.`);
  }

  if (request.presetId) {
    const preset = catalog.presets.find((item) => item.id === request.presetId);
    if (!preset) return incompatible(request, "The requested saved preset is unknown.");
    const configuration = catalog.saved.find((item) => item.runId === preset.runId);
    if (!configuration || !validSha256(configuration.resultSha256)) {
      return incompatible(request, "The requested preset has no valid frozen saved configuration.");
    }
    const run = registry.runs.find((item) => item.id === preset.runId);
    if (!run || !registryMatches(configuration, run, requiredSlots)) {
      return incompatible(request, "The requested preset no longer matches its saved artifact.");
    }
    if (request.year !== configuration.year || request.dateId !== configuration.dateId ||
        !sameComponents(request.components, configuration.components, requiredSlots)) {
      return incompatible(request, "The preset is an exact saved choice and conflicts with the requested components.");
    }
    return ready(request, configuration, run);
  }

  const exact = catalog.saved.find((item) =>
    item.year === request.year && item.dateId === request.dateId &&
    sameComponents(item.components, components, requiredSlots));
  if (exact) {
    const run = registry.runs.find((item) => item.id === exact.runId);
    if (!run || !registryMatches(exact, run, requiredSlots)) {
      return incompatible(request, "The saved configuration is stale or its registry identity has changed.");
    }
    return ready(request, exact, run);
  }

  // Current computation metadata describes precisely the three named slots.
  // Do not let it authorize a selection if a future catalog adds a fourth slot.
  const computationSlots = ["base", "polling", "uncertainty"];
  const requirement = requiredSlots.length === computationSlots.length &&
    computationSlots.every((slot) => requiredSlots.includes(slot)) &&
    catalog.computation.find((item) =>
    item.baseId === components.base &&
    item.years.includes(request.year) &&
    item.polling.includes(components.polling) &&
    item.uncertainty.includes(components.uncertainty));
  if (requirement) {
    return {
      state: "requires_computation",
      request,
      configurationId: null,
      resultId: null,
      configuration: null,
      run: null,
      reason: requirement.reason,
      requiredSteps: requirement.requiredSteps,
      evidence: requirement.evidence,
      outputs: requirement.outputs,
      executionAvailable: false,
    };
  }
  return incompatible(request, "No saved forecast or declared computation requirement matches this exact component selection.");
}

/** Build a client request from one frozen saved run without inventing bindings. */
export function requestForRun(catalog: CompositionCatalog, entry: ExplorerRunEntry): CompositionRequest {
  const requiredSlots = slotIds(catalog);
  const configuration = catalog.saved.find((item) => item.runId === entry.id);
  if (!requiredSlots || !configuration || !registryMatches(configuration, entry, requiredSlots)) {
    throw new Error("The selected run is not a verified saved configuration in this catalog.");
  }
  return {
    catalogId: catalog.id,
    year: configuration.year,
    dateId: configuration.dateId,
    components: { ...configuration.components },
  };
}
