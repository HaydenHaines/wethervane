import type { ExplorerRunEntry } from "./explorer-types";

export type ComponentChoices = Record<string, string>;

export interface CompositionRequest {
  catalogId: string;
  year: number;
  dateId: string;
  components: ComponentChoices;
  /** Presets are exact shortcuts; their scientific bindings remain server-owned. */
  presetId?: string;
}

export interface ComponentOption {
  id: string;
  label: string;
  description: string;
  years: number[];
  availability: "saved" | "requires_computation" | "unavailable";
  experimental: boolean;
  implementationId?: string;
}

export interface ComponentSlot {
  id: string;
  label: string;
  description: string;
  options: ComponentOption[];
}

export interface ConfigurationEvidence {
  status: "preserved" | "experimental" | "historical_reconstruction";
  implementation: string;
  numerical: string;
  predictive: string;
  scope: string;
}

export interface SavedConfiguration {
  id: string;
  runId: string;
  resultSha256: string;
  year: number;
  dateId: string;
  components: ComponentChoices;
  targetId: string;
  lawId: string;
  outputs: string[];
  cutoffs: { label: string; at: string }[];
  evidence: ConfigurationEvidence;
}

export interface CompositionPreset {
  id: string;
  label: string;
  description: string;
  runId: string;
  forecastSha256: string;
}

/** Metadata-only preflight for an implemented research family, never permission to run a fit. */
export interface ComputationRequirement {
  baseId: string;
  years: number[];
  polling: string[];
  uncertainty: string[];
  targetId: string;
  reason: string;
  requiredSteps: string[];
  outputs: string[];
  evidence: ConfigurationEvidence;
}

export interface UnavailableComponent {
  id: string;
  label: string;
  reason: string;
}

export interface CompositionCatalog {
  schemaVersion: 1;
  id: string;
  sourceCatalog: { id: string; sha256: string };
  sources: Record<string, string>;
  defaultPresetId: string;
  execution: { available: false; reason: string };
  presets: CompositionPreset[];
  slots: ComponentSlot[];
  saved: SavedConfiguration[];
  computation: ComputationRequirement[];
  unavailable: UnavailableComponent[];
}

export interface CompositionResolution {
  state: "ready" | "requires_computation" | "incompatible";
  request: CompositionRequest;
  /** Only a ready result has a fully bound saved configuration identity. */
  configurationId: string | null;
  resultId: string | null;
  configuration: SavedConfiguration | null;
  run: ExplorerRunEntry | null;
  reason: string;
  requiredSteps: string[];
  evidence: ConfigurationEvidence | null;
  outputs: string[];
  executionAvailable: false;
}

export interface SavedView {
  id: string;
  name: string;
  configurationId: string;
  runId: string;
  resultSha256: string;
  scenarioId: string | null;
  raceId: string | null;
  compare: boolean;
  baselineRunId: string | null;
  mapView: "forecast" | "actual";
}
