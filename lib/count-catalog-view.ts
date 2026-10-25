import { parseCountCatalogView, type CountCatalogView, type CountRole } from "./count-catalog.ts";
import { MODEL_SLICES, type ModelSliceId } from "./model-slices.ts";

/** Scientific responsibilities stay stable; count-specific copy avoids aggregate-result claims. */
const COUNT_ROLE_COPY = {
  population: { label: "Population & geography", purpose: "Defines how many people are represented, where they live and which dates apply, allowing for uncertainty in those numbers." },
  community: { label: "Community composition", purpose: "Describes groups of people, how they overlap and how their makeup can change." },
  fundamentals: { label: "Earlier elections & candidate information", purpose: "Starts with earlier elections and information about candidates." },
  environment: { label: "Election environment", purpose: "Allows shared election-wide shifts and additional variation between contests." },
  measurement: { label: "Surveys & participation reports", purpose: "Uses surveys and reported participation totals while allowing for errors in them." },
  participation: { label: "Turnout & participation", purpose: "Models how many people take part in an election, allowing uncertainty about turnout." },
  choice: { label: "Candidate choice", purpose: "Models choices among named candidates and other specified responses, such as blank or invalid marks." },
  counts_rules: { label: "Vote counts & election rules", purpose: "Turns participation and candidate choices into whole-number votes under the recorded contest rules." },
} satisfies Record<CountRole, { label: string; purpose: string }>;

export interface CountCatalogRolePresentation {
  id: ModelSliceId;
  label: string;
  purpose: string;
  requiredLabel: string;
  providers: {
    providerId: string;
    version: string;
    instanceId: string;
    options: [string, string][];
    limitations: string[];
  }[];
}

export interface CountCatalogPresentation {
  schema: "CountCatalogPresentationV1";
  title: string;
  scopeLabel: string;
  trainingLabel: string;
  electionDate: string;
  electionDateLabel: string;
  informationCutoff: string;
  informationCutoffLabel: string;
  states: { id: "preparation" | "result" | "evidence" | "site"; label: string; value: string; description: string }[];
  roles: CountCatalogRolePresentation[];
  technical: Pick<CountCatalogView, "targetId" | "bindings"> & {
    configurationSha256: string;
    requestSha256: string;
    outputContractId: string;
    sourceProblemSha256: string;
    resolutionStatus: "requires_computation";
    inferenceEligibility: "not_established_for_current_execution";
  };
}

/** Metadata-only presentation: unsupported or stale DTOs never become a forecast. */
export function countCatalogPresentation(value: unknown): CountCatalogPresentation | null {
  const catalog = parseCountCatalogView(value);
  if (!catalog) return null;
  const date = new Intl.DateTimeFormat("en-US", { month: "long", day: "numeric", year: "numeric", timeZone: "UTC" });
  const cutoff = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "long", year: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit", timeZone: "UTC", hour12: false });
  const contestCount = catalog.scope.predictionContestIds.length;
  return {
    schema: "CountCatalogPresentationV1",
    title: `${catalog.scope.state} count-model research`,
    scopeLabel: `Prepared for the ${catalog.scope.predictionCycle} election · ${contestCount === 1 ? "One" : contestCount} ${catalog.scope.state} ${contestCount === 1 ? "contest" : "contests"} with a specified candidate list`,
    trainingLabel: catalog.scope.trainingCycles.join(" and "),
    electionDate: catalog.scope.predictionElectionDate,
    electionDateLabel: date.format(new Date(`${catalog.scope.predictionElectionDate}T00:00:00Z`)),
    informationCutoff: catalog.prepared.informationCutoff,
    informationCutoffLabel: `${cutoff.format(new Date(catalog.prepared.informationCutoff))} UTC`,
    states: [
      { id: "preparation", label: "Recorded setup", value: "Prepared snapshot", description: "All eight components are specified together in one recorded research setup." },
      { id: "result", label: "Calculation results", value: "No completed forecast supplied", description: "The recorded setup has no completed count forecast in this catalog." },
      { id: "evidence", label: "Testing & evidence", value: "Forecast accuracy not assessed here", description: "This metadata supplies no accuracy evaluation. Software checks alone do not establish forecast accuracy." },
      { id: "site", label: "Site availability", value: "Private preview", description: "Count forecasts are unavailable on the public site. This catalog cannot schedule a calculation." },
    ],
    roles: MODEL_SLICES.map(({ id }) => {
      const role = catalog.roles.find(row => row.role === id)!;
      return {
        id, ...COUNT_ROLE_COPY[id], requiredLabel: "Required in this setup",
        providers: role.providers.map(provider => ({
          providerId: provider.providerId, version: provider.version, instanceId: provider.instanceId,
          options: Object.entries(catalog.prepared.selections.find(selection => selection.instanceId === provider.instanceId)!.options),
          limitations: [...provider.limitations],
        })),
      };
    }),
    technical: {
      targetId: catalog.targetId, configurationSha256: catalog.prepared.configurationSha256,
      requestSha256: catalog.prepared.requestSha256, outputContractId: catalog.prepared.outputContractId,
      sourceProblemSha256: catalog.scope.sourceProblemSha256, bindings: { ...catalog.bindings },
      resolutionStatus: catalog.prepared.status, inferenceEligibility: catalog.statuses.inferenceEligibility,
    },
  };
}
