import type { ComponentStudyCatalog, ComponentStudyRequest, ComponentStudyResult } from "./component-study";
import { canonicalStudySelections } from "./component-study.ts";
import type { ExplorerRace } from "./explorer-types";
import { MODEL_SLICES, type ModelSliceView } from "./model-slices.ts";

const copy: Record<string, { label: string; description: string; options: Record<string, string> }> = {
  structure: { label: "Earlier presidential voting", description: "Learns how strongly earlier presidential voting predicts a state's Senate result. Other settings are not connected to reviewed study results yet.", options: { learned: "Learn the relationship", fixed_one: "Use a fixed one-to-one relationship" } },
  candidate: { label: "Incumbency adjustment", description: "Accounts for which party's candidate already held the seat. Removing this term requires estimating the model's settings together again; it cannot be subtracted from an existing result.", options: { signed_incumbency: "Include incumbency", off: "Remove incumbency" } },
  race: { label: "Race-to-race variation", description: "Allows each race to differ from the starting estimate, including occasional large differences. This uncertainty is required by the study.", options: { included: "Race-to-race variation · Required" } },
  carryover: { label: "National patterns from earlier elections", description: "Learns how earlier national voting patterns carry into this election. Other settings need reviewed bindings before the site can use them.", options: { learned: "Learn the carryover", off: "Remove carryover", fixed_one: "Carry over one-to-one", reference_state_slope: "Match the state relationship" } },
  context: { label: "President's party and midterm effect", description: "The saved study omits this extra predictor. Shared election-wide variation remains included. Alternative settings are not connected to reviewed study results yet.", options: { off: "Not used in this study", party: "President's party", party_midterm: "President's party and midterm" } },
  cycle: { label: "Shared election-wide variation", description: "Allows races to move together because of influences shared across the election. This uncertainty remains included even when the extra presidential predictor is off.", options: { included: "Shared election-wide variation · Required" } },
  poll: { label: "Polling method", description: "Adds historical poll records using a fixed bell-shaped model of survey error. Switching polls changes the complete saved study result and its verified input bindings.", options: { identity: "Without polling evidence", fixed_gaussian: "Fixed survey-error model" } },
};

export function componentStudySlices(catalog: ComponentStudyCatalog, request: ComponentStudyRequest): ModelSliceView[] {
  const canonical = canonicalStudySelections(catalog.catalog, request.selections) || [];
  const example = catalog.studies.find((entry) => entry.id === request.studyId);
  const ready = canonicalStudySelections(catalog.catalog, example?.selections) || [];
  return MODEL_SLICES.map((slice) => {
    const role = catalog.catalog.roles.find((entry) => entry.role === slice.id);
    return {
      id: slice.id, label: slice.label, purpose: slice.purpose,
      unavailableReason: slice.id === "counts_rules"
        ? "This study predicts race probabilities and two-party shares. A separate vote-count model, chamber control and tipping points are unavailable."
        : slice.unavailable,
      controls: (role?.providers || []).map((provider) => {
        const id = provider.default_instance_id, text = copy[id];
        const [key, schema] = Object.entries(provider.options_schema.properties)[0] || [];
        const value = key ? canonical.find((choice) => choice.instance_id === id)?.options[key] || schema.default : "included";
        const saved = key ? ready.find((choice) => choice.instance_id === id)?.options[key] : "included";
        const toggle = id === "poll" ? { label: "Include polling evidence", offValue: "identity", onValue: "fixed_gaussian" } : undefined;
        return {
          id, label: text.label, description: text.description, value,
          required: !["candidate", "carryover", "context", "poll"].includes(id), ...(toggle ? { toggle } : {}),
          options: (schema?.enum || ["included"]).map((option) => ({
            id: option, label: text.options[option] || option,
            description: id === "poll" && option === "identity" ? "No polling evidence is used in this study result." : text.description,
            years: [2018], experimental: true,
            availability: id === "poll" || option === saved ? "saved" : id === "candidate" && option === "off" ? "requires_computation" : "unavailable",
          })),
        };
      }),
    };
  });
}

/** Reuse geographic/probability views without inventing a chamber bundle or interval levels.
 * Observed outcome here is the D/R vote leader only. Pass these races to the
 * race scorer and forecast map; do not use generic actual-winner map wording.
 */
export function studyRaces(result: ComponentStudyResult): ExplorerRace[] {
  const sources = result.sourceAttribution.flatMap((source) => source.urls);
  return result.races.map((race) => {
    const winner = race.observed.democraticVotes === race.observed.republicanVotes ? null
      : race.candidates.find((c) => c.party === (race.observed.democraticVotes > race.observed.republicanVotes ? "D" : "R"))!;
    return {
      id: race.id, state: race.state, seatId: race.seatId, electionLabel: "Regular election · D/R study", notes: race.notes,
      outcomes: race.candidates.map((c) => ({ id: c.id, name: c.name, party: c.party, kind: "candidate" as const,
        members: [{ id: `${race.id}:${c.id}`, name: c.name, party: c.party, incumbent: c.incumbent, sourceUrls: sources }], winProbability: c.winProbability })),
      ...(winner ? { observed: { outcomeId: winner.id, winnerName: winner.name, winnerParty: winner.party, sourceUrls: sources } } : {}),
    };
  });
}

export function studyShareError(result: ComponentStudyResult): number {
  return result.races.reduce((total, race) => total + Math.abs(race.candidates.find((c) => c.party === "D")!.share.mean - race.observed.democraticTwoPartyShare), 0) / result.races.length;
}

export function comparableStudies(a: ComponentStudyResult, b: ComponentStudyResult): boolean {
  if (a.informationCutoff !== b.informationCutoff || a.targetId !== b.targetId || a.outputContractId !== b.outputContractId || a.races.length !== b.races.length) return false;
  return a.races.every((race) => {
    const other = b.races.find((r) => r.id === race.id);
    return other && race.observed.democraticVotes === other.observed.democraticVotes && race.observed.republicanVotes === other.observed.republicanVotes &&
      race.candidates.every((c) => other.candidates.some((o) => o.id === c.id && o.name === c.name && o.share.denominator === c.share.denominator));
  });
}

export interface StudySavedView { name: string; studyId: string; sha256: string; raceId: string; compare: boolean }

/** Translate the reviewed source labels for readers; retain URLs and the exact artifact. */
export function studySourceCopy(source: ComponentStudyResult["sourceAttribution"][number]) {
  const descriptions: Record<string, { label: string; description: string }> = {
    "Source-bound research panel": { label: "Candidate and race records", description: "Names and included races come from the study's saved 2018 data. Candidate names were assembled from final election records; their availability on the study date has not been verified." },
    "Observed two-party outcomes": { label: "Election results used for scoring", description: "Democratic and Republican vote counts come from a separately saved set of election results, including documented corrections to official and MIT Election Data and Science Lab records. Votes for other candidates are excluded." },
    "MEDSL historical source documentation": { label: "MIT Election Data and Science Lab", description: "Documentation for the historical election returns. These links describe the underlying collection; the saved study also records fingerprints identifying its exact data files." },
  };
  return { ...source, ...descriptions[source.label] };
}

export function parseStudySavedViews(raw: string | null, catalog: ComponentStudyCatalog): StudySavedView[] {
  try {
    const parsed = JSON.parse(raw || "[]");
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((item): item is StudySavedView => item && typeof item.name === "string" && item.name.trim().length > 0 && item.name.length <= 80 &&
      typeof item.raceId === "string" && item.raceId.length < 100 && typeof item.compare === "boolean" &&
      catalog.studies.some((entry) => entry.id === item.studyId && entry.sha256 === item.sha256)).slice(-30);
  } catch { return []; }
}
