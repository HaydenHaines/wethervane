import type { HouseLiveView, HouseLiveRaceView } from "./house-live-view";
import type { HouseLiveConfiguration } from "./house-live-result";

type Party = "D" | "R" | "O";
type SeatParty = "D" | "R" | "U";
type ControlParty = "D" | "R" | "unresolved";
type Slot = keyof HouseLiveConfiguration;
const PARTIES: readonly Party[] = ["D", "R", "O"];
const SEAT_PARTIES: readonly SeatParty[] = ["D", "R", "U"];

export interface HouseLiveComparisonValue {
  primary: number;
  reference: number;
  /** Primary minus reference; descriptive, with no cross-run uncertainty interval. */
  difference: number;
}

export interface HouseLiveComparisonRunContext {
  id: string;
  label: string;
  resultSha256: string;
  issuedAt: string;
  cutoffAt: string;
  strictAsOf: boolean;
  configuration: HouseLiveConfiguration;
  sourceCutoffs: { role: string; at: string }[];
  totalPollCount: number;
  /** Each run has its own joint evidence. These rows are never paired across runs. */
  jointDraws: { sha256: string; count: number; chainCount: number };
  qualification: { inputStatus: string; numericalStatus: string; publicationStatus: string; evidenceHashes: string[] };
}

export interface HouseLiveComparisonSummary {
  primary: HouseLiveComparisonRunContext;
  reference: HouseLiveComparisonRunContext;
  contextDifferences: {
    modelSlots: Slot[];
    sourceCutoffsDiffer: boolean;
    pollCounts: { raceId: string; primary: number; reference: number }[];
    methods: { raceId: string; primary: string; reference: string }[];
  };
  races: {
    raceId: string;
    districtKey: string;
    outcomes: { party: Party; label: string | null; supported: boolean; probability: HouseLiveComparisonValue }[];
  }[];
  chamber: {
    status: "comparable";
    scenarioId: string;
    threshold: 218;
    /** Native O seats remain unassigned U; no caucus allocation is inferred. */
    expectedSeats: Record<SeatParty, HouseLiveComparisonValue>;
    control: Record<ControlParty, HouseLiveComparisonValue>;
  } | { status: "unavailable"; reason: string };
  conditionalPairs: { status: "not_compared"; reason: string };
  interpretation: string;
}

export type HouseLiveComparison =
  | { status: "comparable"; summary: HouseLiveComparisonSummary }
  | { status: "incomparable"; reason: string };

const value = (primary: number, reference: number): HouseLiveComparisonValue => ({
  primary, reference, difference: primary - reference,
});
const sorted = (values: readonly string[]) => [...values].sort();
const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
const cutoffSet = (view: HouseLiveView) => view.run.sourceCutoffs
  .map(cutoff => `${cutoff.role}\u0000${Date.parse(cutoff.at)}`).sort();

function candidateSupport(race: HouseLiveRaceView) {
  return race.candidates.map(candidate => ({
    id: candidate.id, name: candidate.name, partyLabel: candidate.partyLabel,
    aggregationParty: candidate.aggregationParty, ballotStatus: candidate.ballotStatus,
    incumbent: candidate.incumbent,
  })).sort((a, b) => a.id.localeCompare(b.id));
}

function outcomeSupport(race: HouseLiveRaceView) {
  return race.outcomes.map(outcome => ({
    id: outcome.id, party: outcome.party, kind: outcome.kind,
    memberIds: sorted(outcome.members.map(member => member.id)),
  })).sort((a, b) => a.id.localeCompare(b.id));
}

function raceMap(view: HouseLiveView): Map<string, HouseLiveRaceView> | null {
  if (view.races.length !== 435) return null;
  const byId = new Map<string, HouseLiveRaceView>();
  const districts = new Set<string>();
  for (const race of view.races) {
    if (byId.has(race.id) || districts.has(race.district.districtKey) ||
      race.seatId !== race.district.districtKey || race.district.cycle !== 2026 ||
      race.district.mapVintage !== view.mapVintage || race.observed !== undefined) return null;
    byId.set(race.id, race);
    districts.add(race.district.districtKey);
  }
  return byId;
}

function context(view: HouseLiveView): HouseLiveComparisonRunContext {
  return {
    id: view.run.id, label: view.run.label, resultSha256: view.run.resultSha256,
    issuedAt: view.run.issuedAt, cutoffAt: view.run.cutoffAt, strictAsOf: view.run.strictAsOf,
    configuration: { ...view.run.configuration },
    sourceCutoffs: view.run.sourceCutoffs.map(cutoff => ({ ...cutoff })),
    totalPollCount: view.races.reduce((total, race) => total + (race.pollCount ?? 0), 0),
    jointDraws: { sha256: view.jointDraws.sha256, count: view.jointDraws.count,
      chainCount: view.jointDraws.chainCount },
    qualification: { inputStatus: view.qualification.inputStatus,
      numericalStatus: view.qualification.numericalStatus,
      publicationStatus: view.qualification.publicationStatus,
      evidenceHashes: [...view.qualification.evidenceHashes] },
  };
}

function compareChamber(primary: HouseLiveView, reference: HouseLiveView): HouseLiveComparisonSummary["chamber"] {
  const a = primary.chamber, b = reference.chamber;
  if (!a || !b || !primary.chamberPrecision || !reference.chamberPrecision)
    return { status: "unavailable", reason: "A qualified chamber result is unavailable for one or both runs." };
  const expectedRetained = { D: 0, R: 0, U: 0 };
  if (a.id !== "party_majority_others_unassigned" || b.id !== a.id ||
    a.thresholds.D !== 218 || a.thresholds.R !== 218 ||
    b.thresholds.D !== 218 || b.thresholds.R !== 218 ||
    SEAT_PARTIES.some(party => a.retainedSeats[party] !== expectedRetained[party] ||
      b.retainedSeats[party] !== expectedRetained[party]) ||
    !same(sorted(Object.keys(a.seats)), sorted(SEAT_PARTIES)) ||
    !same(sorted(Object.keys(b.seats)), sorted(SEAT_PARTIES)))
    return { status: "unavailable", reason: "The chamber majority threshold or unassigned-seat scenario differs." };
  return {
    status: "comparable", scenarioId: a.id, threshold: 218,
    expectedSeats: {
      D: value(a.seats.D.expectedSeats, b.seats.D.expectedSeats),
      R: value(a.seats.R.expectedSeats, b.seats.R.expectedSeats),
      U: value(a.seats.U.expectedSeats, b.seats.U.expectedSeats),
    },
    control: {
      D: value(a.control.democraticMajorityProbability, b.control.democraticMajorityProbability),
      R: value(a.control.republicanMajorityProbability, b.control.republicanMajorityProbability),
      unresolved: value(a.control.unresolvedOrganizingProbability, b.control.unresolvedOrganizingProbability),
    },
  };
}

/** Compare only two separately authenticated current-model display projections. */
export function compareHouseLiveViews(primary: HouseLiveView, reference: HouseLiveView): HouseLiveComparison {
  if (primary.run.cycle !== 2026 || reference.run.cycle !== 2026 ||
    primary.run.historical || reference.run.historical ||
    primary.run.electionDate !== reference.run.electionDate)
    return { status: "incomparable", reason: "Both runs must forecast the same 2026 House election." };
  if (primary.run.resultSha256 === reference.run.resultSha256)
    return { status: "incomparable", reason: "The two selections identify the same result artifact." };
  if (Date.parse(primary.run.cutoffAt) !== Date.parse(reference.run.cutoffAt))
    return { status: "incomparable", reason: "The forecast cutoff timestamps differ." };
  if (primary.mapVintage !== reference.mapVintage || primary.geographySha256 !== reference.geographySha256)
    return { status: "incomparable", reason: "The district map or geography artifact differs." };
  const first = raceMap(primary), second = raceMap(reference);
  if (!first || !second || first.size !== second.size)
    return { status: "incomparable", reason: "Both runs need the same complete 435-district roster." };
  const pollCounts: HouseLiveComparisonSummary["contextDifferences"]["pollCounts"] = [];
  const methods: HouseLiveComparisonSummary["contextDifferences"]["methods"] = [];
  const races: HouseLiveComparisonSummary["races"] = [];
  for (const [raceId, a] of first) {
    const b = second.get(raceId);
    if (!b || a.state !== b.state || a.district.districtKey !== b.district.districtKey ||
      a.ruleFamily !== b.ruleFamily ||
      !same(candidateSupport(a), candidateSupport(b)) || !same(outcomeSupport(a), outcomeSupport(b)))
      return { status: "incomparable", reason: `District support, candidates, or election rules differ for ${raceId}.` };
    const aOutcomes = new Map(a.outcomes.map(outcome => [outcome.party, outcome]));
    const bOutcomes = new Map(b.outcomes.map(outcome => [outcome.party, outcome]));
    races.push({ raceId, districtKey: a.district.districtKey,
      outcomes: PARTIES.map(party => ({
        party, label: aOutcomes.get(party)?.name ?? null, supported: aOutcomes.has(party),
        probability: value(aOutcomes.get(party)?.winProbability ?? 0,
          bOutcomes.get(party)?.winProbability ?? 0),
      })) });
    const aPolls = a.pollCount ?? 0, bPolls = b.pollCount ?? 0;
    if (aPolls !== bPolls) pollCounts.push({ raceId, primary: aPolls, reference: bPolls });
    if (a.method !== b.method) methods.push({ raceId, primary: a.method, reference: b.method });
  }
  const slots: Slot[] = ["base", "polling", "uncertainty"];
  return {
    status: "comparable",
    summary: {
      primary: context(primary), reference: context(reference),
      contextDifferences: {
        modelSlots: slots.filter(slot => primary.run.configuration[slot] !== reference.run.configuration[slot]),
        sourceCutoffsDiffer: !same(cutoffSet(primary), cutoffSet(reference)),
        pollCounts, methods,
      },
      races,
      chamber: compareChamber(primary, reference),
      conditionalPairs: { status: "not_compared",
        reason: "Named-candidate shares are conditional on a separate two-contender denominator and are not compared with group winning chances." },
      interpretation: "Differences are selected run minus reference run. Current outcomes are unobserved; these differences do not score accuracy, establish a better model, or measure uncertainty between runs. Each run keeps its own joint-draw evidence.",
    },
  };
}
