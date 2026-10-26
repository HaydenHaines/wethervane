/** Pure projection of authenticated dated forecasts. Loaders retain artifact admission authority. */
import type { ExplorerBundle, ExplorerOutcome, ExplorerRace, ExplorerRunEntry, ExplorerScenario } from "./explorer-types.ts";
import type { HouseLiveRaceView, HouseLiveView } from "./house-live-view.ts";
import { HOUSE_LIVE_EXPECTED_DISTRICT_KEYS } from "./house-live-result.ts";

export interface ElectionNightPair {
  firstCandidateId: string; secondCandidateId: string;
  firstCandidateName: string; secondCandidateName: string;
  denominator: string; units: "percentage_points"; orientation: "first_minus_second";
  meanMarginPp: number; medianMarginPp: number; range90: { lower: number; upper: number };
  firstExceedsHalf: number;
  winnerRelationship: "same_event" | "separate_conditional_event";
  closeWithin2Pp: number | null; firstAtLeast10Pp: number | null; secondAtLeast10Pp: number | null;
  qualifications: string[];
}
export interface ElectionNightRace {
  id: string; state: string; seatId: string; electionLabel: string;
  ruleFamily: string | null; method: string | null; notes: string[]; sourceUrls: string[];
  /** Unchanged listed modeled outcomes. A group's probability is never allocated among its members. */
  outcomes: ExplorerOutcome[];
  pair: ElectionNightPair | null;
  modalPick: { outcomeId: string; name: string; kind: "candidate" | "group"; probability: number; tiedOutcomeIds: string[] };
}
export interface ElectionNightForecast {
  schema: "ElectionNightForecastV1"; chamber: "senate" | "house";
  editionStatus: "dated_snapshot";
  run: {
    id: string; label: string; year: 2026; configuration: Record<string, string>; artifactSha256: string;
    issuedAt: string; cutoffAt: string; strictAsOf: boolean; sourceCutoffs: { label: string; at: string }[];
    targetId: string | null; lawId: string | null;
  };
  qualification: string[]; sourceUrls: string[]; sourceHashes: Record<string, string>;
  sourceNotes: { label: string; url: string; notes: string[] }[];
  races: ElectionNightRace[]; chamberScenario: ExplorerScenario | null;
  summary: {
    raceCount: number; pairRaceCount: number; expectedModalMisses: number; topTieRaceCount: number;
    tiePolicy: "lexicographic_outcome_id"; interpretation: string;
  };
}
const DENOMINATOR = "Share of votes for the two named principal contenders";
const supportNote = "Probabilities describe listed modeled outcomes; roster corrections require a new dated edition.";
const hash = (v: unknown): v is string => typeof v === "string" && /^[0-9a-f]{64}$/.test(v);
const fraction = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v) && v >= 0 && v <= 1;
const text = (v: unknown): v is string => typeof v === "string" && v.trim().length > 0;
const near = (a: number, b: number) => Math.abs(a - b) <= 1e-8;
function requireValue(value: unknown, message: string): asserts value { if (!value) throw Error(`Election-night forecast: ${message}`); }
function strings(value: unknown): asserts value is string[] {
  requireValue(Array.isArray(value) && value.every(text), "invalid text list");
}
function dates(issued: string, cutoff: string, strict: boolean, sources: { label: string; at: string }[]): void {
  requireValue(typeof issued === "string" && typeof cutoff === "string" && Number.isFinite(Date.parse(issued)) &&
    Number.isFinite(Date.parse(cutoff)) && Date.parse(cutoff) <= Date.parse(issued) && typeof strict === "boolean", "invalid dated edition");
  requireValue(Array.isArray(sources) && sources.every(s => text(s.label) && typeof s.at === "string" &&
    Number.isFinite(Date.parse(s.at)) && Date.parse(s.at) <= Date.parse(cutoff)), "invalid source cutoffs");
}
function normalized(value: unknown): string {
  return JSON.stringify(value, (_, v) => v && typeof v === "object" && !Array.isArray(v)
    ? Object.fromEntries(Object.entries(v).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)) : v);
}
function candidateUrls(outcomes: ExplorerOutcome[]): string[] {
  return [...new Set(outcomes.flatMap(o => o.members.flatMap(m => m.sourceUrls)))];
}
function winnerRace(race: ExplorerRace, chamber: "senate" | "house"): ElectionNightRace {
  requireValue(text(race.id) && text(race.state) && text(race.seatId) && text(race.electionLabel) &&
    !("observed" in race), "unexpected current race identity or observed result");
  strings(race.notes);
  requireValue(Array.isArray(race.outcomes) && race.outcomes.length > 0, "missing listed winner support");
  const ids = new Set<string>(), members = new Set<string>();
  for (const outcome of race.outcomes) {
    requireValue(text(outcome.id) && !ids.has(outcome.id) && text(outcome.name) && text(outcome.party) &&
      ["candidate", "group"].includes(outcome.kind) && fraction(outcome.winProbability) &&
      Array.isArray(outcome.members) && outcome.members.length > 0, "invalid winner outcome");
    ids.add(outcome.id);
    requireValue(outcome.kind !== "candidate" || outcome.members.length === 1 && outcome.name === outcome.members[0].name,
      "candidate outcome must match its singleton member");
    for (const member of outcome.members) {
      requireValue(text(member.id) && !members.has(member.id) && text(member.name) && text(member.party), "invalid or repeated outcome member");
      members.add(member.id); strings(member.sourceUrls);
      requireValue(member.sourceUrls.every(url => /^https?:\/\//.test(url)), "invalid candidate source URL");
    }
    if (chamber === "house") requireValue(!("share" in outcome), "House winner groups cannot carry pair shares");
  }
  requireValue(near(race.outcomes.reduce((s, o) => s + o.winProbability, 0), 1), "winner probabilities must sum to one");
  const maximum = Math.max(...race.outcomes.map(o => o.winProbability));
  // Exact equality identifies exact top ties. Sorting uses code-unit order, never locale or input order.
  const ties = race.outcomes.filter(o => o.winProbability === maximum).map(o => o.id).sort();
  const picked = race.outcomes.find(o => o.id === ties[0])!;
  return { id: race.id, state: race.state, seatId: race.seatId, electionLabel: race.electionLabel,
    ruleFamily: null, method: null, notes: [...race.notes], sourceUrls: candidateUrls(race.outcomes),
    outcomes: structuredClone(race.outcomes), pair: null,
    modalPick: { outcomeId: picked.id, name: picked.name, kind: picked.kind, probability: maximum, tiedOutcomeIds: ties } };
}
function pair(firstId: string, secondId: string, firstName: string, secondName: string, denominator: string,
  mean: number, q: { p05: number; p50: number; p95: number }, firstExceedsHalf: number,
  events: [number | null, number | null, number | null], qualifications: string[],
  winnerRelationship: ElectionNightPair["winnerRelationship"]): ElectionNightPair {
  requireValue(text(firstId) && text(secondId) && firstId !== secondId && text(firstName) && text(secondName) &&
    denominator === DENOMINATOR && fraction(mean) && fraction(q.p05) && fraction(q.p50) && fraction(q.p95) &&
    q.p05 <= q.p50 && q.p50 <= q.p95 && fraction(firstExceedsHalf) &&
    events.every(p => p === null || fraction(p)), "invalid named conditional pair");
  strings(qualifications);
  if (events.every(p => p !== null)) {
    const [close, first, second] = events as [number, number, number];
    requireValue(close + first + second <= 1 + 1e-8 && first <= firstExceedsHalf + 1e-8 &&
      second <= 1 - firstExceedsHalf + 1e-8, "inconsistent conditional pair events");
  }
  const margin = (s: number) => 100 * (2 * s - 1);
  return { firstCandidateId: firstId, secondCandidateId: secondId, firstCandidateName: firstName,
    secondCandidateName: secondName, denominator, units: "percentage_points", orientation: "first_minus_second",
    meanMarginPp: margin(mean), medianMarginPp: margin(q.p50), range90: { lower: margin(q.p05), upper: margin(q.p95) },
    firstExceedsHalf, winnerRelationship, closeWithin2Pp: events[0], firstAtLeast10Pp: events[1], secondAtLeast10Pp: events[2],
    qualifications: [...qualifications] };
}
function finish(value: Omit<ElectionNightForecast, "schema" | "editionStatus" | "summary">): ElectionNightForecast {
  const ids = value.races.map(r => r.id);
  requireValue(new Set(ids).size === ids.length, "duplicate race identity");
  return { schema: "ElectionNightForecastV1", editionStatus: "dated_snapshot", ...value,
    summary: { raceCount: value.races.length, pairRaceCount: value.races.filter(r => r.pair !== null).length,
      expectedModalMisses: value.races.reduce((sum, r) => sum + (1 - r.modalPick.probability), 0),
      topTieRaceCount: value.races.filter(r => r.modalPick.tiedOutcomeIds.length > 1).length,
      tiePolicy: "lexicographic_outcome_id",
      interpretation: "Model-implied expected number of modal-outcome misses across the listed races. Linearity of expectation requires no independence assumption. This is not expected margin error or a predictive interval for misses." } };
}
function senateConfiguration(entry: ExplorerRunEntry): void {
  const c = entry.components;
  const base = ["recent_calendar_history", "senate_history_refreshed_v1", "senate_presidential_slope_v1", "senate_relative_lean_v1", "senate_relative_context_v1"];
  if (entry.id.startsWith("current-comparison-")) {
    requireValue(base.includes(c.base) && ["none", "standard"].includes(c.polling) && c.uncertainty === "standard" &&
      entry.id === `current-comparison-${c.base}-${c.polling === "none" ? "f" : "fp"}` &&
      entry.targetId === "senate2026_principal_pair_fundamentals_comparison_v1" &&
      entry.lawId === "principal_pair_empirical_logit_shared_normal_student_t5_v1", "unsupported Senate comparison identity");
  } else {
    const id = c.polling === "none" ? "current-fundamentals" : c.uncertainty === "standard" ? "current-polling" : `current-${c.uncertainty}`;
    requireValue(c.base === "fundamentals" && ["none", "standard"].includes(c.polling) &&
      ["standard", "cycle-low", "cycle-high", "error-low", "error-high"].includes(c.uncertainty) &&
      (c.polling !== "none" || c.uncertainty === "standard") && entry.id === id &&
      entry.targetId === (c.polling === "none" ? "national_senate_aggregate_baseline_v1" : "national_senate_aggregate_poll_v1") &&
      entry.lawId === "senate_principal_pair_aggregate_v1", "unsupported Senate saved identity");
  }
}
export function toSenateElectionNightForecast(bundle: ExplorerBundle, entry: ExplorerRunEntry): ElectionNightForecast {
  const { href, sha256, ...metadata } = entry;
  requireValue(bundle.schemaVersion === 1 && bundle.run.year === 2026 && bundle.run.historical === false &&
    bundle.historicalNote === null && normalized(bundle.run) === normalized(metadata) && hash(sha256) &&
    href === `/data/explorer/${entry.id}.json` && bundle.races.length === 35, "Senate current bundle/entry identity mismatch");
  senateConfiguration(entry);
  const sourceCutoffs = (bundle.run.inputCutoffs ?? []).map(s => ({ label: s.label, at: s.at }));
  dates(bundle.run.issuedAt, bundle.run.cutoffAt, bundle.run.strictAsOf, sourceCutoffs);
  requireValue(bundle.provenance && hash(bundle.provenance.sources.forecastSha256) &&
    hash(bundle.provenance.sources.jointDrawsSha256) && Object.values(bundle.provenance.sources).every(hash), "invalid Senate source hashes");
  strings(bundle.limitations);
  const races = bundle.races.map(race => {
    requireValue(race.id.startsWith(`2026:${race.state}:senate:`) && /^2026:[A-Z]{2}:senate:(regular|special)$/.test(race.id) &&
      new RegExp(`^US-SEN-${race.state}-CLASS-[123]$`).test(race.seatId) && race.outcomes.length === 2,
      "unexpected Senate race support");
    const projected = winnerRace(race, "senate"), [first, second] = race.outcomes;
    requireValue(first.kind === "candidate" && second.kind === "candidate" &&
      first.members[0].id === first.id && second.members[0].id === second.id &&
      first.members[0].name === first.name && second.members[0].name === second.name &&
      first.share && second.share && first.share.denominator === second.share.denominator &&
      fraction(second.share.mean) && near(first.share.mean + second.share.mean, 1) &&
      near(first.share.quantiles.p05 + second.share.quantiles.p95, 1) &&
      near(first.share.quantiles.p50 + second.share.quantiles.p50, 1) &&
      near(first.share.quantiles.p95 + second.share.quantiles.p05, 1), "Senate named-pair identity/share mismatch");
    projected.pair = pair(first.id, second.id, first.name, second.name, first.share.denominator,
      // The current Senate export validates first_wins == shares > .5 in
      // scripts/export_tipping_points.py::_flatten_draws. These are the same event.
      first.share.mean, first.share.quantiles, first.winProbability, [null, null, null], race.notes, "same_event");
    return projected;
  });
  requireValue(Array.isArray(bundle.scenarios) && new Set(bundle.scenarios.map(s => s.id)).size === bundle.scenarios.length,
    "duplicate Senate chamber scenario");
  const scenario = bundle.scenarios.find(s => s.id === bundle.defaultScenarioId);
  requireValue(scenario && !("performance" in scenario), "missing current default Senate chamber scenario");
  return finish({ chamber: "senate", run: { id: entry.id, label: entry.label, year: 2026,
    configuration: { ...entry.components }, artifactSha256: sha256, issuedAt: entry.issuedAt, cutoffAt: entry.cutoffAt,
    strictAsOf: entry.strictAsOf, sourceCutoffs, targetId: entry.targetId, lawId: entry.lawId },
    qualification: [...bundle.limitations, entry.uncertaintyDescription,
      ...(entry.researchEvidence ? [entry.researchEvidence.status, entry.researchEvidence.numericalStatus,
        entry.researchEvidence.forecastSkill, entry.researchEvidence.qualification] : []), supportNote],
    sourceUrls: [...new Set(races.flatMap(r => r.sourceUrls))], sourceHashes: { ...bundle.provenance.sources },
    sourceNotes: [], races, chamberScenario: structuredClone(scenario) });
}
export function toHouseElectionNightForecast(view: HouseLiveView): ElectionNightForecast {
  const run = view.run, c = run.configuration;
  const bases = ["fundamentals", "recent_calendar_history", "house_history_refreshed_v1", "house_presidential_slope_v1", "house_relative_lean_v1", "house_relative_context_v1"];
  const id = c.base === "fundamentals" ? c.uncertainty !== "standard" ? `house-current-${c.uncertainty}` :
    `house-current-${c.polling === "none" ? "fundamentals" : "polling"}` : `house-current-comparison-${c.base}-${c.polling === "none" ? "f" : "fp"}`;
  requireValue(view.schema === "HouseLiveDisplayV1" && run.cycle === 2026 && run.historical === false && run.id === id &&
    bases.includes(c.base) && ["none", "standard"].includes(c.polling) &&
    ["standard", "cycle-low", "cycle-high", "error-low", "error-high"].includes(c.uncertainty) &&
    (c.uncertainty === "standard" || c.base === "fundamentals" && c.polling === "standard") &&
    run.electionDate === "2026-11-03" && view.mapVintage === "house_election_map_2026" && hash(run.resultSha256) &&
    view.races.length === 435 && view.qualification.publicationStatus === "reviewed", "House current run identity mismatch");
  const sourceCutoffs = run.sourceCutoffs.map(s => ({ label: s.role, at: s.at }));
  dates(run.issuedAt, run.cutoffAt, run.strictAsOf, sourceCutoffs);
  const expected = new Set(HOUSE_LIVE_EXPECTED_DISTRICT_KEYS);
  const races = view.races.map((race: HouseLiveRaceView) => {
    requireValue(expected.has(race.seatId) && race.id === `2026:${race.seatId}:house:regular` &&
      race.state === race.seatId.slice(0, 2) && race.district.districtKey === race.seatId &&
      race.district.cycle === 2026 && race.district.mapVintage === view.mapVintage &&
      ["plurality", "ranked_choice", "majority_runoff", "top_two_plurality"].includes(race.ruleFamily) && text(race.method), "House race/district identity mismatch");
    const projected = winnerRace(race, "house");
    projected.ruleFamily = race.ruleFamily; projected.method = race.method;
    requireValue(new Set(race.candidates.map(m => m.id)).size === race.candidates.length, "duplicate House candidate identity");
    const members = race.outcomes.flatMap(o => o.members);
    requireValue(members.length === race.candidates.length && members.every(m => race.candidates.some(c =>
      c.id === m.id && c.name === m.name && c.aggregationParty === m.party)) &&
      race.outcomes.every(o => ["D", "R", "O"].includes(o.party) && o.id === `${race.id}:${o.party}` &&
        o.members.every(m => m.party === o.party)), "House candidate/winner support mismatch");
    const conditional = race.conditionalPair;
    requireValue(conditional === null || typeof conditional === "object", "explicit House pair availability required");
    if (conditional) {
      requireValue(race.candidates.some(m => m.id === conditional.firstCandidateId && m.name === conditional.firstCandidateName) &&
        race.candidates.some(m => m.id === conditional.secondCandidateId && m.name === conditional.secondCandidateName), "House named-pair identity mismatch");
      projected.pair = pair(conditional.firstCandidateId, conditional.secondCandidateId, conditional.firstCandidateName,
        conditional.secondCandidateName, conditional.denominator, conditional.mean, conditional.quantiles, conditional.firstExceedsHalf,
        [conditional.marginWithinTwoPoints, conditional.firstMarginAtLeastTenPoints, conditional.secondMarginAtLeastTenPoints], conditional.qualifications,
        "separate_conditional_event");
    }
    return projected;
  });
  requireValue(view.chamber === null || view.chamber.id === "party_majority_others_unassigned" && !("performance" in view.chamber), "unexpected House chamber scenario");
  requireValue(hash(view.jointDraws.sha256) && hash(view.geographySha256) &&
    normalized(view.jointDraws.orderedRaceIds) === normalized(view.races.map(r => r.id)), "House joint/race identity mismatch");
  const sourceHashes: Record<string, string> = { jointDrawsSha256: view.jointDraws.sha256, geographySha256: view.geographySha256 };
  view.sources.forEach(s => { requireValue(hash(s.sha256) && /^https?:\/\//.test(s.url), "invalid House provenance"); sourceHashes[s.id] = s.sha256; });
  view.qualification.evidenceHashes.forEach((sha, i) => { requireValue(hash(sha), "invalid House evidence hash"); sourceHashes[`evidence:${i}`] = sha; });
  return finish({ chamber: "house", run: { id: run.id, label: run.label, year: 2026,
    configuration: { ...c }, artifactSha256: run.resultSha256, issuedAt: run.issuedAt, cutoffAt: run.cutoffAt,
    strictAsOf: run.strictAsOf, sourceCutoffs, targetId: null, lawId: null },
    qualification: [view.qualification.inputStatus, view.qualification.numericalStatus, view.qualification.publicationStatus,
      view.qualification.forecastSkill, ...view.qualification.limitations,
      ...view.historicalEvidence.map(h => h.qualification), supportNote],
    sourceUrls: [...new Set([...races.flatMap(r => r.sourceUrls), ...view.sources.map(s => s.url), ...view.historicalEvidence.map(h => h.url)])],
    sourceNotes: view.sources.filter(s => s.limitations?.length).map(s => ({ label: s.label, url: s.url, notes: [...s.limitations!] })),
    sourceHashes, races, chamberScenario: structuredClone(view.chamber) });
}
