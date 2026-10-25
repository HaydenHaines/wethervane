/** Browser response shape checks only; server artifact authentication/review is separate. */
import type { DistrictGeography } from "./district-geography";
import { joinDistrictGeography, parseDistrictKey } from "./district-geography.ts";
import { HOUSE_LIVE_EXPECTED_DISTRICT_KEYS, houseLiveConfigurationKey, parseHouseLiveConfiguration } from "./house-live-result.ts";
import type { ExplorerOutcome, ExplorerRace, ExplorerScenario } from "./explorer-types";
import { HOUSE_LIVE_ALLOWED_CONFIGURATION_KEYS } from "./house-live-catalog.ts";
import type { HouseLiveCatalogView, HouseLiveRaceView, HouseLiveView } from "./house-live-view.ts";

export interface HouseLivePublication {
  schema: "HouseLivePublicationV1";
  status: "available";
  catalog: HouseLiveCatalogView;
  view: HouseLiveView;
  geography: DistrictGeography | null;
}

const expectedConfigurations = new Set<string>(HOUSE_LIVE_ALLOWED_CONFIGURATION_KEYS);
const object = (value: unknown): value is Record<string, unknown> => !!value && typeof value === "object" && !Array.isArray(value);
const text = (value: unknown): value is string => typeof value === "string" && value.length > 0 && value.length <= 4000 && value === value.trim() && !/[\u0000-\u001f\u007f]/.test(value);
const safeId = (value: unknown): value is string => text(value) && /^[A-Za-z0-9][A-Za-z0-9:._-]{0,199}$/.test(value);
const exactKeys = (value: Record<string, unknown>, keys: readonly string[]) => Object.keys(value).length === keys.length && keys.every(key => key in value);
const hash = (value: unknown): value is string => typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
const fraction = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;
const sum = (values: number[]) => values.reduce((total, value) => total + value, 0);
const near = (a: number, b: number, epsilon = 1e-8) => Math.abs(a - b) <= epsilon;
function utcTimestamp(value: unknown): value is string {
  if (typeof value !== "string") return false;
  const match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?(?:Z|\+00:00)$/.exec(value);
  if (!match) return false;
  const [, ys, ms, ds, hs, mins, ss] = match;
  const year = Number(ys), month = Number(ms), day = Number(ds), hour = Number(hs), minute = Number(mins), second = Number(ss);
  const leap = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
  const days = [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  return month >= 1 && month <= 12 && day >= 1 && day <= days[month - 1] && hour <= 23 && minute <= 59 && second <= 59 && Number.isFinite(Date.parse(value));
}
function noObserved2026(value: unknown): boolean {
  if (Array.isArray(value)) return value.every(noObserved2026);
  if (!object(value)) return true;
  for (const [key, child] of Object.entries(value)) {
    if (["observed", "observedOutcome", "observed_outcome", "actualWinner", "actual_winner", "actualResult", "actual_result", "performance"].includes(key)) return false;
    if (!noObserved2026(child)) return false;
  }
  return true;
}
function parseCatalog(value: unknown): HouseLiveCatalogView | null {
  if (!object(value) || !Array.isArray(value.entries) || value.entries.length !== 16 || typeof value.defaultId !== "string") return null;
  const seenIds = new Set<string>(), seenConfigurations = new Set<string>();
  const entries: HouseLiveCatalogView["entries"] = [];
  for (const raw of value.entries) {
    if (!object(raw) || !safeId(raw.id) || !text(raw.label) || !hash(raw.resultSha256) || seenIds.has(raw.id)) return null;
    const configuration = parseHouseLiveConfiguration(raw.configuration);
    if (!configuration) return null;
    const key = houseLiveConfigurationKey(configuration);
    if (!expectedConfigurations.has(key) || seenConfigurations.has(key)) return null;
    seenIds.add(raw.id); seenConfigurations.add(key);
    entries.push({ id: raw.id, label: raw.label, configuration: { ...configuration }, resultSha256: raw.resultSha256 });
  }
  if (seenConfigurations.size !== expectedConfigurations.size || !seenIds.has(value.defaultId)) return null;
  const defaultEntry = entries.find(entry => entry.id === value.defaultId);
  if (!defaultEntry || houseLiveConfigurationKey(defaultEntry.configuration) !== "fundamentals|standard|standard") return null;
  return { defaultId: value.defaultId, entries };
}

function parseRace(value: unknown, mapVintage: string, sourceUrls: ReadonlySet<string>): HouseLiveRaceView | null {
  if (!object(value) || !text(value.id) || !text(value.state) || !text(value.seatId) || !text(value.electionLabel) ||
    !Number.isSafeInteger(value.pollCount) || (value.pollCount as number) < 0 || !Array.isArray(value.notes) || !value.notes.every(text) ||
    !object(value.district) || value.district.cycle !== 2026 || value.district.districtKey !== value.seatId || value.district.mapVintage !== mapVintage ||
    !["plurality", "ranked_choice", "majority_runoff", "top_two_plurality"].includes(String(value.ruleFamily)) || !text(value.method) ||
    !Array.isArray(value.candidates) || !value.candidates.length || !Array.isArray(value.outcomes) || !value.outcomes.length || value.outcomes.length > 3 ||
    !(value.conditionalPair === null || object(value.conditionalPair))) return null;
  const parsedKey = parseDistrictKey(value.seatId);
  if (!parsedKey || parsedKey.state !== value.state || value.id !== `2026:${value.seatId}:house:regular`) return null;
  const candidates = new Map<string, HouseLiveRaceView["candidates"][number]>();
  for (const raw of value.candidates) {
    if (!object(raw) || !safeId(raw.id) || candidates.has(raw.id) || !text(raw.name) ||
      !(raw.partyLabel === null || text(raw.partyLabel)) || !["D", "R", "O"].includes(String(raw.aggregationParty)) ||
      !["printed", "declared_writein"].includes(String(raw.ballotStatus)) || typeof raw.incumbent !== "boolean" ||
      (raw.partyLabel === null && raw.ballotStatus !== "declared_writein") ||
      (value.seatId === "TX-23" && /\b(?:tony|antonio)[- _]+gonzales\b/i.test(`${raw.name} ${raw.id}`)) ||
      !Array.isArray(raw.sourceUrls) || !raw.sourceUrls.length || !raw.sourceUrls.every(url => text(url) && /^https?:\/\//i.test(url)) ||
      new Set(raw.sourceUrls).size !== raw.sourceUrls.length || !raw.sourceUrls.every(url => sourceUrls.has(url as string))) return null;
    candidates.set(raw.id, { id: raw.id, name: raw.name, partyLabel: raw.partyLabel as string | null,
      aggregationParty: raw.aggregationParty as "D" | "R" | "O", ballotStatus: raw.ballotStatus as "printed" | "declared_writein",
      incumbent: raw.incumbent, sourceUrls: [...new Set(raw.sourceUrls as string[])] });
  }
  const outcomes: ExplorerOutcome[] = [], outcomeParties = new Set<string>(), covered = new Set<string>();
  let probabilityMass = 0;
  for (const raw of value.outcomes) {
    if (!object(raw) || !text(raw.id) || !text(raw.name) || !["D", "R", "O"].includes(String(raw.party)) ||
      outcomeParties.has(String(raw.party)) || !["candidate", "group"].includes(String(raw.kind)) || !fraction(raw.winProbability) ||
      !Array.isArray(raw.members) || !raw.members.length || (raw.kind === "candidate" && raw.members.length !== 1) || "share" in raw) return null;
    if (raw.id !== `${value.id}:${raw.party}`) return null;
    const members: ExplorerOutcome["members"] = [];
    for (const member of raw.members) {
      if (!object(member) || !text(member.id) || !text(member.name) || !["D", "R", "O"].includes(String(member.party)) ||
        typeof member.incumbent !== "boolean" || !Array.isArray(member.sourceUrls) || !member.sourceUrls.every(text) ||
        covered.has(member.id)) return null;
      const candidate = candidates.get(member.id);
      if (!candidate || candidate.name !== member.name || candidate.aggregationParty !== member.party || candidate.incumbent !== member.incumbent ||
        JSON.stringify(candidate.sourceUrls) !== JSON.stringify([...new Set(member.sourceUrls as string[])])) return null;
      if (member.party !== raw.party) return null;
      covered.add(member.id);
      members.push({ id: member.id as string, name: member.name as string, party: member.party as string,
        incumbent: member.incumbent as boolean, sourceUrls: [...member.sourceUrls as string[]] });
    }
    if ((raw.kind === "candidate" && raw.name !== members[0].name) ||
      (raw.kind === "group" && members.length === 1 && raw.name === members[0].name)) return null;
    outcomeParties.add(String(raw.party)); probabilityMass += raw.winProbability as number;
    outcomes.push({ id: raw.id, name: raw.name, party: raw.party as string, kind: raw.kind as "candidate" | "group", members, winProbability: raw.winProbability as number });
  }
  if (covered.size !== candidates.size || !near(probabilityMass, 1, 1e-8)) return null;
  let conditionalPair: HouseLiveRaceView["conditionalPair"] = null;
  if (value.conditionalPair !== null) {
    const pair = value.conditionalPair;
    const quantiles = pair.quantiles;
    const pairKeys = ["p025", "p05", "p10", "p50", "p90", "p95", "p975"];
    if (!text(pair.firstCandidateId) || !text(pair.secondCandidateId) || pair.firstCandidateId === pair.secondCandidateId ||
      !candidates.has(pair.firstCandidateId) || !candidates.has(pair.secondCandidateId) ||
      pair.firstCandidateName !== candidates.get(pair.firstCandidateId)!.name || pair.secondCandidateName !== candidates.get(pair.secondCandidateId)!.name ||
      pair.denominator !== "Share of votes for the two named principal contenders" || !fraction(pair.mean) || !object(quantiles) || Object.keys(quantiles).length !== pairKeys.length ||
      !pairKeys.every(key => fraction(quantiles[key])) || !pairKeys.every((key, i) => i === 0 || (quantiles[pairKeys[i - 1]] as number) <= (quantiles[key] as number)) ||
      ![pair.firstExceedsHalf, pair.marginWithinTwoPoints, pair.firstMarginAtLeastTenPoints, pair.secondMarginAtLeastTenPoints].every(fraction) ||
      !Array.isArray(pair.qualifications) || !pair.qualifications.every(text) ||
      (pair.firstMarginAtLeastTenPoints as number) + (pair.secondMarginAtLeastTenPoints as number) + (pair.marginWithinTwoPoints as number) > 1 + 1e-8 ||
      (pair.firstMarginAtLeastTenPoints as number) > (pair.firstExceedsHalf as number) + 1e-8 ||
      (pair.secondMarginAtLeastTenPoints as number) > 1 - (pair.firstExceedsHalf as number) + 1e-8) return null;
    conditionalPair = { firstCandidateId: pair.firstCandidateId, secondCandidateId: pair.secondCandidateId,
      firstCandidateName: pair.firstCandidateName, secondCandidateName: pair.secondCandidateName,
      denominator: pair.denominator, mean: pair.mean,
      quantiles: Object.fromEntries(pairKeys.map(key => [key, quantiles[key]])) as HouseLiveRaceView["conditionalPair"] extends infer T ? T extends { quantiles: infer Q } ? Q : never : never,
      firstExceedsHalf: pair.firstExceedsHalf as number, marginWithinTwoPoints: pair.marginWithinTwoPoints as number,
      firstMarginAtLeastTenPoints: pair.firstMarginAtLeastTenPoints as number, secondMarginAtLeastTenPoints: pair.secondMarginAtLeastTenPoints as number,
      qualifications: [...pair.qualifications] };
  }
  return { id: value.id, state: value.state, seatId: value.seatId, electionLabel: value.electionLabel,
    outcomes, pollCount: value.pollCount as number, notes: [...value.notes], district: { cycle: 2026, districtKey: value.seatId, mapVintage },
    ruleFamily: value.ruleFamily as HouseLiveRaceView["ruleFamily"], method: value.method,
    candidates: [...candidates.values()], conditionalPair };
}

function parseChamber(value: unknown): ExplorerScenario | null | undefined {
  if (value === null) return null;
  if (!object(value) || !text(value.id) || !text(value.label) || !text(value.description) || !object(value.thresholds) ||
    !exactKeys(value.thresholds, ["D", "R"]) ||
    value.thresholds.D !== 218 || value.thresholds.R !== 218 || !object(value.retainedSeats) ||
    !exactKeys(value.retainedSeats, ["D", "R", "U"]) ||
    value.retainedSeats.D !== 0 || value.retainedSeats.R !== 0 || value.retainedSeats.U !== 0 || !object(value.control) ||
    !exactKeys(value.control, ["description", "democraticMajorityProbability", "republicanMajorityProbability", "unresolvedOrganizingProbability"]) ||
    !text(value.control.description) || ![value.control.democraticMajorityProbability, value.control.republicanMajorityProbability,
      value.control.unresolvedOrganizingProbability].every(fraction) ||
    !near(sum([value.control.democraticMajorityProbability as number, value.control.republicanMajorityProbability as number,
      value.control.unresolvedOrganizingProbability as number]), 1) || !object(value.seats) || !exactKeys(value.seats, ["D", "R", "U"])) return undefined;
  const seats: Record<string, unknown> = {};
  const expectedSeatTotals: number[] = [];
  for (const party of ["D", "R", "U"] as const) {
    const distribution = value.seats[party];
    if (!object(distribution) || !Number.isFinite(distribution.expectedSeats) || (distribution.expectedSeats as number) < 0 ||
      !Array.isArray(distribution.pmf) || distribution.pmf.length !== 436 || !distribution.pmf.every(fraction) ||
      !near(sum(distribution.pmf as number[]), 1) || !object(distribution.seatQuantiles) ||
      ![distribution.seatQuantiles.p05, distribution.seatQuantiles.p50, distribution.seatQuantiles.p95].every(q => typeof q === "number" && Number.isFinite(q) && q >= 0 && q <= 435) ||
      !((distribution.seatQuantiles.p05 as number) <= (distribution.seatQuantiles.p50 as number) && (distribution.seatQuantiles.p50 as number) <= (distribution.seatQuantiles.p95 as number))) return undefined;
    const expected = (distribution.pmf as number[]).reduce((acc, chance, number) => acc + chance * number, 0);
    if (!near(expected, distribution.expectedSeats as number, 1e-6)) return undefined;
    expectedSeatTotals.push(distribution.expectedSeats as number);
    seats[party] = { expectedSeats: distribution.expectedSeats, pmf: [...distribution.pmf], seatQuantiles: {
      p05: distribution.seatQuantiles.p05, p50: distribution.seatQuantiles.p50, p95: distribution.seatQuantiles.p95 } };
  }
  if (!near(sum(expectedSeatTotals), 435, 1e-6)) return undefined;
  const dDistribution = value.seats.D as Record<string, unknown>;
  const rDistribution = value.seats.R as Record<string, unknown>;
  const dMajority = sum((dDistribution.pmf as number[]).slice(218));
  const rMajority = sum((rDistribution.pmf as number[]).slice(218));
  if (!near(value.control.democraticMajorityProbability as number, dMajority) ||
    !near(value.control.republicanMajorityProbability as number, rMajority)) return undefined;
  return { id: value.id, label: value.label, description: value.description, thresholds: { D: 218, R: 218 },
    retainedSeats: { D: 0, R: 0, U: 0 }, control: { description: value.control.description,
      democraticMajorityProbability: value.control.democraticMajorityProbability as number,
      republicanMajorityProbability: value.control.republicanMajorityProbability as number,
      unresolvedOrganizingProbability: value.control.unresolvedOrganizingProbability as number },
    seats: seats as ExplorerScenario["seats"] };
}

function parseView(value: unknown, catalog: HouseLiveCatalogView): HouseLiveView | null {
  if (!object(value) || value.schema !== "HouseLiveDisplayV1" || !object(value.run) || !safeId(value.run.id) || !text(value.run.label) ||
    value.run.cycle !== 2026 || value.run.historical !== false || !parseHouseLiveConfiguration(value.run.configuration) ||
    !hash(value.run.resultSha256) || !utcTimestamp(value.run.issuedAt) || !utcTimestamp(value.run.cutoffAt) ||
    Date.parse(value.run.cutoffAt) > Date.parse(value.run.issuedAt) || value.run.electionDate !== "2026-11-03" || typeof value.run.strictAsOf !== "boolean" ||
    !Array.isArray(value.run.sourceCutoffs) || !value.run.sourceCutoffs.length || !text(value.mapVintage) || value.mapVintage !== "house_election_map_2026" ||
    !Array.isArray(value.races) || value.races.length !== 435 || value.tipping !== null || !object(value.qualification) ||
    !text(value.qualification.inputStatus) || !text(value.qualification.numericalStatus) || !text(value.qualification.publicationStatus) ||
    !["unobserved_2026", "not_established"].includes(String(value.qualification.forecastSkill)) ||
    value.qualification.publicationStatus !== "reviewed" ||
    !Array.isArray(value.qualification.limitations) || !value.qualification.limitations.every(text) ||
    !Array.isArray(value.qualification.evidenceHashes) || !value.qualification.evidenceHashes.length || !value.qualification.evidenceHashes.every(hash) ||
    !object(value.jointDraws) || !hash(value.jointDraws.sha256) || !Number.isSafeInteger(value.jointDraws.count) || (value.jointDraws.count as number) <= 0 ||
    !Number.isSafeInteger(value.jointDraws.chainCount) || (value.jointDraws.chainCount as number) <= 0 ||
    (value.jointDraws.count as number) % (value.jointDraws.chainCount as number) !== 0 || !Array.isArray(value.jointDraws.orderedRaceIds) ||
    value.jointDraws.orderedRaceIds.length !== 435 || !value.jointDraws.orderedRaceIds.every(text) || !text(value.jointDraws.identitySemantics) ||
    !hash(value.geographySha256) || !Array.isArray(value.sources) || !value.sources.length || !Array.isArray(value.historicalEvidence) || !value.historicalEvidence.length ||
    !(value.chamberPrecision === null || object(value.chamberPrecision))) return null;
  const run = value.run;
  const jointDraws = value.jointDraws;
  const precision = value.chamberPrecision;
  const sourceCutoffs = run.sourceCutoffs as unknown[];
  const orderedRaceIds = jointDraws.orderedRaceIds as string[];
  const configuration = parseHouseLiveConfiguration(run.configuration)!;
  const entry = catalog.entries.find(candidate => candidate.id === run.id);
  if (!entry || entry.resultSha256 !== value.run.resultSha256 || houseLiveConfigurationKey(entry.configuration) !== houseLiveConfigurationKey(configuration)) return null;
  if (!sourceCutoffs.every(row => object(row) && text(row.role) && utcTimestamp(row.at) && Date.parse(row.at) <= Date.parse(run.cutoffAt as string))) return null;
  const sourceIds = new Set<string>();
  const sources = value.sources.map(source => {
    if (!object(source) || !text(source.id) || !text(source.label) || !text(source.url) || !/^https?:\/\//i.test(source.url) ||
      sourceIds.has(source.id) || !utcTimestamp(source.retrieved_at) || Date.parse(source.retrieved_at) > Date.parse(run.cutoffAt as string) ||
      !hash(source.sha256) || !text(source.role)) return null;
    sourceIds.add(source.id);
    return { id: source.id, label: source.label, url: source.url, retrieved_at: source.retrieved_at, sha256: source.sha256, role: source.role };
  });
  if (sources.some(item => !item)) return null;
  const sourceUrlSet = new Set((sources as HouseLiveView["sources"]).map(source => source.url));
  if (value.chamber === null && !value.qualification.limitations.some(note => /chamber|control|seat total/i.test(note) &&
    /unavailable|unqualified|failed|withheld|not published|not qualified/i.test(note))) return null;
  if (value.chamber === null && value.chamberPrecision !== null) return null;
  if (value.chamber !== null && value.chamberPrecision === null) return null;
  const chamber = parseChamber(value.chamber);
  if (chamber === undefined) return null;
  if (precision !== null && (!object(precision) || !Number.isFinite(precision.maxSeatMeanMcse) || (precision.maxSeatMeanMcse as number) < 0 ||
    !fraction(precision.maxControlMcse) || !hash(precision.evidenceSha256))) return null;
  const races = value.races.map(race => parseRace(race, value.mapVintage as string, sourceUrlSet));
  if (races.some(race => !race)) return null;
  const parsedRaces = races as HouseLiveRaceView[];
  const raceKeys = new Set(parsedRaces.map(race => race.seatId));
  if (raceKeys.size !== HOUSE_LIVE_EXPECTED_DISTRICT_KEYS.length || HOUSE_LIVE_EXPECTED_DISTRICT_KEYS.some(key => !raceKeys.has(key)) ||
    new Set(parsedRaces.map(race => race.id)).size !== 435 ||
    new Set(orderedRaceIds).size !== 435 || orderedRaceIds.some((id, index) => id !== parsedRaces[index].id)) return null;
  const historicalEvidence = value.historicalEvidence.map(item => object(item) && text(item.label) && text(item.url) && /^https?:\/\//i.test(item.url) && text(item.qualification)
    ? { label: item.label, url: item.url, qualification: item.qualification } : null);
  if (historicalEvidence.some(item => !item)) return null;
  return { schema: "HouseLiveDisplayV1", run: { id: entry.id, label: entry.label, cycle: 2026, historical: false,
    configuration, resultSha256: entry.resultSha256, issuedAt: run.issuedAt as string, cutoffAt: run.cutoffAt as string,
    electionDate: "2026-11-03", strictAsOf: run.strictAsOf as boolean,
    sourceCutoffs: sourceCutoffs.map(row => ({ role: (row as Record<string, string>).role, at: (row as Record<string, string>).at })) },
    mapVintage: value.mapVintage, races: parsedRaces, chamber, tipping: null,
    qualification: { inputStatus: value.qualification.inputStatus, numericalStatus: value.qualification.numericalStatus,
      publicationStatus: value.qualification.publicationStatus as string, forecastSkill: value.qualification.forecastSkill as string,
      limitations: [...value.qualification.limitations], evidenceHashes: [...value.qualification.evidenceHashes] },
    chamberPrecision: precision === null ? null : { maxSeatMeanMcse: precision.maxSeatMeanMcse as number,
      maxControlMcse: precision.maxControlMcse as number, evidenceSha256: precision.evidenceSha256 as string },
    jointDraws: { sha256: jointDraws.sha256 as string, count: jointDraws.count as number, chainCount: jointDraws.chainCount as number,
      orderedRaceIds: [...orderedRaceIds], identitySemantics: jointDraws.identitySemantics as string },
    geographySha256: value.geographySha256, sources: sources as HouseLiveView["sources"],
    historicalEvidence: historicalEvidence as HouseLiveView["historicalEvidence"] };
}

function parseGeography(value: unknown, view: HouseLiveView): DistrictGeography | null | undefined {
  if (value === null) return null;
  if (!object(value) || !object(value.metadata) || !object(value.geojson) || value.geojson.type !== "FeatureCollection" ||
    !Array.isArray(value.geojson.features) || value.metadata.cycle !== 2026 || value.metadata.mapVintage !== view.mapVintage ||
    value.metadata.admission !== "verified" || value.metadata.sourceSha256 !== view.geographySha256 || !hash(value.metadata.sourceSha256)) return undefined;
  const features = value.geojson.features.map(feature => {
    if (!object(feature) || feature.type !== "Feature" || !object(feature.properties) || !text(feature.properties.districtKey) ||
      !object(feature.geometry) || !["Polygon", "MultiPolygon"].includes(String(feature.geometry.type))) return null;
    return { type: "Feature" as const, properties: { districtKey: feature.properties.districtKey },
      geometry: { type: feature.geometry.type as "Polygon" | "MultiPolygon", coordinates: structuredClone(feature.geometry.coordinates) } };
  });
  if (features.some(feature => !feature)) return undefined;
  const geography: DistrictGeography = { metadata: { cycle: 2026, mapVintage: view.mapVintage, admission: "verified",
    sourceSha256: view.geographySha256, ...(text(value.metadata.sourceName) ? { sourceName: value.metadata.sourceName } : {}),
    ...(text(value.metadata.sourceUrl) && /^https?:\/\//i.test(value.metadata.sourceUrl) ? { sourceUrl: value.metadata.sourceUrl } : {}),
    ...(text(value.metadata.sourceNote) ? { sourceNote: value.metadata.sourceNote } : {}) },
    geojson: { type: "FeatureCollection", features: features as unknown as DistrictGeography["geojson"]["features"] } };
  const joined = joinDistrictGeography(view.races.map(race => ({ race, districtKey: race.district.districtKey })),
    { cycle: 2026, mapVintage: view.mapVintage }, geography);
  return joined.status === "ready" && joined.matches.length === 435 ? geography : undefined;
}

/** Validate public browser response shape and selected-run consistency; never authenticates publication. */
export function parseHouseLivePublication(value: unknown): HouseLivePublication | null {
  if (!object(value) || value.schema !== "HouseLivePublicationV1" || value.status !== "available" || !noObserved2026(value)) return null;
  const catalog = parseCatalog(value.catalog);
  if (!catalog) return null;
  const view = parseView(value.view, catalog);
  if (!view) return null;
  const geography = parseGeography(value.geography, view);
  if (geography === undefined) return null;
  return { schema: "HouseLivePublicationV1", status: "available", catalog, view, geography };
}
