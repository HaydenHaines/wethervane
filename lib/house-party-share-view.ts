/** Private frontend fixtures only; this is not a backend publication schema. */
export type PartyShareEstimate =
  | { status: "modeled"; intervalKind: "central"; mean: number; intervals: { level: number; lower: number; upper: number }[] }
  | { status: "structural"; value: 0 | 1; absentParty: "D" | "R"; denominatorStatus: "positive_present_party"; reason: string }
  | { status: "unavailable"; reason: string };
export interface HousePartyShareFixture {
  fixtureOnly: true;
  district: string;
  stage: "first_ballot";
  units: "share_fraction";
  sourceStatus: "invented_fixture";
  context: string;
  estimate: PartyShareEstimate;
  observed: { status: "reported"; democraticVotes: number; republicanVotes: number } | { status: "unavailable"; reason: string };
  candidateWinner: { status: "unavailable"; reason: string };
}
export interface HousePartyShareView extends Omit<HousePartyShareFixture, "observed"> {
  observed: { status: "reported"; numerator: number; denominator: number; share: number } | { status: "unavailable"; reason: string };
}
const text = (v: unknown): v is string => typeof v === "string" && v.trim().length > 0;
const fraction = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v) && v >= 0 && v <= 1;
const unavailable = (reason: string): PartyShareEstimate => ({ status: "unavailable", reason });
/** Invalid numerical evidence is withheld; no missing value is replaced by zero. */
export function housePartyShareView(input: HousePartyShareFixture): HousePartyShareView {
  if (input.fixtureOnly !== true || input.sourceStatus !== "invented_fixture" || input.stage !== "first_ballot" || input.units !== "share_fraction" || !/^[A-Z]{2}-\d{2}$/.test(input.district) || !text(input.context) || input.candidateWinner?.status !== "unavailable" || !text(input.candidateWinner.reason))
    throw new TypeError("An explicitly invented first-ballot presentation fixture is required.");
  const e = input.estimate;
  let estimate: PartyShareEstimate = unavailable("Share estimate is missing or invalid.");
  if (e?.status === "unavailable" && text(e.reason)) estimate = { ...e };
  if (e?.status === "structural" && (e.value === 0 || e.value === 1) && e.absentParty === (e.value === 0 ? "D" : "R") && e.denominatorStatus === "positive_present_party" && text(e.reason)) estimate = { ...e };
  if (e?.status === "modeled" && e.intervalKind === "central" && fraction(e.mean) && Array.isArray(e.intervals) && e.intervals.length > 0 && e.intervals.every(i => i && fraction(i.level) && i.level > 0 && i.level < 1 && fraction(i.lower) && fraction(i.upper) && i.lower <= i.upper) && new Set(e.intervals.map(i => i.level)).size === e.intervals.length) {
    const intervals = [...e.intervals].sort((a, b) => a.level - b.level);
    if (intervals.every((i, index) => !index || (i.lower <= intervals[index - 1].lower && i.upper >= intervals[index - 1].upper))) estimate = { status: "modeled", intervalKind: "central", mean: e.mean, intervals: intervals.map(i => ({ ...i })) };
  }
  const o = input.observed;
  let observed: HousePartyShareView["observed"] = { status: "unavailable", reason: "Observed vote counts are missing or invalid." };
  if (o?.status === "unavailable" && text(o.reason)) observed = { ...o };
  if (o?.status === "reported" && Number.isSafeInteger(o.democraticVotes) && o.democraticVotes >= 0 && Number.isSafeInteger(o.republicanVotes) && o.republicanVotes >= 0 && Number.isSafeInteger(o.democraticVotes + o.republicanVotes)) {
    const denominator = o.democraticVotes + o.republicanVotes;
    observed = denominator === 0 ? { status: "unavailable", reason: "No Democratic or Republican votes: D + R is zero, so the observed share is undefined." } : { status: "reported", numerator: o.democraticVotes, denominator, share: o.democraticVotes / denominator };
  }
  return { ...input, estimate, observed, candidateWinner: { ...input.candidateWinner } };
}
