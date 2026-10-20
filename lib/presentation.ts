import type { RaceForecast } from "./site-types";
import type { ExplorerRace, ExplorerOutcome } from "./explorer-types";

export const STATE_NAMES: Record<string, string> = {
  AL: "Alabama",
  AK: "Alaska",
  AZ: "Arizona",
  AR: "Arkansas",
  CA: "California",
  CO: "Colorado",
  CT: "Connecticut",
  DE: "Delaware",
  DC: "District of Columbia",
  FL: "Florida",
  GA: "Georgia",
  HI: "Hawaii",
  ID: "Idaho",
  IL: "Illinois",
  IN: "Indiana",
  IA: "Iowa",
  KS: "Kansas",
  KY: "Kentucky",
  LA: "Louisiana",
  ME: "Maine",
  MD: "Maryland",
  MA: "Massachusetts",
  MI: "Michigan",
  MN: "Minnesota",
  MS: "Mississippi",
  MO: "Missouri",
  MT: "Montana",
  NE: "Nebraska",
  NV: "Nevada",
  NH: "New Hampshire",
  NJ: "New Jersey",
  NM: "New Mexico",
  NY: "New York",
  NC: "North Carolina",
  ND: "North Dakota",
  OH: "Ohio",
  OK: "Oklahoma",
  OR: "Oregon",
  PA: "Pennsylvania",
  RI: "Rhode Island",
  SC: "South Carolina",
  SD: "South Dakota",
  TN: "Tennessee",
  TX: "Texas",
  UT: "Utah",
  VT: "Vermont",
  VA: "Virginia",
  WA: "Washington",
  WV: "West Virginia",
  WI: "Wisconsin",
  WY: "Wyoming",
};
export const COLORS = {
  D: "#3e6683",
  R: "#a3553f",
  I: "#28776f",
  likelyD: "#92afc1",
  likelyR: "#d4a28c",
  likelyI: "#83b9ad",
  O: "#726e66",
  likelyO: "#beb9ad",
  competitive: "#7b658e",
  inactive: "#d8ddd4",
  unresolved: "#a3a998",
};
export const partyName = (party: string) =>
  (
    ({
      D: "Democratic",
      R: "Republican",
      I: "Independent",
      IND: "Independent",
      LIB: "Libertarian",
      GRE: "Green",
      W: "Write-in",
      O: "Other / residual",
      DFL: "DFL (Democratic)",
      DNL: "Democratic–NPL",
    }) as Record<string, string>
  )[party] || party;
export const partyColor = (party: string) =>
  COLORS[party as "D" | "R" | "I" | "O"] ||
  (party === "IND" ? COLORS.I : COLORS.O);
export const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
export const dateLabel = (value: string) =>
  new Intl.DateTimeFormat("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  }).format(new Date(value));

export const probability = (value: number) =>
  value > 0 && value < 0.001
    ? "<0.1%"
    : value < 1 && value > 0.999
      ? ">99.9%"
      : percent(value);

export const candidateShortName = (name: string) =>
  name.includes(",")
    ? name.split(",")[0].trim()
    : name.trim().split(/\s+/).at(-1) || name;

export function leadingOutcome(race: ExplorerRace): ExplorerOutcome {
  return [...race.outcomes].sort(
    (a, b) => b.winProbability - a.winProbability || a.id.localeCompare(b.id),
  )[0];
}

type DisplayRace = RaceForecast | ExplorerRace;
function leader(race: DisplayRace): { party: string; winProbability: number } {
  if ("outcomes" in race) {
    const outcome = leadingOutcome(race);
    return {
      party: visualParty(outcome),
      winProbability: outcome.winProbability,
    };
  }
  const first = race.firstWinProbability >= race.secondWinProbability;
  return {
    party: race.candidates[first ? 0 : 1].party,
    winProbability: first
      ? race.firstWinProbability
      : race.secondWinProbability,
  };
}
export function raceRating(race: DisplayRace): "Safe" | "Likely" | "Tossup" {
  const chance = leader(race).winProbability;
  return chance >= 0.98 ? "Safe" : chance >= 0.65 ? "Likely" : "Tossup";
}
export function raceColor(race: DisplayRace): string {
  const rating = raceRating(race);
  if (rating === "Tossup") return COLORS.competitive;
  const party = leader(race).party;
  return rating === "Safe"
    ? partyColor(party)
    : COLORS[
        `likely${party}` as "likelyD" | "likelyR" | "likelyI" | "likelyO"
      ] || COLORS.likelyI;
}
export function raceTextColor(race: DisplayRace): string {
  return raceRating(race) === "Tossup"
    ? COLORS.competitive
    : partyColor(leader(race).party);
}
export const competitiveOrder = (a: ExplorerRace, b: ExplorerRace) =>
  leadingOutcome(a).winProbability - leadingOutcome(b).winProbability ||
  a.id.localeCompare(b.id);
export const ppChange = (value: number) =>
  `${value >= 0 ? "+" : "−"}${Math.abs(value * 100).toFixed(1)} pp`;

export function visualParty(outcome: ExplorerOutcome): "D" | "R" | "I" | "O" {
  if (outcome.id === "D" || outcome.id === "R") return outcome.id;
  if (outcome.party === "D" || outcome.party === "R") return outcome.party;
  return outcome.party === "I" || outcome.party === "IND" ? "I" : "O";
}
export const outcomeColor = (outcome: ExplorerOutcome) =>
  partyColor(visualParty(outcome));
