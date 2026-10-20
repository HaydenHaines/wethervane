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
  R: "#b46c54",
  I: "#33877e",
  competitive: "#857096",
  inactive: "#d8ddd4",
  unresolved: "#a3a998",
};
export const partyName = (party: string) =>
  ({ D: "Democratic", R: "Republican", I: "Independent" })[party] || party;
export const partyColor = (party: string) =>
  COLORS[party as "D" | "R" | "I"] || COLORS.I;
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
